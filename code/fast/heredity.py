"""Collective genealogy and heredity at the new level.

Reproduction alone is not an evolutionary transition. Godfrey-Smith's minimum
for a Darwinian individual is reproduction *and* heredity *and* variation in
fitness. VitaNova has been shown to reproduce at the collective level; this
measures whether the new level also inherits.

Collectives are tracked by membership overlap between samples, using the stable
boid ids the fast engine carries. A fission event is one collective whose members
end up in two or more collectives at the next sample, with the children together
retaining most of the parent. Each collective's life-history trait is its **size
at its own fission** -- measured when it divides, never at birth, or a child
would resemble its parent merely by being made of it.

Heredity is then the parent-offspring regression of that trait. Two controls:
  - a permutation null that reshuffles which child belongs to which parent;
  - a partial correlation removing the shared time trend, since collectives late
    in a run are larger for reasons that have nothing to do with ancestry.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import runfast
import vitafast

MIN_CLUSTER = 10
MIN_CHILD = 6
KEEP = 0.6


def centroid(w, uids):
    m = np.isin(w.uid, list(uids))
    return (float(w.x[m].mean()), float(w.y[m].mean())) if m.any() else (np.nan, np.nan)


def track(seed, steps=12000, every=25, max_boids=1500):
    """Run one simulation and return its collective genealogy."""
    w = vitafast.Fast(seed=seed, MAX_BOIDS=max_boids)
    prev = []                      # [(cid, set(uids), size)]
    coll = {}                      # cid -> record
    next_cid = 0
    fissions = []

    for i in range(steps):
        if i % every == 0:
            cs = runfast.clusters(w.x, w.y, min_size=MIN_CLUSTER)
            cur = [set(w.uid[c].tolist()) for c in cs]

            main, overlap = [], []
            for s in cur:
                best, bo = None, 0
                for cid, ps, _ in prev:
                    o = len(s & ps)
                    if o > bo:
                        best, bo = cid, o
                main.append(best)
                overlap.append(bo)

            groups = {}
            for j, p in enumerate(main):
                groups.setdefault(p, []).append(j)

            new_prev = []
            for p, js in groups.items():
                if p is None:
                    for j in js:
                        coll[next_cid] = dict(cid=next_cid, birth=i, parent=None,
                                              birth_size=len(cur[j]),
                                              max_size=len(cur[j]),
                                              fission=None, fission_size=None,
                                              n_children=0, fission_pop=None,
                                              fission_share=None,
                                              birth_xy=centroid(w, cur[j]),
                                              fission_xy=None)
                        new_prev.append((next_cid, cur[j], len(cur[j])))
                        next_cid += 1
                    continue

                psize = dict((c, s) for c, _, s in prev)[p]
                if len(js) == 1:
                    j = js[0]
                    coll[p]["max_size"] = max(coll[p]["max_size"], len(cur[j]))
                    new_prev.append((p, cur[j], len(cur[j])))
                else:
                    kids = [j for j in js if len(cur[j]) >= MIN_CHILD]
                    kept = sum(len(cur[j]) for j in kids)
                    if len(kids) >= 2 and kept >= KEEP * psize:
                        coll[p]["fission"] = i
                        coll[p]["fission_size"] = psize
                        coll[p]["fission_pop"] = int(w.n)
                        coll[p]["fission_share"] = psize / max(int(w.n), 1)
                        pset = dict((c, u) for c, u, _ in prev)[p]
                        coll[p]["fission_xy"] = centroid(w, pset)
                        coll[p]["n_children"] = len(kids)
                        fissions.append((seed, p, i, psize, len(kids)))
                        for j in kids:
                            coll[next_cid] = dict(
                                cid=next_cid, birth=i, parent=p,
                                birth_size=len(cur[j]), max_size=len(cur[j]),
                                fission=None, fission_size=None, n_children=0)
                            new_prev.append((next_cid, cur[j], len(cur[j])))
                            next_cid += 1
                    else:
                        j = max(js, key=lambda k: len(cur[k]))
                        coll[p]["max_size"] = max(coll[p]["max_size"], len(cur[j]))
                        new_prev.append((p, cur[j], len(cur[j])))
            prev = new_prev
        if w.n == 0:
            break
        w.step()
    return coll, fissions


def pairs_from(coll, seed):
    """(parent trait, child trait, child fission step) for complete pairs."""
    out = []
    for c in coll.values():
        if c["fission"] is None or c["parent"] is None:
            continue
        p = coll.get(c["parent"])
        if p is None or p["fission_size"] is None:
            continue
        out.append((seed, p["cid"], c["cid"], p["fission_size"],
                    c["fission_size"], c["fission"], c["fission"] - c["birth"],
                    c["fission_pop"], c["fission_share"], p["fission_share"],
                    *(c.get("birth_xy") or (np.nan, np.nan)),
                    *(c.get("fission_xy") or (np.nan, np.nan)),
                    *(p.get("fission_xy") or (np.nan, np.nan))))
    return out


def _job(a):
    seed, steps, every, mx = a
    try:
        coll, fis = track(seed, steps, every, mx)
        return seed, pairs_from(coll, seed), len(fis), len(coll)
    except Exception as e:
        return seed, [], 0, 0


def regression(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return dict(n=len(x), slope=np.nan, r=np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return dict(n=len(x), slope=float(slope), intercept=float(intercept),
                r=float(r))


def partial_corr(x, y, z):
    """Correlation of x and y after removing their linear dependence on z."""
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1]), regression(rx, ry)["slope"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1-60")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--max-boids", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--out", default="runs/heredity.json")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    print(f"{len(seeds)} seeds x {a.steps} steps on {a.workers} workers",
          flush=True)
    t0, all_pairs, n_fis, n_coll = time.time(), [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, (s, a.steps, a.every, a.max_boids))
                for s in seeds]
        for k, f in enumerate(as_completed(futs), 1):
            seed, pairs, nf, nc = f.result()
            all_pairs += pairs
            n_fis += nf
            n_coll += nc
            if k % 10 == 0:
                print(f"  {k}/{len(seeds)} seeds, {n_fis} fissions, "
                      f"{len(all_pairs)} parent-child pairs "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_coll} collectives, {n_fis} fission events, "
          f"{len(all_pairs)} complete parent-child pairs")
    if len(all_pairs) < 10:
        print("too few pairs for a regression")
        json.dump(all_pairs, open(a.out, "w"))
        return

    arr = np.array([(p[3], p[4], p[5], p[6], p[7], p[8], p[9])
                    for p in all_pairs], float)
    ps, cs, cstep, clife = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    cpop, cshare, pshare = arr[:, 4], arr[:, 5], arr[:, 6]
    print(f"\ncollective's share of the whole population at its own fission:")
    print(f"  parent {pshare.mean():.2f} +/- {pshare.std():.2f}, "
          f"child {cshare.mean():.2f} +/- {cshare.std():.2f}")
    print(f"  pairs where BOTH are under half the population: "
          f"{int(((pshare<0.5)&(cshare<0.5)).sum())} of {len(ps)}")
    print(f"\nparent size at fission: {ps.mean():.1f} +/- {ps.std():.1f} "
          f"(range {ps.min():.0f}-{ps.max():.0f})")
    print(f"child  size at fission: {cs.mean():.1f} +/- {cs.std():.1f}")

    reg = regression(ps, cs)
    print(f"\nPARENT-OFFSPRING REGRESSION on size at fission")
    print(f"  n = {reg['n']} pairs")
    print(f"  slope = {reg['slope']:+.3f}   r = {reg['r']:+.3f}")

    rng = np.random.default_rng(0)
    null = np.array([regression(ps, rng.permutation(cs))["slope"]
                     for _ in range(5000)])
    p_perm = float((np.abs(null) >= abs(reg["slope"])).mean())
    print(f"  permutation null slope: {null.mean():+.3f} +/- {null.std():.3f}")
    print(f"  p (two-sided) = {p_perm:.4f}")

    pr, pslope = partial_corr(ps, cs, cstep)
    print(f"\n  controlling for fission step (shared time trend):")
    print(f"    partial r = {pr:+.3f}, slope = {pslope:+.3f}")
    pr2, ps2 = partial_corr(ps, cs, cpop)
    print(f"  controlling for total population at the child's fission:")
    print(f"    partial r = {pr2:+.3f}, slope = {ps2:+.3f}")
    sub = (pshare < 0.5) & (cshare < 0.5)
    if sub.sum() >= 10:
        r3 = regression(ps[sub], cs[sub])
        print(f"  restricted to collectives under half the population "
              f"(n={int(sub.sum())}):")
        print(f"    slope = {r3['slope']:+.3f}, r = {r3['r']:+.3f}")
    else:
        print(f"  too few pairs where both are under half the population "
              f"({int(sub.sum())})")

    lr = regression(ps, clife)
    print(f"\n  parent size vs child lifetime: r = {lr['r']:+.3f} (n={lr['n']})")

    json.dump(dict(pairs=[list(p) for p in all_pairs],
                   slope=reg["slope"], r=reg["r"], p_perm=p_perm,
                   partial_r=pr, n_fissions=n_fis, n_collectives=n_coll),
              open(a.out, "w"), indent=2)
    print("\nsaved:", a.out)


if __name__ == "__main__":
    main()
