# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Is the collective-level Darwinian population visible to selection?

The manuscript establishes, separately, that (a) a collective-level trait
(size at fission) is transmitted from parent to offspring and (b) reproductive
success varies sharply among collectives (most leave nothing, the rest two or
three). What it never measures is whether those two facts CONNECT: does the
heritable trait covary with reproductive success? That covariance is the
selection differential of the Price equation at the collective level, and it
decides between two different papers:

  cov ~ 0   a Darwinian population that selection cannot see -- heredity and
            fitness variation both present, selection differential absent;
  cov != 0  collective-level selection already operates, unannounced, and the
            manuscript's "none possible" needs correcting.

Measurements, all with the SEED as the statistical unit (collectives within a
run share a history and are not independent):

  1. Fitness distribution and the opportunity for selection I = var(w)/mean(w)^2.
  2. Selection differential S on size, for every collective (standardised
     within seed). Reported for birth_size and max_size separately, with the
     caveat that max_size -> division is partly mechanical (the detector needs
     two children of >= MIN_CHILD, so a dividing collective cannot be small).
  3. The compounding test, which is the decisive one: does the PARENT's
     heritable trait (its size at its own fission) predict the CHILD's
     reproductive success? Heredity (their slope 0.72) times this covariance is
     what a response to selection would run on. Permutation null shuffles
     children among parents within seed.
  4. Heritability of fitness itself: parent n_children vs child n_children.

Censoring: a collective born near the end of a run never had time to divide.
A collective is scoreable for fitness if it divided, or if it was born at
least T95 steps before the end, where T95 is the 95th percentile of observed
birth-to-fission times.

Usage (from the repository root):
  $env:PYTHONPATH="code\\fast;code\\reproduce"
  py analysis\\selection_differential.py --seeds 1-32 --steps 9000 --workers 6
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("fast", "reproduce"):
    sys.path.insert(0, os.path.join(_HERE, "..", "code", _p))

import heredity  # noqa: E402


# ---------------------------------------------------------------- census
def _job(a):
    seed, steps, every, mx = a
    try:
        coll, _ = heredity.track(seed, steps, every, mx)
        rows = []
        for c in coll.values():
            rows.append(dict(
                seed=seed, cid=c["cid"], parent=c["parent"], birth=c["birth"],
                birth_size=c["birth_size"], max_size=c["max_size"],
                fission=c["fission"], fission_size=c["fission_size"],
                n_children=c["n_children"]))
        return seed, rows, None
    except Exception as e:
        return seed, [], f"{type(e).__name__}: {e}"


def census(seeds, steps, every, max_boids, workers):
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, (s, steps, every, max_boids)) for s in seeds]
        for k, f in enumerate(as_completed(futs), 1):
            seed, r, err = f.result()
            if err:
                print(f"  seed {seed}: {err}", flush=True)
            rows += r
            if k % 4 == 0 or k == len(seeds):
                print(f"  {k}/{len(seeds)} seeds, {len(rows)} collectives "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------- helpers
def boot_ci(vals, n=10000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(vals, len(vals)).mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_seed(rows, fn, min_n=3):
    """fn(list of rows for one seed) -> value or None; returns per-seed array."""
    out = {}
    for r in rows:
        out.setdefault(r["seed"], []).append(r)
    vals = []
    for s, rr in sorted(out.items()):
        v = fn(rr)
        if v is not None:
            vals.append((s, v))
    return vals


def sel_diff(rr, trait, min_n=5):
    """Standardised selection differential within one seed."""
    z = np.array([r[trait] for r in rr], float)
    w = np.array([r["n_children"] for r in rr], float)
    if len(z) < min_n or z.std() == 0 or w.mean() == 0:
        return None
    return float(np.cov((z - z.mean()) / z.std(), w / w.mean())[0, 1])


def summarize(name, vals):
    v = np.array([x for _, x in vals], float)
    if len(v) < 3:
        print(f"  {name}: only {len(v)} usable seeds")
        return None
    lo, hi = boot_ci(v)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    pos = int((v > 0).sum())
    print(f"  {name}: mean {v.mean():+.3f}  [bootstrap 95% CI {lo:+.3f}, "
          f"{hi:+.3f}]  t={t:+.2f}  positive in {pos}/{len(v)} seeds")
    return dict(mean=float(v.mean()), ci=[lo, hi], t=float(t),
                n_seeds=len(v), n_positive=pos)


def perm_p(x, y, seeds, stat, n=5000, rng_seed=0):
    """Two-sided permutation p, shuffling y within seed."""
    rng = np.random.default_rng(rng_seed)
    obs = stat(x, y)
    null = np.empty(n)
    seeds = np.asarray(seeds)
    y = np.asarray(y, float)
    for i in range(n):
        yp = y.copy()
        for s in np.unique(seeds):
            m = seeds == s
            yp[m] = rng.permutation(yp[m])
        null[i] = stat(x, yp)
    return obs, float((np.abs(null) >= abs(obs)).mean()), null


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def center_within(v, seeds):
    v = np.asarray(v, float).copy()
    seeds = np.asarray(seeds)
    for s in np.unique(seeds):
        m = seeds == s
        v[m] -= v[m].mean()
    return v


# ---------------------------------------------------------------- analyses
def analyse(rows, steps, out_path):
    res = {}
    fis = [r for r in rows if r["fission"] is not None]
    waits = np.array([r["fission"] - r["birth"] for r in fis], float)
    t95 = float(np.percentile(waits, 95)) if len(waits) else 0.0
    print(f"\n{len(rows)} collectives, {len(fis)} divided; birth-to-fission "
          f"median {np.median(waits):.0f}, 95th pct {t95:.0f} steps")
    res["t95"] = t95

    score = [r for r in rows
             if r["fission"] is not None or r["birth"] <= steps - t95]
    w = np.array([r["n_children"] for r in score], float)
    print(f"\n[1] FITNESS among {len(score)} scoreable collectives")
    for k in sorted(set(w.astype(int))):
        print(f"      {k} children: {int((w == k).sum())}")
    I = float(w.var() / w.mean() ** 2) if w.mean() > 0 else np.nan
    print(f"      opportunity for selection I = {I:.2f}")
    res["I"] = I

    print(f"\n[2] SELECTION DIFFERENTIAL on size (standardised, per seed)")
    res["S"] = {}
    for trait in ("birth_size", "max_size"):
        vals = per_seed(score, lambda rr, t=trait: sel_diff(rr, t))
        res["S"][trait] = summarize(f"S' on {trait}", vals)
    print("      (max_size -> division is partly mechanical; birth_size is the"
          " cleaner trait)")

    print(f"\n[3] COMPOUNDING TEST: parent fission_size -> child fitness")
    kids = [r for r in score if r["parent"] is not None]
    by_id = {(r["seed"], r["cid"]): r for r in rows}
    xx, yy, yb, ss, bb = [], [], [], [], []
    for r in kids:
        p = by_id.get((r["seed"], r["parent"]))
        if p is None or p["fission_size"] is None:
            continue
        xx.append(p["fission_size"])
        yy.append(r["n_children"])
        yb.append(1.0 if r["fission"] is not None else 0.0)
        ss.append(r["seed"])
        bb.append(r["birth"])
    xx, yy, yb = np.array(xx, float), np.array(yy, float), np.array(yb, float)
    ss, bb = np.array(ss), np.array(bb, float)
    print(f"      {len(xx)} scoreable children with a measured parent;"
          f" {int(yb.sum())} of them divided")
    res["compounding"] = dict(n=len(xx), n_divided=int(yb.sum()))
    if len(xx) >= 20:
        xc = center_within(xx, ss)
        for label, y in (("child n_children", yy), ("child divided", yb)):
            yc = center_within(y, ss)
            b, p, _ = perm_p(xc, yc, ss, slope)
            r_ = float(np.corrcoef(xc, yc)[0, 1])
            print(f"      {label} ~ parent size (within-seed): "
                  f"slope {b:+.4f} per member, r {r_:+.3f}, perm p = {p:.4f}")
            res["compounding"][label] = dict(slope=b, r=r_, p=p)
        # time control: residualise both on child's birth step
        xr = xc - np.polyval(np.polyfit(bb, xc, 1), bb)
        yr = center_within(yy, ss)
        yr = yr - np.polyval(np.polyfit(bb, yr, 1), bb)
        b2, p2, _ = perm_p(xr, yr, ss, slope)
        print(f"      ... controlling child birth step: slope {b2:+.4f}, "
              f"perm p = {p2:.4f}")
        res["compounding"]["time_controlled"] = dict(slope=b2, p=p2)

    print(f"\n[4] HERITABILITY OF FITNESS: parent n_children -> child n_children")
    m = np.array([by_id[(s, int(c))]["n_children"] > 0
                  for s, c in zip(ss, [r["parent"] for r in kids
                                       if by_id.get((r["seed"], r["parent"]))
                                       and by_id[(r["seed"], r["parent"])]
                                       ["fission_size"] is not None])])
    px = np.array([by_id[(r["seed"], r["parent"])]["n_children"]
                   for r in kids
                   if by_id.get((r["seed"], r["parent"]))
                   and by_id[(r["seed"], r["parent"])]["fission_size"]
                   is not None], float)
    if len(px) >= 20 and px.std() > 0:
        pxc, yyc = center_within(px, ss), center_within(yy, ss)
        b, p, _ = perm_p(pxc, yyc, ss, slope)
        print(f"      slope {b:+.4f}, perm p = {p:.4f} "
              f"(parent n_children range {px.min():.0f}-{px.max():.0f})")
        res["fitness_heritability"] = dict(slope=b, p=p)
    else:
        print("      not enough variance in parent n_children")

    json.dump(res, open(out_path, "w"), indent=2)
    print(f"\nsaved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1-32")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--max-boids", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--census", default=os.path.join(_HERE, "..", "results",
                                                     "selection_census.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results",
                                                  "selection_differential.json"))
    ap.add_argument("--reuse", action="store_true",
                    help="skip simulation, analyse the stored census")
    a = ap.parse_args()

    if a.reuse and os.path.exists(a.census):
        rows = json.load(open(a.census))
        print(f"reusing stored census: {len(rows)} collectives")
    else:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
        print(f"{len(seeds)} seeds x {a.steps} steps on {a.workers} workers",
              flush=True)
        rows = census(seeds, a.steps, a.every, a.max_boids, a.workers)
        json.dump(rows, open(a.census, "w"))
        print(f"census saved: {a.census}")

    analyse(rows, a.steps, a.out)


if __name__ == "__main__":
    main()
