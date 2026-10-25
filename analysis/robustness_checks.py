# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Robustness checks and recomputed core numbers from the stored 32-seed census.

Everything here reads results/selection_census.json (written by
selection_differential.py) -- no simulation.

  A. Recurrence: how many of the 32 seeds show the phenomenon at each stage
     (collectives formed / at least one division / a lineage >= 3 generations).
  B. Lineage depth distribution and the deepest chain, on THIS dataset.
  C. Heredity regression recomputed on THIS dataset (raw, within-seed centred,
     permutation null) so the paper's numbers come from one census.
  D. Threshold-artefact check: the fission detector needs children >= MIN_CHILD
     and clusters >= MIN_CLUSTER, so small collectives cannot REGISTER a
     division. Rerun the compounding test restricted to children born well
     above the detection floor (birth_size >= 15, >= 20): if parent size still
     predicts child division there, the covariance is not a detection artefact.
  E. Response check: S > 0 and h > 0 predict size at fission should rise along
     lineages. Regress fission_size on generation depth, within seed.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(_HERE, "..", "results", "selection_census.json")
OUT = os.path.join(_HERE, "..", "results", "robustness_checks.json")
STEPS = 12000


def center_within(v, seeds):
    v = np.asarray(v, float).copy()
    seeds = np.asarray(seeds)
    for s in np.unique(seeds):
        m = seeds == s
        v[m] -= v[m].mean()
    return v


def slope_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])


def perm_p(x, y, seeds, n=5000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs, _ = slope_r(x, y)
    seeds = np.asarray(seeds)
    y = np.asarray(y, float)
    null = np.empty(n)
    for i in range(n):
        yp = y.copy()
        for s in np.unique(seeds):
            m = seeds == s
            yp[m] = rng.permutation(yp[m])
        null[i], _ = slope_r(x, yp)
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main():
    rows = json.load(open(CENSUS))
    res = {}
    by_id = {(r["seed"], r["cid"]): r for r in rows}
    seeds = sorted(set(r["seed"] for r in rows))

    # ---- A. recurrence per seed
    print(f"[A] RECURRENCE across {len(seeds)} seeds, {len(rows)} collectives")
    formed = [s for s in seeds if any(r["seed"] == s for r in rows)]
    fiss_seeds = [s for s in seeds
                  if any(r["seed"] == s and r["fission"] is not None
                         for r in rows)]

    # generation depth per collective (root = 0)
    depth = {}
    def get_depth(key):
        if key in depth:
            return depth[key]
        r = by_id[key]
        d = 0 if r["parent"] is None else \
            get_depth((r["seed"], r["parent"])) + 1
        depth[key] = d
        return d
    for k in by_id:
        get_depth(k)
    max_depth_seed = {}
    for (s, c), d in depth.items():
        max_depth_seed[s] = max(max_depth_seed.get(s, 0), d)
    deep_seeds = [s for s, d in max_depth_seed.items() if d >= 3]
    n_fissions = sum(1 for r in rows if r["fission"] is not None)
    print(f"    collectives formed in {len(formed)}/{len(seeds)} seeds")
    print(f"    >=1 division in {len(fiss_seeds)}/{len(seeds)} seeds "
          f"({n_fissions} division events)")
    print(f"    lineage >=3 collective generations in "
          f"{len(deep_seeds)}/{len(seeds)} seeds")
    res["recurrence"] = dict(n_seeds=len(seeds), formed=len(formed),
                             divided=len(fiss_seeds), deep3=len(deep_seeds),
                             n_fissions=n_fissions, n_collectives=len(rows))

    # ---- B. lineage depth
    dmax = max(max_depth_seed.values())
    hist = np.bincount([max_depth_seed[s] for s in seeds])
    print(f"\n[B] LINEAGE DEPTH: deepest {dmax} collective generations; "
          f"per-seed max depth histogram {hist.tolist()}")
    res["lineage"] = dict(deepest=int(dmax),
                          per_seed_max=[int(max_depth_seed[s]) for s in seeds])

    # ---- C. heredity recomputed (both fissioned)
    px, cx, ss, cstep = [], [], [], []
    for r in rows:
        if r["fission"] is None or r["parent"] is None:
            continue
        p = by_id.get((r["seed"], r["parent"]))
        if p is None or p["fission_size"] is None:
            continue
        px.append(p["fission_size"]); cx.append(r["fission_size"])
        ss.append(r["seed"]); cstep.append(r["fission"])
    px, cx = np.array(px, float), np.array(cx, float)
    ss, cstep = np.array(ss), np.array(cstep, float)
    b_raw, r_raw = slope_r(px, cx)
    pxc, cxc = center_within(px, ss), center_within(cx, ss)
    b_c, p_c = perm_p(pxc, cxc, ss)
    # also partial out child's fission step
    xr = pxc - np.polyval(np.polyfit(cstep, pxc, 1), cstep)
    yr = cxc - np.polyval(np.polyfit(cstep, cxc, 1), cstep)
    b_t, p_t = perm_p(xr, yr, ss)
    print(f"\n[C] HEREDITY on this census: {len(px)} parent-child pairs")
    print(f"    raw slope {b_raw:+.2f} (r {r_raw:+.2f}); within-seed "
          f"{b_c:+.2f} (perm p {p_c:.4f}); + time control {b_t:+.2f} "
          f"(perm p {p_t:.4f})")
    res["heredity"] = dict(n=len(px), slope_raw=b_raw, r_raw=r_raw,
                           slope_centred=b_c, p_centred=p_c,
                           slope_time=b_t, p_time=p_t)

    # ---- D. threshold-artefact check
    waits = [r["fission"] - r["birth"] for r in rows if r["fission"] is not None]
    t95 = float(np.percentile(waits, 95))
    score = [r for r in rows
             if r["fission"] is not None or r["birth"] <= STEPS - t95]
    print(f"\n[D] DETECTION-FLOOR CHECK (compounding test, restricted)")
    res["threshold"] = {}
    for floor in (0, 15, 20, 25):
        xx, yy, sss = [], [], []
        for r in score:
            if r["parent"] is None or r["birth_size"] < floor:
                continue
            p = by_id.get((r["seed"], r["parent"]))
            if p is None or p["fission_size"] is None:
                continue
            xx.append(p["fission_size"])
            yy.append(1.0 if r["fission"] is not None else 0.0)
            sss.append(r["seed"])
        if len(xx) < 30:
            print(f"    child birth_size >= {floor}: n={len(xx)}, too few")
            continue
        xc, yc = center_within(xx, sss), center_within(yy, sss)
        b, p = perm_p(xc, yc, sss)
        _, r_ = slope_r(xc, yc)
        frac = np.mean(yy)
        print(f"    child birth_size >= {floor:>2}: n={len(xx):>4}, "
              f"P(divide)={frac:.2f}, slope {b:+.5f}, r {r_:+.3f}, "
              f"perm p {p:.4f}")
        res["threshold"][floor] = dict(n=len(xx), p_divide=float(frac),
                                       slope=b, r=r_, p=p)

    # ---- E. response: fission_size vs generation depth
    gx, gy, gs = [], [], []
    for r in rows:
        if r["fission_size"] is None:
            continue
        gx.append(depth[(r["seed"], r["cid"])])
        gy.append(r["fission_size"])
        gs.append(r["seed"])
    gx, gy = np.array(gx, float), np.array(gy, float)
    gxc, gyc = center_within(gx, gs), center_within(gy, gs)
    b_g, p_g = perm_p(gxc, gyc, np.array(gs))
    _, r_g = slope_r(gxc, gyc)
    print(f"\n[E] RESPONSE: size at fission vs generation depth "
          f"({len(gx)} fissioning collectives)")
    print(f"    within-seed slope {b_g:+.2f} members/generation, r {r_g:+.3f},"
          f" perm p {p_g:.4f}")
    by_gen = {}
    for d, v in zip(gx.astype(int), gy):
        by_gen.setdefault(d, []).append(v)
    for d in sorted(by_gen):
        if len(by_gen[d]) >= 5:
            print(f"      gen {d:>2}: n={len(by_gen[d]):>3}, "
                  f"median {np.median(by_gen[d]):.0f}")
    res["response"] = dict(n=len(gx), slope=b_g, r=r_g, p=p_g,
                           by_gen={int(d): [len(v), float(np.median(v))]
                                   for d, v in by_gen.items()})

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
