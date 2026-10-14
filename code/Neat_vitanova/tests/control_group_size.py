# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Control: is the evolved milling state real, or just a small-group artefact?

Angular momentum is easier to achieve in a small group, and the evolved
controller settled on groups of about nine. This measures M against group size
in the *fixed-rule* model, so the evolved result can be compared at a matched
size rather than against the largest group of a 120-boid run."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flock import Flock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALI = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
COH = [0.25, 0.5, 1.0, 2.0, 4.0]
SIZES = [9, 15, 25, 40]


def cell(a):
    n, w_ali, w_coh, seed = a
    f = Flock(n=n, seed=seed, w_sep=2.0, w_ali=w_ali, w_coh=w_coh,
              r_sep=15.0, r_ali=25.0, r_coh=60.0, noise=0.03)
    P, M, D, G = [], [], [], []
    for i in range(500):
        f.step()
        if i >= 300 and i % 5 == 0:
            cs = f.clusters()
            if not cs:
                continue
            p, m = f.group_order(cs[0])
            P.append(p); M.append(m); D.append(f.nnd()); G.append(len(cs[0]))
    if not M:
        return n, w_ali, w_coh, 0.0, 0.0, 0.0, 0.0
    return (n, w_ali, w_coh, float(np.mean(P)), float(np.mean(M)),
            float(np.mean(D)), float(np.mean(G)))


if __name__ == "__main__":
    jobs = [(n, a, c, s) for n in SIZES for a in ALI for c in COH
            for s in (0, 1, 2)]
    res = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for r in pool.map(cell, jobs, chunksize=4):
            res.append(r)
    best = {}
    for n, a, c, P, M, D, G in res:
        if D < 4.0:                      # collapsed onto a point, not a mill
            continue
        key = n
        if key not in best or M > best[key][0]:
            best[key] = (M, P, D, G, a, c)
    print(f"{'flock n':>8}{'best M':>9}{'P there':>9}{'NND':>7}{'group':>7}"
          f"{'w_ali':>7}{'w_coh':>7}")
    out = {}
    for n in SIZES:
        if n not in best:
            print(f"{n:>8}{'none':>9}"); continue
        M, P, D, G, a, c = best[n]
        out[str(n)] = dict(M=round(M, 3), P=round(P, 3), nnd=round(D, 1),
                           group=round(G, 1), w_ali=a, w_coh=c)
        print(f"{n:>8}{M:>9.2f}{P:>9.2f}{D:>7.1f}{G:>7.0f}{a:>7}{c:>7}")
    nf = json.load(open(os.path.join(HERE, "data", "neat_flock.json")))
    ev = nf["milling"]
    print(f"\nevolved controller: M={ev['M']}  P={ev['P']}  "
          f"group={ev['group']}  NND={ev['nnd']}")
    ref = out.get("9") or out.get("15")
    if ref:
        print(f"fixed rules at a matched group size: M={ref['M']}  P={ref['P']}")
        print("evolved beats fixed at matched size:",
              ev["M"] > ref["M"] + 0.05)
    json.dump(dict(fixed_by_size=out, evolved=ev),
              open(os.path.join(HERE, "data", "control_group_size.json"), "w"),
              indent=2)
