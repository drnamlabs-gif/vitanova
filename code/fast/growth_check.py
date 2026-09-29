"""Distributional check: how fast does the population grow, original vs fast?

Trajectories cannot be compared after a few hundred steps, so compare the step
at which the population first reaches a threshold, across many seeds.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import runfast

THRESH = 300


def first_reach(step, n, thresh=THRESH):
    k = np.flatnonzero(np.asarray(n) >= thresh)
    return int(step[k[0]]) if len(k) else None


def fast_job(seed):
    r = runfast.run(seed, steps=12000, every=25, max_boids=THRESH + 20)
    return seed, first_reach(r["step"], r["n"]), int(np.max(r["n"]))


if __name__ == "__main__":
    orig = {}
    for p in sorted(glob.glob("../reproduce/runs/sweep/*.npz")):
        z = np.load(p)
        s = int(os.path.basename(p)[4:-4])
        orig[s] = first_reach(z["step"], z["n_boids"])
    seeds = sorted(orig)
    print(f"original: {len(seeds)} seeds available (capped at 500 boids)")

    fast = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(fast_job, s) for s in seeds]):
            s, step, nmax = f.result()
            fast[s] = step

    both = [s for s in seeds if orig[s] is not None and fast[s] is not None]
    o = np.array([orig[s] for s in both], float)
    f_ = np.array([fast[s] for s in both], float)
    print(f"\nseeds where both reached n={THRESH}: {len(both)} of {len(seeds)}")
    print(f"  original reached it at step: median {np.median(o):.0f}, "
          f"IQR {np.percentile(o,25):.0f}-{np.percentile(o,75):.0f}")
    print(f"  fast     reached it at step: median {np.median(f_):.0f}, "
          f"IQR {np.percentile(f_,25):.0f}-{np.percentile(f_,75):.0f}")
    print(f"  fast / original (median ratio): {np.median(f_/o):.2f}")
    print(f"  fast faster in {100*(f_<o).mean():.0f}% of seeds")
    no = sum(1 for s in seeds if orig[s] is None)
    nf = sum(1 for s in seeds if fast[s] is None)
    print(f"  never reached {THRESH}: original {no}, fast {nf}")
