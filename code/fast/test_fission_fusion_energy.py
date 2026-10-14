# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Does the fission/fusion balance respond to energy, as in mitochondrial networks?

In cells the balance is not fixed: high energy demand favours a fused,
connected network, while stress and damage favour fragmentation. If VitaNova's
collectives only resemble that picture, the ratio will be flat. If the
resemblance has mechanism behind it, the ratio will move with the energy
available to the agents.

Energy availability is varied through the worker gain, the single term that
sets how fast an agent accumulates.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vitafast                                   # noqa: E402
from runfast import clusters                      # noqa: E402

GAINS = [1.05, 1.1, 1.2, 1.4, 1.8]
SEEDS = [7, 11, 12, 13, 14, 15]
STEPS, SNAP, MIN_SIZE, SHARE = 9000, 50, 10, 0.25


def one(a):
    gain, seed = a
    w = vitafast.Fast(seed=seed, MAX_BOIDS=700, WORKER_ENERGY_GAIN=gain)
    prev, fis, fus = None, 0, 0
    pops, t0 = [], time.time()
    for i in range(STEPS):
        if i % SNAP == 0:
            cur = [set(w.uid[c].tolist())
                   for c in clusters(w.x, w.y, min_size=MIN_SIZE)]
            if prev and cur:
                for c in cur:
                    if sum(1 for p in prev if len(c & p) / len(c) >= SHARE) >= 2:
                        fus += 1
                for p in prev:
                    if sum(1 for c in cur if len(c & p) / len(p) >= SHARE) >= 2:
                        fis += 1
            prev = cur
            pops.append(w.n)
        if w.n == 0:
            break
        w.step()
    return dict(gain=gain, seed=seed, fission=fis, fusion=fus,
                ratio=(fus / fis if fis else None),
                n_max=int(max(pops) if pops else 0),
                mean_E=float(w.E.mean()) if w.n else 0.0,
                survived=bool(max(pops) > 150 if pops else False),
                wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    jobs = [(g, s) for g in GAINS for s in SEEDS]
    print(f"{len(jobs)} runs, worker energy gain {GAINS}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "runs", "fission_fusion_energy.json"),
                         "w"), indent=2)
    print(f"\n{'energy gain':>12}{'runs':>6}{'fissions':>10}{'fusions':>9}"
          f"{'fusion/fission':>16}{'peak n':>9}")
    print("-" * 64)
    xs, ys = [], []
    for g in GAINS:
        r = [x for x in rows if x["gain"] == g and x["survived"] and x["fission"]]
        if not r:
            print(f"{g:>12}{0:>6}")
            continue
        F = sum(x["fission"] for x in r); U = sum(x["fusion"] for x in r)
        print(f"{g:>12}{len(r):>6}{F:>10}{U:>9}{U/F:>16.2f}"
              f"{np.mean([x['n_max'] for x in r]):>9.0f}")
        for x in r:
            if x["ratio"] is not None:
                xs.append(g); ys.append(x["ratio"])
    if len(xs) > 4:
        from scipy import stats
        sp = stats.spearmanr(xs, ys)
        print(f"\nSpearman(energy gain, fusion/fission) = {sp.statistic:+.2f}, "
              f"p = {sp.pvalue:.4f}   (n={len(xs)} runs)")
        print("mitochondrial prediction: more energy -> more fusion, "
              "a connected network; less energy -> fragmentation")
