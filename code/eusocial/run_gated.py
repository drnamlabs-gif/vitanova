"""Shape of the soldier-time distribution, with and without the gate."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "fast"))
import gated as G  # noqa: E402


def one(a):
    gate, seed, steps = a
    w = G.Gated(seed=seed, MAX_BOIDS=600, gate=gate)
    t0 = time.time()
    for _ in range(steps):
        if w.n == 0:
            break
        w.step()
    f = w.soldier_fractions()
    if w.n < 20 or len(f) < 50:
        return dict(gate=gate, seed=seed, survived=False)
    off = {}
    for _, p, _ in w.births:
        off[int(p)] = off.get(int(p), 0) + 1
    ids = [u for u in w.seen if w.seen[u] >= 100]
    frac = np.array([w.sol_time[u] / w.seen[u] for u in ids])
    kids = np.array([off.get(u, 0) for u in ids], float)
    hist, _ = np.histogram(f, bins=np.linspace(0, 1, 11))
    return dict(gate=gate, seed=seed, survived=True, n=int(w.n),
                births=len(w.births), tracked=len(f),
                mean=round(float(f.mean()), 3), sd=round(float(f.std()), 3),
                near_zero=round(float((f < 0.1).mean()), 3),
                near_one=round(float((f > 0.9).mean()), 3),
                middle=round(float(((f >= 0.1) & (f <= 0.9)).mean()), 3),
                corr=round(float(np.corrcoef(frac, kids)[0, 1])
                           if frac.std() > 0 and kids.std() > 0 else 0.0, 3),
                kids_of_specialists=round(float(kids[frac > 0.9].mean())
                                          if (frac > 0.9).any() else 0.0, 3),
                kids_of_workers=round(float(kids[frac < 0.1].mean())
                                      if (frac < 0.1).any() else 0.0, 3),
                hist=hist.tolist(), wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    steps = int(os.environ.get("STEPS", 8000))
    seeds = list(range(1, int(os.environ.get("NSEED", 12)) + 1))
    jobs = [(g, s, steps) for g in (True, False) for s in seeds]
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 8 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "gated_runs.json"), "w"), indent=2)
    print(f"\n{'gate':>6}{'survived':>10}{'mean':>8}{'<0.1':>8}{'0.1-0.9':>10}"
          f"{'>0.9':>8}{'corr':>8}{'kids: spec':>12}{'kids: wrk':>11}")
    print("-" * 82)
    for g in (True, False):
        s = [r for r in rows if r["gate"] == g and r["survived"]]
        if not s:
            print(f"{str(g):>6}{0:>10}")
            continue
        f_ = lambda k: np.mean([r[k] for r in s])
        print(f"{str(g):>6}{f'{len(s)}/{len(seeds)}':>10}{f_('mean'):>8.3f}"
              f"{f_('near_zero'):>8.2f}{f_('middle'):>10.2f}{f_('near_one'):>8.2f}"
              f"{f_('corr'):>8.2f}{f_('kids_of_specialists'):>12.2f}"
              f"{f_('kids_of_workers'):>11.2f}")
    print("\n'<0.1' almost never a soldier, '>0.9' almost always one.")
    print("A large middle would mean genuinely plastic roles; two peaks and an")
    print("empty middle means individuals are committed to one role for life.")
