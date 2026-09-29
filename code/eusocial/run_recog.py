"""Does nestmate recognition stop collectives drifting back together?

Fission and fusion are counted directly, by tracking which uids belong to which
cluster from one sample to the next: two clusters whose members end up in one
cluster is a fusion, one cluster whose members end up in two is a fission.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "fast"), os.path.join(HERE, "..", "reproduce")):
    sys.path.insert(0, p)
import shape, crowded as CR, recog as RG, vitafast as VF     # noqa: E402
from runfast import clusters                                  # noqa: E402

MIN = 10
R_GUARD = 1.5 * VF.PERCEPTION_RADIUS


def match(prev, cur):
    """(fusions, fissions) between two lists of uid-sets"""
    fu = fi = 0
    for c in cur:
        hits = sum(1 for p in prev if len(p & c) >= 0.5 * len(p))
        if hits >= 2:
            fu += 1
    for p in prev:
        hits = sum(1 for c in cur if len(p & c) >= 0.5 * len(c))
        if hits >= 2:
            fi += 1
    return fu, fi


def one(a):
    mode, gamma, d0, seed, steps = a
    w = (CR.Crowded(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
         if gamma == 0 else
         RG.Recog(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26,
                  mode=mode, gamma=gamma, recog_d0=d0))
    prev, fus, fis, ring, win, big = None, 0, 0, [], [], []
    misfire = []
    t0 = time.time()
    for i in range(steps):
        if w.n == 0:
            break
        if i % 25 == 0:
            cs = clusters(w.x, w.y, min_size=MIN)
            sets = [set(w.uid[c].tolist()) for c in cs]
            if prev is not None and i >= 500:
                f1, f2 = match(prev, sets)
                fus += f1; fis += f2
            prev = sets
            if len(cs) == 1 and hasattr(w, "_stranger") and len(w._stranger):
                # only nestmates in contact here, so any push is a misfire
                misfire.append(float(np.mean(w._stranger)))
            if cs:
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                big.append(len(cs[0]))
                win.append(20 <= len(cs[0]) <= 55)
                ring.append(bool(d and d["annularity"] > 0.5
                                 and d["r_max"] < R_GUARD))
            else:
                big.append(0); win.append(False); ring.append(False)
        w.step()
    return dict(mode=mode, gamma=gamma, d0=d0, seed=seed, fusions=fus, fissions=fis,
                ring_frac=float(np.mean(ring)), in_window=float(np.mean(win)),
                largest_med=float(np.median([b for b in big if b])) if any(big) else 0.0,
                largest_max=int(max(big)) if big else 0,
                n_end=int(w.n), wall=round(time.time() - t0, 1),
                misfire=float(np.mean(misfire)) if misfire else 0.0)


if __name__ == "__main__":
    cfgs = []
    for spec in sys.argv[1].split(","):
        m, g, d0 = spec.split(":")
        cfgs.append((m, float(g), float(d0)))
    seeds = [int(s) for s in sys.argv[2].split(",")]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 9000
    jobs = [(m, g, d0, s, steps) for m, g, d0 in cfgs for s in seeds]
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        fs = [ex.submit(one, j) for j in jobs]
        for k, f in enumerate(as_completed(fs), 1):
            res.append(f.result())
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "recog_runs.json"), "w"), indent=1)

    print(f"\n{'mode':>8} {'gamma':>6} {'d0':>5} {'fusions':>9} {'fissions':>9} {'fu/fi':>7} "
          f"{'% ring':>8} {'% window':>9} {'largest':>8} {'misfire':>8}")
    print("-" * 78)
    for m, g, d0 in cfgs:
        rs = [r for r in res if r["mode"] == m and r["gamma"] == g
              and r["d0"] == d0]
        lab = "none" if g == 0 else m
        print(f"{lab:>8} {g:>6.1f} {d0:>5.2f} "
              f"{np.median([r['fusions'] for r in rs]):>9.1f} "
              f"{np.median([r['fissions'] for r in rs]):>9.1f} "
              f"{(sum(r['fusions'] for r in rs)/max(sum(r['fissions'] for r in rs),1)):>7.2f} "
              f"{np.mean([r['ring_frac'] for r in rs]):>7.0%} "
              f"{np.mean([r['in_window'] for r in rs]):>8.0%} "
              f"{np.median([r['largest_med'] for r in rs]):>8.0f} "
              f"{np.mean([r['misfire'] for r in rs]):>8.4f}")
