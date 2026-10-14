# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Sweep the crowding coefficient: does it hold the collective in the ring window
without abolishing fission?"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "fast"), os.path.join(HERE, "..", "reproduce")):
    sys.path.insert(0, p)
import shape, crowded as CR, nocaste as NC, vitafast as VF   # noqa: E402
from runfast import clusters, verdict                        # noqa: E402

R_GUARD = 1.5 * VF.PERCEPTION_RADIUS      # a ring must fit this


def one(a):
    (beta, K), seed, steps, outdir = a
    w = (NC.NoCaste(seed=seed, MAX_BOIDS=1200) if beta == 0 else
         CR.Crowded(seed=seed, MAX_BOIDS=1200, crowd_beta=beta, crowd_k=K))
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity",
                           "ring", "big_r")}
    snaps, t0 = {}, time.time()
    for i in range(steps):
        if i % 25 == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i); rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["big_r"].append(d["r_max"] if d else 0.0)
            # a genuine ring: annular AND still inside the geometry that allows it
            rec["ring"].append(bool(d and d["annularity"] > 0.5
                                    and d["r_max"] < R_GUARD))
        if i % 100 == 0 and outdir:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    R["seed"] = seed; R["wall"] = time.time() - t0
    v = verdict(R)
    if outdir and R["largest"].max() > 20:
        pay = {k: val for k, val in R.items() if isinstance(val, np.ndarray)}
        for s, sn in snaps.items():
            for k2, arr in sn.items():
                pay[f"snap_{s}_{k2}"] = arr
        pay["n_boids"] = pay["n"]; pay["n_flocks"] = pay["n_clusters"]
        pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, detect_every=25,
                 label=f"crowding beta={beta} K={K}, seed {seed}")))
        np.savez_compressed(
            os.path.join(outdir, f"crowd_b{beta}k{K:g}_seed{seed}.npz"), **pay)
    lg = R["largest"]
    return dict(beta=beta, K=K, seed=seed,
                survived=bool(R["n"][-1] > 0),
                n_max=int(R["n"].max()), n_end=int(R["n"][-1]),
                largest_max=int(lg.max()),
                largest_med=float(np.median(lg[lg > 0])) if (lg > 0).any() else 0.0,
                in_window=float(((lg >= 20) & (lg <= 55)).mean()),
                ring_frac=float(R["ring"].mean()),
                annular=round(float(R["annularity"].max()), 3),
                divided=bool(v["divided"]), parent=v["parent_size"],
                max_clusters=int(R["n_clusters"].max()),
                wall=round(R["wall"], 1))


if __name__ == "__main__":
    betas = [tuple(float(v) for v in p.split(":")) for p in sys.argv[1].split(",")]
    seeds = [int(s) for s in sys.argv[2].split(",")]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 9000
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    jobs = [(b, s, steps, out) for b in betas for s in seeds]
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        fs = [ex.submit(one, j) for j in jobs]
        for k, f in enumerate(as_completed(fs), 1):
            res.append(f.result())
            print(f"  {k}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "crowd_runs.json"), "w"), indent=1)

    print(f"\n{'beta:K':>9} {'surv':>6} {'largest med':>12} {'largest max':>12} "
          f"{'% in window':>12} {'% ring':>8} {'divided':>8} {'parent':>7} {'max cl':>9}")
    print("-" * 94)
    for b in betas:
        rs = [r for r in res if (r["beta"], r["K"]) == b and r["survived"]]
        lab = f"{b[0]:g}:{b[1]:g}"
        if not rs:
            print(f"{lab:>9} {'0/'+str(len(seeds)):>6}"); continue
        dv = [r for r in rs if r["divided"]]
        print(f"{lab:>9} {str(len(rs))+'/'+str(len(seeds)):>6} "
              f"{np.median([r['largest_med'] for r in rs]):>12.0f} "
              f"{np.median([r['largest_max'] for r in rs]):>12.0f} "
              f"{np.mean([r['in_window'] for r in rs]):>11.0%} "
              f"{np.mean([r['ring_frac'] for r in rs]):>7.0%} "
              f"{str(len(dv))+'/'+str(len(rs)):>8} "
              f"{np.median([r['parent'] for r in dv]) if dv else 0:>7.0f} "
              f"{np.median([r['max_clusters'] for r in rs]):>9.0f}")
