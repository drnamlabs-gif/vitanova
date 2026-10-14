# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Ring and collective division with castes deleted."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fast"))
sys.path.insert(0, os.path.join(HERE, "..", "reproduce"))
import shape                                    # noqa: E402
import nocaste as NC                            # noqa: E402
import vitafast as VF                           # noqa: E402
from runfast import clusters, verdict           # noqa: E402


def one(a):
    castes, seed, steps, outdir = a
    w = (VF.Fast(seed=seed, MAX_BOIDS=600) if castes
         else NC.NoCaste(seed=seed, MAX_BOIDS=600))
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity")}
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
        if i % 100 == 0 and outdir:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    R["seed"] = seed
    R["wall"] = time.time() - t0
    v = verdict(R)
    if outdir and R["n"].max() > 150:
        pay = {k: val for k, val in R.items() if isinstance(val, np.ndarray)}
        for s, sn in snaps.items():
            for k2, arr in sn.items():
                pay[f"snap_{s}_{k2}"] = arr
        pay["n_boids"] = pay["n"]; pay["n_flocks"] = pay["n_clusters"]
        pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, detect_every=25,
                 label=("with castes" if castes else "castes removed")
                 + f", seed {seed}")))
        tag = "castes" if castes else "nocaste"
        np.savez_compressed(os.path.join(outdir, f"{tag}_seed{seed}.npz"), **pay)
    return dict(castes=castes, seed=seed,
                survived=bool(R["n"].max() > 150), n_max=int(R["n"].max()),
                largest=int(R["largest"].max()),
                annular=round(float(R["annularity"].max()), 3),
                divided=bool(v["divided"]), parent=v["parent_size"],
                division_at=v["division_at"], wall=round(R["wall"], 1))


if __name__ == "__main__":
    outdir = os.path.join(HERE, "data"); os.makedirs(outdir, exist_ok=True)
    steps = int(os.environ.get("STEPS", 9000))
    seeds = list(range(1, int(os.environ.get("NSEED", 16)) + 1))
    jobs = [(c, s, steps, outdir) for c in (False, True) for s in seeds]
    print(f"{len(jobs)} runs: castes removed vs the standard model", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "nocaste_runs.json"), "w"), indent=2)
    print(f"\n{'model':>16}{'survived':>11}{'annularity':>13}{'divided':>10}"
          f"{'parent size':>14}{'peak n':>9}")
    print("-" * 74)
    for c in (True, False):
        g = [r for r in rows if r["castes"] == c]
        s = [r for r in g if r["survived"]]
        if not s:
            print(f"{'with castes' if c else 'castes removed':>16}"
                  f"{f'0/{len(g)}':>11}")
            continue
        d = [r for r in s if r["divided"]]
        ps = [r["parent"] for r in d]
        print(f"{'with castes' if c else 'castes removed':>16}"
              f"{f'{len(s)}/{len(g)}':>11}"
              f"{np.median([r['annular'] for r in s]):>13.2f}"
              f"{f'{len(d)}/{len(s)}':>10}"
              f"{(f'{np.mean(ps):.1f}' if ps else '—'):>14}"
              f"{np.mean([r['n_max'] for r in s]):>9.0f}")
