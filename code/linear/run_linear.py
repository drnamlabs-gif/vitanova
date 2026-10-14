# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Run the linear-tanh controller and read the coefficients that evolve.

The point of writing the controller as a 6 x 11 matrix is that the result can
be read. Each entry says how strongly one sensory quantity pushes one steering
rule, and a coefficient that is consistently large and of consistent sign
across independent replicates is a statement about what the evolved solution
does, not a weight in a black box."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../fast", "../reproduce"):
    sys.path.insert(0, os.path.join(HERE, d))
import shape                                    # noqa: E402
import linear_vita as L                         # noqa: E402
from runfast import clusters, verdict           # noqa: E402


def run(seed, steps=9000, every=25, max_boids=600, mask=None, outdir=None):
    w = L.LinearVita(seed=seed, MAX_BOIDS=max_boids, mask=mask)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity",
                           "hollowness", "mean_E", "soldier_frac")}
    snaps, t0 = {}, time.time()
    for i in range(steps):
        if i % every == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i); rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["hollowness"].append(d["hollowness"] if d else 0.0)
            rec["mean_E"].append(float(w.E.mean()) if w.n else 0.0)
            rec["soldier_frac"].append(float(w.sol.mean()) if w.n else 0.0)
        if i % 100 == 0 and outdir:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    R["wall"] = time.time() - t0
    R["seed"] = seed
    R["G_final"] = w.G.copy() if w.n else np.zeros((0, L.GENOME))
    if outdir and R["n"].max() > 150:
        pay = {k: v for k, v in R.items() if isinstance(v, np.ndarray)}
        for s, sn in snaps.items():
            for k, a in sn.items():
                pay[f"snap_{s}_{k}"] = a
        pay["n_boids"] = pay["n"]; pay["n_flocks"] = pay["n_clusters"]
        pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, label=f"linear-tanh seed {seed}", detect_every=every)))
        np.savez_compressed(os.path.join(outdir, f"lin_seed{seed}.npz"), **pay)
    return R


def _job(a):
    seed, steps, mx, outdir = a
    R = run(seed, steps=steps, max_boids=mx, outdir=outdir)
    v = verdict(R)
    G = R["G_final"]
    return dict(seed=seed, n_end=int(R["n"][-1]), n_max=int(R["n"].max()),
                largest=int(R["largest"].max()),
                annular=round(float(R["annularity"].max()), 2),
                divided=bool(v["divided"]), parent=v["parent_size"],
                division_at=v["division_at"],
                G_mean=(G.mean(0).tolist() if len(G) else None),
                wall=round(R["wall"], 1))


if __name__ == "__main__":
    outdir = os.path.join(HERE, "data"); os.makedirs(outdir, exist_ok=True)
    seeds = list(range(1, 13))
    print(f"{len(seeds)} runs, linear-tanh controller, 66 coefficients",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for f in as_completed([pool.submit(_job, (s, 9000, 600, outdir))
                               for s in seeds]):
            rows.append(f.result()); r = rows[-1]
            print(f"  seed {r['seed']:>3}: n {r['n_end']:>4}  largest "
                  f"{r['largest']:>4}  annular {r['annular']:.2f}  "
                  f"{'division at '+str(r['division_at'])+' parent '+str(r['parent']) if r['divided'] else 'no division'}"
                  f"  {r['wall']}s", flush=True)
    rows.sort(key=lambda r: r["seed"])
    json.dump(rows, open(os.path.join(HERE, "linear_runs.json"), "w"), indent=2)
    d = [r for r in rows if r["divided"]]
    ps = [r["parent"] for r in d]
    an = [r["annular"] for r in rows if r["n_max"] > 150]
    print(f"\ndivision in {len(d)}/{len(rows)} runs;  parent size "
          f"{np.mean(ps):.1f} +/- {np.std(ps):.1f}" if ps else "\nno divisions")
    print(f"peak annularity, surviving runs: median {np.median(an):.2f} (n={len(an)})")
    print("reference -- fixed 11-10-6 network: parent 17.0 +/- 2.2, annularity 0.92")
    print("saved linear_runs.json")
