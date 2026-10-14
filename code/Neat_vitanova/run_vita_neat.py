# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Run NEAT-VitaNova and record what appears.

No fitness function is involved anywhere. The metrics below are observation
only: cluster structure, the shape statistic used elsewhere in this project, and
the complexity of the controllers that happen to survive."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reproduce"))
import shape                      # noqa: E402
import vita_neat as V             # noqa: E402

def clusters(x, y, link=V.RADIUS, min_size=10):
    n = len(x)
    if n == 0:
        return []
    adj = ((x[None, :] - x[:, None]) ** 2 +
           (y[None, :] - y[:, None]) ** 2) <= link * link
    seen = np.zeros(n, bool); out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop(); comp.append(j)
            nxt = np.flatnonzero(adj[j] & ~seen)
            seen[nxt] = True; stack.extend(nxt.tolist())
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def run(seed, steps=12000, every=25, snap_every=100, max_boids=600, out=None):
    w = V.VitaNeat(seed=seed, max_boids=max_boids)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity",
                           "hollowness", "core_frac", "mean_E", "soldier_frac",
                           "hidden", "conns", "topologies")}
    snaps = {}
    t0 = time.time()
    for i in range(steps):
        if i % every == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            dd = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            h, c = w.complexity()
            sig = {(b.n_hidden, b.n_conn) for b in w.brains}
            rec["step"].append(i); rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(dd["annularity"] if dd else 0.0)
            rec["hollowness"].append(dd["hollowness"] if dd else 0.0)
            rec["core_frac"].append(dd["core_frac"] if dd else 0.0)
            rec["mean_E"].append(float(w.E.mean()) if w.n else 0.0)
            rec["soldier_frac"].append(float(w.sol.mean()) if w.n else 0.0)
            rec["hidden"].append(h); rec["conns"].append(c)
            rec["topologies"].append(len(sig))
        if snap_every and i % snap_every == 0:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    res = {k: np.array(v) for k, v in rec.items()}
    res.update(seed=seed, wall=time.time() - t0, births=len(w.births),
               deaths=len(w.deaths), extinct=(w.n == 0))
    if out:
        pay = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
        for s, sn in snaps.items():
            for k, a in sn.items():
                pay[f"snap_{s}_{k}"] = a
        pay["n_boids"] = pay["n"]; pay["n_flocks"] = pay["n_clusters"]
        pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, label=f"NEAT-VitaNova seed {seed}", detect_every=every,
                 wall=res["wall"], births=res["births"], deaths=res["deaths"])))
        np.savez_compressed(out, **pay)
    return res


def _job(a):
    seed, steps, mx, outdir = a
    r = run(seed, steps=steps, max_boids=mx,
            out=os.path.join(outdir, f"neat_seed{seed}.npz") if outdir else None)
    return dict(seed=seed, n_end=int(r["n"][-1]), n_max=int(r["n"].max()),
                largest_max=int(r["largest"].max()),
                annular_max=round(float(r["annularity"].max()), 2),
                hidden_end=round(float(r["hidden"][-1]), 2),
                hidden_max=round(float(r["hidden"].max()), 2),
                conns_end=round(float(r["conns"][-1]), 1),
                topologies_max=int(r["topologies"].max()),
                births=r["births"], extinct=bool(r["extinct"]),
                wall=round(r["wall"], 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-6")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--max-boids", type=int, default=600)
    ap.add_argument("--outdir", default=os.path.join(HERE, "data", "neat_runs"))
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"{len(seeds)} runs x {a.steps} steps, no fitness function", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(_job, (s, a.steps, a.max_boids, a.outdir))
                               for s in seeds]):
            rows.append(f.result()); r = rows[-1]
            print(f"  seed {r['seed']:>3}: n {r['n_end']:>4} (max {r['n_max']:>4})  "
                  f"largest {r['largest_max']:>4}  annular {r['annular_max']:.2f}  "
                  f"hidden {r['hidden_end']:.2f} (max {r['hidden_max']:.2f})  "
                  f"topologies {r['topologies_max']:>3}  births {r['births']:>5}  "
                  f"{r['wall']}s", flush=True)
    rows.sort(key=lambda r: r["seed"])
    json.dump(rows, open(os.path.join(HERE, "data", "neat_vita_runs.json"), "w"),
              indent=2)
    print("\nsaved data/neat_vita_runs.json")
