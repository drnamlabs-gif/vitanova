"""Does structure accumulate, and is it kept?

Two arms, both with mutation matched to the original network.

A  standard structural rate (0.02 per birth), run long enough for lineages to
   reach ~60 generations. Drift alone predicts about 1.2 hidden nodes.
B  raised structural rate (0.10 per birth) over the usual length. Drift alone
   predicts about 1.8 hidden nodes in ~18 generations.

Comparing what is observed against the drift expectation is the test. Below it
means structure is being removed, above it means it is being kept.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "reproduce"))
import shape                                # noqa: E402
import vita_neat as V                       # noqa: E402
from run_vita_neat import clusters          # noqa: E402

ARMS = {"A_long":  dict(steps=40000, add_node_rate=0.02, max_boids=400),
        "B_fast":  dict(steps=12000, add_node_rate=0.10, max_boids=400)}
SEEDS = [int(x) for x in os.environ.get('SEEDS', '1,3,5,7').split(',')]


def one(a):
    arm, seed, outdir = a
    p = ARMS[arm]
    cfg = V.neat_config(add_node_rate=p["add_node_rate"])
    w = V.VitaNeat(seed=seed, max_boids=p["max_boids"], cfg=cfg, init_hidden=0)
    rec = {k: [] for k in ("step", "n", "largest", "annularity", "hidden",
                           "conns", "topologies", "depth")}
    snaps, t0 = {}, time.time()
    pm = {}
    for i in range(p["steps"]):
        if i % 100 == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            h, c = w.complexity()
            for u, pa in zip(w.uid, w.par):
                pm[int(u)] = int(pa)
            dep = []
            for u in w.uid[:200]:
                u = int(u); k = 0
                while pm.get(u, -1) >= 0 and k < 500:
                    u = pm[u]; k += 1
                dep.append(k)
            rec["step"].append(i); rec["n"].append(w.n)
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["hidden"].append(h); rec["conns"].append(c)
            rec["topologies"].append(len({(b.n_hidden, b.n_conn) for b in w.brains}))
            rec["depth"].append(float(np.median(dep)) if dep else 0.0)
        if i % 500 == 0:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    if outdir and R["n"].max() > 150:
        pay = dict(R)
        for s, sn in snaps.items():
            for k, arr in sn.items():
                pay[f"snap_{s}_{k}"] = arr
        pay["n_boids"] = pay["n"]; pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, arm=arm, label=f"{arm} seed {seed}", detect_every=100)))
        np.savez_compressed(os.path.join(outdir, f"{arm}_seed{seed}.npz"), **pay)
    gens = float(R["depth"][-1]) if len(R["depth"]) else 0.0
    return dict(arm=arm, seed=seed, survived=bool(R["n"][-1] > 10),
                n_max=int(R["n"].max()), largest=int(R["largest"].max()),
                annular=round(float(R["annularity"].max()), 2),
                hidden_end=round(float(R["hidden"][-1]), 2),
                conns_end=round(float(R["conns"][-1]), 1),
                generations=round(gens, 1),
                drift_expectation=round(gens * p["add_node_rate"], 2),
                topologies=int(R["topologies"].max()),
                wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    outdir = os.path.join(HERE, "data", "matched_runs")
    os.makedirs(outdir, exist_ok=True)
    jobs = [(arm, s, outdir) for arm in ARMS for s in SEEDS]
    print(f"{len(jobs)} runs, mutation matched to the original network",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(one, j) for j in jobs]):
            rows.append(f.result()); r = rows[-1]
            print(f"  {r['arm']:<7} seed {r['seed']}: "
                  f"{'ok  ' if r['survived'] else 'DIED'}  "
                  f"gens {r['generations']:>5}  hidden {r['hidden_end']:>5.2f}  "
                  f"drift {r['drift_expectation']:>5.2f}  "
                  f"annular {r['annular']:.2f}  n {r['n_max']:>4}  {r['wall']}s",
                  flush=True)
    json.dump(rows, open(os.path.join(HERE, "data", "matched_sweep.json"), "w"),
              indent=2)
    print(f"\n{'arm':<9}{'runs':>6}{'generations':>13}{'hidden observed':>17}"
          f"{'drift expected':>16}{'ratio':>8}{'annular':>9}")
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm and r["survived"]]
        if not g:
            print(f"{arm:<9}{0:>6}"); continue
        h = np.mean([r["hidden_end"] for r in g])
        e = np.mean([r["drift_expectation"] for r in g])
        print(f"{arm:<9}{len(g):>6}{np.mean([r['generations'] for r in g]):>13.1f}"
              f"{h:>17.2f}{e:>16.2f}{h/max(e,1e-9):>8.2f}"
              f"{np.mean([r['annular'] for r in g]):>9.2f}")
    print("\nsaved data/matched_sweep.json")
