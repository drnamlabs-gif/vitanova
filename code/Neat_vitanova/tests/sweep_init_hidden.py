# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""What can NEAT do once it starts with hidden nodes?

With a minimal start, most populations never reproduce at all: a boid needs the
grouping bonus to reach the reproductive threshold within its lifespan, and a
linear controller with random weights does not group. Seeding hidden nodes gives
the architecture the fixed network already has. Sweeping the number seeded, up
to the original's ten, separates "NEAT adds nothing" from "NEAT never got
started".
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reproduce"))
import shape                                   # noqa: E402
import vita_neat as V                          # noqa: E402
from run_vita_neat import clusters             # noqa: E402

HIDDEN = [0, 1, 2, 5, 10]
SEEDS = [1, 2, 3, 4, 5, 6]
STEPS, MAXB, EVERY = 10000, 500, 25


def one(a):
    H, seed, outdir = a
    t0 = time.time()
    w = V.VitaNeat(seed=seed, max_boids=MAXB, init_hidden=H)
    rec = {k: [] for k in ("step", "n", "largest", "annularity", "hidden",
                           "conns", "topologies", "soldier")}
    snaps = {}
    for i in range(STEPS):
        if i % EVERY == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            h, c = w.complexity()
            rec["step"].append(i); rec["n"].append(w.n)
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["hidden"].append(h); rec["conns"].append(c)
            rec["topologies"].append(len({(b.n_hidden, b.n_conn) for b in w.brains}))
            rec["soldier"].append(float(w.sol.mean()) if w.n else 0.0)
        if i % 200 == 0 and outdir:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    if outdir and R["n"].max() > 150:
        pay = {k: v for k, v in R.items()}
        for s, sn in snaps.items():
            for k, arr in sn.items():
                pay[f"snap_{s}_{k}"] = arr
        pay["n_boids"] = pay["n"]; pay["n_flocks"] = pay["largest"] * 0
        pay["ring_score"] = pay["annularity"]
        pay["meta_json"] = np.array(json.dumps(
            dict(seed=seed, label=f"NEAT init_hidden={H} seed {seed}",
                 detect_every=EVERY, init_hidden=H)))
        np.savez_compressed(os.path.join(outdir, f"h{H}_seed{seed}.npz"), **pay)
    return dict(H=H, seed=seed, n_end=int(R["n"][-1]), n_max=int(R["n"].max()),
                largest_max=int(R["largest"].max()),
                annular_max=round(float(R["annularity"].max()), 2),
                hidden_start=round(float(R["hidden"][0]), 2),
                hidden_end=round(float(R["hidden"][-1]), 2),
                conns_end=round(float(R["conns"][-1]), 1),
                topologies_max=int(R["topologies"].max()),
                births=len(w.births),
                survived=bool(R["n"][-1] > 10),
                wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    outdir = os.path.join(HERE, "data", "hidden_runs")
    os.makedirs(outdir, exist_ok=True)
    jobs = [(H, s, outdir) for H in HIDDEN for s in SEEDS]
    print(f"{len(jobs)} runs: hidden {HIDDEN} x {len(SEEDS)} seeds, "
          f"{STEPS} steps, no fitness function", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(one, j) for j in jobs]):
            rows.append(f.result()); r = rows[-1]
            print(f"  H={r['H']:>2} seed {r['seed']}: "
                  f"{'survived' if r['survived'] else 'DIED   '}  "
                  f"n {r['n_end']:>3}  largest {r['largest_max']:>3}  "
                  f"annular {r['annular_max']:.2f}  "
                  f"hidden {r['hidden_start']:.1f}->{r['hidden_end']:.2f}  "
                  f"topo {r['topologies_max']:>2}  births {r['births']:>4}  "
                  f"{r['wall']}s", flush=True)
    rows.sort(key=lambda r: (r["H"], r["seed"]))
    json.dump(rows, open(os.path.join(HERE, "data", "init_hidden_sweep.json"),
                         "w"), indent=2)
    print(f"\n{'init hidden':>12}{'survived':>10}{'peak n':>9}{'largest':>9}"
          f"{'annular':>9}{'hidden end':>12}{'topologies':>12}{'births':>9}")
    for H in HIDDEN:
        g = [r for r in rows if r["H"] == H]
        sv = [r for r in g if r["survived"]]
        f = lambda k: np.mean([r[k] for r in (sv or g)])
        print(f"{H:>12}{len(sv)}/{len(g):<9}{f('n_max'):>9.0f}"
              f"{f('largest_max'):>9.0f}{f('annular_max'):>9.2f}"
              f"{f('hidden_end'):>12.2f}{f('topologies_max'):>12.1f}"
              f"{f('births'):>9.0f}")
    print("\nsaved data/init_hidden_sweep.json")
