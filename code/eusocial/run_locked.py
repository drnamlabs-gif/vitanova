"""Ring, division and characteristic size with the role locked at birth."""
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
import locked as L                              # noqa: E402
from runfast import clusters, verdict           # noqa: E402


def one(a):
    lock, seed, steps = a
    w = L.Locked(seed=seed, MAX_BOIDS=600, lock_role=lock)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity")}
    t0 = time.time()
    for i in range(steps):
        if i % 25 == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i); rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    R["seed"] = seed
    R["wall"] = time.time() - t0
    v = verdict(R)
    return dict(lock=lock, seed=seed,
                survived=bool(R["n"].max() > 150),
                n_max=int(R["n"].max()), largest=int(R["largest"].max()),
                annular=round(float(R["annularity"].max()), 3),
                divided=bool(v["divided"]), parent=v["parent_size"],
                division_at=v["division_at"],
                soldier_frac=round(float(w.sol.mean()) if w.n else 0.0, 3),
                role_flips=w.role_flips, wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    steps = int(os.environ.get("STEPS", 9000))
    seeds = list(range(1, int(os.environ.get("NSEED", 16)) + 1))
    jobs = [(l, s, steps) for l in (True, False) for s in seeds]
    print(f"{len(jobs)} runs: role locked at birth vs recomputed every step",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "locked_runs.json"), "w"), indent=2)
    print(f"\n{'role':>10}{'survived':>11}{'annularity':>13}{'divided':>10}"
          f"{'parent size':>14}{'role flips':>12}")
    print("-" * 72)
    for l in (True, False):
        g = [r for r in rows if r["lock"] == l]
        s = [r for r in g if r["survived"]]
        if not s:
            print(f"{'locked' if l else 'plastic':>10}{f'0/{len(g)}':>11}")
            continue
        d = [r for r in s if r["divided"]]
        ps = [r["parent"] for r in d]
        print(f"{'locked' if l else 'plastic':>10}{f'{len(s)}/{len(g)}':>11}"
              f"{np.median([r['annular'] for r in s]):>13.2f}"
              f"{f'{len(d)}/{len(s)}':>10}"
              f"{(f'{np.mean(ps):.1f}' if ps else '—'):>14}"
              f"{np.mean([r['role_flips'] for r in s]):>12.0f}")
    print("\nreference: plastic roles give annularity ~0.9 and parent size ~17")
