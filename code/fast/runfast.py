# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Run the vectorised model, detect ring formation and division, sweep seeds.

Division here needs all of: a sustained single cluster of at least MIN_PARENT
that is growing, then at least two sustained clusters of at least MIN_CHILD,
retaining at least KEEP of the parent's membership. That last condition is what
separates a division from a structure falling apart.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "reproduce"))
import shape  # noqa: E402
import vitafast  # noqa: E402

MIN_PARENT, MIN_CHILD, HOLD, KEEP = 15, 6, 3, 0.7
LINK = vitafast.FLOCK_THRESHOLD


def clusters(x, y, link=LINK, min_size=10):
    n = len(x)
    if n == 0:
        return []
    adj = ((x[None, :] - x[:, None]) ** 2 +
           (y[None, :] - y[:, None]) ** 2) <= link * link
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            nxt = np.flatnonzero(adj[j] & ~seen)
            seen[nxt] = True
            stack.extend(nxt.tolist())
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def run(seed, steps=8000, every=25, snap_every=0, max_boids=5000, **ov):
    w = vitafast.Fast(seed=seed, MAX_BOIDS=max_boids, **ov)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity",
                           "hollowness", "core_frac", "mean_E", "soldier_frac")}
    snaps = {}
    t0 = time.time()
    for i in range(steps):
        if i % every == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i)
            rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["hollowness"].append(d["hollowness"] if d else 0.0)
            rec["core_frac"].append(d["core_frac"] if d else 0.0)
            rec["mean_E"].append(float(w.E.mean()) if w.n else 0.0)
            rec["soldier_frac"].append(float(w.sol.mean()) if w.n else 0.0)
        if snap_every and i % snap_every == 0:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()
    out = {k: np.array(v) for k, v in rec.items()}
    out["wall"] = time.time() - t0
    out["seed"] = seed
    out["snapshots"] = snaps
    out["births"] = w.births
    return out


def verdict(r):
    step, big, largest, n = (r["step"], np.array(r["n_clusters"]),
                             np.array(r["largest"]), np.array(r["n"]))

    def sustained(mask):
        keep, run_len = [], 0
        for i, v in enumerate(mask):
            run_len = run_len + 1 if v else 0
            if run_len >= HOLD:
                keep.append(i)
        return keep

    single = sustained((big == 1) & (largest >= MIN_PARENT))
    div_at = parent = children = None
    for i in single:
        j = max(0, i - 2 * HOLD)
        if largest[i] <= largest[j]:
            continue
        after = sustained(big[i:] >= 2)
        if not after:
            continue
        k = i + after[0]
        if n[k] >= KEEP * largest[i] and largest[k] >= MIN_CHILD:
            div_at, parent, children = int(step[k]), int(largest[i]), int(n[k])
            break
    return dict(seed=r["seed"], n_max=int(n.max()), n_end=int(n[-1]),
                largest_max=int(largest.max()),
                annular_max=round(float(r["annularity"].max()), 2),
                division_at=div_at, parent_size=parent, children_total=children,
                divided=div_at is not None, wall=round(r["wall"], 1))


def _job(a):
    seed, steps, mx = a
    try:
        return verdict(run(seed, steps=steps, max_boids=mx))
    except Exception as e:
        return dict(seed=seed, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--max-boids", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--report", default="runs/fast_sweep.json")
    a = ap.parse_args()
    lo, hi = (a.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)

    print(f"{len(seeds)} seeds x {a.steps} steps on {a.workers} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_job, (s, a.steps, a.max_boids))
                               for s in seeds]):
            rows.append(f.result())
            r = rows[-1]
            tag = "DIVIDED" if r.get("divided") else ("err" if "error" in r else "-")
            print(f"  seed {r['seed']:>4} {tag:>8}  "
                  f"parent={r.get('parent_size')} at {r.get('division_at')}  "
                  f"annular={r.get('annular_max')}  "
                  f"n_max={r.get('n_max')}  {r.get('wall')}s", flush=True)
    rows.sort(key=lambda r: r["seed"])
    div = [r for r in rows if r.get("divided")]
    sizes = np.array([r["parent_size"] for r in div]) if div else np.array([])
    print(f"\ndivided: {len(div)}/{len(rows)} seeds in {time.time()-t0:.0f}s total")
    if len(sizes):
        print(f"parent size at division: median {np.median(sizes):.0f}, "
              f"mean {sizes.mean():.1f} +/- {sizes.std():.1f}, "
              f"range {sizes.min()}-{sizes.max()}")
        print("  original engine gave 53, 53, 58 (seeds 2024, 19, 16)")
    json.dump(rows, open(a.report, "w"), indent=2)
    print("report:", a.report)


if __name__ == "__main__":
    main()
