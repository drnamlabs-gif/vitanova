"""Run the minimal DEB world and record structure, without any group-level rule
entering the simulation. Clustering is measurement only -- it never feeds back.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "reproduce"))
import shape  # noqa: E402

import deb  # noqa: E402


def clusters(x, y, link=25.0, min_size=10):
    """Single-linkage components, computed only for reporting."""
    n = len(x)
    if n == 0:
        return []
    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    adj = d2 <= link * link
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        frontier, comp = [i], []
        seen[i] = True
        while frontier:
            j = frontier.pop()
            comp.append(j)
            nxt = np.flatnonzero(adj[j] & ~seen)
            seen[nxt] = True
            frontier.extend(nxt.tolist())
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return out


def run(seed, steps, every=25, snap_every=100, out=None, verbose=True,
        **overrides):
    w = deb.World(seed=seed, **overrides)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity",
                           "hollowness", "core_frac", "mean_V", "mean_R",
                           "resource", "births", "deaths")}
    snaps = {}
    t0 = time.time()
    for i in range(steps):
        if i % every == 0:
            cs = clusters(w.x, w.y)
            big = max(cs, key=len) if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i)
            rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
            rec["hollowness"].append(d["hollowness"] if d else 0.0)
            rec["core_frac"].append(d["core_frac"] if d else 0.0)
            rec["mean_V"].append(float(w.V.mean()) if w.n else 0.0)
            rec["mean_R"].append(float(w.R.mean()) if w.n else 0.0)
            rec["resource"].append(float(w.res.mean()))
            rec["births"].append(len(w.births))
            rec["deaths"].append(len(w.deaths))
            if verbose and i % (every * 20) == 0:
                print(f"  step {i:5d} n={w.n:4d} clusters={len(cs):2d} "
                      f"largest={rec['largest'][-1]:4d} "
                      f"annular={rec['annularity'][-1]:.2f} "
                      f"V={rec['mean_V'][-1]:.2f} res={rec['resource'][-1]:.3f}",
                      flush=True)
        if snap_every and i % snap_every == 0:
            snaps[i] = w.snapshot()
        if w.n == 0:
            break
        w.step()

    res = {k: np.array(v) for k, v in rec.items()}
    res["wall_seconds"] = time.time() - t0
    res["seed"] = seed
    res["overrides"] = overrides
    if out:
        payload = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
        for s, snap in snaps.items():
            for k, a in snap.items():
                payload[f"snap_{s}_{k}"] = a
        payload["meta_json"] = np.array(json.dumps(
            {"seed": seed, "label": f"DEB seed {seed} {overrides or ''}".strip(),
             "detect_every": every, "wall_seconds": res["wall_seconds"],
             "overrides": {k: float(v) for k, v in overrides.items()}}))
        # names the shared plot/viewer tools expect
        payload["n_boids"] = payload.pop("n")
        payload["n_flocks"] = payload.pop("n_clusters")
        payload["ring_score"] = payload["annularity"]
        np.savez_compressed(out, **payload)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[],
                    metavar="NAME=VALUE", help="override a Params field")
    args = ap.parse_args()

    ov = {}
    for item in args.set:
        k, v = item.split("=", 1)
        ov[k] = float(v)
    r = run(args.seed, args.steps, args.every, args.snapshot_every,
            args.out, **ov)
    i = int(np.argmax(r["annularity"])) if len(r["annularity"]) else 0
    print(f"\nwall {r['wall_seconds']:.0f}s   final n={r['n'][-1]}   "
          f"peak annularity={r['annularity'].max():.2f} at step {r['step'][i]} "
          f"(largest cluster {r['largest'][i]})")
    print(f"max clusters={r['n_clusters'].max()}  "
          f"max largest={r['largest'].max()}  "
          f"peak hollowness={r['hollowness'].max():.2f}")
    if args.out:
        print("saved:", args.out)


if __name__ == "__main__":
    main()
