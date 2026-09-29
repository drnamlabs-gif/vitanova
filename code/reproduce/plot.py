"""Redraw VitaNova figures from stored snapshots, never from a re-run.

Reads the .npz written by `vitanova.py run --snapshot-every N` and produces
the two figure kinds the ALIFE 2024 paper uses: the simulation snapshot
(boids coloured by role, predators as red triangles) and the flock
count / average size time series.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 800, 600  # 160*BOID_SIZE, 120*BOID_SIZE from real_eti.py


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    steps = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("snap_")})
    return z, meta, steps


def densest_window(x, y, half=60):
    """Centre of the densest square window -- where the collective is."""
    best, cx, cy = -1, float(np.mean(x)), float(np.mean(y))
    for px, py in zip(x, y):
        n = int(((np.abs(x - px) <= half) & (np.abs(y - py) <= half)).sum())
        if n > best:
            best, cx, cy = n, float(px), float(py)
    return cx, cy


def draw_snapshot(z, step, ax, zoom=0):
    g = lambda k: z[f"snap_{step}_{k}"]
    x, y, dx, dy = g("x"), g("y"), g("dx"), g("dy")
    soldier = g("soldier")
    ax.set_facecolor("black")
    ax.quiver(x[~soldier], y[~soldier], dx[~soldier], dy[~soldier],
              color="#2ecc40", scale=40, width=0.004, headwidth=3,
              label=f"worker ({(~soldier).sum()})")
    ax.quiver(x[soldier], y[soldier], dx[soldier], dy[soldier],
              color="#0074d9", scale=40, width=0.004, headwidth=3,
              label=f"soldier ({soldier.sum()})")
    ax.quiver(g("px"), g("py"), g("pdx"), g("pdy"),
              color="#ff4136", scale=20, width=0.009, headwidth=3,
              label="predator")
    if zoom and len(x):
        cx, cy = densest_window(x, y, zoom)
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy + zoom, cy - zoom)
        inside = int(((np.abs(x - cx) <= zoom) & (np.abs(y - cy) <= zoom)).sum())
        ax.set_title(f"step {step}  ({inside} of {len(x)} boids)", fontsize=9)
    else:
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(HEIGHT, 0)      # pygame's y axis points down
        ax.set_title(f"step {step}  ({len(x)} boids)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def figure_snapshots(z, steps, out, cols=4, zoom=0):
    rows = int(np.ceil(len(steps) / cols))
    side = 2.8 if zoom else 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, side * rows))
    for ax, step in zip(np.ravel(axes), steps):
        draw_snapshot(z, step, ax, zoom=zoom)
    for ax in np.ravel(axes)[len(steps):]:
        ax.axis("off")
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    print("wrote", out)


def figure_dynamics(z, meta, out):
    step = z["step"]
    have = set(z.files)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(step, z["n_flocks"], lw=1, label="all flocks")
    if "big_flocks" in have:
        ax[0].plot(step, z["big_flocks"], lw=1.5, label="flocks >= 10")
    ax[0].set_xlabel("time step"); ax[0].set_ylabel("count")
    ax[0].set_title("Flock count"); ax[0].legend(fontsize=8)

    ax[1].plot(step, z["largest"], lw=1.2, color="#0074d9",
               label="largest flock")
    ax[1].plot(step, z["n_boids"], lw=1, color="#aaaaaa", label="population")
    ax[1].set_xlabel("time step"); ax[1].set_ylabel("members")
    ax[1].set_title("Largest flock size vs population"); ax[1].legend(fontsize=8)

    ax[2].plot(step, z["ring_score"], lw=1.2, color="#b10dc9", label="shape score")
    for key, lab in (("coverage", "angular coverage"), ("hollow", "hollowness"),
                     ("hollowness", "hollowness"), ("cv", "radial CV"),
                     ("core_frac", "core fraction")):
        if key in have:
            ax[2].plot(step, z[key], lw=0.8, alpha=.6, label=lab)
    ax[2].set_xlabel("time step"); ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Ring shape of the largest flock"); ax[2].legend(fontsize=8)

    fig.suptitle(meta.get("label") or f"seed {meta['seed']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--zoom", type=int, default=0,
                    help="half-width in pixels to crop around the densest cluster")
    ap.add_argument("--snapshots", default=None,
                    help="comma-separated steps; default: 8 spread over the run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z, meta, available = load(args.npz)
    tag = os.path.splitext(os.path.basename(args.npz))[0]

    figure_dynamics(z, meta, os.path.join(args.outdir, f"{tag}_dynamics.png"))

    if available:
        if args.snapshots:
            want = [int(s) for s in args.snapshots.split(",")]
            steps = [min(available, key=lambda a: abs(a - w)) for w in want]
        else:
            steps = [available[i] for i in
                     np.linspace(0, len(available) - 1, min(8, len(available))).astype(int)]
        suffix = f"_zoom{args.zoom}" if args.zoom else ""
        figure_snapshots(z, steps,
                         os.path.join(args.outdir, f"{tag}_snapshots{suffix}.png"),
                         zoom=args.zoom)
    else:
        print("no snapshots stored in", args.npz)


if __name__ == "__main__":
    main()
