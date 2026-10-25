# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The phenomenon, twice: formation -> aggregation -> ring -> division in two
independent runs (seeds 7 and 25), each with its annularity/size trace.

Column 1 shows the whole arena early in the run; columns 2-4 zoom on the focal
collective (the one whose fission the census recorded). In the division panel
the two daughters are identified by membership overlap with the parent and
shaded separately. Column 5 is the run's annularity (largest collective, only
when it holds >= 15 members) and largest-collective size, with the four
snapshot moments marked -- the same convention as Figure 1g of the manuscript.
"""

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("fast", "reproduce"):
    sys.path.insert(0, os.path.join(_HERE, "..", "code", _p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

import runfast
import shape
import style
import vitafast

OUT = os.path.join(_HERE, "..", "paper", "figures", "fig_stages.pdf")
ZOOM = 170          # half-width of the zoom window, px
MIN_TRACE = 15      # annularity only when the largest collective holds this

# seed -> snapshot steps (early, aggregation, ring, division); the census
# recorded the root fission at step 2850 (seed 7) and 2700 (seed 25).
PLAN = {7: (200, 1500, 2750, 2900),
        25: (200, 1500, 2600, 2750)}


def capture(seed, steps, snaps):
    w = vitafast.Fast(seed=seed, MAX_BOIDS=1500)
    want = set(snaps)
    shots, trace = {}, {"step": [], "ann": [], "largest": []}
    for i in range(steps + 1):
        if i % 25 == 0:
            cs = runfast.clusters(w.x, w.y)
            big = cs[0] if cs else None
            ok = big is not None and len(big) >= MIN_TRACE
            d = shape.annularity(w.x[big], w.y[big]) if ok else None
            trace["step"].append(i)
            trace["ann"].append(d["annularity"] if d else np.nan)
            trace["largest"].append(len(big) if big is not None else 0)
        if i in want:
            shots[i] = (w.snapshot(), [c.copy() for c in
                                       runfast.clusters(w.x, w.y)])
        if w.n == 0:
            break
        w.step()
    return shots, trace


def hull(ax, x, y, color):
    if len(x) < 3:
        return
    pts = np.column_stack([x, y])
    try:
        h = ConvexHull(pts)
        ax.add_patch(Polygon(pts[h.vertices], closed=True, facecolor=color,
                             edgecolor="none", alpha=1.0, zorder=0))
    except Exception:
        pass


def scatter_agents(ax, s, size):
    wk = ~s["soldier"]
    ax.scatter(s["x"][wk], s["y"][wk], s=size, c=style.WORKER, lw=0, zorder=2)
    ax.scatter(s["x"][~wk], s["y"][~wk], s=size, c=style.SOLDIER, lw=0,
               zorder=2)
    ax.scatter(s["px"], s["py"], s=4 * size, c=style.PREDATOR, marker="v",
               lw=0, zorder=3)


def main():
    style.use()
    fig, axes = plt.subplots(2, 5, figsize=(10.2, 3.9),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25],
                                              wspace=0.08, hspace=0.3))
    names = ["dispersed (whole arena)", "aggregation", "ring", "division"]
    for row, (seed, snaps) in enumerate(PLAN.items()):
        print(f"seed {seed}: running {max(snaps)} steps...", flush=True)
        shots, trace = capture(seed, max(snaps), snaps)

        # focal collective: largest cluster at the ring step; its daughters at
        # the division step are the clusters sharing most of its members.
        s_ring, cs_ring = shots[snaps[2]]
        focal = cs_ring[0]
        focal_uids = set(s_ring["uid"][focal].tolist())
        s_div, cs_div = shots[snaps[3]]
        ov = sorted(cs_div, key=lambda c: -len(focal_uids &
                                               set(s_div["uid"][c].tolist())))
        daughters = ov[:2]

        for col, t in enumerate(snaps):
            ax = axes[row, col]
            snap, cs = shots[t]
            if col == 0:
                scatter_agents(ax, snap, 2.0)
                ax.set_xlim(0, vitafast.WIDTH)
                ax.set_ylim(0, vitafast.HEIGHT)
            else:
                if col == 3:
                    for dtr, c in zip(daughters, (style.HULL_A, style.HULL_B)):
                        hull(ax, snap["x"][dtr], snap["y"][dtr], c)
                    members = np.concatenate(daughters)
                elif col == 2:
                    hull(ax, snap["x"][focal], snap["y"][focal], style.HULL_A)
                    members = focal
                else:
                    members = cs[0] if cs else np.arange(len(snap["x"]))
                cx = float(np.mean(snap["x"][members]))
                cy = float(np.mean(snap["y"][members]))
                scatter_agents(ax, snap, 6.0)
                ax.set_xlim(cx - ZOOM, cx + ZOOM)
                ax.set_ylim(cy - ZOOM, cy + ZOOM)
            ax.set_aspect("equal")
            style.bare(ax)
            ax.set_title(f"{names[col]}  ·  step {t}", fontsize=7.5, pad=2)
            if col == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=9)

        ax = axes[row, 4]
        st = np.array(trace["step"], float)
        ax.plot(st, trace["ann"], color=style.ACC, lw=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("annularity", color=style.ACC)
        # twin axis matches the established Fig-1g convention of the paper
        ax2 = ax.twinx()
        ax2.plot(st, np.maximum(trace["largest"], 1), color=style.MUTED,
                 lw=1.0, ls="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("largest collective", color=style.MUTED)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        for t in snaps:
            ax.axvline(t, color=style.GRID, lw=0.7, zorder=0)
        if row == 1:
            ax.set_xlabel("step")
    for ext in (".pdf", ".png"):
        fig.savefig(OUT.replace(".pdf", ext))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
