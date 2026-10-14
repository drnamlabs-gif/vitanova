# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Homepage animation: one full life cycle of a collective (seed 7).

Left panel: the whole arena. Right panel: a camera that tracks the focal
collective through dispersal -> aggregation -> ring -> division (the census
recorded the root fission of seed 7 at step 2850). After the division the two
daughter collectives are shaded separately, identified by membership overlap
with the ring-stage parent, the same rule as fig_stages.py.

Output: media/vitanova_division.gif (dark theme, for drnamlabs.com).
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
from PIL import Image

import runfast
import vitafast

SEED = 7
STEPS = 3150          # fission at 2850; run a little past it
SKIP = 12             # capture every SKIP steps
RING_STEP = 2750      # focal collective defined here (largest cluster)
DIV_STEP = 2850       # census fission step
ZOOM = 185            # half-width of the tracking camera, px
HOLD_END = 18         # extra frames frozen on the final division frame

OUTDIR = os.path.join(_HERE, "..", "media")
OUT = os.path.join(OUTDIR, "vitanova_division.gif")

# dark theme
BG = "#0e1420"
PANEL = "#111a2b"
EDGE = "#2a3850"
TXT = "#dce5f2"
MUT = "#7d8aa0"
WORKER = "#6fb1e8"
SOLDIER = "#f2a93b"
PREDATOR = "#e05252"
HULL_A = "#2b4a6f"
HULL_B = "#6f5426"

STAGES = [(0, "dispersed"), (500, "aggregation"), (2050, "ring forming"),
          (2600, "ring"), (DIV_STEP, "division")]


def stage_name(step):
    name = STAGES[0][1]
    for t, s in STAGES:
        if step >= t:
            name = s
    return name


def capture():
    w = vitafast.Fast(seed=SEED, MAX_BOIDS=1500)
    frames = []
    for i in range(STEPS + 1):
        if i % SKIP == 0:
            frames.append((i, w.snapshot(),
                           [c.copy() for c in runfast.clusters(w.x, w.y)]))
        if w.n == 0:
            break
        w.step()
    return frames


def hull(ax, x, y, color, alpha):
    if len(x) < 3:
        return
    pts = np.column_stack([x, y])
    try:
        h = ConvexHull(pts)
        ax.add_patch(Polygon(pts[h.vertices], closed=True, facecolor=color,
                             edgecolor="none", alpha=alpha, zorder=1))
    except Exception:
        pass


def scatter_agents(ax, s, size):
    wk = ~s["soldier"]
    ax.scatter(s["x"][wk], s["y"][wk], s=size, c=WORKER, lw=0, zorder=3)
    ax.scatter(s["x"][~wk], s["y"][~wk], s=size, c=SOLDIER, lw=0, zorder=3)
    ax.scatter(s["px"], s["py"], s=3.5 * size, c=PREDATOR, marker="v",
               lw=0, zorder=4)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"seed {SEED}: running {STEPS} steps...", flush=True)
    frames = capture()

    # focal collective = largest cluster at the ring step
    ring = min(frames, key=lambda f: abs(f[0] - RING_STEP))
    focal_uids = set(ring[1]["uid"][ring[2][0]].tolist())

    images = []
    cam = None  # smoothed camera centre
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    for k, (step, s, cs) in enumerate(frames):
        fig = plt.figure(figsize=(8.0, 4.3), facecolor=BG)
        axL = fig.add_axes([0.035, 0.10, 0.44, 0.80])
        axR = fig.add_axes([0.525, 0.10, 0.44, 0.80])
        for ax in (axL, axR):
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EDGE); sp.set_linewidth(1.0)
            ax.set_aspect("equal")

        # ---- left: whole arena
        scatter_agents(axL, s, 1.6)
        axL.set_xlim(0, vitafast.WIDTH)
        axL.set_ylim(0, vitafast.HEIGHT)
        axL.set_title("arena", color=MUT, fontsize=9, pad=4)

        # ---- camera target: members carrying the focal uids, else largest
        uid = s["uid"]
        in_focal = np.isin(uid, list(focal_uids))
        if in_focal.sum() >= 5:
            tx, ty = float(np.mean(s["x"][in_focal])), \
                     float(np.mean(s["y"][in_focal]))
        elif cs:
            tx, ty = float(np.mean(s["x"][cs[0]])), \
                     float(np.mean(s["y"][cs[0]]))
        else:
            tx, ty = float(np.mean(s["x"])), float(np.mean(s["y"]))
        cam = (tx, ty) if cam is None else \
            (0.85 * cam[0] + 0.15 * tx, 0.85 * cam[1] + 0.15 * ty)

        # ---- right: tracking camera; shade parent ring, then daughters
        if step >= DIV_STEP:
            ov = sorted(cs, key=lambda c: -len(focal_uids &
                                               set(uid[c].tolist())))
            for dtr, col in zip(ov[:2], (HULL_A, HULL_B)):
                hull(axR, s["x"][dtr], s["y"][dtr], col, 0.9)
        elif step >= RING_STEP - 300 and cs:
            big = cs[0]
            if len(focal_uids & set(uid[big].tolist())) >= 5:
                hull(axR, s["x"][big], s["y"][big], HULL_A, 0.55)
        scatter_agents(axR, s, 7.0)
        axR.set_xlim(cam[0] - ZOOM, cam[0] + ZOOM)
        axR.set_ylim(cam[1] - ZOOM, cam[1] + ZOOM)
        axR.set_title(stage_name(step), color=TXT, fontsize=10, pad=4)

        # camera box on the arena panel
        axL.add_patch(plt.Rectangle((cam[0] - ZOOM, cam[1] - ZOOM),
                                    2 * ZOOM, 2 * ZOOM, fill=False,
                                    edgecolor=MUT, lw=0.8, zorder=5))

        fig.text(0.035, 0.945, "VitaNova", color=TXT, fontsize=12,
                 fontweight="bold")
        fig.text(0.175, 0.945, "a collective forms a ring and divides "
                 "(no fitness function)", color=MUT, fontsize=9)
        fig.text(0.965, 0.945, f"step {step}", color=MUT, fontsize=9,
                 ha="right")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(buf.copy()))
        plt.close(fig)
        if k % 40 == 0:
            print(f"  frame {k}/{len(frames)} (step {step})", flush=True)

    images += [images[-1]] * HOLD_END
    pal = [im.quantize(colors=128, method=Image.MEDIANCUT, dither=0)
           for im in images]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=70,
                loop=0, optimize=True)
    print(f"saved: {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, "
          f"{len(pal)} frames)")


if __name__ == "__main__":
    main()
