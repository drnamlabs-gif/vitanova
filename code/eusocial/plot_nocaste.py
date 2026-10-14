# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../reproduce"); sys.path.insert(0, "../fast")
from shape import annularity
from runfast import clusters

def snap(d, s):
    return {k: d[f"snap_{s}_{k}"] for k in
            ("x","y","dx","dy","energy","age","soldier","uid","parent","px","py")}

def biggest(S, rank=0):
    cs = clusters(S["x"], S["y"], min_size=8)
    if len(cs) <= rank: return None
    m = np.zeros(len(S["x"]), bool); m[cs[rank]] = True
    return m

def panel(ax, S, title, zoom=True):
    m = biggest(S)
    x, y = S["x"], S["y"]
    ax.scatter(S["px"], S["py"], s=46, marker="^", c="#c0392b", zorder=4, lw=0)
    if m is None or not zoom:
        ax.scatter(x, y, s=5, c="#2c3e50", lw=0)
        ax.set_xlim(0, 800); ax.set_ylim(600, 0)
    else:
        ax.scatter(x[~m], y[~m], s=5, c="#b0bec5", lw=0)
        cx, cy = x[m].mean(), y[m].mean()
        r = np.hypot(x[m]-cx, y[m]-cy)
        # colour by radius so the hollow core is legible
        ax.scatter(x[m], y[m], s=12, c=r, cmap="viridis", lw=0, zorder=3)
        # velocity arrows
        ax.quiver(x[m], y[m], S["dx"][m], -S["dy"][m], color="#37474f",
                  width=0.004, scale=28, alpha=0.55, zorder=2)
        h = max(r.max()*1.9, 45)
        ax.set_xlim(cx-h, cx+h); ax.set_ylim(cy+h, cy-h)
        a = annularity(x[m], y[m])
        title += f"\nn={m.sum()}" + (
            f"  A={a['annularity']:.2f}  hollow={a['hollowness']:.2f}" if a else "")
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

seed  = int(sys.argv[1])
steps = [int(s) for s in sys.argv[2].split(",")]
zoom  = sys.argv[3] != "wide"
d = np.load(f"data/nocaste_seed{seed}.npz", allow_pickle=True)
fig, axes = plt.subplots(1, len(steps), figsize=(2.6*len(steps), 3.1))
for ax, s in zip(np.atleast_1d(axes), steps):
    panel(ax, snap(d, s), f"step {s}", zoom)
fig.suptitle(f"VitaNova with castes DELETED - seed {seed}", fontsize=10, y=1.0)
fig.tight_layout()
out = f"figures/nocaste_seed{seed}_{'zoom' if zoom else 'wide'}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
