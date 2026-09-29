import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../reproduce"); sys.path.insert(0, "../fast")
from shape import annularity
from runfast import clusters

def snap(d, s):
    return {k: d[f"snap_{s}_{k}"] for k in ("x","y","dx","dy","soldier","px","py")}

seed  = int(sys.argv[1]); steps = [int(s) for s in sys.argv[2].split(",")]
d = np.load(f"data/nocaste_seed{seed}.npz", allow_pickle=True)
fig, axes = plt.subplots(1, len(steps), figsize=(2.9*len(steps), 3.3))
for ax, s in zip(np.atleast_1d(axes), steps):
    S = snap(d, s); x, y = S["x"], S["y"]
    cs = clusters(x, y, min_size=8)[:2]
    cols = ["#1f77b4", "#d62728"]
    lab = []
    for i, c in enumerate(cs):
        ax.quiver(x[c], y[c], S["dx"][c], -S["dy"][c], color=cols[i],
                  width=0.005, scale=26, alpha=0.5, zorder=2)
        ax.scatter(x[c], y[c], s=16, c=cols[i], lw=0, zorder=3)
        a = annularity(x[c], y[c])
        lab.append(f"n={len(c)}" + (f" A={a['annularity']:.2f}" if a else ""))
    ax.scatter(S["px"], S["py"], s=52, marker="^", c="#7f7f7f", zorder=4, lw=0)
    if cs:
        allc = np.concatenate(cs)
        cx, cy = x[allc].mean(), y[allc].mean()
        h = max(np.abs(x[allc]-cx).max(), np.abs(y[allc]-cy).max())*1.35 + 12
        ax.set_xlim(cx-h, cx+h); ax.set_ylim(cy+h, cy-h)
    ax.set_title(f"step {s}\n" + "   ".join(lab), fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.suptitle(f"Colony fission with castes DELETED - seed {seed}  "
             "(blue = parent, red = daughter)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f"figures/nocaste_fission_seed{seed}.png", dpi=150, bbox_inches="tight")
print("saved figures/nocaste_fission_seed%d.png" % seed)
