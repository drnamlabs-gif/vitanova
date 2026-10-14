# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../reproduce"); sys.path.insert(0, "../fast")
from shape import annularity
from runfast import clusters

def best_frame(tag, seed):
    d = np.load(f"data/{tag}_seed{seed}.npz", allow_pickle=True)
    steps = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("snap_")})
    best = None
    for s in steps:
        x, y = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
        cs = clusters(x, y, min_size=10)
        if not cs: continue
        c = cs[0]
        a = annularity(x[c], y[c])
        if a and (best is None or a["annularity"] > best[0]):
            best = (a["annularity"], s, c, d)
    return best

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (tag, seed, name) in zip(axes, [
        ("castes", 13, "castes, plastic"), ("castes", 14, "castes, plastic"),
        ("nocaste", 4, "castes DELETED"), ("nocaste", 15, "castes DELETED")]):
    A, s, c, d = best_frame(tag, seed)
    x, y = d[f"snap_{s}_x"][c], d[f"snap_{s}_y"][c]
    sol = d[f"snap_{s}_soldier"][c].astype(bool)
    cx, cy = x.mean(), y.mean()
    ax.quiver(x, y, d[f"snap_{s}_dx"][c], -d[f"snap_{s}_dy"][c],
              color="#455a64", width=0.005, scale=26, alpha=.45, zorder=2)
    ax.scatter(x[~sol], y[~sol], s=20, c="#1f77b4", lw=0, zorder=3, label="worker")
    if sol.any():
        ax.scatter(x[sol], y[sol], s=20, c="#d62728", lw=0, zorder=3, label="soldier")
    h = np.hypot(x-cx, y-cy).max()*1.3 + 8
    ax.set_xlim(cx-h, cx+h); ax.set_ylim(cy+h, cy-h)
    ax.set_title(f"{name}\nseed {seed}, step {s}: n={len(c)}, A={A:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", framealpha=.85)
fig.suptitle("The ring does not need castes, and does not need plasticity", fontsize=11, y=1.03)
fig.tight_layout(); fig.savefig("figures/nocaste_compare.png", dpi=150, bbox_inches="tight")
print("saved figures/nocaste_compare.png")
