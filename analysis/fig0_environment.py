# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure 1 - the environment: arena, agents, castes, predators, ranges."""
import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/fast")
sys.path.insert(0, ".")
import style; style.use()
from style import WORKER, SOLDIER, PREDATOR, INK, MUTED, GRID, panel_label, bare

W, H, R, PROX = 800, 600, 25.0, 15.0
D = "/Volumes/DATA/ETI Presentation/ETI/eusocial/data/castes_seed13.npz"
d = np.load(D, allow_pickle=True)
STEP = 900

x, y = d[f"snap_{STEP}_x"], d[f"snap_{STEP}_y"]
dx, dy = d[f"snap_{STEP}_dx"], d[f"snap_{STEP}_dy"]
sol = d[f"snap_{STEP}_soldier"].astype(bool)
px, py = d[f"snap_{STEP}_px"], d[f"snap_{STEP}_py"]

fig = plt.figure(figsize=(6.9, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.14)

# (a) the whole arena
a = fig.add_subplot(gs[0, 0])
a.add_patch(plt.Rectangle((0, 0), W, H, fill=False, lw=1.0, ec=INK))
a.quiver(x, y, dx, -dy, color=MUTED, width=.0035, scale=34, alpha=.6, zorder=2)
a.scatter(x[~sol], y[~sol], s=13, c=WORKER, lw=0, zorder=3, label="worker")
a.scatter(x[sol], y[sol], s=13, c=SOLDIER, lw=0, zorder=3, label="soldier")
a.scatter(px, py, s=44, marker="^", c=PREDATOR, lw=0, zorder=4, label="predator")
a.set_xlim(-25, W + 25); a.set_ylim(H + 25, -25)
a.set_aspect("equal"); bare(a)
for sp in a.spines.values(): sp.set_visible(False)
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=7.6,
         handletextpad=.35, columnspacing=1.5)
a.plot([20, 20 + 100], [H + 8] * 2, color=INK, lw=1.2, clip_on=False)
a.text(70, H + 20, "100 px", ha="center", va="top", fontsize=7)
panel_label(a, "a", dx=-0.01, dy=1.20)

# (b) one agent's world
b = fig.add_subplot(gs[0, 1])
# focus on the agent with the most neighbours, so the ranges show something
dd = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
i = int(((dd < R) & ~np.eye(len(x), dtype=bool)).sum(1).argmax())
cx, cy = x[i], y[i]
HALF = 46
m = (np.abs(x - cx) < HALF) & (np.abs(y - cy) < HALF)
b.add_patch(Circle((x[i], y[i]), R, fill=True, fc="#eef3f8", ec=WORKER,
                   lw=.9, ls=(0, (4, 2)), zorder=1))
b.add_patch(Circle((x[i], y[i]), PROX, fill=True, fc="#e3ebf3", ec=MUTED,
                   lw=.8, ls=(0, (2, 2)), zorder=1))
b.quiver(x[m], y[m], dx[m], -dy[m], color=MUTED, width=.006, scale=26,
         alpha=.65, zorder=2)
b.scatter(x[m & ~sol], y[m & ~sol], s=22, c=WORKER, lw=0, zorder=3)
b.scatter(x[m & sol], y[m & sol], s=22, c=SOLDIER, lw=0, zorder=3)
b.scatter([x[i]], [y[i]], s=52, facecolor="none", edgecolor=INK, lw=1.3, zorder=5)
pm = (np.abs(px - cx) < HALF) & (np.abs(py - cy) < HALF)
b.scatter(px[pm], py[pm], s=52, marker="^", c=PREDATOR, lw=0, zorder=4)
b.annotate("perception,\nseparation,\nalignment,\ncohesion:\n$R=25$ px",
           xy=(x[i] + R * .72, y[i] - R * .72), xytext=(x[i] + 26, y[i] - 30),
           fontsize=6.8, color=WORKER, linespacing=1.25,
           arrowprops=dict(arrowstyle="-", lw=.7, color=WORKER))
b.annotate("energy subsidy\ncounts neighbours\nwithin 15 px",
           xy=(x[i] - PROX * .72, y[i] + PROX * .72), xytext=(x[i] - 43, y[i] + 26),
           fontsize=6.8, color="#5d6670", linespacing=1.25,
           arrowprops=dict(arrowstyle="-", lw=.7, color=MUTED))
b.set_xlim(cx - HALF, cx + HALF); b.set_ylim(cy + HALF, cy - HALF)
b.set_aspect("equal"); bare(b)
for sp in b.spines.values(): sp.set_visible(False)
panel_label(b, "b", dx=-0.01, dy=1.20)

fig.savefig("figures/fig0_environment.pdf", dpi=300)
fig.savefig("figures/fig0_environment.png", dpi=200)
print(f"step {STEP}: {len(x)} agents ({sol.sum()} soldiers), {len(px)} predators")
print(f"focal agent has {int(((dd[i] < R).sum())-1)} neighbours within R")
