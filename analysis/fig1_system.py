# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure 1 - the system and what it does, in one figure."""
import sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/fast")
sys.path.insert(0, ".")
from shape import annularity
from runfast import clusters
import style; style.use()
from style import (WORKER, SOLDIER, PREDATOR, HULL_A, HULL_B, INK, MUTED,
                   panel_label, bare)

W, H, R, PROX = 800, 600, 25.0, 15.0
D = "/Volumes/DATA/ETI Presentation/ETI/eusocial/data/castes_seed13.npz"
d = np.load(D, allow_pickle=True)
ENV, PANELS = 900, [1200, 1800, 2400, 3000]
steps = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("snap_")})

fig = plt.figure(figsize=(6.9, 6.4))
gs = fig.add_gridspec(3, 4, height_ratios=[1.45, 1.0, 0.95],
                      hspace=0.30, wspace=0.16)

# (a) arena
x, y = d[f"snap_{ENV}_x"], d[f"snap_{ENV}_y"]
dx, dy = d[f"snap_{ENV}_dx"], d[f"snap_{ENV}_dy"]
sol = d[f"snap_{ENV}_soldier"].astype(bool)
px, py = d[f"snap_{ENV}_px"], d[f"snap_{ENV}_py"]
a = fig.add_subplot(gs[0, 0:3])
a.add_patch(plt.Rectangle((0, 0), W, H, fill=False, lw=1.0, ec=INK))
a.quiver(x, y, dx, -dy, color=MUTED, width=.0035, scale=34, alpha=.6, zorder=2)
a.scatter(x[~sol], y[~sol], s=13, c=WORKER, lw=0, zorder=3, label="worker")
a.scatter(x[sol], y[sol], s=13, c=SOLDIER, lw=0, zorder=3, label="soldier")
a.scatter(px, py, s=40, marker="^", c=PREDATOR, lw=0, zorder=4, label="predator")
a.set_xlim(-25, W + 25); a.set_ylim(H + 25, -25)
a.set_aspect("equal"); bare(a)
for sp in a.spines.values(): sp.set_visible(False)
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=7.6,
         handletextpad=.35, columnspacing=1.6)
a.plot([20, 120], [H + 6] * 2, color=INK, lw=1.2, clip_on=False)
a.text(70, H + 18, "100 px", ha="center", va="top", fontsize=7)
panel_label(a, "a", dx=-0.01, dy=1.17)

# (b) one agent's ranges
b = fig.add_subplot(gs[0, 3])
dd = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
i = int(((dd < R) & ~np.eye(len(x), dtype=bool)).sum(1).argmax())
cx, cy, HALF = x[i], y[i], 44
m = (np.abs(x - cx) < HALF) & (np.abs(y - cy) < HALF)
b.add_patch(Circle((cx, cy), R, fc="#eef3f8", ec=WORKER, lw=.9,
                   ls=(0, (4, 2)), zorder=1))
b.add_patch(Circle((cx, cy), PROX, fc="#e3ebf3", ec=MUTED, lw=.8,
                   ls=(0, (2, 2)), zorder=1))
b.quiver(x[m], y[m], dx[m], -dy[m], color=MUTED, width=.007, scale=26,
         alpha=.65, zorder=2)
b.scatter(x[m & ~sol], y[m & ~sol], s=18, c=WORKER, lw=0, zorder=3)
b.scatter(x[m & sol], y[m & sol], s=18, c=SOLDIER, lw=0, zorder=3)
b.scatter([cx], [cy], s=44, facecolor="none", edgecolor=INK, lw=1.2, zorder=5)
b.text(cx, cy - R - 4, "$R=25$", ha="center", va="bottom", fontsize=7,
       color=WORKER)
b.text(cx, cy + PROX + 2, "15", ha="center", va="top", fontsize=6.6,
       color="#5d6670")
b.set_xlim(cx - HALF, cx + HALF); b.set_ylim(cy + HALF, cy - HALF)
b.set_aspect("equal"); bare(b)
for sp in b.spines.values(): sp.set_visible(False)
panel_label(b, "b", dx=-0.03, dy=1.17)

# (c-f) four states
S, A, L = [], [], []
for s in steps:
    xx, yy = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
    cs = clusters(xx, yy, min_size=10)
    q = annularity(xx[cs[0]], yy[cs[0]]) if cs else None
    S.append(s); L.append(len(cs[0]) if cs else 0)
    A.append(q["annularity"] if q else 0.0)
S, A, L = map(np.array, (S, A, L))
CAP = ["aggregation", "ring", "ring, grown", "fission"]
for k, st in enumerate(PANELS):
    ax = fig.add_subplot(gs[1, k])
    xx, yy = d[f"snap_{st}_x"], d[f"snap_{st}_y"]
    ddx, ddy = d[f"snap_{st}_dx"], d[f"snap_{st}_dy"]
    ss = d[f"snap_{st}_soldier"].astype(bool)
    cs = clusters(xx, yy, min_size=10)
    for j, c in enumerate(cs[:2]):
        mx, my = xx[c].mean(), yy[c].mean()
        rj = np.hypot(xx[c] - mx, yy[c] - my).max() + 5
        ax.add_patch(Circle((mx, my), rj, fc=[HULL_A, HULL_B][j], ec="none",
                            zorder=1, alpha=.85))
        ax.quiver(xx[c], yy[c], ddx[c], -ddy[c], color=MUTED, width=.010,
                  scale=22, alpha=.55, zorder=2)
        sc = ss[c]
        ax.scatter(xx[c][~sc], yy[c][~sc], s=9, c=WORKER, lw=0, zorder=3)
        ax.scatter(xx[c][sc], yy[c][sc], s=9, c=SOLDIER, lw=0, zorder=3)
    ax.scatter(d[f"snap_{st}_px"], d[f"snap_{st}_py"], s=20, marker="^",
               c=PREDATOR, lw=0, zorder=4)
    allc = np.concatenate(cs[:2]) if cs else np.arange(len(xx))
    mx, my = xx[allc].mean(), yy[allc].mean()
    h = np.hypot(xx[allc] - mx, yy[allc] - my).max() * 1.3 + 6
    ax.set_xlim(mx - h, mx + h); ax.set_ylim(my + h, my - h)
    ax.set_aspect("equal"); bare(ax)
    kk = int(np.argmin(abs(S - st)))
    sub = (f"$n={L[kk]}$, $A={A[kk]:.2f}$" if len(cs) < 2
           else f"$n={len(cs[0])}+{len(cs[1])}$")
    ax.set_title(f"{CAP[k]}\n{sub}", fontsize=7.4, pad=2.5, linespacing=1.25)
    panel_label(ax, "cdef"[k], dx=-0.03, dy=1.36)

# (g) time course
ax = fig.add_subplot(gs[2, :])
ax.plot(S, A, color=WORKER, lw=1.3, label="ring structure $A$")
ax.set_ylabel("annularity $A$", color=WORKER); ax.set_ylim(-.03, 1.03)
ax.set_xlabel("step"); ax.set_xlim(0, S.max())
ax.tick_params(axis="y", colors=WORKER)
ax2 = ax.twinx()
ax2.plot(S, L, color=SOLDIER, lw=1.1, ls=(0, (4, 1.6)), label="largest collective")
ax2.set_yscale("log"); ax2.set_ylabel("members", color=SOLDIER)
ax2.tick_params(axis="y", colors=SOLDIER); ax2.grid(False)
ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(SOLDIER)
for k, st in enumerate(PANELS):
    ax.axvline(st, color=INK, lw=.6, alpha=.45)
    ax.text(st, 1.045, "cdef"[k], ha="center", fontsize=8, fontweight="bold")
lost = S[np.argmax((S > 3000) & (A < .05))]
ax.axvspan(lost, S.max(), color="#f2f2f2", zorder=0)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center", fontsize=7.5,
          bbox_to_anchor=(0.74, 0.44), framealpha=0.0)
panel_label(ax, "g", dx=-0.055, dy=1.14)

fig.savefig("figures/fig1_system.pdf", dpi=300)
fig.savefig("figures/fig1_system.png", dpi=200)
print(f"merged: arena step {ENV} ({len(x)} agents, {sol.sum()} soldiers), "
      f"focal agent {int((dd[i] < R).sum())-1} neighbours; A peak {A.max():.2f}")
