"""Figure 1 - the phenomenon and its transience, in one run."""
import sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/fast")
sys.path.insert(0, ".")
from shape import annularity
from runfast import clusters
import style; style.use()
from style import (ACC, ACC2, MUTED, INK, WORKER, SOLDIER, PREDATOR,
                   HULL_A, HULL_B, panel_label, bare)

SEED, PANELS = 13, [1200, 1800, 2400, 3000]
d = np.load(f"/Volumes/DATA/ETI Presentation/ETI/eusocial/data/castes_seed{SEED}.npz",
            allow_pickle=True)
steps = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("snap_")})

# series
S, A, L, N = [], [], [], []
for s in steps:
    x, y = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
    cs = clusters(x, y, min_size=10)
    a = annularity(x[cs[0]], y[cs[0]]) if cs else None
    S.append(s); N.append(len(x))
    L.append(len(cs[0]) if cs else 0)
    A.append(a["annularity"] if a else 0.0)
S, A, L, N = map(np.array, (S, A, L, N))

fig = plt.figure(figsize=(6.9, 3.9))
gs = fig.add_gridspec(2, 4, height_ratios=[1.30, 0.95], hspace=0.16, wspace=0.10)

CAP = ["aggregation", "ring", "ring, grown", "fission"]
for i, st in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i])
    x, y = d[f"snap_{st}_x"], d[f"snap_{st}_y"]
    dx, dy = d[f"snap_{st}_dx"], d[f"snap_{st}_dy"]
    sol = d[f"snap_{st}_soldier"].astype(bool)
    cs = clusters(x, y, min_size=10)
    for j, c in enumerate(cs[:2]):
        # shaded disc marks which collective a group of agents belongs to
        cxj, cyj = x[c].mean(), y[c].mean()
        rj = np.hypot(x[c] - cxj, y[c] - cyj).max() + 5
        ax.add_patch(plt.Circle((cxj, cyj), rj, fc=[HULL_A, HULL_B][j],
                                ec="none", zorder=1, alpha=.85))
        ax.quiver(x[c], y[c], dx[c], -dy[c], color=MUTED, width=.010,
                  scale=22, alpha=.55, zorder=2)
        sc = sol[c]
        ax.scatter(x[c][~sc], y[c][~sc], s=9, c=WORKER, lw=0, zorder=3)
        ax.scatter(x[c][sc], y[c][sc], s=9, c=SOLDIER, lw=0, zorder=3)
    ax.scatter(d[f"snap_{st}_px"], d[f"snap_{st}_py"], s=22, marker="^",
               c=PREDATOR, lw=0, zorder=4)
    allc = np.concatenate(cs[:2]) if cs else np.arange(len(x))
    cx, cy = x[allc].mean(), y[allc].mean()
    h = np.hypot(x[allc] - cx, y[allc] - cy).max() * 1.35 + 6
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h)
    ax.set_aspect("equal"); bare(ax)
    k = int(np.argmin(abs(S - st)))
    sub = f"$n={L[k]}$, $A={A[k]:.2f}$" if len(cs) < 2 else \
          f"$n={len(cs[0])}+{len(cs[1])}$"
    ax.set_title(f"{CAP[i]}\n{sub}", fontsize=7.8, pad=3, linespacing=1.3)
    panel_label(ax, "abcd"[i], dx=-0.02, dy=1.34)

ax = fig.add_subplot(gs[1, :])
ax.plot(S, A, color=WORKER, lw=1.3, label="ring structure $A$")
ax.set_ylabel("annularity $A$", color=WORKER); ax.set_ylim(-.03, 1.03)
ax.set_xlabel("step"); ax.set_xlim(0, S.max())
ax.tick_params(axis="y", colors=WORKER)
ax2 = ax.twinx()
ax2.plot(S, L, color=SOLDIER, lw=1.1, ls=(0, (4, 1.6)),
         label="largest collective")
ax2.set_yscale("log"); ax2.set_ylabel("members", color=SOLDIER)
ax2.tick_params(axis="y", colors=SOLDIER); ax2.grid(False)
ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(SOLDIER)
for i, st in enumerate(PANELS):
    ax.axvline(st, color=INK, lw=.6, alpha=.45)
    ax.text(st, 1.045, "abcd"[i], ha="center", fontsize=8, fontweight="bold")
lost = S[np.argmax((S > 3000) & (A < .05))]
ax.axvspan(lost, S.max(), color="#f2f2f2", zorder=0)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center", fontsize=7.5,
          bbox_to_anchor=(0.74, 0.44), framealpha=0.0)
panel_label(ax, "e", dx=-0.055, dy=1.16)
fig.savefig("figures/fig1_phenomenon.pdf")
fig.savefig("figures/fig1_phenomenon.png", dpi=200)
print("fig1 written; A peak", A.max(), "at", S[A.argmax()], "| final largest", L[-1])
