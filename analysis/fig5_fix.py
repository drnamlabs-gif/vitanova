"""Figure 5 - a locally sensed maintenance cost makes the organised state persistent."""
import json, re, sys, glob, numpy as np, matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

E = "/Volumes/DATA/ETI Presentation/ETI/eusocial"
base = {r["seed"]: r for r in json.load(open(f"{E}/crowd_base.json"))}
cost = {r["seed"]: r for r in json.load(open(f"{E}/crowd_cost.json"))}
live = sorted(s for s in base if base[s]["largest_max"] > 100)

fig = plt.figure(figsize=(6.9, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.42)

# (a,b) size traces
for j, (tag, name) in enumerate([("0.0k0", "no maintenance cost"),
                                 ("0.08k26", "with maintenance cost")]):
    ax = fig.add_subplot(gs[0, j])
    ax.axhspan(20, 55, color="#fdf1d6", zorder=0)
    for f in sorted(glob.glob(f"{E}/data/crowd_b{tag}_seed*.npz")):
        d = np.load(f, allow_pickle=True)
        ax.plot(d["step"], np.where(d["largest"] > 0, d["largest"], np.nan),
                lw=0.7, alpha=.75, color=ACC if j else ACC2)
    ax.set_yscale("log"); ax.set_ylim(8, 2600); ax.set_xlim(0, 9000)
    ax.set_xlabel("step"); ax.set_title(name, fontsize=8.2, pad=4)
    if j == 0:
        ax.set_ylabel("largest collective")
        ax.text(8750, 22, "window", fontsize=7, color="#8a6d17", ha="right",
                va="bottom")
    ax.set_xticks([0, 4000, 8000])
    panel_label(ax, "ab"[j], dx=-0.26, dy=1.18)

# (c) equilibrium size is predictable
axc = fig.add_subplot(gs[0, 2])
grid = []
for ln in open(f"{E}/crowd_grid.log"):
    m = re.match(r"\s*([\d.]+):(\d+)\s+\d+/\d+\s+(\d+)\s", ln)
    if m:
        grid.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
for beta, mk, col in ((0.03, "o", ACC), (0.05, "s", ACC2), (0.08, "^", ACC3)):
    g = [(k, o) for b, k, o in grid if b == beta]
    if not g: continue
    K = np.array([x[0] for x in g]); O = np.array([x[1] for x in g])
    axc.scatter(K + 0.3 / beta, O, marker=mk, s=22, color=col,
                label=f"$\\beta={beta}$", zorder=3)
lim = [24, 72]
axc.plot(lim, lim, color=INK, lw=0.8, ls="--", zorder=1)
axc.set_xlim(*lim); axc.set_ylim(*lim)
axc.set_xlabel("predicted $n^{*}=K+\\bar{g}/\\beta$")
axc.set_ylabel("observed median size")
axc.legend(loc="upper left", fontsize=7)
axc.set_aspect("equal")
panel_label(axc, "c", dx=-0.30, dy=1.18)

# (d,e,f) paired outcomes
def paired(ax, key, title, pct=True, lo=None, hi=None):
    x = np.array([base[s][key] for s in live], float)
    y = np.array([cost[s][key] for s in live], float)
    for a, b in zip(x, y):
        ax.plot([0, 1], [a, b], color=MUTED, lw=0.7, alpha=.7, zorder=1)
    ax.scatter(np.zeros_like(x), x, s=16, color=ACC2, zorder=3)
    ax.scatter(np.ones_like(y), y, s=16, color=ACC, zorder=3)
    ax.plot([0, 1], [np.median(x), np.median(y)], color=INK, lw=1.8, zorder=4)
    p = wilcoxon(x, y).pvalue
    ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1])
    ax.set_xticklabels(["off", "on"])
    ax.set_title(f"{title}\n$p={p:.3f}$", fontsize=8, pad=4)
    if pct: ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    if lo is not None: ax.set_ylim(lo, hi)
    return p

d1 = fig.add_subplot(gs[1, 0])
paired(d1, "in_window", "time in the size window")
panel_label(d1, "d", dx=-0.30, dy=1.20)
d2 = fig.add_subplot(gs[1, 1])
paired(d2, "ring_frac", "time organised")
panel_label(d2, "e", dx=-0.30, dy=1.20)
d3 = fig.add_subplot(gs[1, 2])
x = np.array([base[s]["parent"] for s in live if base[s]["divided"]], float)
y = np.array([cost[s]["parent"] for s in live if cost[s]["divided"]], float)
d3.boxplot([x, y], widths=.45, patch_artist=True,
           boxprops=dict(facecolor="#eef1f4", lw=.8),
           medianprops=dict(color=ACC2, lw=1.4),
           whiskerprops=dict(lw=.8), capprops=dict(lw=.8),
           flierprops=dict(marker="o", ms=2.5, mfc=MUTED, mec="none"))
d3.set_xticklabels(["off", "on"])
nb = sum(base[s]["divided"] for s in live); nc = sum(cost[s]["divided"] for s in live)
d3.set_title("size at fission", fontsize=8, pad=4)
d3.set_yscale("log"); d3.set_ylabel("members")
panel_label(d3, "f", dx=-0.30, dy=1.20)
for a in (d1, d2, d3):
    a.set_xlabel("maintenance cost")

fig.savefig("figures/fig5_fix.pdf", dpi=300)
fig.savefig("figures/fig5_fix.png", dpi=200)
for k in ("largest_med", "largest_max", "in_window", "ring_frac"):
    x = np.array([base[s][k] for s in live], float); y = np.array([cost[s][k] for s in live], float)
    print(f"{k:>12}: {np.median(x):8.3f} -> {np.median(y):8.3f}  p={wilcoxon(x,y).pvalue:.4f}")
print("fission:", nb, "vs", nc, "of", len(live))
