"""Figure 4 - the organised state is a transient, and what sets its window."""
import json, sys, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

SRC = "/Volumes/DATA/ETI Presentation/ETI/eusocial/size_window.json"
rows = json.load(open(SRC))
R_PERC = 25.0
n  = np.array([r["n"] for r in rows], float)
A  = np.array([r["A"] for r in rows], float)
rx = np.array([r["r_max"] for r in rows], float)

fig = plt.figure(figsize=(6.9, 4.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.52, wspace=0.28)

# (a) structure against size
a = fig.add_subplot(gs[0, 0])
edges = np.array([10, 15, 20, 25, 30, 40, 50, 65, 85, 110, 150, 220, 320, 600])
xs, ys, lo, hi = [], [], [], []
for l, h in zip(edges[:-1], edges[1:]):
    m = (n >= l) & (n < h)
    if m.sum() < 8: continue
    xs.append(np.sqrt(l * h)); ys.append(np.median(A[m]))
    lo.append(np.percentile(A[m], 25)); hi.append(np.percentile(A[m], 75))
a.axvspan(20, 55, color="#fdf1d6", zorder=0)
a.fill_between(xs, lo, hi, color=ACC, alpha=.14, lw=0)
a.plot(xs, ys, "o-", color=ACC)
a.set_xscale("log"); a.set_xlabel("collective size $n$")
a.set_ylabel("annularity $A$"); a.set_ylim(-0.03, 1.0)
a.text(33, .92, "window", ha="center", fontsize=7.2, color="#8a6d17")
a.set_xticks([10, 30, 100, 300]); a.set_xticklabels(["10", "30", "100", "300"])
panel_label(a, "a", dx=-0.22, dy=1.15)

# (b) structure against spatial extent
b = fig.add_subplot(gs[0, 1])
sub = np.random.default_rng(0).permutation(len(n))[:2600]
sc = b.scatter(rx[sub] / R_PERC, A[sub], c=np.log10(n[sub]), s=3.2, lw=0,
               cmap="viridis", alpha=.55, rasterized=True)
b.axvline(1.0, color=ACC2, lw=1.1, ls="--")
b.text(1.08, .95, "$r=R$", fontsize=7.5, color=ACC2)
b.set_xscale("log"); b.set_xlim(.35, 7)
b.set_xticks([0.5, 1, 2, 4]); b.set_xticklabels(["0.5", "1", "2", "4"])
b.set_xlabel("collective radius / perception radius")
b.set_ylabel("annularity $A$"); b.set_ylim(-0.03, 1.0)
cb = fig.colorbar(sc, ax=b, pad=0.02, fraction=0.045)
cb.set_label("$\\log_{10} n$", fontsize=7.5); cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.6)
panel_label(b, "b", dx=-0.22, dy=1.15)

# (c) every run loses the structure and none recovers it
c = fig.add_subplot(gs[1, :])
tr = defaultdict(dict)
for r in rows:
    k = (r["tag"], r["seed"])
    if r["step"] not in tr[k] or r["n"] > tr[k][r["step"]]["n"]:
        tr[k][r["step"]] = r
keys = sorted(tr, key=lambda k: -np.mean(
    [(v["A"] > .5) and (v["r_max"] < 1.5 * R_PERC) for v in tr[k].values()]))
STEPS = np.arange(0, 9000, 100)
M = np.zeros((len(keys), len(STEPS)))
for i, k in enumerate(keys):
    for j, s in enumerate(STEPS):
        v = tr[k].get(s)
        M[i, j] = 1.0 if (v and v["A"] > .5 and v["r_max"] < 1.5 * R_PERC) else 0.0
c.imshow(M, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(
    ["#f0f2f4", ACC]), interpolation="nearest",
    extent=[0, 9000, len(keys), 0])
c.set_yticks([]); c.set_xlabel("step"); c.grid(False)
c.set_ylabel(f"{len(keys)} runs")
frac = M.mean()

panel_label(c, "c", dx=-0.055, dy=1.20)
fig.savefig("figures/fig4_transient.pdf", dpi=300)
fig.savefig("figures/fig4_transient.png", dpi=200)
last = M[:, -1].sum()
print(f"{len(keys)} runs, organised fraction {frac:.3f}, runs organised at the end: {last:.0f}")
print("median per-run fraction:", np.median(M.mean(1)).round(3))
