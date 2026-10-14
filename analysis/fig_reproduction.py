# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Collective reproduction is repeatable, not a single lucky run."""
import sys, glob, re, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/fast")
sys.path.insert(0, ".")
from runfast import clusters
import style; style.use()
from style import (ACC, ACC2, MUTED, INK, WORKER, SOLDIER,
                   HULL_A, HULL_B, panel_label, bare)

W, H = 800, 600
C = dict(fission=4, fsize=5)
MIN_MEDIAN_R = 8.0     # a collective whose members are piled at the centre
                       # renders as a dot; require real radial extent

events, per_run, n_runs, all_sizes = [], {}, 0, []
for f in sorted(glob.glob("../shared/data/seed*.npz")):
    seed = int(re.search(r"seed(\d+)", f).group(1))
    d = np.load(f, allow_pickle=True)
    col = d["collectives"]
    k = int((col[:, C["fission"]] >= 0).sum())
    per_run[seed] = k
    n_runs += 1
    all_sizes += [float(v) for v in col[col[:, C["fission"]] >= 0, C["fsize"]] if v > 0]
    if k == 0:
        continue
    steps = sorted({int(x.split("_")[1]) for x in d.files if x.startswith("snap_")})
    for row in col:
        fs = row[C["fission"]]
        if fs < 0:
            continue
        s = min(steps, key=lambda t: abs(t - fs))
        x, y = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
        cs = [c for c in clusters(x, y, min_size=20)
              if np.median(np.hypot(x[c] - x[c].mean(),
                                    y[c] - y[c].mean())) > MIN_MEDIAN_R]
        if len(cs) < 2:
            continue
        cen = [(x[c].mean(), y[c].mean(), c) for c in cs]
        best, bd = None, 1e9
        for i in range(len(cen)):
            for j in range(i + 1, len(cen)):
                dd = np.hypot(cen[i][0] - cen[j][0], cen[i][1] - cen[j][1])
                if dd < bd:
                    bd, best = dd, (cen[i], cen[j])
        if best is None or not (25 < bd < 200):
            continue
        a, b = sorted(best, key=lambda t: -len(t[2]))
        events.append(dict(seed=seed, step=s, file=f, sep=bd, big=a[2], small=b[2],
                           cx=(a[0] + b[0]) / 2, cy=(a[1] + b[1]) / 2,
                           n_big=len(a[2]), n_small=len(b[2]),
                           parent=float(row[C["fsize"]])))

div_runs = sorted(s for s, k in per_run.items() if k)
best_per_seed = {}
for e in events:
    sc = min(e["n_small"] / e["n_big"], 1.0) / (1 + e["n_big"] / 200)
    if e["seed"] not in best_per_seed or sc > best_per_seed[e["seed"]][0]:
        best_per_seed[e["seed"]] = (sc, e)
gallery = [best_per_seed[s][1] for s in div_runs if s in best_per_seed]

fig = plt.figure(figsize=(6.9, 4.6))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.25], hspace=0.45, wspace=0.30)

for k, e in enumerate(gallery[:6]):
    ax = fig.add_subplot(gs[k // 3, (k % 3) * 2:(k % 3) * 2 + 2])
    d = np.load(e["file"], allow_pickle=True)
    x, y = d[f"snap_{e['step']}_x"], d[f"snap_{e['step']}_y"]
    sol = d[f"snap_{e['step']}_sol"].astype(bool)
    for c, hc in ((e["big"], HULL_A), (e["small"], HULL_B)):
        cxj, cyj = x[c].mean(), y[c].mean()
        rj = np.hypot(x[c] - cxj, y[c] - cyj).max() + 4
        ax.add_patch(plt.Circle((cxj, cyj), rj, fc=hc, ec="none", zorder=1,
                                alpha=.85))
        sc = sol[c]
        ax.scatter(x[c][~sc], y[c][~sc], s=7, c=WORKER, lw=0, zorder=3)
        ax.scatter(x[c][sc], y[c][sc], s=7, c=SOLDIER, lw=0, zorder=3)
    allc = np.concatenate([e["big"], e["small"]])
    cx, cy = x[allc].mean(), y[allc].mean()
    h = max(np.abs(x[allc] - cx).max(), np.abs(y[allc] - cy).max()) * 1.15 + 10
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h)
    ax.set_aspect("equal"); bare(ax)
    ax.set_title(f"seed {e['seed']}, step {e['step']}\n"
                 f"${e['n_big']} + {e['n_small']}$ members",
                 fontsize=7.2, pad=2.5, linespacing=1.25)
    ax.plot([cx - h + 8, cx - h + 8 + 50], [cy + h - 10] * 2, color=INK, lw=1.2)
    if k == 0:
        ax.text(cx - h + 8, cy + h - 16, "50 px", fontsize=6.2, va="bottom")

axm = fig.add_subplot(gs[2, 0:2])
axm.add_patch(plt.Rectangle((0, 0), W, H, fill=False, lw=.8, ec=INK))
axm.scatter([e["cx"] for e in events], [e["cy"] for e in events],
            s=8, c=WORKER, lw=0, alpha=.5)
axm.set_xlim(-35, W + 35); axm.set_ylim(H + 35, -35)
axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([]); axm.grid(False)
for sp in axm.spines.values(): sp.set_visible(False)
axm.set_xlabel("where in the arena")
panel_label(axm, "g", dx=-0.06, dy=1.14)

axh = fig.add_subplot(gs[2, 2:4])
p = np.array(all_sizes, float)
axh.hist(p, bins=np.logspace(np.log10(p.min()), np.log10(p.max()), 18),
         color=WORKER, edgecolor="white", linewidth=.5)
axh.set_xscale("log"); axh.set_xlabel("members at division")
axh.set_ylabel("divisions")
panel_label(axh, "h", dx=-0.26, dy=1.14)

axr = fig.add_subplot(gs[2, 4:6])
ks = [per_run[s] for s in sorted(per_run, key=lambda s: -per_run[s])]
axr.bar(range(len(ks)), ks, color=[WORKER if v else "#d8dde3" for v in ks],
        width=.75, edgecolor="none")
axr.set_xlabel(f"divisions in each of {n_runs} runs")
axr.set_xticks([])
panel_label(axr, "i", dx=-0.13, dy=1.14)
panel_label(fig.axes[0], "a", dx=-0.03, dy=1.32)

fig.savefig("figures/fig_reproduction.pdf", dpi=300)
fig.savefig("figures/fig_reproduction.png", dpi=200)
print(f"{len(events)} clean division frames; galleries from seeds {[e['seed'] for e in gallery]}")
print(f"divisions: {sum(per_run.values())} recorded in {len(div_runs)}/{n_runs} runs "
      f"({div_runs})")
print(f"members at division (all {len(p)} records): median {np.median(p):.0f}, "
      f"range {p.min():.0f}-{p.max():.0f}, IQR {np.percentile(p,25):.0f}-{np.percentile(p,75):.0f}")
print(f"arena span x {min(e['cx'] for e in events):.0f}-{max(e['cx'] for e in events):.0f}, "
      f"y {min(e['cy'] for e in events):.0f}-{max(e['cy'] for e in events):.0f}")
