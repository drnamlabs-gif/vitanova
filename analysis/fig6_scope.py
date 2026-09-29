"""Figure 6 - collectives are chimeras, and size regulation makes fusion survivable."""
import json, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

S = json.load(open("scope.json"))
comp, fus = S["composition"], S["fusions"]
eff = np.array([c["effective"] for c in comp])
lin = np.array([c["lineages"] for c in comp])
shr = np.array([c["largest_share"] for c in comp])

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.42})

# (a) how many lineages a collective contains
a = ax[0]
bins = np.arange(0.5, 13.5, 1)
a.hist(lin, bins=bins, color=ACC, edgecolor="white", linewidth=.6)
a.axvline(1, color=ACC2, lw=1.2, ls="--")
a.set_xlabel("founding lineages per collective")
a.set_ylabel(f"collectives (of {len(comp)})")
a.set_xlim(0, 12)
panel_label(a, "a", dx=-0.30, dy=1.16)

# (b) composition of a sample, sorted by dominance
b = ax[1]
idx = np.argsort(-shr)[::max(1, len(comp) // 55)][:55]
order = sorted(idx, key=lambda i: -shr[i])
cmap = plt.get_cmap("Blues")
for k, i in enumerate(order):
    bot = 0.0
    for j, s in enumerate(comp[i]["shares"]):
        b.bar(k, s, bottom=bot, width=1.0,
              color=cmap(0.85 - 0.11 * j), edgecolor="none")
        bot += s
    if bot < 1: b.bar(k, 1 - bot, bottom=bot, width=1.0, color="#dde3ea",
                      edgecolor="none")
b.axhline(shr.mean(), color=ACC2, lw=1.1)

b.set_xlim(-.5, len(order) - .5); b.set_ylim(0, 1)
b.set_xticks([]); b.set_xlabel("collectives, ordered by dominance")
b.set_ylabel("share of members")
b.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
b.grid(False)
panel_label(b, "b", dx=-0.28, dy=1.16)

# (c) size recovery after a fusion
c = ax[2]
c.axhspan(20, 55, color="#fdf1d6", zorder=0)
T = np.arange(0, 60) * 25
for f in fus:
    t = np.array(f["traj"], float)
    c.plot(T[:len(t)], t, color=MUTED, lw=.8, alpha=.7, zorder=2)
M = np.full((len(fus), 60), np.nan)
for i, f in enumerate(fus):
    M[i, :len(f["traj"])] = f["traj"]
c.plot(T, np.nanmedian(M, 0), color=ACC2, lw=1.8, zorder=3)
c.set_xlabel("steps after a fusion"); c.set_ylabel("members")
c.set_xlim(0, 1475)
last = M[:, -1][~np.isnan(M[:, -1])]
alive = last > 0

panel_label(c, "c", dx=-0.30, dy=1.16)

fig.savefig("figures/fig6_scope.pdf", dpi=300)
fig.savefig("figures/fig6_scope.png", dpi=200)
print(f"lineages median {np.median(lin):.0f} mean {lin.mean():.1f}; "
      f"effective {eff.mean():.2f}; largest share {shr.mean():.0%}; "
      f"clonal {(lin==1).mean():.1%}")
pre = np.array([f["pre"] for f in fus], float)
alive = last > 0
print(f"fusion: combined {np.median(pre):.0f} (range {pre.min():.0f}-{pre.max():.0f}); "
      f"after 1500 steps median {np.median(last[alive]):.0f} among survivors; "
      f"{int((alive & (last<=55)).sum())}/{len(last)} inside window, "
      f"{int((~alive).sum())} extinct, {int((last>55).sum())} above window")
