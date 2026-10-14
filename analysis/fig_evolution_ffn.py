# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Selection shapes the collective; it does not make it reproduce."""
import sys, json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, f_oneway
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

L = "/Volumes/DATA/ETI Presentation/ETI/linear"
arms = json.load(open(f"{L}/evolution_ffn.json"))
gv   = json.load(open(f"{L}/genotype_variance_ffn.json"))

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.46})

# (a) structure responds to selection
ARMS = [("evolving", "full", ACC), ("no-mutation", "no\nmutation", ACC3),
        ("clonal", "clonal", ACC2)]
a = ax[0]
for i, (arm, lab, col) in enumerate(ARMS):
    est = [x for x in arms if x["arm"] == arm and x["largest"] >= 20]
    v = np.array([x["peak_A"] for x in est], float)
    xs = np.full(len(v), i) + np.random.default_rng(i).normal(0, .075, len(v))
    a.scatter(xs, v, s=9, c=col, lw=0, alpha=.65, zorder=3)
    a.plot([i - .27, i + .27], [np.median(v)] * 2, color=INK, lw=1.9, zorder=4)
    allr = [x for x in arms if x["arm"] == arm]
    nd = sum(1 for x in allr if x["divided"])
    a.text(i, -0.115, f"{nd}/{len(allr)}", ha="center", fontsize=7, color=MUTED)
    a.text(i, -0.215, f"{len(est)}/{len(allr)}", ha="center", fontsize=7,
           color=MUTED)
a.text(-0.56, -0.165, "divided", fontsize=6.4, color=MUTED, ha="left")
a.text(-0.56, -0.265, "formed", fontsize=6.4, color=MUTED, ha="left")
a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in ARMS],
                                          fontsize=7.4, linespacing=1.15)
a.set_ylabel("peak ring structure $A$"); a.set_ylim(-0.34, 1.02)
panel_label(a, "a", dx=-0.30, dy=1.14)

# (b) the collective trait is heritable
b = ax[1]
G = sorted({x["genotype"] for x in gv}); W = sorted({x["world"] for x in gv})
M = np.zeros((len(G), len(W)))
for x in gv: M[x["genotype"], x["world"]] = x["peak_A"]
order = np.argsort(-M.mean(1))
for r, gi in enumerate(order):
    b.scatter(np.full(len(W), r) + np.random.default_rng(gi).normal(0, .10, len(W)),
              M[gi], s=7, c=ACC, lw=0, alpha=.6, zorder=3)
    b.plot([r - .3, r + .3], [M[gi].mean()] * 2, color=INK, lw=1.4, zorder=4)
tot = M.mean()
ss_g = len(W) * ((M.mean(1) - tot) ** 2).sum()
ss_w = len(G) * ((M.mean(0) - tot) ** 2).sum()
ss_t = ((M - tot) ** 2).sum()
b.set_xticks([]); b.set_xlabel(f"{len(G)} genotypes, {len(W)} worlds each")
b.set_ylabel("peak ring structure $A$"); b.set_ylim(-0.03, 1.02)
b.set_title(f"genotype {ss_g/ss_t:.0%}, world {ss_w/ss_t:.0%}",
            fontsize=7.8, pad=3)
panel_label(b, "b", dx=-0.30, dy=1.14)

# (c) and it predicts reproductive success
c = ax[2]
A = np.array([M[g].mean() for g in G])
D = np.array([np.mean([x["divided"] for x in gv if x["genotype"] == g]) for g in G])
c.scatter(A, D, s=26, c=ACC, lw=0, zorder=3)
s = spearmanr(A, D)
c.set_xlabel("peak ring structure $A$"); c.set_ylabel("runs that divided")
c.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
c.set_xlim(-0.05, 0.92); c.set_ylim(-0.06, 1.08)
c.set_title(f"Spearman $\\rho={s.statistic:+.2f}$, $p={s.pvalue:.3f}$",
            fontsize=7.8, pad=3)
panel_label(c, "c", dx=-0.32, dy=1.14)

fig.savefig("figures/fig_evolution_ffn.pdf", dpi=300)
fig.savefig("figures/fig_evolution_ffn.png", dpi=200)

for arm, lab, _ in ARMS:
    est=[x for x in arms if x["arm"]==arm and x["largest"]>=20]
    v=np.array([x["peak_A"] for x in est],float)
    allr=[x for x in arms if x["arm"]==arm]
    print(f"{arm:>12}: {len(est):>2}/40 formed, median A {np.median(v):.2f}, "
          f"divided {sum(1 for x in allr if x['divided'])}/40 (all runs)")
ev=[x["peak_A"] for x in arms if x["arm"]=="evolving" and x["largest"]>=20]
cl=[x["peak_A"] for x in arms if x["arm"]=="clonal" and x["largest"]>=20]
nm=[x["peak_A"] for x in arms if x["arm"]=="no-mutation" and x["largest"]>=20]
print(f"  evolving vs clonal p={mannwhitneyu(ev,cl).pvalue:.4f}; "
      f"vs no-mutation p={mannwhitneyu(ev,nm).pvalue:.3f}")
print(f"  variance: genotype {ss_g/ss_t:.1%}, world {ss_w/ss_t:.1%}")
print(f"  A vs division rate across genotypes: rho={s.statistic:+.3f}, p={s.pvalue:.4f}")
