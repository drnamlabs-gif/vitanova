"""Selection happens; it changes who is present, not what happens."""
import sys, json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

L = "/Volumes/DATA/ETI Presentation/ETI/linear"
E = "/Volumes/DATA/ETI Presentation/ETI/eusocial"
noev = json.load(open(f"{L}/no_evolution.json"))
lin  = json.load(open(f"{L}/lineage_sorting.json"))
res  = json.load(open(f"{L}/resurrection.json"))
eco  = json.load(open(f"{E}/no_ecology.json"))

fig, ax = plt.subplots(1, 4, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.52})

# (a) does the morphology need evolution at all?
ARMS = [("evolving", "full"), ("no-mutation", "no\nmutation"),
        ("clonal", "clonal")]
a = ax[0]
for i, (arm, lab) in enumerate(ARMS):
    g = [x for x in noev if x["arm"] == arm and x["largest"] >= 20]
    v = np.array([x["annular"] for x in g], float)
    if not len(v): continue
    xs = np.full(len(v), i) + np.random.default_rng(i).normal(0, .07, len(v))
    a.scatter(xs, v, s=7, c=[ACC, ACC3, ACC2][i], lw=0, alpha=.6, zorder=3)
    a.plot([i - .26, i + .26], [np.median(v)] * 2, color=INK, lw=1.8, zorder=4)
    nd = sum(1 for x in g if x["divided"])
    a.text(i, -0.115, f"{nd}/{len(g)}", ha="center", fontsize=7, color=MUTED)
a.set_xticks(range(3))
a.set_xticklabels([l for _, l in ARMS], fontsize=7.4, linespacing=1.15)
a.set_ylabel("ring structure $A$"); a.set_ylim(-0.26, 1.02)
a.text(-0.52, -0.175, "divided", fontsize=6.6, color=MUTED, ha="left")
panel_label(a, "a", dx=-0.30, dy=1.14)

# (b) selection is nonetheless strong: founding lineages are sorted away
b = ax[1]
surv = [x for x in lin if x.get("survived") and x.get("effective_lineages")]
ev = np.array([x["effective_lineages"] for x in surv], float)
b.hist(ev, bins=np.arange(0.5, 12.5, 1), color=ACC, edgecolor="white", lw=.6)
b.set_xlabel("effective founding lineages\nsurviving (from 100)", linespacing=1.2)
b.set_ylabel(f"runs (of {len(surv)})")
b.set_xlim(0, 12)
panel_label(b, "b", dx=-0.30, dy=1.14)

# (c) but descendants are no better than their ancestors
c = ax[2]
anc = np.array([r["ancestor"]["annularity"] for r in res], float)
des = np.array([r["descendant"]["annularity"] for r in res], float)
for p, q in zip(anc, des):
    c.plot([0, 1], [p, q], color=MUTED, lw=.7, alpha=.65, zorder=1)
c.scatter(np.zeros_like(anc), anc, s=14, color=ACC2, zorder=3)
c.scatter(np.ones_like(des), des, s=14, color=ACC, zorder=3)
c.plot([0, 1], [np.median(anc), np.median(des)], color=INK, lw=1.9, zorder=4)
c.set_xlim(-0.35, 1.35); c.set_xticks([0, 1])
c.set_xticklabels(["founding\ngenotype", "descendant\ngenotype"], fontsize=7,
                  linespacing=1.15)
c.set_ylabel("ring structure $A$")
p = wilcoxon(anc, des).pvalue
c.set_title(f"$n={len(res)}$, $p={p:.2f}$", fontsize=8, pad=3)
panel_label(c, "c", dx=-0.30, dy=1.14)

# (d) the ecology, by contrast, is what the phenomenon needs
e = ax[3]
for i, (flag, lab) in enumerate([(True, "ecology\non"), (False, "ecology\noff")]):
    v = np.array([x["peak_A"] for x in eco if x["subsidy"] == flag], float)
    xs = np.full(len(v), i) + np.random.default_rng(7 + i).normal(0, .07, len(v))
    e.scatter(xs, v, s=9, c=[ACC, ACC2][i], lw=0, alpha=.7, zorder=3)
    e.plot([i - .26, i + .26], [np.median(v)] * 2, color=INK, lw=1.8, zorder=4)
e.set_xticks([0, 1]); e.set_xticklabels(["ecology\non", "ecology\noff"],
                  fontsize=7.4, linespacing=1.15)
e.set_ylabel("peak ring structure $A$"); e.set_ylim(-0.05, 1.02)
panel_label(e, "d", dx=-0.34, dy=1.14)

fig.savefig("figures/fig_evolution.pdf", dpi=300)
fig.savefig("figures/fig_evolution.png", dpi=200)

print("(a) arms:")
for arm, _ in ARMS:
    g=[x for x in noev if x["arm"]==arm and x["largest"]>=20]
    tot=len([x for x in noev if x["arm"]==arm])
    v=np.array([x["annular"] for x in g],float)
    print(f"   {arm:>12}: {len(g):>3}/{tot} established, median A {np.median(v):.2f}, "
          f"divided {sum(1 for x in g if x['divided'])}/{len(g)}")
ge=[x for x in noev if x["arm"]=="evolving" and x["largest"]>=20]
gc=[x for x in noev if x["arm"]=="clonal" and x["largest"]>=20]
print(f"   evolving vs clonal annularity: Mann-Whitney p="
      f"{mannwhitneyu([x['annular'] for x in ge],[x['annular'] for x in gc]).pvalue:.4f}")
print(f"(b) effective lineages: median {np.median(ev):.2f}, "
      f"max {ev.max():.1f}, from 100 founders, n={len(surv)} runs")
print(f"(c) resurrection: ancestor {np.median(anc):.3f} vs descendant "
      f"{np.median(des):.3f}, Wilcoxon p={p:.3f}, n={len(res)}")
on=np.array([x["peak_A"] for x in eco if x["subsidy"]],float)
off=np.array([x["peak_A"] for x in eco if not x["subsidy"]],float)
lg=np.array([x["largest"] for x in eco if not x["subsidy"]],float)
print(f"(d) ecology on: median peak A {np.median(on):.2f} (n={len(on)}); "
      f"off: {np.median(off):.2f}, largest collective ever {lg.max():.0f}, "
      f"Mann-Whitney p={mannwhitneyu(on,off).pvalue:.4f}")
