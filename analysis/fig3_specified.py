# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure 3 - the radial structure is not written into the energy rules."""
import json, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, panel_label

R = json.load(open("../p3_specified_vs_emergent/results.json"))
sh = R["shells"]
mid = np.array([np.mean([float(v) for v in s["shell"].split("-")]) for s in sh])
nb = np.array([s["neighbours"] for s in sh])
en = np.array([s["energy"] for s in sh])
mem = np.array([s["members"] for s in sh], float)

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.35), gridspec_kw={"wspace": 0.58})

# (a) what varies with radius, and what does not
a = ax[0]
a.plot(mid, nb, "o-", color=ACC, label="neighbours")
a.set_xlabel("radius (fraction of collective radius)")
a.set_ylabel("neighbours within 15 px", color=ACC)
a.tick_params(axis="y", colors=ACC); a.set_ylim(0, 14)
a2 = a.twinx()
a2.plot(mid, en, "s--", color=ACC2, label="energy")
a2.set_ylabel("mean energy", color=ACC2); a2.set_ylim(0, 170)
a2.tick_params(axis="y", colors=ACC2); a2.grid(False)
a2.spines["right"].set_visible(True); a2.spines["right"].set_color(ACC2)
a2.axhline(150, color=ACC2, lw=0.6, ls=":", alpha=0.7)
panel_label(a, "a", dx=-0.30, dy=1.16)

# (b) births track membership: no growth zone, no germ line
b = ax[1]
vals = [[R["members_inner_half"], 1 - R["members_inner_half"]],
        [R["births_inner_half"], 1 - R["births_inner_half"]]]
lab = ["members", "births"]
for i, v in enumerate(vals):
    b.barh(i, v[0], height=.5, color=ACC, edgecolor="none")
    b.barh(i, v[1], left=v[0], height=.5, color="#d8dde3", edgecolor="none")
    b.text(v[0] / 2, i, f"{v[0]:.0%}", ha="center", va="center",
           color="white", fontsize=8)
    b.text(v[0] + v[1] / 2, i, f"{v[1]:.0%}", ha="center", va="center",
           color="#5d6670", fontsize=8)
b.set_yticks([])
for i, t in enumerate(lab):
    b.text(0.012, i + 0.34, t, ha="left", va="bottom", fontsize=8)
b.set_ylim(-0.6, 1.85)
b.set_xlim(0, 1); b.set_xticks([0, .5, 1]); b.set_xticklabels(["0", "50%", "100%"])
b.set_xlabel("inner half (dark) vs outer half")
b.grid(False); b.spines["left"].set_visible(False)

panel_label(b, "b", dx=-0.26, dy=1.16)

# (c) division is not triggered by an energy surplus
c = ax[2]
v = [R["frac_over_elsewhere"], R["frac_over_near_division"]]
c.bar([0, 1], v, width=.5, color=[MUTED, ACC2], edgecolor="none")
for i, x in enumerate(v):
    c.text(i, x + .008, f"{x:.1%}", ha="center", fontsize=8)
c.set_xticks([0, 1]); c.set_xticklabels(["elsewhere", "near a\ndivision"])
c.set_ylabel("members above the\nreproduction threshold")
c.set_ylim(0, .26)
c.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
panel_label(c, "c", dx=-0.34, dy=1.16)

fig.savefig("figures/fig3_specified.pdf")
fig.savefig("figures/fig3_specified.png", dpi=200)
print("shell energy range", en.min(), en.max(), "| neighbours", nb.max(), "->", nb.min())
print("births inner", R["births_inner_half"], "members inner", R["members_inner_half"])
