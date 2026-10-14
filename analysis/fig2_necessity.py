# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure 2 - which hand-coded group-level terms are necessary."""
import json, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, MUTED, INK, GRID, panel_label

R = json.load(open("../p4_minimal_ecology/results.json"))["cells"]
FACTORS = ["allee", "mixing", "defence", "ceiling"]
NICE = {"allee": "worker energy subsidy", "mixing": "soldier subsidy",
        "defence": "predators flee soldiers", "ceiling": "crowding ceiling"}

rows = []
for cell, v in R.items():
    on = set() if cell == "none" else set(cell.split("+"))
    rows.append((on, v["divided"], v["n"], v["largest"], v["peak_n"]))
rows.sort(key=lambda r: ("allee" in r[0], r[1], r[3]))
N_NO_ALLEE = sum(1 for r in rows if "allee" not in r[0])

fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.5),
                         gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.05})
ax0, ax1 = axes
y = np.arange(len(rows))

# left: which terms are switched on
ax0.set_xlim(-0.6, len(FACTORS) - 0.4); ax0.set_ylim(-0.7, len(rows) - 0.3)
ax0.grid(False)
for sp in ax0.spines.values():
    sp.set_visible(False)
for j, f in enumerate(FACTORS):
    ax0.text(j, len(rows) - 0.1, NICE[f], rotation=38, ha="left", va="bottom",
             fontsize=7.4)
for i, (on, *_ ) in enumerate(rows):
    ax0.plot([0, len(FACTORS) - 1], [i, i], color=GRID, lw=0.6, zorder=0)
    for j, f in enumerate(FACTORS):
        ax0.scatter(j, i, s=32, zorder=2,
                    c=(ACC if f in on else "white"),
                    edgecolors=(ACC if f in on else "#c3c9d0"), linewidths=0.9)
ax0.set_xticks([]); ax0.set_yticks([])

# right: outcome
ax1.set_ylim(-0.7, len(rows) - 0.3); ax1.set_yticks([])
div = np.array([r[1] for r in rows], float)
ax1.barh(y, div, height=0.55, color=[ACC2 if d else "#d8dde3" for d in div],
         edgecolor="none", zorder=2)
for i, (on, d, n, lg, pk) in enumerate(rows):
    ax1.text(d + 0.12, i, f"{d}/{n}", va="center", fontsize=7.6,
             color=INK if d else MUTED)
    ax1.text(5.9, i, f"{lg:.0f}", va="center", ha="right", fontsize=7.4,
             color=MUTED)
ax1.set_xlim(0, 6.3)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xlabel("replicates in which a collective divided (of 6)")
ax1.text(5.9, len(rows) - 0.1, "largest\ncollective", ha="right", va="bottom",
         fontsize=7.4, color=MUTED, linespacing=1.15)
ax1.axhspan(-0.7, N_NO_ALLEE - 0.5, color="#f4f6f7", zorder=0)
ax0.axhspan(-0.7, N_NO_ALLEE - 0.5, color="#f4f6f7", zorder=0)

panel_label(ax0, "", dx=0, dy=1)
fig.savefig("figures/fig2_necessity.pdf")
fig.savefig("figures/fig2_necessity.png", dpi=200)
tot_no = sum(v["divided"] for k, v in R.items() if "allee" not in k)
n_no = sum(v["n"] for k, v in R.items() if "allee" not in k)
with_mix = sum(v["divided"] for k, v in R.items() if "allee" in k and "mixing" in k)
wo_mix = sum(v["divided"] for k, v in R.items() if "allee" in k and "mixing" not in k)
print(f"without allee: {tot_no} divisions in {n_no} runs")
print(f"with allee: mixing {with_mix}/24, no mixing {wo_mix}/24")
