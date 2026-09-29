"""Largest-collective size and ring state over time, beta=0 vs beta>0."""
import sys, glob, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

tags = sys.argv[1].split(",")
fig, axes = plt.subplots(2, len(tags), figsize=(4.2*len(tags), 5.6),
                         sharex=True, sharey="row")
axes = np.atleast_2d(axes)
if axes.shape[0] == 1: axes = axes.T
for j, b in enumerate(tags):
    fs = sorted(glob.glob(f"data/crowd_b{b}_seed*.npz"))
    ax0, ax1 = axes[0, j], axes[1, j]
    ax0.axhspan(20, 55, color="#ffd54f", alpha=.35, zorder=0)
    fr = []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        s, lg = d["step"], d["largest"]
        ax0.plot(s, lg, lw=.9, alpha=.8)
        if "ring" in d.files:
            r = d["ring"].astype(float); fr.append(r.mean())
            ax1.plot(s, np.convolve(r, np.ones(8)/8, mode="same"), lw=.9, alpha=.8)
    ax0.set_yscale("log"); ax0.set_ylim(8, 1300)
    ax0.set_title(("no cost" if b.endswith("k0") else b.replace("k", ", K=").replace("0.08", "$\\beta$=0.08")), fontsize=10)
    ax0.grid(alpha=.25)
    ax1.set_ylim(-.05, 1.05); ax1.grid(alpha=.25)
    ax1.set_xlabel("step")
    if j == 0:
        ax0.set_ylabel("size of largest collective")
        ax1.set_ylabel("in a genuine ring state\n(smoothed)")
    if fr: ax1.text(.03, .92, f"mean {np.mean(fr):.0%}", transform=ax1.transAxes,
                    fontsize=9, va="top")
axes[0, 0].text(200, 30, "ring window", fontsize=8, color="#7a5c00")
fig.suptitle("A density-dependent maintenance cost turns the ring from a transient\ninto a persistent state, without abolishing colony fission",
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig("figures/crowded_traces.png", dpi=150, bbox_inches="tight")
print("saved figures/crowded_traces.png")
