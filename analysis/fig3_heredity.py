"""Figure 3 - collective size is inherited across the lineage."""
import sys, glob, re, json, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

C = dict(cid=0, birth=1, parent=2, max_size=3, fission=4, fsize=5)
D = "/Volumes/DATA/ETI Presentation/papers/shared/data"
rng = np.random.default_rng(0)

P, K, REP, PH = [], [], [], []
for f in sorted(glob.glob(f"{D}/seed*.npz")):
    sd = int(re.search(r"seed(\d+)", f).group(1))
    col = np.load(f, allow_pickle=True)["collectives"]
    fsz = {int(r[C["cid"]]): float(r[C["fsize"]]) for r in col}
    fis = {int(r[C["cid"]]): float(r[C["fission"]]) for r in col}
    for r in col:
        c, p = int(r[C["cid"]]), int(r[C["parent"]])
        if p < 0 or fsz.get(p, -1) <= 0 or fsz.get(c, -1) <= 0: continue
        P.append(fsz[p]); K.append(fsz[c]); REP.append(sd)
        PH.append(int(fis[c] // 3000))
P, K = np.array(P, float), np.array(K, float)
REP, PH = np.array(REP), np.array(PH)

def slope(x, y):
    return np.polyfit(x, y, 1)[0], np.corrcoef(x, y)[0, 1]

def centred(x, y, groups):
    xc, yc = x.copy(), y.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() > 1:
            xc[m] -= x[m].mean(); yc[m] -= y[m].mean()
    return xc, yc

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.42})

# (a) the regression
s0, r0 = slope(np.log10(P), np.log10(K))
a = ax[0]
a.scatter(P, K, s=9, c=ACC, lw=0, alpha=.45, zorder=3)
xs = np.logspace(np.log10(P.min()), np.log10(P.max()), 50)
a.plot(xs, 10 ** (np.polyval(np.polyfit(np.log10(P), np.log10(K), 1),
                             np.log10(xs))), color=ACC2, lw=1.6, zorder=4)
a.plot(xs, xs, color=INK, lw=.8, ls="--", zorder=2)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("parent size at its division")
a.set_ylabel("daughter size at\nits own division", linespacing=1.15)
a.set_title(f"$n={len(P)}$, slope ${s0:+.2f}$, $r={r0:+.2f}$", fontsize=7.8, pad=3)
panel_label(a, "a", dx=-0.32, dy=1.16)

# (b) the control ladder
b = ax[1]
lad = [("none", *slope(np.log10(P), np.log10(K)))]
for lab, g in (("replicate", REP), ("phase", PH)):
    xc, yc = centred(np.log10(P), np.log10(K), g)
    lad.append((lab, *slope(xc, yc)))
both = np.array([f"{r}_{p}" for r, p in zip(REP, PH)])
xc, yc = centred(np.log10(P), np.log10(K), both)
lad.append(("replicate\n$\\times$ phase", *slope(xc, yc)))
ys = np.arange(len(lad))
b.barh(ys, [l[1] for l in lad], color=ACC, height=.6, lw=0)
b.set_yticks(ys); b.set_yticklabels([l[0] for l in lad], fontsize=7,
                                    linespacing=1.1)
b.invert_yaxis(); b.set_xlabel("parent--offspring slope")
b.set_xlim(0, 1.0); b.grid(False, axis="y")
panel_label(b, "b", dx=-0.42, dy=1.16)

# (c) permutation null
c = ax[2]
# null: shuffle which daughter goes with which parent, within replicate x phase,
# so the group structure and the marginal distributions are preserved
obs = lad[-1][1]
groups = [np.flatnonzero(both == g) for g in np.unique(both)]
lp, lk = np.log10(P), np.log10(K)
null = []
for _ in range(2000):
    ks = lk.copy()
    for idx in groups:
        if len(idx) > 1: ks[idx] = lk[rng.permutation(idx)]
    xc2, yc2 = centred(lp, ks, both)
    null.append(slope(xc2, yc2)[0])
null = np.array(null)
c.hist(null, bins=34, color="#d8dde3", edgecolor="white", lw=.4)
c.axvline(obs, color=ACC2, lw=1.8)
c.set_xlabel("slope"); c.set_ylabel("permutations")
p = (np.abs(null) >= abs(obs)).mean()
c.set_title(f"observed ${obs:+.2f}$, null ${null.mean():+.2f}\\pm{null.std():.2f}$,"
            f" $p<{max(p,1/len(null)):.4f}$", fontsize=7, pad=3)
panel_label(c, "c", dx=-0.34, dy=1.16)

fig.savefig("figures/fig3_heredity.pdf", dpi=300)
fig.savefig("figures/fig3_heredity.png", dpi=200)
for l in lad: print(f"  control {l[0]:>20}: slope {l[1]:+.3f}, r {l[2]:+.3f}")
print(f"  n={len(P)} parent-offspring pairs across {len(np.unique(REP))} runs")
print(f"  permutation null {null.mean():+.3f} +/- {null.std():.3f}, "
      f"observed {obs:+.3f}, p<{max(p,1/len(null)):.4f}")
