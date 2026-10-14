# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure 2 - collective division compounds into lineages."""
import sys, glob, re, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, ".")
import style; style.use()
from style import ACC, ACC2, ACC3, MUTED, INK, GRID, panel_label

C = dict(cid=0, birth=1, parent=2, max_size=3, fission=4, fsize=5)
D = "/Volumes/DATA/ETI Presentation/papers/shared/data"

def load(seed):
    col = np.load(f"{D}/seed{seed}.npz", allow_pickle=True)["collectives"]
    par = {int(r[C["cid"]]): int(r[C["parent"]]) for r in col}
    born = {int(r[C["cid"]]): float(r[C["birth"]]) for r in col}
    fis = {int(r[C["cid"]]): float(r[C["fission"]]) for r in col}
    fsz = {int(r[C["cid"]]): float(r[C["fsize"]]) for r in col}
    dep = {}
    for c in par:
        d, p, seen = 0, par[c], {c}
        while p >= 0 and p in par and p not in seen:
            d += 1; seen.add(p); p = par[p]
        dep[c] = d
    return par, born, fis, fsz, dep

fig = plt.figure(figsize=(6.9, 3.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2.0, 1, 1],
                      hspace=0.55, wspace=0.42)

# (a) the genealogy of one run
SEED = 14
par, born, fis, fsz, dep = load(SEED)
a = fig.add_subplot(gs[:, 0])
end = {c: (fis[c] if fis[c] >= 0 else 8950) for c in par}
ys = {}
for c in sorted(par, key=lambda k: (dep[k], born[k])):
    ys[c] = dep[c] + 0.16 * (sum(1 for k in ys if dep[k] == dep[c]) % 5 - 2)
for c in par:
    a.plot([born[c], end[c]], [ys[c]] * 2, color=ACC, lw=1.0, alpha=.75,
           solid_capstyle="round", zorder=2)
    p = par[c]
    if p >= 0 and p in ys:
        a.plot([born[c], born[c]], [ys[p], ys[c]], color=MUTED, lw=.6,
               alpha=.7, zorder=1)
a.set_xlabel("step"); a.set_ylabel("collective generation")
a.set_xlim(0, 9000); a.set_ylim(-0.8, max(dep.values()) + 0.8)
a.set_title(f"seed {SEED}: {len(par)} collectives, "
            f"{max(dep.values())} generations", fontsize=8, pad=4)
panel_label(a, "a", dx=-0.13, dy=1.10)

# (b) lineage depth across runs
b = fig.add_subplot(gs[0, 1:])
rows = []
for f in sorted(glob.glob(f"{D}/seed*.npz")):
    sd = int(re.search(r"seed(\d+)", f).group(1))
    p2, _, fi, _, d2 = load(sd)
    rows.append((sd, len(p2), sum(1 for v in fi.values() if v >= 0),
                 max(d2.values()) if d2 else 0))
rows.sort(key=lambda r: -r[3])
b.bar(range(len(rows)), [r[3] for r in rows],
      color=[ACC if r[3] else "#d8dde3" for r in rows], width=.75, lw=0)
b.set_xticks([]); b.set_xlabel("16 independent runs")
b.set_ylabel("deepest\nlineage", linespacing=1.15)
panel_label(b, "b", dx=-0.17, dy=1.16)

# (c) offspring per collective
c = fig.add_subplot(gs[1, 1])
kids = []
for f in sorted(glob.glob(f"{D}/seed*.npz")):
    sd = int(re.search(r"seed(\d+)", f).group(1))
    p2, *_ = load(sd)
    cnt = {k: 0 for k in p2}
    for k, p in p2.items():
        if p >= 0 and p in cnt: cnt[p] += 1
    kids += list(cnt.values())
kids = np.array(kids, float)
vals, cts = np.unique(kids, return_counts=True)
c.bar(vals, cts, color=ACC, width=.65, lw=0)
c.set_xlabel("offspring per\ncollective", linespacing=1.15)
c.set_ylabel(f"collectives")
c.set_xticks([0, 1, 2, 3])
panel_label(c, "c", dx=-0.42, dy=1.16)

# (d) daughter / parent size
d = fig.add_subplot(gs[1, 2])
ratio = []
for f in sorted(glob.glob(f"{D}/seed*.npz")):
    sd = int(re.search(r"seed(\d+)", f).group(1))
    p2, _, _, fz, _ = load(sd)
    for k, p in p2.items():
        if p >= 0 and fz.get(p, -1) > 0 and fz.get(k, -1) > 0:
            ratio.append(fz[k] / fz[p])
ratio = np.array(ratio, float)
d.hist(ratio, bins=np.linspace(0, 2.2, 22), color=ACC, edgecolor="white", lw=.5)
d.axvline(1.0, color=ACC2, lw=1.2, ls="--")
d.set_xlabel("daughter size /\nparent size", linespacing=1.15)
d.set_ylabel("pairs")
panel_label(d, "d", dx=-0.42, dy=1.16)

fig.savefig("figures/fig2_lineage.pdf", dpi=300)
fig.savefig("figures/fig2_lineage.png", dpi=200)
print(f"seed {SEED}: {len(par)} collectives, deepest {max(dep.values())} generations")
print(f"across 16 runs: deepest lineages {[r[3] for r in rows]}")
print(f"offspring per collective: mean {kids.mean():.2f}, var {kids.var():.2f}, "
      f"I={kids.var()/kids.mean()**2:.2f}, {int((kids==0).sum())}/{len(kids)} leave none")
print(f"daughter/parent size ratio: median {np.median(ratio):.2f} (n={len(ratio)})")
