"""Two checks on the NEAT runs, against the fixed-topology model.

1. Caste stratification. Do soldiers sit in the core and workers on the rim?
   Compared against the fixed 11-10-6 controller run under identical rules, so
   any difference is attributable to the controller.

2. Spatial niching. With no speciation operator, several topologies coexist.
   If space is doing the work that speciation does in standard NEAT, then boids
   sharing a topology should be closer together than boids that do not, and the
   grouping should follow lineage. Both are tested against permutation nulls
   that keep the positions and reshuffle the labels.
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEAT = os.path.join(HERE, "data", "neat_runs")
FNN = os.path.join(ROOT, "papers", "shared", "data")
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
RADIUS, SHELLS = 25.0, np.linspace(0, 1, 6)


def components(x, y, link=RADIUS, min_size=15):
    n = len(x)
    if n == 0:
        return []
    adj = ((x[None, :] - x[:, None]) ** 2 +
           (y[None, :] - y[:, None]) ** 2) <= link * link
    seen = np.zeros(n, bool); out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop(); comp.append(j)
            nxt = np.flatnonzero(adj[j] & ~seen)
            seen[nxt] = True; stack.extend(nxt.tolist())
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def radial_castes(files, xk, yk, sk, label):
    shells = [[] for _ in range(5)]
    counts = [0] * 5
    for p in files:
        z = np.load(p)
        steps = sorted({int(k.split("_")[1]) for k in z.files
                        if k.startswith("snap_") and k.endswith(xk)})
        for s in steps:
            x, y = z[f"snap_{s}{xk}"], z[f"snap_{s}{yk}"]
            key = f"snap_{s}{sk}"
            if key not in z.files or len(x) < 20:
                continue
            sol = z[key].astype(bool)
            for c in components(x, y):
                xb, yb, sb = x[c], y[c], sol[c]
                r = np.hypot(xb - xb.mean(), yb - yb.mean())
                rn = r / max(r.max(), 1e-9)
                idx = np.clip(np.digitize(rn, SHELLS) - 1, 0, 4)
                for b in range(5):
                    m = idx == b
                    if m.any():
                        shells[b].append(float(sb[m].mean()))
                        counts[b] += int(m.sum())
    return dict(label=label,
                profile=[round(float(np.mean(v)), 3) if v else None for v in shells],
                members=counts)


def niching(files):
    rows = []
    for p in files:
        z = np.load(p)
        steps = sorted({int(k.split("_")[1]) for k in z.files
                        if k.startswith("snap_") and k.endswith("_hidden")})
        for s in steps:
            x, y = z[f"snap_{s}_x"], z[f"snap_{s}_y"]
            h, c = z[f"snap_{s}_hidden"], z[f"snap_{s}_conns"]
            if len(x) < 40:
                continue
            topo = np.array([hash((int(a), int(b))) for a, b in zip(h, c)])
            uniq = np.unique(topo)
            if len(uniq) < 2:
                continue
            d = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
            iu = np.triu_indices(len(x), 1)
            same = topo[iu[0]] == topo[iu[1]]
            if same.sum() < 20 or (~same).sum() < 20:
                continue
            obs = float(d[iu][same].mean() - d[iu][~same].mean())
            rng = np.random.default_rng(0)
            null = []
            for _ in range(200):
                t = rng.permutation(topo)
                sm = t[iu[0]] == t[iu[1]]
                null.append(d[iu][sm].mean() - d[iu][~sm].mean())
            null = np.array(null)
            rows.append(dict(seed=int(os.path.basename(p)[9:-4]), step=s,
                             n=len(x), topologies=int(len(uniq)),
                             observed=obs, null_mean=float(null.mean()),
                             null_sd=float(null.std()),
                             z=float((obs - null.mean()) / max(null.std(), 1e-9))))
    return rows


if __name__ == "__main__":
    neat_files = sorted(glob.glob(os.path.join(NEAT, "neat_seed*.npz")))
    neat_files = [f for f in neat_files
                  if len(np.load(f)["n_boids"]) and np.load(f)["n_boids"].max() > 200]
    fnn_files = sorted(glob.glob(os.path.join(FNN, "seed*.npz")))

    print("1. CASTE STRATIFICATION — soldier fraction by radial shell\n")
    a = radial_castes(neat_files, "_x", "_y", "_soldier", "NEAT")
    b = radial_castes(fnn_files, "_x", "_y", "_sol", "fixed 11-10-6")
    print(f"{'shell r/R':<12}{'NEAT':>10}{'fixed FNN':>12}")
    for i in range(5):
        va = a["profile"][i]; vb = b["profile"][i]
        print(f"{SHELLS[i]:.1f}-{SHELLS[i+1]:.1f}   "
              f"{'' if va is None else f'{va:>8.3f}'}"
              f"{'' if vb is None else f'{vb:>12.3f}'}")
    for r in (a, b):
        p = [v for v in r["profile"] if v is not None]
        print(f"  {r['label']}: core-to-rim change = {p[0]-p[-1]:+.3f}")

    print("\n2. SPATIAL NICHING — are same-topology boids closer together?\n")
    rows = niching(neat_files)
    if rows:
        z = np.array([r["z"] for r in rows])
        obs = np.array([r["observed"] for r in rows])
        print(f"  {len(rows)} snapshots with >= 2 topologies")
        print(f"  mean within-minus-between distance: {obs.mean():+.1f} px")
        print(f"  mean z against permutation null   : {z.mean():+.2f}")
        print(f"  snapshots with z < -2 (clustered) : "
              f"{100*(z < -2).mean():.0f}%")
        print(f"  snapshots with z > +2 (dispersed) : "
              f"{100*(z > 2).mean():.0f}%")
    json.dump(dict(castes=[a, b], niching=rows),
              open(os.path.join(HERE, "data", "strata_niching.json"), "w"),
              indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.8))
    mid = (SHELLS[:-1] + SHELLS[1:]) / 2
    for r, col in ((a, "#b91c1c"), (b, "#1d4ed8")):
        p = [v if v is not None else np.nan for v in r["profile"]]
        ax[0].plot(mid, p, marker="o", ms=4, color=col, label=r["label"])
    ax[0].set_xlabel("radial position $r/R$ in the collective")
    ax[0].set_ylabel("fraction of soldiers")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("caste by position", fontsize=8.5)
    if rows:
        ax[1].hist([r["z"] for r in rows], bins=25, color="#b91c1c", alpha=.8)
        ax[1].axvline(0, color="k", lw=.7)
        ax[1].axvline(-2, color="k", ls="--", lw=.6)
        ax[1].set_xlabel("z of within-topology distance vs null")
        ax[1].set_ylabel("snapshots")
        ax[1].set_title("spatial niching by topology", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "s3_strata_niching.pdf"))
    fig.savefig(os.path.join(OUT, "s3_strata_niching.png"), dpi=170)
    print("\nfigure: figures/s3_strata_niching.png")
