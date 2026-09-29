"""Is there a genetic signal that separates one collective from another?

Nestmate recognition can only work if within-collective genetic distance is
smaller than between-collective distance.  Measured live, since the saved
snapshots do not carry genomes.
"""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "fast"), os.path.join(HERE, "..", "reproduce")):
    sys.path.insert(0, p)
import crowded as CR                      # noqa: E402
from runfast import clusters              # noqa: E402


def gdist(G, i, j):
    """mean per-gene |difference|, normalised by the genome's own spread"""
    return np.abs(G[i] - G[j]).mean() / max(G.std(), 1e-9)


rows = []
for seed in (1, 3, 4, 9, 12, 15):
    w = CR.Crowded(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
    for i in range(9000):
        if w.n == 0:
            break
        if i % 250 == 0 and i >= 1000:
            cs = clusters(w.x, w.y, min_size=8)
            if len(cs) >= 2:
                a, b = cs[0], cs[1]
                wa = np.mean([gdist(w.G, p, q) for p in a[:12] for q in a[:12] if p != q])
                wb = np.mean([gdist(w.G, p, q) for p in b[:12] for q in b[:12] if p != q])
                bt = np.mean([gdist(w.G, p, q) for p in a[:12] for q in b[:12]])
                rows.append((seed, i, len(a), len(b), wa, wb, bt))
        w.step()

r = np.array([x[4:] for x in rows])
within = np.concatenate([r[:, 0], r[:, 1]])
between = r[:, 2]
print(f"{len(rows)} frames with two collectives, from 6 seeds\n")
print(f"  within-collective  genetic distance: {within.mean():.4f} +/- {within.std():.4f}")
print(f"  between-collective genetic distance: {between.mean():.4f} +/- {between.std():.4f}")
print(f"  ratio between/within               : {between.mean()/within.mean():.3f}")
from scipy.stats import wilcoxon
paired_w = (r[:, 0] + r[:, 1]) / 2
print(f"\n  paired Wilcoxon (between vs within, n={len(r)}): "
      f"p = {wilcoxon(between, paired_w).pvalue:.2g}")
print(f"  frames where between > within: {(between > paired_w).mean():.0%}")
sep = (between - paired_w) / paired_w
print(f"  median relative separation: {np.median(sep):+.1%}")
