"""Map the flocking substrate onto the polarisation / angular-momentum plane.

Rather than assert three hand-picked regimes, sweep the alignment and cohesion
weights and locate the states. A genuine milling torus is high angular momentum,
low polarisation, and *not* collapsed (nearest-neighbour distance well above the
separation radius)."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flock import Flock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALI = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
COH = [0.25, 0.5, 1.0, 2.0, 4.0]


def cell(a):
    w_ali, w_coh, seed = a
    f = Flock(n=120, seed=seed, w_sep=2.0, w_ali=w_ali, w_coh=w_coh,
              r_sep=15.0, r_ali=25.0, r_coh=60.0, noise=0.05)
    P, M, D, G, N = [], [], [], [], []
    for i in range(1500):
        f.step()
        if i >= 800 and i % 5 == 0:
            p, m, k = f.largest_group_order()     # per-group, not global
            P.append(p); M.append(m); D.append(f.nnd())
            G.append(k); N.append(len(f.clusters()))
    return (w_ali, w_coh, seed, float(np.mean(P)), float(np.mean(M)),
            float(np.mean(D)), float(np.mean(G)), float(np.mean(N)))


if __name__ == "__main__":
    jobs = [(a, c, s) for a in ALI for c in COH for s in (0, 1)]
    res = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(cell, j) for j in jobs]):
            res.append(f.result())
    grid = {}
    for a, c, s, p, m, d, g, nc in res:
        grid.setdefault((a, c), []).append((p, m, d, g, nc))
    P = np.array([[np.mean([v[0] for v in grid[(a, c)]]) for c in COH] for a in ALI])
    M = np.array([[np.mean([v[1] for v in grid[(a, c)]]) for c in COH] for a in ALI])
    D = np.array([[np.mean([v[2] for v in grid[(a, c)]]) for c in COH] for a in ALI])
    G = np.array([[np.mean([v[3] for v in grid[(a, c)]]) for c in COH] for a in ALI])
    NC = np.array([[np.mean([v[4] for v in grid[(a, c)]]) for c in COH] for a in ALI])

    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.1))
    for k, (Z, lab) in enumerate([(P, "polarisation $P$ (largest group)"),
                                  (M, "angular momentum $M$ (largest group)"),
                                  (D, "nearest-neighbour distance"),
                                  (G, "largest group size")]):
        im = ax[k].imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        ax[k].set_xticks(range(len(COH))); ax[k].set_xticklabels(COH, fontsize=7)
        ax[k].set_yticks(range(len(ALI))); ax[k].set_yticklabels(ALI, fontsize=7)
        ax[k].set_xlabel("cohesion weight"); ax[k].set_ylabel("alignment weight")
        ax[k].set_title(lab, fontsize=9)
        fig.colorbar(im, ax=ax[k], fraction=.046)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "s1_phase_diagram.pdf"))

    best = max(((a, c, np.mean([v[1] for v in grid[(a, c)]]),
                 np.mean([v[0] for v in grid[(a, c)]]),
                 np.mean([v[2] for v in grid[(a, c)]]))
                for a in ALI for c in COH),
               key=lambda t: t[2] - t[3] + min(t[4] / 15.0, 1.0))
    out = dict(alignment=ALI, cohesion=COH, P=P.tolist(), M=M.tolist(),
               nnd=D.tolist(), group=G.tolist(), n_clusters=NC.tolist(),
               best_torus=dict(w_ali=best[0], w_coh=best[1],
                               M=round(best[2], 3), P=round(best[3], 3),
                               nnd=round(best[4], 2)))
    json.dump(out, open(os.path.join(HERE, "data", "phase_diagram.json"), "w"),
              indent=2)
    print(f"{'w_ali':>6}{'w_coh':>7}{'P':>7}{'M':>7}{'NND':>7}{'group':>7}{'#cl':>6}")
    for i, a in enumerate(ALI):
        for j, c in enumerate(COH):
            print(f"{a:>6}{c:>7}{P[i,j]:>7.2f}{M[i,j]:>7.2f}{D[i,j]:>7.1f}"
                  f"{G[i,j]:>7.0f}{NC[i,j]:>6.1f}")
    print("\nbest milling candidate:", json.dumps(out["best_torus"]))
