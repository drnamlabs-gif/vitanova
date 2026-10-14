# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Ancestor versus descendant.

The standard test in experimental evolution: revive the ancestor and compare it
with its descendant under identical conditions. Here the ancestor is the
original, unmutated controller of whichever founding lineage came to dominate a
run, and the descendant is that lineage's controller at the end of the run.

Both are then run clonally -- every boid carrying the same controller, no
mutation, no variation -- so the comparison measures the controller itself and
nothing else. If descendants outperform ancestors, mutation improved them and
there is adaptation. If they match, the run was sorting and nothing more.
"""
import json, os, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../fast", "../reproduce"):
    sys.path.insert(0, os.path.join(HERE, d))
import shape                                # noqa: E402
import linear_vita as L                     # noqa: E402
from runfast import clusters                # noqa: E402

STEPS, MAXB = 9000, 600


def evaluate(genome, seed, steps=6000):
    """Run one controller clonally and score it."""
    w = L.LinearVita(seed=seed, MAX_BOIDS=MAXB, MUTATION_RATE=0.0)
    w.G = np.repeat(np.asarray(genome, float)[None, :], w.n, axis=0)
    ann, npop = [], []
    for i in range(steps):
        if i % 100 == 0:
            cs = clusters(w.x, w.y)
            if cs:
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                ann.append(d["annularity"] if d else 0.0)
            npop.append(w.n)
        if w.n == 0:
            break
        w.step()
    return dict(survived=bool(w.n > 20), n_end=int(w.n),
                n_max=int(max(npop) if npop else 0),
                annularity=round(float(max(ann) if ann else 0.0), 3),
                births=len(w.births))


def harvest(seed):
    """Run the evolving model, then return the winning lineage's ancestral and
    descendant controllers."""
    w = L.LinearVita(seed=seed, MAX_BOIDS=MAXB)
    G0 = w.G.copy()                      # the hundred founding controllers
    for i in range(STEPS):
        if w.n == 0:
            break
        w.step()
    if w.n < 20:
        return None
    pm = {int(c): int(p) for _, p, c in w.births}
    cache = {}
    def founder(u):
        path = []
        while True:
            if u in cache:
                root = cache[u]; break
            p = pm.get(u, -1)
            if p < 0:
                root = u; break
            path.append(u); u = p
        for q in path:
            cache[q] = root
        return root
    roots = [founder(int(u)) for u in w.uid]
    win, count = Counter(roots).most_common(1)[0]
    if win >= len(G0):
        return None
    members = np.flatnonzero(np.array(roots) == win)
    return dict(seed=seed, winner=int(win), share=count / len(roots),
                ancestor=G0[win].tolist(),
                descendant=w.G[members].mean(0).tolist(),
                drift=float(np.linalg.norm(w.G[members].mean(0) - G0[win])))


def _job(seed):
    h = harvest(seed)
    if h is None:
        return None
    out = dict(seed=seed, share=round(h["share"], 2),
               drift=round(h["drift"], 2), ancestor=[], descendant=[])
    for tag, g in (("ancestor", h["ancestor"]), ("descendant", h["descendant"])):
        scores = [evaluate(g, s) for s in (101, 102, 103)]
        out[tag] = dict(
            survived=sum(s["survived"] for s in scores),
            annularity=round(float(np.mean([s["annularity"] for s in scores])), 3),
            n_max=round(float(np.mean([s["n_max"] for s in scores])), 1))
    return out


if __name__ == "__main__":
    seeds = list(range(1, 61))
    print(f"harvesting winners from {len(seeds)} runs, then reviving ancestors",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(_job, s) for s in seeds]), 1):
            r = f.result()
            if r:
                rows.append(r)
            if k % 10 == 0:
                print(f"  {k}/{len(seeds)} runs, {len(rows)} with a winner",
                      flush=True)
    json.dump(rows, open(os.path.join(HERE, "resurrection.json"), "w"), indent=2)
    if not rows:
        print("no winners harvested"); sys.exit()
    A = np.array([[r["ancestor"]["annularity"], r["ancestor"]["survived"],
                   r["ancestor"]["n_max"]] for r in rows])
    D = np.array([[r["descendant"]["annularity"], r["descendant"]["survived"],
                   r["descendant"]["n_max"]] for r in rows])
    print(f"\n{len(rows)} winning lineages revived, each scored on 3 fresh worlds")
    print(f"{'':<14}{'annularity':>12}{'survived /3':>14}{'peak population':>18}")
    print("-" * 60)
    print(f"{'ancestor':<14}{A[:,0].mean():>12.3f}{A[:,1].mean():>14.2f}{A[:,2].mean():>18.1f}")
    print(f"{'descendant':<14}{D[:,0].mean():>12.3f}{D[:,1].mean():>14.2f}{D[:,2].mean():>18.1f}")
    from scipy import stats
    for i, lab in enumerate(("annularity", "survival", "peak population")):
        try:
            p = stats.wilcoxon(D[:, i], A[:, i], alternative="greater").pvalue
        except Exception:
            p = float("nan")
        print(f"  descendant > ancestor on {lab:<16} p = {p:.3f}")
    print(f"\nmean genome distance travelled by the winning lineage: "
          f"{np.mean([r['drift'] for r in rows]):.2f}")
