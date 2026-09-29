"""Selection, or a lottery?

If selection is doing the work, the hundred founding lineages should be culled
hard: a few controllers out-reproduce the rest and their descendants take over.
If it is only a lottery -- any viable lineage will do -- the survivors should
thin out no faster than drift alone would thin them.

The clonal arm is the control that makes this measurable. Every boid there
carries an identical controller, so nothing can be selected *on*; whatever
lineage collapse happens in that arm is pure drift. The comparison is therefore
between lineage collapse with genetic variation present and lineage collapse
with it absent, in an otherwise identical world.
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

STEPS, MAXB, N0 = 9000, 600, 100


def one(a):
    arm, seed = a
    kw = dict(MAX_BOIDS=MAXB)
    if arm != "evolving":
        kw["MUTATION_RATE"] = 0.0
    w = L.LinearVita(seed=seed, **kw)
    if arm == "clonal":
        g = np.random.default_rng(seed).standard_normal(L.GENOME)
        w.G = np.repeat(g[None, :], w.n, axis=0)
    ann = []
    for i in range(STEPS):
        if i % 100 == 0:
            cs = clusters(w.x, w.y)
            if cs:
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                ann.append(d["annularity"] if d else 0.0)
        if w.n == 0:
            break
        w.step()
    if w.n < 20:
        return dict(arm=arm, seed=seed, survived=False)

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
    cnt = Counter(roots)
    share = np.array(sorted(cnt.values(), reverse=True), float) / len(roots)
    simpson = float((share ** 2).sum())          # 1 = one lineage owns everything
    return dict(arm=arm, seed=seed, survived=True, n_end=int(w.n),
                founders_left=len(cnt), founders_start=N0,
                largest_share=round(float(share[0]), 3),
                simpson=round(simpson, 3),
                effective_lineages=round(1.0 / simpson, 2),
                generations=round(float(np.mean(
                    [len(str(1)) for _ in range(1)])), 0),
                annularity=round(float(max(ann) if ann else 0.0), 2),
                births=len(w.births))


if __name__ == "__main__":
    lo, hi = int(os.environ.get("LO", 1)), int(os.environ.get("HI", 81))
    arms = os.environ.get("ARMS", "evolving,no-mutation,clonal").split(",")
    jobs = [(a, s) for a in arms for s in range(lo, hi)]
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "lineage_sorting.json"), "w"),
              indent=2)
    print(f"\n{'arm':<13}{'survivors':>11}{'founders left':>15}"
          f"{'largest share':>15}{'effective lineages':>20}{'annularity':>12}")
    print("-" * 88)
    for a in arms:
        s = [r for r in rows if r["arm"] == a and r["survived"]]
        if not s:
            print(f"{a:<13}{0:>11}")
            continue
        f_ = lambda k: np.mean([r[k] for r in s])
        print(f"{a:<13}{len(s):>11}{f_('founders_left'):>15.1f}"
              f"{f_('largest_share'):>15.2f}{f_('effective_lineages'):>20.2f}"
              f"{f_('annularity'):>12.2f}")
    print(f"\nall runs start from {N0} founding lineages")
