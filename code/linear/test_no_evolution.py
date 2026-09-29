"""Does evolution contribute anything to the morphology?

Three arms, identical in every other respect.

  evolving   the normal model: variation at the start, mutation at every birth,
             and differential reproduction. Evolution operates.
  no-mutation  initial random variation is kept, but offspring are exact clones.
             Selection can still sort the initial genomes; no new variation.
  clonal     every boid starts with the *same* random controller and offspring
             are exact clones. There is no variation at all, so there is no
             selection and no evolution of any kind.

If rings and divisions appear in the clonal arm, then the morphology is a
property of the rules and the ecology, and evolution is not producing it.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../fast", "../reproduce"):
    sys.path.insert(0, os.path.join(HERE, d))
import shape                                # noqa: E402
import linear_vita as L                     # noqa: E402
from runfast import clusters, verdict       # noqa: E402

ARMS = ("evolving", "no-mutation", "clonal")
STEPS, MAXB, EVERY = 9000, 600, 25


def one(a):
    arm, seed = a
    kw = dict(MAX_BOIDS=MAXB)
    if arm != "evolving":
        kw["MUTATION_RATE"] = 0.0          # offspring are exact clones
    w = L.LinearVita(seed=seed, **kw)
    if arm == "clonal":                    # one controller for everybody
        g = np.random.default_rng(seed).standard_normal(L.GENOME)
        w.G = np.repeat(g[None, :], w.n, axis=0)
    rec = {k: [] for k in ("step", "n", "n_clusters", "largest", "annularity")}
    t0 = time.time()
    for i in range(STEPS):
        if i % EVERY == 0:
            cs = clusters(w.x, w.y)
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            rec["step"].append(i); rec["n"].append(w.n)
            rec["n_clusters"].append(len(cs))
            rec["largest"].append(len(big) if big is not None else 0)
            rec["annularity"].append(d["annularity"] if d else 0.0)
        if w.n == 0:
            break
        w.step()
    R = {k: np.array(v) for k, v in rec.items()}
    R["seed"] = seed
    R["wall"] = time.time() - t0
    v = verdict(R)
    # how much genetic variation is left at the end
    spread = float(np.mean(np.std(w.G, axis=0))) if w.n > 1 else 0.0
    return dict(arm=arm, seed=seed, n_max=int(R["n"].max()),
                n_end=int(R["n"][-1]), largest=int(R["largest"].max()),
                annular=round(float(R["annularity"].max()), 2),
                divided=bool(v["divided"]), parent=v["parent_size"],
                genetic_sd=round(spread, 3),
                survived=bool(R["n"].max() > 150),
                wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    lo = int(os.environ.get("LO", 1)); hi = int(os.environ.get("HI", 17))
    arms = os.environ.get("ARMS", ",".join(ARMS)).split(",")
    seeds = list(range(lo, hi))
    jobs = [(arm, s) for arm in arms for s in seeds]
    print(f"{len(jobs)} runs: {len(ARMS)} arms x {len(seeds)} seeds", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 12 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    path = os.path.join(HERE, "no_evolution.json")
    if os.environ.get("APPEND") and os.path.exists(path):
        rows = json.load(open(path)) + rows
    json.dump(rows, open(path, "w"), indent=2)
    print(f"\n{'arm':<13}{'survived':>10}{'annularity (survivors)':>26}"
          f"{'divisions':>11}{'parent':>9}{'genetic sd':>12}")
    print("-" * 82)
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        sv = [r for r in g if r["survived"]]
        an = [r["annular"] for r in sv]
        dv = [r for r in sv if r["divided"]]
        ps = [r["parent"] for r in dv]
        print(f"{arm:<13}{len(sv)}/{len(g):<8}"
              f"{(f'{np.median(an):.2f}  (max {max(an):.2f})' if an else '—'):>26}"
              f"{f'{len(dv)}/{len(sv)}' if sv else '—':>11}"
              f"{(f'{np.mean(ps):.0f}' if ps else '—'):>9}"
              f"{np.mean([r['genetic_sd'] for r in sv]) if sv else 0:>12.3f}")
    print("\nreference, fixed 11-10-6 network: annularity 0.92, parent 17.0 +/- 2.2")
