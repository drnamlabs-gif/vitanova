"""Step 2, on the exact VitaNova substrate.

Same question as before but now with every flocking parameter identical to
evo_cooperative_flock.py, so the only difference between the two arms is the
controller: constant tendencies versus a NEAT network whose topology evolves.

Arm 1  sweep constant (T_sep, T_ali, T_coh) -- what fixed tendencies can do.
Arm 2  evolve a NEAT controller for the same objective.
Compared at matched group size, because angular momentum is easier in a small
group.
"""
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat import Config
from population import Population
from vecnet import VectorNetwork
from vita_substrate import VitaFlock, N_IN, N_OUT, BOID_COUNT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS, BURN = 800, 500
GRID = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]


def rollout(ctl, seed):
    f = VitaFlock(ctl, n=BOID_COUNT, seed=seed)
    P, M, G, D = [], [], [], []
    for i in range(STEPS):
        f.step()
        if i >= BURN and i % 10 == 0:
            m = f.measure()
            P.append(m["P"]); M.append(m["M"])
            G.append(m["group"]); D.append(m["nnd"])
    return tuple(float(np.mean(a)) for a in (P, M, G, D))


def milling_score(P, M, G, D):
    return float(max(0.0, M - 0.5 * P) * (G / BOID_COUNT) * min(1.0, D / 4.0))


def _const(t):
    vals = [rollout(np.array(t), s) for s in (0, 1)]
    P, M, G, D = (float(np.mean([v[k] for v in vals])) for k in range(4))
    return t, P, M, G, D, milling_score(P, M, G, D)


def _genome(g):
    # two initial conditions, not one: with a single seed evolution overfits
    # the starting configuration rather than finding a controller
    vals = [rollout(VectorNetwork(g), s) for s in (0, 1)]
    P, M, G, D = (float(np.mean([v[k] for v in vals])) for k in range(4))
    return milling_score(P, M, G, D)


def evolve(generations=40, size=150, workers=12):
    cfg = Config(n_inputs=N_IN, n_outputs=N_OUT, output_activation="tanh",
                 add_node_rate=0.05, add_connection_rate=0.12,
                 compat_threshold=3.0, target_species=8)
    pop = Population(cfg, size=size, seed=0)
    hist = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for gen in range(generations):
            for g, fv in zip(pop.genomes, pool.map(_genome, pop.genomes,
                                                   chunksize=1)):
                g.fitness = fv
            best = max(pop.genomes, key=lambda g: g.fitness)
            h, c = best.complexity()
            pop.speciate()
            hist.append(dict(gen=gen, best=best.fitness, species=len(pop.species),
                             hidden=h, conns=c,
                             threshold=round(cfg.compat_threshold, 2),
                             mean_hidden=float(np.mean([g.complexity()[0]
                                                        for g in pop.genomes]))))
            if gen % 5 == 0:
                print(f"  gen {gen:3d}  best {best.fitness:.3f}  "
                      f"species {len(pop.species):2d}  "
                      f"thr {cfg.compat_threshold:.2f}  "
                      f"best hidden {h}  mean hidden "
                      f"{hist[-1]['mean_hidden']:.2f}", flush=True)
            pop.reproduce(); pop.generation += 1
    return max(pop.genomes, key=lambda g: g.fitness), hist


if __name__ == "__main__":
    t0 = time.time()
    print("arm 1: constant tendencies")
    combos = [t for t in itertools.product(GRID, repeat=3)]
    with ProcessPoolExecutor(max_workers=12) as pool:
        const = list(pool.map(_const, combos, chunksize=4))
    const.sort(key=lambda r: -r[5])
    print(f"  {'T_sep':>6}{'T_ali':>7}{'T_coh':>7}{'P':>7}{'M':>7}{'group':>7}{'NND':>7}")
    for t, P, M, G, D, s in const[:6]:
        print(f"  {t[0]:>6}{t[1]:>7}{t[2]:>7}{P:>7.2f}{M:>7.2f}{G:>7.0f}{D:>7.1f}")
    bestP = max(const, key=lambda r: r[1])
    print(f"  most polarised: P={bestP[1]:.2f} at T={bestP[0]}, group={bestP[3]:.0f}")

    print(f"\narm 2: NEAT ({time.time()-t0:.0f}s so far)")
    best, hist = evolve()
    vals = [rollout(VectorNetwork(best), s) for s in (2, 3, 4, 5, 6)]   # held out
    P, M, G, D = (float(np.mean([v[k] for v in vals])) for k in range(4))
    h, c = best.complexity()
    print(f"  held out over 5 seeds: P={P:.2f} M={M:.2f} group={G:.0f} NND={D:.1f}"
          f"  ({h} hidden, {c} connections)")

    # matched comparison: constant tendencies whose group size is closest
    match = min(const, key=lambda r: abs(r[3] - G))
    print(f"\n  best constant overall:        M={const[0][2]:.2f} "
          f"(P={const[0][1]:.2f}, group={const[0][3]:.0f})")
    print(f"  constant at matched group:    M={match[2]:.2f} "
          f"(P={match[1]:.2f}, group={match[3]:.0f}, T={match[0]})")
    print(f"  NEAT:                         M={M:.2f} (P={P:.2f}, group={G:.0f})")
    verdict = M > max(const[0][2], match[2]) + 0.05
    print(f"\n  NEAT reaches milling that constant tendencies cannot: {verdict}")

    json.dump(dict(constant=[dict(T=list(t), P=P2, M=M2, group=G2, nnd=D2,
                                  score=s) for t, P2, M2, G2, D2, s in const],
                   neat=dict(P=P, M=M, group=G, nnd=D, hidden=h, connections=c,
                             history=hist),
                   verdict=bool(verdict), wall=round(time.time() - t0, 1)),
              open(os.path.join(HERE, "data", "vita_neat_step2.json"), "w"),
              indent=2)
    print("saved data/vita_neat_step2.json")
