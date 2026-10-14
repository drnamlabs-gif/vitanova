# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 2: can evolved tendency controllers reach collective states, including
one that fixed weights could not?

Objective A (school): maximise within-group polarisation. A sanity check --
fixed rules already do this, so NEAT should too.

Objective B (milling): maximise within-group angular momentum while keeping
polarisation low. Step 1 swept the fixed-weight model and found no milling state
anywhere: the best non-collapsed group reached M = 0.20. If evolution finds a
controller that beats that, then context-dependent tendencies reach collective
states that constant tendencies cannot.

Both objectives are external fitness functions. They exist only to test the
machinery and are removed in step 3.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat import Config
from population import Population
from vecnet import VectorNetwork
from neat_flock import NeatFlock, N_IN, N_OUT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS, BURN, NBOIDS, SEEDS = 500, 300, 40, (0, 1)


def rollout(genome, seed):
    f = NeatFlock(VectorNetwork(genome), n=NBOIDS, seed=seed)
    P, M, G, D = [], [], [], []
    for i in range(STEPS):
        f.step()
        if i >= BURN and i % 5 == 0:
            m = f.measure()
            P.append(m["P"]); M.append(m["M"])
            G.append(m["group"]); D.append(m["nnd"])
    return (float(np.mean(P)), float(np.mean(M)),
            float(np.mean(G)), float(np.mean(D)))


def score(genome, objective):
    vals = [rollout(genome, s) for s in SEEDS]
    P = np.mean([v[0] for v in vals])
    M = np.mean([v[1] for v in vals])
    G = np.mean([v[2] for v in vals]) / NBOIDS
    D = np.mean([v[3] for v in vals])
    collapse = min(1.0, D / 6.0)            # discourage piling onto one point
    if objective == "school":
        return float(P * G * collapse)
    return float(max(0.0, M - 0.5 * P) * G * collapse)


def _eval(args):
    return score(*args)


def evolve(objective, generations=40, size=60, seed=0, workers=12):
    cfg = Config(n_inputs=N_IN, n_outputs=N_OUT, output_activation="tanh",
                 add_node_rate=0.04, add_connection_rate=0.10,
                 compat_threshold=3.0)
    pop = Population(cfg, size=size, seed=seed)
    history = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for gen in range(generations):
            fits = list(pool.map(_eval, [(g, objective) for g in pop.genomes],
                                 chunksize=2))
            for g, fv in zip(pop.genomes, fits):
                g.fitness = fv
            best = max(pop.genomes, key=lambda g: g.fitness)
            h, c = best.complexity()
            pop.speciate()
            history.append(dict(generation=gen, best=best.fitness,
                                mean=float(np.mean(fits)),
                                species=len(pop.species), hidden=h, conns=c))
            if gen % 5 == 0:
                print(f"  gen {gen:3d}  best {best.fitness:.3f}  "
                      f"species {len(pop.species):2d}  hidden {h}  conns {c}",
                      flush=True)
            pop.reproduce()
            pop.generation += 1
    best = max(pop.genomes, key=lambda g: g.fitness)
    return best, history


if __name__ == "__main__":
    out = {}
    for objective in ("school", "milling"):
        print(f"\n=== objective: {objective} ===", flush=True)
        t0 = time.time()
        best, hist = evolve(objective)
        vals = [rollout(best, s) for s in (0, 1, 2, 3, 4)]
        P = float(np.mean([v[0] for v in vals]))
        M = float(np.mean([v[1] for v in vals]))
        G = float(np.mean([v[2] for v in vals]))
        D = float(np.mean([v[3] for v in vals]))
        h, c = best.complexity()
        out[objective] = dict(P=round(P, 3), M=round(M, 3), group=round(G, 1),
                              nnd=round(D, 1), hidden=h, connections=c,
                              fitness=round(best.fitness, 3),
                              wall=round(time.time() - t0, 1), history=hist,
                              genome=dict(
                                  nodes={str(k): v[0] for k, v in best.nodes.items()},
                                  conns=[[c2.a, c2.b, round(c2.w, 3), c2.enabled]
                                         for c2 in best.conns.values()]))
        print(f"  -> held-out over 5 seeds: P={P:.2f}  M={M:.2f}  "
              f"group={G:.0f}/{NBOIDS}  NND={D:.1f}  "
              f"({h} hidden, {c} connections)")
    out["fixed_rule_reference"] = dict(
        note="best non-collapsed group over the step-1 sweep",
        max_M=0.20, max_P=0.96)
    json.dump(out, open(os.path.join(HERE, "data", "neat_flock.json"), "w"),
              indent=2)
    print("\nfixed rules reached M = 0.20 at best;  "
          f"NEAT milling controller reached M = {out['milling']['M']}")
    print("saved data/neat_flock.json")
