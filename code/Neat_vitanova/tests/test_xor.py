"""The standard NEAT correctness check.

XOR is not linearly separable, so a network with no hidden node cannot solve it.
A correct NEAT must therefore *grow structure* to succeed, which makes this a
test of the structural mutation and crossover machinery and not only of weight
search. Reported: how often it solves, how long it takes, and how much structure
it adds."""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat import Config, Network
from population import Population

XOR = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]


def evaluate(g):
    net = Network(g)
    err = sum((net(list(x))[0] - y) ** 2 for x, y in XOR)
    return 4.0 - err


def solved(g):
    net = Network(g)
    return all((net(list(x))[0] > 0.5) == (y > 0.5) for x, y in XOR)


def main(runs=20, generations=300, size=150):
    out, t0 = [], time.time()
    for r in range(runs):
        cfg = Config(n_inputs=2, n_outputs=1)
        pop = Population(cfg, size=size, seed=r)
        best = pop.run(evaluate, generations, target=3.9)
        ok = solved(best)
        h, c = best.complexity()
        out.append(dict(run=r, solved=bool(ok), generations=pop.generation,
                        fitness=round(best.fitness, 4), hidden=h, connections=c,
                        history=pop.history))
        print(f"run {r:2d}: {'solved' if ok else 'FAILED'} "
              f"in {pop.generation:3d} generations, fitness {best.fitness:.3f}, "
              f"{h} hidden nodes, {c} connections", flush=True)
    n_ok = sum(o["solved"] for o in out)
    gens = [o["generations"] for o in out if o["solved"]]
    hid = [o["hidden"] for o in out if o["solved"]]
    summary = dict(runs=runs, solved=n_ok, rate=n_ok / runs,
                   median_generations=statistics.median(gens) if gens else None,
                   mean_hidden_nodes=round(statistics.fmean(hid), 2) if hid else None,
                   wall_seconds=round(time.time() - t0, 1))
    print("\n" + json.dumps(summary, indent=2))
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(summary=summary, runs=out),
              open(os.path.join(d, "xor_benchmark.json"), "w"))
    print("saved data/xor_benchmark.json")
    return n_ok == runs


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
