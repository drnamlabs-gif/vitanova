"""Why is there only ever one species?

Compatibility distance is  c_E*E/N + c_D*D/N + c_W*Wbar,  with N the size of the
larger genome (taken as 1 below 20 genes). The structural terms are divided by
N; the weight term is not. So the same threshold means very different things for
a 2-gene XOR genome and a 27-gene flocking genome, and structural differences
become nearly invisible once genomes are large.
"""
import os, sys, random, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat import Config, Genome, Innovations

def sample(n_in, n_out, k=200, seed=0):
    cfg = Config(n_inputs=n_in, n_outputs=n_out)
    rng = random.Random(seed)
    inn = Innovations(start_node=n_in + n_out)
    gs = [Genome.minimal(cfg, inn, rng) for _ in range(k)]
    d = [gs[i].distance(gs[j]) for i in range(k) for j in range(i + 1, k)]
    return cfg, gs, inn, rng, np.array(d)

print("distance between two freshly initialised genomes\n")
print(f"{'task':<22}{'genes':>7}{'N used':>8}{'median d':>10}{'90th':>8}{'thr 3.0':>9}{'thr 1.5':>9}")
for name, (ni, no) in {"XOR (2 in, 1 out)": (2, 1),
                       "flocking (9 in, 3 out)": (9, 3),
                       "VitaNova (11 in, 6 out)": (11, 6)}.items():
    cfg, gs, inn, rng, d = sample(ni, no)
    genes = len(gs[0].conns)
    N = 1 if genes < 20 else genes
    print(f"{name:<22}{genes:>7}{N:>8}{np.median(d):>10.2f}{np.percentile(d,90):>8.2f}"
          f"{100*(d>3.0).mean():>8.0f}%{100*(d>1.5).mean():>8.0f}%")

print("\nwhat one structural mutation is worth, against weight differences")
cfg, gs, inn, rng, d = sample(9, 3)
a = gs[0]
for label, mut in (("add one connection", "conn"), ("add one node", "node")):
    b = a.copy()
    if mut == "conn":
        b.mutate_add_connection(inn, rng)
    else:
        b.mutate_add_node(inn, rng)
    print(f"  {label:<22} distance from parent = {a.distance(b):.3f}")
print(f"  {'two random genomes':<22} distance             = {np.median(d):.3f}")
print("\n=> a structural change is worth far less than ordinary weight variation,")
print("   so speciation cannot protect it. The threshold has to be set relative")
print("   to the distance distribution, not fixed at a value borrowed from XOR.")
