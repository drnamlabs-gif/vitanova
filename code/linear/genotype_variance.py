# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Does the genotype determine the collective phenotype at all?

In the clonal arm every agent shares one controller, so a run is fully specified
by (genotype, initial condition). Those were confounded in the earlier test: the
same seed drew both. Here they are crossed, so the variance in collective
structure can be partitioned between them. If genotype explains little, then
selection has no collective-level variation to act on, which would explain why
removing evolution changes nothing.
"""
import os, sys, json
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__))
for d in ("..","../fast","../reproduce"): sys.path.insert(0, os.path.join(HERE,d))
import shape, linear_vita as L
from runfast import clusters

GENOTYPES, WORLDS, STEPS = 12, 8, 9000

def one(a):
    gi, wi = a
    g = np.random.default_rng(10_000 + gi).standard_normal(L.GENOME)
    w = L.LinearVita(seed=20_000 + wi, MAX_BOIDS=600, MUTATION_RATE=0.0)
    w.G = np.repeat(g[None, :], w.n, axis=0)
    best, largest = 0.0, 0
    for i in range(STEPS):
        if w.n == 0: break
        if i % 25 == 0:
            cs = clusters(w.x, w.y, min_size=10)
            if cs:
                largest = max(largest, len(cs[0]))
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                if d: best = max(best, d["annularity"])
        w.step()
    return dict(genotype=gi, world=wi, peak_A=round(best,3), largest=largest,
                n_end=int(w.n))

if __name__ == "__main__":
    jobs=[(g,s) for g in range(GENOTYPES) for s in range(WORLDS)]
    res=[]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(HERE,"genotype_variance.json"),"w"), indent=1)

    A=np.zeros((GENOTYPES,WORLDS))
    for r in res: A[r["genotype"], r["world"]] = r["peak_A"]
    gm, wm, tot = A.mean(1), A.mean(0), A.mean()
    ss_g = WORLDS*((gm-tot)**2).sum()
    ss_w = GENOTYPES*((wm-tot)**2).sum()
    ss_t = ((A-tot)**2).sum()
    ss_e = ss_t - ss_g - ss_w
    from scipy.stats import f_oneway
    fg = f_oneway(*[A[i,:] for i in range(GENOTYPES)])
    fw = f_oneway(*[A[:,j] for j in range(WORLDS)])
    print(f"{GENOTYPES} genotypes x {WORLDS} initial conditions, peak annularity\n")
    print(f"  grand mean                       : {tot:.3f}")
    print(f"  spread across genotypes (sd)     : {gm.std():.3f}  "
          f"range {gm.min():.2f}-{gm.max():.2f}")
    print(f"  spread across initial conditions : {wm.std():.3f}  "
          f"range {wm.min():.2f}-{wm.max():.2f}")
    print()
    print(f"  variance explained by genotype           : {ss_g/ss_t:>5.1%}  "
          f"(one-way F={fg.statistic:.2f}, p={fg.pvalue:.3f})")
    print(f"  variance explained by initial condition  : {ss_w/ss_t:>5.1%}  "
          f"(one-way F={fw.statistic:.2f}, p={fw.pvalue:.3f})")
    print(f"  residual                                 : {ss_e/ss_t:>5.1%}")
