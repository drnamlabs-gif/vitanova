"""Does an evolving flock with no energetic ecology produce the pattern?

The worker energy subsidy is what makes grouping pay. Switching it off leaves
the flocking rules, the evolvable controllers and the predators intact, and
removes the ecology. The question is whether anything resembling the collective
structure appears without it.
"""
import os, sys, json
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE,"..","fast"), os.path.join(HERE,"..","reproduce")):
    sys.path.insert(0,p)
import shape, vitafast as VF
from runfast import clusters

def one(a):
    subsidy, seed = a
    kw = dict(seed=seed, MAX_BOIDS=600)
    if not subsidy:
        kw["WORKER_TOGETHER_ENERGY"] = 0.0
    w = VF.Fast(**kw)
    best, largest, npop = 0.0, 0, []
    for i in range(9000):
        if w.n == 0: break
        if i % 25 == 0:
            cs = clusters(w.x, w.y, min_size=10)
            npop.append(w.n)
            if cs:
                largest = max(largest, len(cs[0]))
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                if d: best = max(best, d["annularity"])
        w.step()
    return dict(subsidy=subsidy, seed=seed, peak_A=round(best,3),
                largest=largest, n_max=int(max(npop)) if npop else 0,
                n_end=int(w.n))

if __name__ == "__main__":
    seeds=[1,3,4,5,7,9,11,12,13,14,15,16]
    jobs=[(s,sd) for s in (True,False) for sd in seeds]
    res=[]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(HERE,"no_ecology.json"),"w"), indent=1)
    print(f"{'ecology':>12} {'runs':>5} {'peak A med':>11} {'peak A max':>11} "
          f"{'largest med':>12} {'n_max med':>10} {'A>0.5':>7}")
    print('-'*72)
    for s,lab in ((True,'subsidy on'),(False,'subsidy off')):
        g=[x for x in res if x['subsidy']==s]
        A=np.array([x['peak_A'] for x in g]); L=np.array([x['largest'] for x in g])
        N=np.array([x['n_max'] for x in g])
        print(f"{lab:>12} {len(g):>5} {np.median(A):>11.2f} {A.max():>11.2f} "
              f"{np.median(L):>12.0f} {np.median(N):>10.0f} {(A>0.5).sum():>4}/{len(g)}")
