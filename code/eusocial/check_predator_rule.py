"""Is the large-group predator rule inert in the caste-free model?

In vitafast._predators the large-group branch sits in an `elif` after the
soldier-reversal branch. In nocaste/crowded nobody is ever a soldier, so the
first branch can never fire and the large-group rule -- a COLLECTIVE-SIZE
dependent rule -- is live throughout the experiments that produced the paper's
main result. This was described in the manuscript as dead code.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE,"..","fast"), os.path.join(HERE,"..","reproduce")):
    sys.path.insert(0,p)
import shape, crowded as CR, vitafast as VF
from runfast import clusters, verdict
R_GUARD=1.5*VF.PERCEPTION_RADIUS

def one(a):
    rule_on, seed, steps = a
    kw = dict(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
    if not rule_on:
        kw["LARGE_GROUP_THRESHOLD"] = None
    w = CR.Crowded(**kw)
    ring, win, big = [], [], []
    for i in range(steps):
        if w.n == 0: break
        if i % 25 == 0:
            cs = clusters(w.x, w.y)
            d = shape.annularity(w.x[cs[0]], w.y[cs[0]]) if cs else None
            big.append(len(cs[0]) if cs else 0)
            win.append(bool(cs) and 20 <= len(cs[0]) <= 55)
            ring.append(bool(d and d["annularity"] > .5 and d["r_max"] < R_GUARD))
        w.step()
    return dict(rule=rule_on, seed=seed, ring=float(np.mean(ring)),
                window=float(np.mean(win)), n_end=int(w.n),
                largest=float(np.median([b for b in big if b]) if any(big) else 0))

if __name__ == "__main__":
    seeds=[1,3,4,5,6,7,9,12,14,15]
    jobs=[(r,s,9000) for r in (True,False) for s in seeds]
    res=[]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(HERE,"predator_rule.json"),"w"), indent=1)
    from scipy.stats import wilcoxon
    A={r["seed"]:r for r in res if r["rule"]}
    B={r["seed"]:r for r in res if not r["rule"]}
    print(f"{'':>26} {'rule ON':>9} {'rule OFF':>9} {'paired p':>10}")
    print("-"*58)
    for k,lab in (("ring","fraction organised"),("window","fraction in window"),
                  ("largest","median largest collective")):
        x=np.array([A[s][k] for s in seeds]); y=np.array([B[s][k] for s in seeds])
        try: p=f"{wilcoxon(x,y).pvalue:.3f}"
        except ValueError: p="tied"
        print(f"{lab:>26} {np.median(x):>9.3f} {np.median(y):>9.3f} {p:>10}")
