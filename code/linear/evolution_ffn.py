# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The three-arm evolution ablation, on the original feed-forward network.

evolving     100 random controllers, mutation at every birth, selection.
no-mutation  100 random controllers kept, offspring are exact clones. Selection
             can sort the standing variation; no new variation appears.
clonal       every agent starts with the SAME random controller and offspring
             are exact clones. No variation, so selection cannot change the
             composition of the population.
"""
import os, sys, json
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__))
for d in ("..","../fast","../reproduce"): sys.path.insert(0, os.path.join(HERE,d))
import shape, vitafast as VF
from runfast import clusters

SEEDS, STEPS = 40, 9000

def one(a):
    arm, seed = a
    kw = dict(seed=seed, MAX_BOIDS=600)
    if arm != "evolving":
        kw["MUTATION_RATE"] = 0.0
    w = VF.Fast(**kw)
    if arm == "clonal":
        g = np.random.default_rng(seed).standard_normal(VF.GENOME)
        w.G = np.repeat(g[None, :], w.n, axis=0)
    best, largest, divided, prev = 0.0, 0, False, 0
    for i in range(STEPS):
        if w.n == 0: break
        if i % 25 == 0:
            cs = clusters(w.x, w.y, min_size=10)
            if len(cs) >= 2 and prev == 1: divided = True
            prev = len(cs)
            if cs:
                largest = max(largest, len(cs[0]))
                d = shape.annularity(w.x[cs[0]], w.y[cs[0]])
                if d: best = max(best, d["annularity"])
        w.step()
    return dict(arm=arm, seed=seed, peak_A=round(best,3), largest=largest,
                divided=bool(divided), n_end=int(w.n))

if __name__ == "__main__":
    arms=("evolving","no-mutation","clonal")
    jobs=[(a,s) for a in arms for s in range(1, SEEDS+1)]
    res=[]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(HERE,"evolution_ffn.json"),"w"), indent=1)
    from scipy.stats import mannwhitneyu, fisher_exact
    print(f"ORIGINAL FFN, {SEEDS} seeds per arm, {STEPS} steps\n")
    print(f"{'arm':>12} {'formed >=20':>12} {'median A':>10} {'divided':>10}")
    print('-'*48)
    S={}
    for a in arms:
        g=[x for x in res if x['arm']==a]
        est=[x for x in g if x['largest']>=20]
        A=np.array([x['peak_A'] for x in est],float)
        nd=sum(1 for x in est if x['divided'])
        S[a]=(est,A,nd)
        print(f"{a:>12} {len(est):>8}/{len(g)} {np.median(A):>10.2f} "
              f"{str(nd)+'/'+str(len(est)):>10}")
    print()
    for a,b in (("evolving","clonal"),("evolving","no-mutation")):
        print(f"  {a} vs {b}: peak A Mann-Whitney p="
              f"{mannwhitneyu(S[a][1],S[b][1]).pvalue:.4f}")
    ea,ca=S['evolving'],S['clonal']
    p=fisher_exact([[ea[2],len(ea[0])-ea[2]],[ca[2],len(ca[0])-ca[2]]])[1]
    print(f"  evolving vs clonal: division rate Fisher p={p:.3f}")
