# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Which model variants produce the ring, and which produce collective division?

Each variant is run on the same seeds for the same number of steps, and scored
by the highest annularity reached by its largest collective, with the geometric
guard that a ring must fit within 1.5 perception radii.
"""
import os, sys, json, time
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
H=os.path.dirname(os.path.abspath(__file__))
for d in ("fast","reproduce","eusocial","linear","deb"):
    sys.path.insert(0, os.path.join(H,d))
import shape, vitafast as VF
from runfast import clusters

SEEDS, STEPS = [1,3,4,7,12,13,14,15], 9000
GUARD = 1.5*VF.PERCEPTION_RADIUS

def build(name, seed):
    if name == "Fast (FFN, castes)":
        return VF.Fast(seed=seed, MAX_BOIDS=600)
    if name == "NoCaste (FFN, no castes)":
        import nocaste as NC; return NC.NoCaste(seed=seed, MAX_BOIDS=600)
    if name == "Linear (no hidden layer)":
        import linear_vita as L; return L.LinearVita(seed=seed, MAX_BOIDS=600)
    if name == "Locked (role fixed at birth)":
        import locked as LK
        return [c for n,c in vars(LK).items() if n=="Locked"][0](seed=seed, MAX_BOIDS=600)
    if name == "Crowded (size ceiling)":
        import crowded as CR
        return CR.Crowded(seed=seed, MAX_BOIDS=600, crowd_beta=0.08, crowd_k=26)
    if name == "No ecology (subsidy off)":
        return VF.Fast(seed=seed, MAX_BOIDS=600, WORKER_TOGETHER_ENERGY=0.0)
    raise KeyError(name)

NAMES = ["Fast (FFN, castes)", "NoCaste (FFN, no castes)", "Linear (no hidden layer)",
         "Locked (role fixed at birth)", "Crowded (size ceiling)",
         "No ecology (subsidy off)"]

def one(a):
    name, seed = a
    try: w = build(name, seed)
    except Exception as e: return dict(name=name, seed=seed, err=str(e)[:60])
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
                if d and d["r_max"] < GUARD: best = max(best, d["annularity"])
        w.step()
    return dict(name=name, seed=seed, peak_A=round(best,3), largest=largest,
                divided=bool(divided), n_end=int(w.n))

if __name__ == "__main__":
    jobs=[(n,s) for n in NAMES for s in SEEDS]
    res=[]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(H,"variant_rings.json"),"w"), indent=1)
    print(f"{len(SEEDS)} seeds x {STEPS} steps per variant\n")
    print(f"{'variant':>30} {'median peak A':>14} {'max':>6} {'ring in':>9} {'divided':>9} {'largest':>8}")
    print('-'*82)
    for n in NAMES:
        g=[x for x in res if x['name']==n and 'err' not in x]
        if not g:
            e=[x for x in res if x['name']==n][0]; print(f"{n:>30}  ERROR {e.get('err')}"); continue
        A=np.array([x['peak_A'] for x in g]); L=np.array([x['largest'] for x in g])
        print(f"{n:>30} {np.median(A):>14.2f} {A.max():>6.2f} "
              f"{str(int((A>0.5).sum()))+'/'+str(len(g)):>9} "
              f"{str(sum(x['divided'] for x in g))+'/'+str(len(g)):>9} {np.median(L):>8.0f}")
