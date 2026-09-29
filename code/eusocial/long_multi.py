"""How long does a collective lineage last, and how deep does it get?

Descent is counted only through divisions. A collective that appears from a
fusion, or by re-forming from scattered agents, starts a new lineage at
generation 0 rather than inheriting one.
"""
import os, sys, json, time
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE,"..","fast"), os.path.join(HERE,"..","reproduce")):
    sys.path.insert(0,p)
import crowded as CR, nocaste as NC
from runfast import clusters

STEPS, SAMPLE = 100_000, 50

def one(a):
    regulated, seed = a
    w = (CR.Crowded(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
         if regulated else NC.NoCaste(seed=seed, MAX_BOIDS=1200))
    prev, nid, divisions, fusions = [], 0, 0, 0
    gen, maxgen, t0 = {}, 0, time.time()
    blobbed = False
    for i in range(STEPS):
        if w.n == 0: break
        if w.n > 500:          # regulation has failed; clustering becomes the
            blobbed = True     # bottleneck and no collective lineage survives
            break
        if i % SAMPLE == 0:
            cs = clusters(w.x, w.y, min_size=10)
            cur = []
            for c in cs:
                s = set(w.uid[c].tolist())
                src = [p for p in prev if len(p[1] & s) >= 0.5 * len(p[1])]
                if len(src) == 1:
                    cur.append((src[0][0], s))
                else:
                    if len(src) >= 2: fusions += 1
                    gen[nid] = 0                     # fusion or re-formation
                    cur.append((nid, s)); nid += 1
            for p in prev:
                kids = [c for c in cur if len(p[1] & c[1]) >= 0.35 * len(p[1])]
                if len(kids) >= 2:
                    divisions += 1
                    for k in kids:
                        if k[0] != p[0]:
                            gen[k[0]] = gen.get(p[0], 0) + 1
                            maxgen = max(maxgen, gen[k[0]])
            prev = cur
        w.step()
    return dict(regulated=regulated, seed=seed, steps_run=i + 1,
                extinct=bool(w.n == 0), blobbed=blobbed,
                divisions=divisions, fusions=fusions,
                collectives=nid, max_generation=maxgen,
                wall=round(time.time() - t0, 1))

if __name__ == "__main__":
    seeds=[1,3,4,5,6,7,9,12,14,15]
    jobs=[(True,s) for s in seeds]   # regulated only; the unregulated
    # model blobs into one component and costs O(n^2) at n>1000, so its division
    # counts are taken from the 9,000-step batch instead
    res=[]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]): res.append(f.result())
    json.dump(res, open(os.path.join(HERE,"long_multi.json"),"w"), indent=1)
    print(f"{'model':>12} {'extinct':>9} {'steps run':>11} {'divisions':>11} "
          f"{'fusions':>9} {'deepest lineage':>16}")
    print('-'*74)
    for r,lab in ((True,'regulated'),):
        g=[x for x in res if x['regulated']==r]
        st=np.array([x['steps_run'] for x in g],float)
        dv=np.array([x['divisions'] for x in g],float)
        fu=np.array([x['fusions'] for x in g],float)
        mg=np.array([x['max_generation'] for x in g],float)
        print(f"{lab:>12} {sum(x['extinct'] for x in g):>6}/{len(g)} "
              f"{np.median(st):>11.0f} {np.median(dv):>11.0f} {np.median(fu):>9.0f} "
              f"{np.median(mg):>10.0f} (max {mg.max():.0f})")
        print()
        print(f"{'seed':>6} {'steps':>9} {'ended':>10} {'divisions':>10} "
              f"{'fusions':>8} {'gen':>5}")
        for x in sorted(g, key=lambda z: z['seed']):
            end = 'extinct' if x['extinct'] else ('blobbed' if x['blobbed'] else 'ran out')
            print(f"{x['seed']:>6} {x['steps_run']:>9} {end:>10} "
                  f"{x['divisions']:>10} {x['fusions']:>8} {x['max_generation']:>5}")
