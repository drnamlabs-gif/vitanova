# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""One seed, run long, to count collective generations.

With the crowding cost the population stays near 30, so a step is cheap and the
run can go far enough to see a collective lineage rather than a single division.
Collectives are tracked by member identity: a division is recorded when the
members of one collective are predominantly found in two at the next sample.
"""
import os, sys, json
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE,"..","fast"), os.path.join(HERE,"..","reproduce")):
    sys.path.insert(0,p)
import shape, crowded as CR, vitafast as VF
from runfast import clusters

SEED   = int(sys.argv[1]) if len(sys.argv) > 1 else 9
STEPS  = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
SAMPLE = 25
GUARD  = 1.5 * VF.PERCEPTION_RADIUS

w = CR.Crowded(seed=SEED, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
prev, next_cid = [], 0          # list of (cid, uid set)
divisions, fusions, log = [], 0, []
gen = {}                        # cid -> collective generation
t0 = __import__("time").time()

for i in range(STEPS):
    if w.n == 0:
        break
    if i % SAMPLE == 0:
        cs = clusters(w.x, w.y, min_size=10)
        cur = []
        for c in cs:
            s = set(w.uid[c].tolist())
            src = [p for p in prev if len(p[1] & s) >= 0.5 * len(p[1])]
            if len(src) == 1:
                cid = src[0][0]
            else:
                cid = next_cid; next_cid += 1
                gen[cid] = (max(gen[p[0]] for p in src) + 1) if src else 0
            cur.append((cid, s))
        # a parent whose members are split across two current collectives
        for p in prev:
            kids = [c for c in cur if len(p[1] & c[1]) >= 0.35 * len(p[1])]
            if len(kids) >= 2:
                divisions.append(dict(step=i, parent=p[0],
                                      parent_size=len(p[1]),
                                      kids=[len(k[1]) for k in kids]))
                for k in kids:
                    if k[0] != p[0]:
                        gen[k[0]] = gen.get(p[0], 0) + 1
        for c in cur:
            srcs = [p for p in prev if len(p[1] & c[1]) >= 0.5 * len(p[1])]
            if len(srcs) >= 2: fusions += 1
        prev = cur
        if i % (SAMPLE * 400) == 0:
            big = cs[0] if cs else None
            d = shape.annularity(w.x[big], w.y[big]) if big is not None else None
            log.append(dict(step=i, n=int(w.n), clusters=len(cs),
                            largest=int(len(big)) if big is not None else 0,
                            A=round(d["annularity"], 3) if d else 0.0,
                            divisions=len(divisions),
                            max_gen=max(gen.values()) if gen else 0))
    w.step()

json.dump(dict(seed=SEED, steps=STEPS, divisions=divisions, fusions=fusions,
               log=log, generations=gen),
          open(os.path.join(HERE, f"long_lineage_seed{SEED}.json"), "w"))
print(f"seed {SEED}, {i+1} steps, {__import__('time').time()-t0:.0f}s")
print(f"  collective divisions : {len(divisions)}")
print(f"  collective fusions   : {fusions}")
print(f"  distinct collectives : {next_cid}")
print(f"  deepest lineage      : {max(gen.values()) if gen else 0} generations")
if divisions:
    ps=np.array([d['parent_size'] for d in divisions],float)
    print(f"  parent size at division: median {np.median(ps):.0f}, "
          f"range {ps.min():.0f}-{ps.max():.0f}")
    st=np.array([d['step'] for d in divisions],float)
    print(f"  first at step {st.min():.0f}, last at {st.max():.0f}, "
          f"mean interval {np.diff(np.sort(st)).mean():.0f} steps" if len(st)>1 else "")
