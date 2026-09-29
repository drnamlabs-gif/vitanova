"""Lineage composition of collectives, and size recovery after a fusion."""
import json, os, sys
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(v, "1")
import numpy as np
E = "/Volumes/DATA/ETI Presentation/ETI"
for p in (f"{E}/eusocial", f"{E}/fast", f"{E}/reproduce"): sys.path.insert(0, p)
import crowded as CR
from runfast import clusters

class Founder(CR.Crowded):
    def __init__(self, **kw):
        super().__init__(**kw); self.fnd = np.arange(self.n)
    def _keep(self, m):
        self.fnd = self.fnd[m]; super()._keep(m)
    def _reproduce(self):
        b = self.n
        par = np.flatnonzero((self.E > self.cfg["REPRODUCTION_ENERGY_THRESHOLD"])
                             & (self.n + 1 <= self.cfg["MAX_BOIDS"]))
        super()._reproduce()
        if self.n > b:
            self.fnd = np.concatenate([self.fnd, self.fnd[par[:self.n - b]]])

SEEDS = [1, 3, 4, 5, 6, 7, 9, 12, 14, 15]
comp, fus = [], []
SAMPLE, HORIZON = 25, 60
for seed in SEEDS:
    w = Founder(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
    hist, prev, events = [], None, []
    for i in range(30000):
        if w.n == 0: break
        if i % SAMPLE == 0:
            cs = clusters(w.x, w.y, min_size=10)
            sets = [set(w.uid[c].tolist()) for c in cs]
            sizes = [len(c) for c in cs]
            hist.append((i, sets, sizes))
            if i % 500 == 0 and i >= 2000:
                for c in cs:
                    _, cnt = np.unique(w.fnd[c], return_counts=True)
                    p = cnt / cnt.sum()
                    comp.append(dict(seed=seed, step=i, n=len(c),
                                     lineages=int(len(cnt)),
                                     effective=float(1 / (p ** 2).sum()),
                                     largest_share=float(p.max()),
                                     shares=sorted(p.tolist(), reverse=True)[:8]))
            if prev is not None:
                for si, c in enumerate(sets):
                    src = [k for k, q in enumerate(prev[1])
                           if len(q & c) >= 0.5 * len(q)]
                    if len(src) >= 2:
                        events.append((len(hist) - 1,
                                       sum(prev[2][k] for k in src), c))
            prev = (i, sets, sizes)
        w.step()
    for hi, pre, uids in events:
        traj = []
        for h in range(hi, min(hi + HORIZON, len(hist))):
            _, sets, sizes = hist[h]
            traj.append(max((len(s & uids), sz) for s, sz in zip(sets, sizes))[1]
                        if sets else 0)
        fus.append(dict(seed=seed, pre=int(pre), traj=[int(t) for t in traj]))

json.dump(dict(composition=comp, fusions=fus), open("scope.json", "w"))
c = np.array([[x["lineages"], x["effective"], x["largest_share"]] for x in comp])
print(f"{len(comp)} collectives: lineages median {np.median(c[:,0]):.0f} "
      f"(mean {c[:,0].mean():.1f}), effective {c[:,1].mean():.2f}, "
      f"largest share {c[:,2].mean():.0%}, single-lineage {(c[:,0]==1).mean():.0%}")
print(f"{len(fus)} fusions")
