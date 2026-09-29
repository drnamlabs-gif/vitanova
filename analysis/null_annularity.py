"""Size-matched null distribution for the annularity statistic.

The paper's headline is a fraction of time spent 'organised' (A > 0.5 with a
radius guard). That number is meaningless without knowing how often the same
criterion fires on configurations that are NOT annular, at the same n.
"""
import sys, numpy as np, json
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
from shape import annularity

rng = np.random.default_rng(0)
NS = [16, 20, 30, 40, 55, 75, 150, 400, 1000]
DRAWS = 400

def uniform_disc(n, R=25):
    r = R * np.sqrt(rng.random(n)); t = rng.uniform(0, 2*np.pi, n)
    return r*np.cos(t), r*np.sin(t)

def centre_heavy(n, R=25):
    r = R * rng.random(n)**2; t = rng.uniform(0, 2*np.pi, n)
    return r*np.cos(t), r*np.sin(t)

def gaussian(n, s=10):
    return rng.normal(0, s, n), rng.normal(0, s, n)

def core_halo(n, R=25):          # monotone, no trough
    k = n//3
    r = np.concatenate([rng.normal(0, 2, k).clip(0), R*rng.random(n-k)])
    t = rng.uniform(0, 2*np.pi, n)
    return r*np.cos(t), r*np.sin(t)

def ring_core(n, R=25):          # the real structure
    k = max(2, n//4)
    r = np.concatenate([np.abs(rng.normal(0, 1.5, k)),
                        rng.normal(R, 2.5, n-k)])
    t = rng.uniform(0, 2*np.pi, n)
    return r*np.cos(t), r*np.sin(t)

CFG = [("uniform disc", uniform_disc), ("centre-heavy", centre_heavy),
       ("gaussian", gaussian), ("core + halo (monotone)", core_halo),
       ("ring + core", ring_core)]

out = {}
print(f"{'configuration':>24} " + " ".join(f"{n:>11}" for n in NS))
print("-"*(25+12*len(NS)))
for name, fn in CFG:
    row, cells = [], {}
    for n in NS:
        A = []
        for _ in range(DRAWS):
            x, y = fn(n)
            a = annularity(x, y)
            A.append(a["annularity"] if a else 0.0)
        A = np.array(A)
        cells[n] = dict(mean=float(A.mean()), frac_gt_half=float((A > .5).mean()))
        row.append(f"{A.mean():.2f}/{(A>.5).mean():.0%}")
    out[name] = cells
    print(f"{name:>24} " + " ".join(f"{v:>11}" for v in row))
json.dump(out, open("null_annularity.json", "w"), indent=1)
print("\ncells are  mean A / fraction with A > 0.5   (400 draws each)")
