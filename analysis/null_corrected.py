# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Compare observed annularity against the size-matched null, properly."""
import sys, json, numpy as np
sys.path.insert(0, "/Volumes/DATA/ETI Presentation/ETI/reproduce")
from shape import annularity
rng = np.random.default_rng(1)

def draw_null(n, kind="uniform", R=25):
    if kind == "uniform":
        r = R*np.sqrt(rng.random(n))
    else:
        k = n//3
        r = np.concatenate([rng.normal(0,2,k).clip(0), R*rng.random(n-k)])
    t = rng.uniform(0, 2*np.pi, n)
    return r*np.cos(t), r*np.sin(t)

rows = json.load(open("/Volumes/DATA/ETI Presentation/ETI/eusocial/size_window.json"))
n = np.array([r["n"] for r in rows]); A = np.array([r["A"] for r in rows])
rx = np.array([r["r_max"] for r in rows])

# 95th percentile of the null, per size, over both non-annular families
sizes = np.unique(np.clip(n, 15, 1200))
grid = np.unique(np.round(np.logspace(np.log10(15), np.log10(1200), 26)).astype(int))
q95 = {}
for g in grid:
    vals = []
    for kind in ("uniform", "halo"):
        for _ in range(300):
            x, y = draw_null(int(g), kind)
            a = annularity(x, y)
            vals.append(a["annularity"] if a else 0.0)
    q95[g] = float(np.percentile(vals, 95))
def thr(k): return q95[grid[np.argmin(abs(grid - k))]]
T = np.array([thr(k) for k in n])

print("null 95th percentile of A, by size:")
for g in grid[::3]: print(f"   n={g:>5}: {q95[g]:.2f}")
guard = rx < 37.5
print()
print(f"observations                                : {len(n)}")
print(f"  exceeding a fixed A > 0.5                 : {(A > .5).mean():.1%}")
print(f"  exceeding A > 0.5 AND radius guard        : {((A > .5) & guard).mean():.1%}")
print(f"  exceeding the size-matched null 95th pct  : {(A > T).mean():.1%}")
print(f"  exceeding null 95th pct AND radius guard  : {((A > T) & guard).mean():.1%}")
m = (n >= 20) & (n <= 55)
print()
print(f"inside the size window 20-55 ({m.sum()} obs):")
print(f"  median observed A                         : {np.median(A[m]):.2f}")
print(f"  median size-matched null 95th pct         : {np.median(T[m]):.2f}")
print(f"  fraction above the null                   : {(A[m] > T[m]).mean():.0%}")
mb = n > 75
print(f"above n=75 ({mb.sum()} obs):")
print(f"  median observed A                         : {np.median(A[mb]):.2f}")
print(f"  median null 95th pct                      : {np.median(T[mb]):.2f}")
print(f"  fraction above the null                   : {(A[mb] > T[mb]).mean():.0%}")
