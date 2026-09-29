"""Does the size window depend on how a collective is delimited?"""
import sys, glob, re, numpy as np
E = "/Volumes/DATA/ETI Presentation/ETI"
sys.path.insert(0, f"{E}/reproduce"); sys.path.insert(0, f"{E}/fast")
from shape import annularity
import runfast

def sweep(link, min_size=10):
    out = []
    for f in sorted(glob.glob(f"{E}/eusocial/data/castes_seed*.npz")
                    + glob.glob(f"{E}/eusocial/data/nocaste_seed*.npz")):
        d = np.load(f, allow_pickle=True)
        for s in sorted({int(k.split("_")[1]) for k in d.files
                         if k.startswith("snap_")}):
            x, y = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
            for c in runfast.clusters(x, y, link=link, min_size=min_size):
                a = annularity(x[c], y[c])
                if a: out.append((len(c), a["annularity"]))
    return np.array(out)

print(f"{'link':>5} {'obs':>6} " + " ".join(f"{b:>9}" for b in
      ("15-19", "20-29", "30-39", "40-54", "55-74", "75-99", "100+")))
print("-" * 76)
bands = [(15, 20), (20, 30), (30, 40), (40, 55), (55, 75), (75, 100), (100, 10**6)]
for link in (20.0, 25.0, 30.0):
    r = sweep(link)
    n, A = r[:, 0], r[:, 1]
    row = f"{link:>5.0f} {len(r):>6} "
    for lo, hi in bands:
        m = (n >= lo) & (n < hi)
        row += f"{np.median(A[m]):>9.2f} " if m.sum() >= 8 else f"{'-':>9} "
    print(row)
    coll = [lo for lo, hi in bands if ((n >= lo) & (n < hi)).sum() >= 8
            and np.median(A[(n >= lo) & (n < hi)]) < 0.15]
    print(f"      collapse (median A < 0.15) from n = {min(coll) if coll else '?'}")
