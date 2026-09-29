"""Ring integrity as a function of collective size, pooled over all runs."""
import sys, glob, re, json, numpy as np
sys.path.insert(0, "../reproduce"); sys.path.insert(0, "../fast")
from shape import annularity
from runfast import clusters

rows = []
for tag in ("castes", "nocaste"):
    for f in sorted(glob.glob(f"data/{tag}_seed*.npz")):
        seed = int(re.search(r"seed(\d+)", f).group(1))
        d = np.load(f, allow_pickle=True)
        steps = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("snap_")})
        for s in steps:
            x, y = d[f"snap_{s}_x"], d[f"snap_{s}_y"]
            for c in clusters(x, y, min_size=10):
                a = annularity(x[c], y[c])
                if not a:
                    continue
                cx, cy = x[c].mean(), y[c].mean()
                r = np.hypot(x[c] - cx, y[c] - cy)
                rows.append(dict(tag=tag, seed=seed, step=s, n=len(c),
                                 A=a["annularity"], hollow=a["hollowness"],
                                 r_trough=a["r_trough"], r_max=a["r_max"],
                                 core_frac=a["core_frac"],
                                 r_med=float(np.median(r)),
                                 # rim = members outside the trough radius
                                 rim=int((r > a["r_trough"]).sum())))
json.dump(rows, open("size_window.json", "w"))
print(f"{len(rows)} cluster observations\n")

n  = np.array([r["n"] for r in rows])
A  = np.array([r["A"] for r in rows])
rim= np.array([r["rim"] for r in rows])
rt = np.array([r["r_trough"] for r in rows])
rx = np.array([r["r_max"] for r in rows])

edges = [10, 15, 20, 30, 40, 55, 75, 100, 150, 250, 400, 10000]
print(f"{'size':>12} {'obs':>5} {'A median':>9} {'frac A>0.5':>11} "
      f"{'r_trough':>9} {'r_max':>7} {'rim n':>7}")
print("-" * 68)
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (n >= lo) & (n < hi)
    if m.sum() < 5: continue
    lab = f"{lo}-{hi-1}" if hi < 10000 else f"{lo}+"
    print(f"{lab:>12} {m.sum():>5} {np.median(A[m]):>9.2f} "
          f"{(A[m] > 0.5).mean():>11.2f} {np.median(rt[m]):>9.1f} "
          f"{np.median(rx[m]):>7.1f} {np.median(rim[m]):>7.0f}")
