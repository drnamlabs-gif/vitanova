"""Small parallel scan over ecological parameters of the DEB world."""
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "2")
import numpy as np
import run_deb

def job(a):
    name, steps, ov = a
    try:
        r = run_deb.run(2024, steps, every=25, snap_every=0, out=None,
                        verbose=False, **ov)
        i = int(np.argmax(r["annularity"]))
        return dict(name=name, n_end=int(r["n"][-1]), n_max=int(r["n"].max()),
                    clusters_max=int(r["n_clusters"].max()),
                    clusters_end=int(r["n_clusters"][-1]),
                    largest_max=int(r["largest"].max()),
                    annular_max=round(float(r["annularity"].max()), 2),
                    annular_at=int(r["step"][i]),
                    hollow_max=round(float(r["hollowness"].max()), 2),
                    res_end=round(float(r["resource"][-1]), 3),
                    V_end=round(float(r["mean_V"][-1]), 2),
                    secs=round(r["wall_seconds"]))
    except Exception as e:
        return dict(name=name, error=f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    jobs = []
    for rg in (0.0010, 0.0020, 0.0040):
        jobs.append((f"R_REGEN={rg}", steps, {"R_REGEN": rg}))
    for pm in (0.04, 0.08):
        jobs.append((f"R_REGEN=0.002,P_M={pm}", steps,
                     {"R_REGEN": 0.002, "P_M": pm}))
    for rr in (6.0, 12.0):
        jobs.append((f"R_REGEN=0.002,R_REPRO={rr}", steps,
                     {"R_REGEN": 0.002, "R_REPRO": rr}))
    jobs.append(("R_REGEN=0.002,N_PRED=0", steps,
                 {"R_REGEN": 0.002, "N_PRED": 0}))
    jobs.append(("R_REGEN=0.002,CELL=50", steps,
                 {"R_REGEN": 0.002, "CELL": 50}))

    rows = []
    with ProcessPoolExecutor(max_workers=9) as pool:
        futs = [pool.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(("  " + r["name"]) if "error" not in r else
                  f"  {r['name']} ERROR {r['error']}", flush=True)
    order = {j[0]: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[r["name"]])
    hdr = f"{'config':<30}{'n_end':>6}{'n_max':>6}{'clus':>5}{'largest':>8}{'annular':>8}{'@step':>7}{'hollow':>7}{'res':>7}{'V':>6}{'s':>5}"
    print("\n" + hdr); print("-"*len(hdr))
    for r in rows:
        if "error" in r: print(f"{r['name']:<30}{r['error']}"); continue
        print(f"{r['name']:<30}{r['n_end']:>6}{r['n_max']:>6}{r['clusters_max']:>5}"
              f"{r['largest_max']:>8}{r['annular_max']:>8.2f}{r['annular_at']:>7}"
              f"{r['hollow_max']:>7.2f}{r['res_end']:>7.3f}{r['V_end']:>6.2f}{r['secs']:>5}")
    json.dump(rows, open("runs/scan.json","w"), indent=2)
