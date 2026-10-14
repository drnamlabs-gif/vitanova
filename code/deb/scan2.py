# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Productivity and predation scan for the DEB world."""
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "2")
import numpy as np
import run_deb

JOBS = ([(f"R_REGEN={r}", {"R_REGEN": r}) for r in (0.004, 0.006, 0.008, 0.012)]
        + [("R_REGEN=0.006,N_PRED=0", {"R_REGEN": 0.006, "N_PRED": 0}),
           ("R_REGEN=0.006,N_PRED=20", {"R_REGEN": 0.006, "N_PRED": 20}),
           ("R_REGEN=0.008,N_PRED=0", {"R_REGEN": 0.008, "N_PRED": 0})])


def job(a):
    name, ov = a
    try:
        r = run_deb.run(2024, int(sys.argv[1]) if len(sys.argv) > 1 else 4000,
                        every=25, snap_every=0, verbose=False, **ov)
        i = int(np.argmax(r["annularity"]))
        return (name, int(r["n"][-1]), int(r["n_clusters"].max()),
                int(r["largest"].max()), float(r["annularity"].max()),
                int(r["step"][i]), float(r["hollowness"].max()),
                round(r["wall_seconds"]))
    except Exception as e:
        return (name, -1, -1, -1, float("nan"), -1, float("nan"), 0)


if __name__ == "__main__":
    rows = []
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(job, j) for j in JOBS]):
            rows.append(f.result())
            print("  done", rows[-1][0], flush=True)
    order = {j[0]: i for i, j in enumerate(JOBS)}
    rows.sort(key=lambda r: order[r[0]])
    print(f"\n{'config':<26}{'n_end':>6}{'clus':>5}{'largest':>8}"
          f"{'annular':>8}{'@step':>7}{'hollow':>7}{'s':>5}")
    print("-" * 72)
    for r in rows:
        print(f"{r[0]:<26}{r[1]:>6}{r[2]:>5}{r[3]:>8}{r[4]:>8.2f}"
              f"{r[5]:>7}{r[6]:>7.2f}{r[7]:>5}")
