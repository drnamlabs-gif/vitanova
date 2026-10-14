# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Sweep seeds through the headless VitaNova driver, looking for the paper's
ring-formation-then-division sequence.

Each worker process loads its own copy of real_eti.py, so the module-level
globals the original relies on (`boids`, `predators`) stay per-run.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import vitanova

_ETI = None


def _worker(job):
    """Run one seed. Keeps one real_eti module per process and re-seeds it."""
    global _ETI
    seed, steps, detect_every, min_flock, ring_thresh, max_boids, outdir = job
    if _ETI is None:
        _ETI = vitanova.load_original()
    try:
        res = vitanova.run(seed, max_steps=steps, detect_every=detect_every,
                           min_flock=min_flock, max_boids=max_boids, eti=_ETI)
    except Exception as exc:  # a run that blows up should not kill the sweep
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
    verdict = vitanova.classify(res, ring_thresh=ring_thresh)
    if outdir:
        vitanova.save(res, os.path.join(outdir, f"seed{seed}.npz"))
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-199",
                    help="e.g. 0-199 or 2024,7,42")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--detect-every", type=int, default=25)
    ap.add_argument("--min-flock", type=int, default=10)
    ap.add_argument("--ring-thresh", type=float, default=0.35)
    ap.add_argument("--max-boids", type=int, default=500,
                    help="stop a run whose population blooms past this (0 = off)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--outdir", default=None,
                    help="also save per-seed metric series here")
    ap.add_argument("--report", default="runs/sweep.json")
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    jobs = [(s, args.steps, args.detect_every, args.min_flock,
             args.ring_thresh, args.max_boids, args.outdir) for s in seeds]

    print(f"sweeping {len(seeds)} seeds x {args.steps} steps "
          f"on {args.workers} workers", flush=True)

    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, j): j[0] for j in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            v = fut.result()
            results.append(v)
            tag = ("FULL SEQUENCE" if v.get("full_sequence") else
                   "ring" if v.get("ring_at") is not None else
                   "err" if "error" in v else "-")
            print(f"[{done}/{len(seeds)}] seed {v['seed']:>5} {tag:>13} "
                  f"ring@{v.get('ring_at')} div@{v.get('division_at')} "
                  f"maxring={v.get('max_ring_score', 0):.2f} "
                  f"maxflock={v.get('max_largest')} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    results.sort(key=lambda v: v["seed"])
    hits = [v for v in results if v.get("full_sequence")]
    rings = [v for v in results if v.get("ring_at") is not None]
    summary = {
        "n_seeds": len(seeds),
        "steps": args.steps,
        "ring_thresh": args.ring_thresh,
        "min_flock": args.min_flock,
        "n_ring": len(rings),
        "n_full_sequence": len(hits),
        "full_sequence_seeds": [v["seed"] for v in hits],
        "ring_seeds": [v["seed"] for v in rings],
        "wall_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nring formed: {len(rings)}/{len(seeds)}   "
          f"full sequence (ring then division): {len(hits)}/{len(seeds)}")
    print("full-sequence seeds:", [v["seed"] for v in hits])
    print("report:", args.report)


if __name__ == "__main__":
    main()
