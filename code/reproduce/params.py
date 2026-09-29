"""One-at-a-time parameter scan on a single seed.

Two groups of parameters are varied. The first four are the hand-coded
group-level reward terms -- the ones that make "nothing selects for the
collective" false as currently written:

  WORKER_TOGETHER_ENERGY    Allee bonus for workers with 2..CRITICAL-1 neighbours
  CRITICAL_GROUP_SIZE       where that bonus flips to a penalty
  NEAR_WORKER_GAIN          soldier bonus for being near a worker
  SOLDIER_REVERSAL_THRESHOLD  predator flees on seeing this many soldiers (None = rule off)
  LARGE_GROUP_THRESHOLD     predator is cautious past this many nearby boids (None = rule off)

The rest are ordinary ecology and genetics. The point of the scan is to see
which knobs the ring-and-division phenomenon actually depends on.

Predator count is handled specially: the module burns PREDATOR_COUNT throwaway
predators at import before building the real population, so changing it would
also change every boid's genotype. Here the burn is always held at the baseline
10, so varying predator count varies only the predators.
"""

import argparse
import copy
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import vitanova

BASELINE = {
    "WORKER_TOGETHER_ENERGY": 0.2,
    "CRITICAL_GROUP_SIZE": 20,
    "NEAR_WORKER_GAIN": 0.6,
    "SOLDIER_REVERSAL_THRESHOLD": 3,
    "LARGE_GROUP_THRESHOLD": 20,
    "PREDATOR_COUNT": 10,
    "MUTATION_RATE": 0.1,
    "MUTATION_STRENGTH": 0.1,
    "REPRODUCTION_ENERGY_THRESHOLD": 200,
    "WORKER_ENERGY_GAIN": 1.1,
    "SOLDIER_ENERGY_GAIN": 0.7,
    "SPEED": 1,
}

SCAN = {
    "WORKER_TOGETHER_ENERGY": [0.0, 0.1, 0.4],
    "CRITICAL_GROUP_SIZE": [10, 40, 10**9],
    "NEAR_WORKER_GAIN": [0.0, 1.2],
    "SOLDIER_REVERSAL_THRESHOLD": [None, 1, 10],
    "LARGE_GROUP_THRESHOLD": [None, 50],
    "PREDATOR_COUNT": [0, 5, 20],
    "MUTATION_RATE": [0.02, 0.3],
    "MUTATION_STRENGTH": [0.03, 0.3],
    "REPRODUCTION_ENERGY_THRESHOLD": [150, 300],
    "WORKER_ENERGY_GAIN": [1.0, 1.3],
    "SOLDIER_ENERGY_GAIN": [0.4, 1.0],
}

PATCHED = {"SOLDIER_REVERSAL_THRESHOLD", "LARGE_GROUP_THRESHOLD", "PREDATOR_COUNT"}


def apply_config(eti, cfg):
    for key, val in cfg.items():
        if key in PATCHED:
            continue
        if not hasattr(eti, key):
            raise KeyError(f"{key} is not a module-level constant")
        setattr(eti, key, val)

    thresh = cfg["SOLDIER_REVERSAL_THRESHOLD"]
    original = eti.Predator.detect_nearby_soldiers

    def detect_nearby_soldiers(self, boids, _o=original, _t=thresh):
        if _t is None:
            return []                       # rule disabled
        seen = _o(self, boids)
        if len(seen) < _t:
            return []
        return (seen * 3)[:3]               # satisfies the inline `>= 3` test

    eti.Predator.detect_nearby_soldiers = detect_nearby_soldiers

    big = cfg["LARGE_GROUP_THRESHOLD"]

    def detect_large_groups(self, boids, group_threshold=None, _g=big):
        if _g is None:
            return False
        near = [b for b in boids
                if self.distance_to(b) < self.large_group_visual_range]
        return len(near) > _g

    eti.Predator.detect_large_groups = detect_large_groups


def init_population(eti, seed, n_predators, burn=10):
    """Like vitanova.init_population, but the throwaway burn is held fixed."""
    random.seed(seed)
    np.random.seed(seed)
    for _ in range(burn):
        eti.Predator(random.randint(0, eti.WIDTH), random.randint(0, eti.HEIGHT))
    boids = eti.create_boids()
    predators = [eti.Predator(random.randint(0, eti.WIDTH),
                              random.randint(0, eti.HEIGHT))
                 for _ in range(n_predators)]
    eti.boids = boids
    eti.predators = predators
    return boids, predators


def run_config(job):
    name, cfg, seed, steps, detect_every, max_boids, outdir = job
    try:
        eti = vitanova.load_original("paper")
        apply_config(eti, cfg)
        boids, predators = init_population(eti, seed, cfg["PREDATOR_COUNT"])

        rec = {k: [] for k in ("step", "n_boids", "n_flocks", "big_flocks",
                               "largest", "ring_score", "coverage", "cv",
                               "hollow", "mean_r", "cx", "cy", "soldier_frac",
                               "mean_energy")}
        bloom_at = None
        t0 = time.time()
        for step in range(steps):
            if step % detect_every == 0:
                flocks = vitanova.flock_sets(
                    eti.identify_flocks(boids, eti.FLOCK_THRESHOLD))
                big = [f for f in flocks if len(f) >= 10]
                m = vitanova.ring_metrics(max(flocks, key=len)) if flocks else None
                rec["step"].append(step)
                rec["n_boids"].append(len(boids))
                rec["n_flocks"].append(len(flocks))
                rec["big_flocks"].append(len(big))
                rec["largest"].append(max((len(f) for f in flocks), default=0))
                rec["soldier_frac"].append(
                    sum(b.role == "soldier" for b in boids) / len(boids)
                    if boids else 0.0)
                rec["mean_energy"].append(
                    float(np.mean([b.energy for b in boids])) if boids else 0.0)
                for k in ("ring_score", "coverage", "cv", "hollow",
                          "mean_r", "cx", "cy"):
                    rec[k].append(m[k] if m else 0.0)
            if not boids:
                break
            if max_boids and len(boids) > max_boids:
                bloom_at = step
                break
            for b in [b for b in boids if b.should_die()]:
                boids.remove(b)
            for b in boids:
                b.update(boids, predators)
            for p in predators:
                p.update(boids)

        res = {k: np.array(v) for k, v in rec.items()}
        res.update(seed=seed, max_steps=steps, detect_every=detect_every,
                   extinct_at=(step if not boids else None), bloom_at=bloom_at,
                   wall_seconds=time.time() - t0, label=name, snapshots={})
        if outdir:
            vitanova.save(res, os.path.join(outdir, f"{name}.npz"))

        v = vitanova.classify(res)
        v.update(config=name,
                 pop_max=int(res["n_boids"].max()) if len(res["n_boids"]) else 0,
                 pop_end=int(res["n_boids"][-1]) if len(res["n_boids"]) else 0,
                 bloom_at=bloom_at)
        return v
    except Exception as exc:
        return {"config": name, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--detect-every", type=int, default=25)
    ap.add_argument("--max-boids", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--outdir", default="runs/params")
    ap.add_argument("--report", default="runs/params.json")
    ap.add_argument("--only", default=None, help="scan just this parameter")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    jobs = [("baseline", copy.deepcopy(BASELINE), args.seed, args.steps,
             args.detect_every, args.max_boids, args.outdir)]
    for key, values in SCAN.items():
        if args.only and key != args.only:
            continue
        for val in values:
            cfg = copy.deepcopy(BASELINE)
            cfg[key] = val
            jobs.append((f"{key}={val}", cfg, args.seed, args.steps,
                         args.detect_every, args.max_boids, args.outdir))

    print(f"seed {args.seed}: {len(jobs)} configs x {args.steps} steps "
          f"on {args.workers} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_config, j) for j in jobs]
        for done, fut in enumerate(as_completed(futs), 1):
            v = fut.result()
            rows.append(v)
            if "error" in v:
                print(f"[{done}/{len(jobs)}] {v['config']:<36} ERROR {v['error']}",
                      flush=True)
            else:
                tag = "RING+DIV" if v.get("full_sequence") else (
                      "ring" if v.get("ring_at") is not None else "-")
                print(f"[{done}/{len(jobs)}] {v['config']:<36}{tag:>9}  "
                      f"maxring={v.get('max_ring_score',0):.2f}  "
                      f"maxflock={v.get('max_largest',0):>3}  "
                      f"pop_max={v.get('pop_max',0):>3}  "
                      f"ring@{v.get('ring_at')} div@{v.get('division_at')}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    order = {j[0]: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order.get(r["config"], 999))
    with open(args.report, "w") as fh:
        json.dump({"seed": args.seed, "steps": args.steps,
                   "baseline": BASELINE, "results": rows}, fh, indent=2)
    print("\nreport:", args.report)


if __name__ == "__main__":
    main()
