"""Ablation ladder for the two-layer ETI claim.

The claim the paper makes is that an evolutionary transition in individuality
needs two layers: components that evolve, and a collective that arises from
them -- with nothing ever selecting on the collective. The scripts in ETI/
happen to contain the controls for exactly that ladder:

    neuronal_flock.py        self-organization only. Fixed random neural nets,
                             no energy, no reproduction, no mutation. Boids are
                             only ever removed (eaten), so the population is
                             monotonically non-increasing.
    evo_flock.py             component evolution of six scalar traits, with
                             resources -- but no predators.
    evo_cooperative_flock.py both layers: evolving neural genotypes under
                             predator pressure. The paper's script.

Measured with one common detector so the three are comparable: does a ring
form, and does it ever *divide*?

These scripts run their simulation at module scope, so the loader stubs pygame
and truncates the source before the main loop, keeping the definitions and the
populations the script builds.
"""

import argparse
import json
import math
import os
import random
import sys
import types

import numpy as np

import vitanova

ETI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Stub:
    """Absorbs any pygame call. Rendering is the only thing being removed."""

    def __getattr__(self, _name):
        return _Stub()

    def __call__(self, *_a, **_k):
        return _Stub()

    def __iter__(self):
        return iter(())


def load_headless(path, stop_before):
    """Exec a script's definitions with pygame stubbed, up to its main loop."""
    src = open(path).read()
    cut = src.index(stop_before)
    ns = {"__name__": "vn_ablation", "__file__": path}
    saved = sys.modules.get("pygame")
    sys.modules["pygame"] = _Stub()
    try:
        exec(compile(src[:cut], path, "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("pygame", None)
        else:
            sys.modules["pygame"] = saved
    return ns


def find_flocks(boids, threshold, min_neighbors):
    """Single-linkage flocks, same semantics as the scripts' identify_flocks."""
    flocks, visited = [], set()
    for boid in boids:
        if boid in visited:
            continue
        neighbors = [o for o in boids
                     if o is not boid
                     and math.hypot(o.x - boid.x, o.y - boid.y) <= threshold]
        if len(neighbors) < min_neighbors:
            continue
        new = set(neighbors)
        new.add(boid)
        visited.update(new)
        for existing in flocks[:]:
            if not new.isdisjoint(existing):
                new |= existing
                flocks.remove(existing)
        flocks.append(new)
    return flocks


# Per-script adapter: where to cut the source, and how to advance one step.
SCRIPTS = {
    "selforg": {
        "path": os.path.join(ETI_DIR, "neuronal_flock.py"),
        "cut": "running = True",
        "label": "neuronal_flock.py (self-organization only)",
        "step": "selforg",
    },
    "traits": {
        "path": os.path.join(ETI_DIR, "evo_flock.py"),
        "cut": "running = True",
        "label": "evo_flock.py (scalar-trait evolution, no predators)",
        "step": "traits",
    },
}


def step_selforg(ns):
    """neuronal_flock.py's loop body, minus the draw calls."""
    boids, predators = ns["boids"], ns["predators"]
    for boid in boids:
        boid.update(boids, predators)
    for predator in predators:
        predator.update(boids)


def step_traits(ns):
    """evo_flock.py's loop body, minus the draw calls."""
    boids, resources = ns["boids"], ns["resources"]
    for boid in boids[:]:
        offspring = boid.reproduce()
        if offspring:
            boids.append(offspring)
        if boid.update(boids, resources):
            boids.remove(boid)
    for resource in resources:
        resource.regenerate_energy()


def run(which, steps, seed=None, detect_every=25, snapshot_every=0):
    cfg = SCRIPTS[which]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ns = load_headless(cfg["path"], cfg["cut"])
    if seed is not None:
        # The script seeds itself at import; re-seed and rebuild so a sweep can
        # vary the seed without editing the file.
        random.seed(seed)
        np.random.seed(seed)

    step_fn = {"selforg": step_selforg, "traits": step_traits}[cfg["step"]]
    boids = ns["boids"]
    threshold = ns.get("FLOCK_THRESHOLD", ns["BOID_SIZE"] * 5)
    min_neighbors = ns.get("FLOCK_SIZE_THRESHOLD", 4)

    rec = {k: [] for k in ("step", "n_boids", "n_flocks", "largest",
                           "ring_score", "coverage", "cv", "hollow", "mean_r")}
    snaps = {}

    for step in range(steps):
        if step % detect_every == 0:
            flocks = [f for f in find_flocks(boids, threshold, min_neighbors)
                      if len(f) >= 10]
            rec["step"].append(step)
            rec["n_boids"].append(len(boids))
            rec["n_flocks"].append(len(flocks))
            rec["largest"].append(max((len(f) for f in flocks), default=0))
            m = vitanova.ring_metrics(max(flocks, key=len)) if flocks else None
            for k in ("ring_score", "coverage", "cv", "mean_r", "hollow"):
                rec[k].append(m[k] if m else 0.0)
        if snapshot_every and step % snapshot_every == 0:
            snaps[step] = snapshot(ns)
        if not boids:
            break
        step_fn(ns)

    out = {k: np.array(v) for k, v in rec.items()}
    out["snapshots"] = snaps
    out["label"] = cfg["label"]
    out["seed"] = seed
    out["max_steps"] = steps
    out["detect_every"] = detect_every
    out["extinct_at"] = None
    out["wall_seconds"] = 0.0
    return out


def snapshot(ns):
    boids, predators = ns["boids"], ns.get("predators", [])
    members = sorted(boids, key=lambda b: (b.x, b.y, id(b)))
    role = lambda b: getattr(b, "role", "worker") == "soldier"
    return {
        "x": np.array([b.x for b in members]),
        "y": np.array([b.y for b in members]),
        "dx": np.array([b.dx for b in members]),
        "dy": np.array([b.dy for b in members]),
        "soldier": np.array([role(b) for b in members]),
        "energy": np.array([getattr(b, "energy", 0.0) for b in members]),
        "age": np.array([getattr(b, "age", 0) for b in members]),
        "px": np.array([p.x for p in predators]),
        "py": np.array([p.y for p in predators]),
        "pdx": np.array([p.dx for p in predators]),
        "pdy": np.array([p.dy for p in predators]),
    }


def verdict(res):
    """Did a ring form, and did the collective ever divide?"""
    n = res["n_boids"]
    flocks = res["n_flocks"]
    grew = bool(len(n) > 1 and n.max() > n[0])
    ring_samples = int((res["ring_score"] >= 0.35).sum())
    single_then_two = False
    seen_single = False
    for f in flocks:
        if f == 1:
            seen_single = True
        elif f >= 2 and seen_single:
            single_then_two = True
            break
    return {
        "script": res["label"],
        "seed": res["seed"],
        "n_boids_start": int(n[0]) if len(n) else 0,
        "n_boids_end": int(n[-1]) if len(n) else 0,
        "population_ever_grew": grew,
        "max_ring_score": round(float(res["ring_score"].max()), 3) if len(res["ring_score"]) else 0.0,
        "ring_samples": ring_samples,
        "max_flock": int(res["largest"].max()) if len(res["largest"]) else 0,
        "one_flock_became_two": single_then_two,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=sorted(SCRIPTS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--detect-every", type=int, default=25)
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run(args.which, args.steps, args.seed, args.detect_every,
              args.snapshot_every)
    print(json.dumps(verdict(res), indent=2))
    if args.out:
        vitanova.save(res, args.out)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
