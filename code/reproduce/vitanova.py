# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Headless, instrumented driver for the original VitaNova simulation.

Loads a VitaNova script *unmodified* and drives it without pygame,
so hundreds of seeds can be swept for the ring-formation / ring-division
sequence reported in the ALIFE 2024 paper.

Dynamics are untouched. Only rendering is dropped: Boid.draw and Predator.draw
read self.x/y/dx/dy/role and consume no RNG, so removing them cannot change a
trajectory. identify_flocks() is likewise side-effect free with respect to the
dynamics -- it only writes boid.isInFlock, which nothing ever reads -- so it is
called on a sampling interval instead of every step.

The one subtlety is RNG consumption order. real_eti.py builds (and discards)
PREDATOR_COUNT predators at module scope before __main__ builds the population
it actually uses. init_population() replicates that, so seed s here gives the
same initial condition as running real_eti.py with seed s.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ETI_DIR = os.path.dirname(HERE)

# The ALIFE 2024 figures were produced by evo_cooperative_flock.py, not by
# real_eti.py -- see README.md for the evidence (window caption, info line,
# MAX_STEPS=4400 matching the figure's x-axis, the stacked two-row plot, and
# savefig('result/flock_size_vs_count.png')).
TARGETS = {
    "paper": os.path.join(ETI_DIR, "evo_cooperative_flock.py"),
    "basic": os.path.join(ETI_DIR, "basic_eti.py"),
    "toroidal": os.path.join(ETI_DIR, "toroidal_evolving_boids.py"),
    "real_eti": os.path.join(ETI_DIR, "original code", "real_eti.py"),
    # The code published with the paper (github.com/namlehai90/Evolving-Neuronal-Flocks).
    # Note it is NOT the script that produced the paper's figures: different window
    # caption, different on-screen info line, and should_die() drops energy death.
    "published": os.path.join(HERE, "published_flock.py"),
}
ORIGINAL = TARGETS["paper"]


def resolve_target(name):
    return TARGETS.get(name, name)


def load_original(path=ORIGINAL):
    """Import a VitaNova script as a module without running its __main__ block."""
    path = resolve_target(path)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    name = "vn_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def flock_sets(flocks):
    """identify_flocks returns a list of sets in some versions, a dict in others."""
    return list(flocks.values()) if isinstance(flocks, dict) else list(flocks)


def init_population(eti, seed):
    """Build the initial population exactly as the script's __main__ does."""
    random.seed(seed)
    np.random.seed(seed)

    # The script spawns PREDATOR_COUNT predators at module scope and then
    # __main__ overwrites them. Those draws still happened, so replicate them
    # or every downstream random number shifts.
    for _ in range(eti.PREDATOR_COUNT):
        px = random.randint(0, eti.WIDTH)
        py = random.randint(0, eti.HEIGHT)
        eti.Predator(px, py)

    boids = eti.create_boids()
    predators = [
        eti.Predator(random.randint(0, eti.WIDTH), random.randint(0, eti.HEIGHT))
        for _ in range(eti.PREDATOR_COUNT)
    ]

    # adjust_energy() and near_worker() read the module-level `boids` global.
    eti.boids = boids
    eti.predators = predators
    return boids, predators


def _ordered(flock):
    """Stable ordering for a flock set: set iteration order is id-based."""
    return sorted(flock, key=lambda b: (b.x, b.y, id(b)))


def ring_metrics(flock):
    """Shape descriptors for one flock, in a bounded box with reflecting walls.

    Returns coverage (how much of the circle is occupied), cv (spread of the
    radial distribution) and hollow (how empty the centre is relative to the
    densest annulus). A ring scores high on all three; a filled blob scores
    low on hollow; an arc scores low on coverage.
    """
    members = _ordered(flock)
    xs = np.array([b.x for b in members])
    ys = np.array([b.y for b in members])
    cx, cy = xs.mean(), ys.mean()
    r = np.hypot(xs - cx, ys - cy)
    mean_r = float(r.mean())
    if mean_r <= 1e-9:
        return None

    cv = float(r.std() / mean_r)

    ang = np.arctan2(ys - cy, xs - cx)
    nbins = 12
    idx = np.floor((ang + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    coverage = float(len(np.unique(idx)) / nbins)

    R = float(r.max())
    edges = np.linspace(0.0, R, 7)
    counts, _ = np.histogram(r, bins=edges)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = counts / np.maximum(areas, 1e-9)
    peak = float(dens.max())
    hollow = float(1.0 - dens[0] / peak) if peak > 0 else 0.0

    return {
        "n": len(members),
        "cx": float(cx),
        "cy": float(cy),
        "mean_r": mean_r,
        "cv": cv,
        "coverage": coverage,
        "hollow": max(hollow, 0.0),
        "ring_score": coverage * max(0.0, 1.0 - min(cv, 1.0)) * max(hollow, 0.0),
    }


def snapshot(boids, predators):
    members = sorted(boids, key=lambda b: (b.x, b.y, id(b)))
    return {
        "x": np.array([b.x for b in members]),
        "y": np.array([b.y for b in members]),
        "dx": np.array([b.dx for b in members]),
        "dy": np.array([b.dy for b in members]),
        "energy": np.array([b.energy for b in members]),
        "age": np.array([b.age for b in members]),
        "soldier": np.array([b.role == "soldier" for b in members]),
        "genotype": np.array([b.genotype for b in members]),
        "px": np.array([p.x for p in predators]),
        "py": np.array([p.y for p in predators]),
        "pdx": np.array([p.dx for p in predators]),
        "pdy": np.array([p.dy for p in predators]),
    }


def run(seed, max_steps=4400, detect_every=25, snapshot_every=0,
        min_flock=10, max_boids=0, eti=None, verbose=False):
    """Run one simulation headless, returning sampled metrics over time."""
    eti = eti or load_original()
    boids, predators = init_population(eti, seed)
    bloom_at = None

    steps, n_boids, n_flocks, big_flocks = [], [], [], []
    largest, soldier_frac, mean_energy = [], [], []
    ring, coverage, cv, hollow, mean_r, cx, cy = [], [], [], [], [], [], []
    snaps = {}

    t0 = time.time()
    for step in range(max_steps):
        if step % detect_every == 0:
            flocks = flock_sets(eti.identify_flocks(boids, eti.FLOCK_THRESHOLD))
            sized = sorted((len(f) for f in flocks), reverse=True)
            big = [f for f in flocks if len(f) >= min_flock]
            steps.append(step)
            n_boids.append(len(boids))
            n_flocks.append(len(flocks))
            big_flocks.append(len(big))
            largest.append(sized[0] if sized else 0)
            soldier_frac.append(
                sum(b.role == "soldier" for b in boids) / len(boids) if boids else 0.0
            )
            mean_energy.append(
                float(np.mean([b.energy for b in boids])) if boids else 0.0
            )
            m = ring_metrics(max(flocks, key=len)) if flocks else None
            ring.append(m["ring_score"] if m else 0.0)
            coverage.append(m["coverage"] if m else 0.0)
            cv.append(m["cv"] if m else 0.0)
            hollow.append(m["hollow"] if m else 0.0)
            mean_r.append(m["mean_r"] if m else 0.0)
            cx.append(m["cx"] if m else 0.0)
            cy.append(m["cy"] if m else 0.0)

        if snapshot_every and step % snapshot_every == 0:
            snaps[step] = snapshot(boids, predators)

        if not boids:
            break
        # Cost is O(N^2) per step, so a population bloom can cost more than the
        # rest of a sweep combined. Stop such a run and record why.
        if max_boids and len(boids) > max_boids:
            bloom_at = step
            break

        to_remove = [b for b in boids if b.should_die()]
        for b in to_remove:
            boids.remove(b)
        # Iterating the live list, exactly as real_eti.py does: reproduce()
        # appends offspring mid-iteration and they are updated the same step.
        for b in boids:
            b.update(boids, predators)
        for p in predators:
            p.update(boids)

        if verbose and step % 500 == 0:
            print(f"  seed {seed} step {step:5d} boids {len(boids):4d} "
                  f"flocks {n_flocks[-1] if n_flocks else 0}", flush=True)

    return {
        "seed": seed,
        "max_steps": max_steps,
        "detect_every": detect_every,
        "wall_seconds": time.time() - t0,
        "extinct_at": step if not boids else None,
        "bloom_at": bloom_at,
        "step": np.array(steps),
        "n_boids": np.array(n_boids),
        "n_flocks": np.array(n_flocks),
        "big_flocks": np.array(big_flocks),
        "largest": np.array(largest),
        "soldier_frac": np.array(soldier_frac),
        "mean_energy": np.array(mean_energy),
        "ring_score": np.array(ring),
        "coverage": np.array(coverage),
        "cv": np.array(cv),
        "hollow": np.array(hollow),
        "mean_r": np.array(mean_r),
        "cx": np.array(cx),
        "cy": np.array(cy),
        "snapshots": snaps,
    }


def classify(res, ring_thresh=0.35, hold=3):
    """Score a run against the sequence the paper describes.

    single_flock : a stretch where exactly one flock of size >= min_flock holds
    ring         : ring_score stays above threshold for `hold` samples while a
                   single flock dominates
    division     : that single flock becomes two sustained flocks afterwards
    """
    big = res["big_flocks"]
    rs = res["ring_score"]

    single = big == 1
    ring_run, ring_at = 0, None
    for i, (s, r) in enumerate(zip(single, rs)):
        if s and r >= ring_thresh:
            ring_run += 1
            if ring_run >= hold and ring_at is None:
                ring_at = int(res["step"][i - hold + 1])
        else:
            ring_run = 0

    div_at = None
    if ring_at is not None:
        start = int(np.argmax(res["step"] >= ring_at))
        run2 = 0
        for i in range(start, len(big)):
            if big[i] >= 2:
                run2 += 1
                if run2 >= hold:
                    div_at = int(res["step"][i - hold + 1])
                    break
            else:
                run2 = 0

    return {
        "seed": res["seed"],
        "survived": res["extinct_at"] is None,
        "extinct_at": res["extinct_at"],
        "bloom_at": res.get("bloom_at"),
        "max_ring_score": float(rs.max()) if len(rs) else 0.0,
        "max_largest": int(res["largest"].max()) if len(res["largest"]) else 0,
        "single_flock_samples": int(single.sum()),
        "ring_at": ring_at,
        "division_at": div_at,
        "full_sequence": ring_at is not None and div_at is not None,
        "wall_seconds": round(res["wall_seconds"], 1),
    }


def save(res, path):
    payload = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    for step, snap in res.get("snapshots", {}).items():
        for key, arr in snap.items():
            payload[f"snap_{step}_{key}"] = arr
    meta = {k: v for k, v in res.items()
            if not isinstance(v, np.ndarray) and k != "snapshots"}
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def selftest(target=ORIGINAL):
    """Verify init_population reproduces the script's own __main__ ordering."""
    eti = load_original(target)
    ref_boids = eti.create_boids()
    ref_preds = [eti.Predator(random.randint(0, eti.WIDTH),
                              random.randint(0, eti.HEIGHT))
                 for _ in range(eti.PREDATOR_COUNT)]
    ref = ([(b.x, b.y, b.dx, b.dy, b.role, float(b.genotype.sum())) for b in ref_boids],
           [(p.x, p.y, p.dx, p.dy) for p in ref_preds])

    boids, preds = init_population(eti, 2024)
    got = ([(b.x, b.y, b.dx, b.dy, b.role, float(b.genotype.sum())) for b in boids],
           [(p.x, p.y, p.dx, p.dy) for p in preds])

    ok = ref == got
    print(f"selftest: initial condition for seed 2024 matches "
          f"{os.path.basename(resolve_target(target))}:", ok)
    if not ok:
        print("  ref boid0:", ref[0][0])
        print("  got boid0:", got[0][0])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest")
    st.add_argument("--target", default="paper")

    r = sub.add_parser("run")
    r.add_argument("--target", default="paper",
                   help="paper | basic | toroidal | real_eti | path to a .py")
    r.add_argument("--seed", type=int, default=2024)
    r.add_argument("--steps", type=int, default=4400)
    r.add_argument("--detect-every", type=int, default=25)
    r.add_argument("--snapshot-every", type=int, default=0)
    r.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "selftest":
        sys.exit(0 if selftest(args.target) else 1)

    res = run(args.seed, args.steps, args.detect_every, args.snapshot_every,
              eti=load_original(args.target), verbose=True)
    verdict = classify(res)
    print(json.dumps(verdict, indent=2))
    if args.out:
        save(res, args.out)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
