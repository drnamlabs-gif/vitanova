"""Instrumented run of the paper model, for the window around a division.

Adds what the plain runs lack: a stable identity per boid, a parent link for
every birth, and a record of every death, plus dense snapshots through the
window of interest. Nothing here touches the dynamics -- identities are written
onto the objects, and the wrapped reproduce() calls the original first.
"""

import argparse
import json
import os

import numpy as np

import vitanova


def run(seed=2024, steps=6500, window=(3500, 6500), snap_every=25,
        coarse_every=100, target="paper"):
    eti = vitanova.load_original(target)
    boids, predators = vitanova.init_population(eti, seed)

    state = {"step": 0, "next_uid": 0}
    for b in boids:
        b._uid = state["next_uid"]; b._parent = -1; b._born = 0
        state["next_uid"] += 1

    births, deaths = [], []
    original_reproduce = eti.Boid.reproduce

    def reproduce(self, boids, _orig=original_reproduce):
        before = len(boids)
        _orig(self, boids)
        if len(boids) > before:
            child = boids[-1]
            child._uid = state["next_uid"]; state["next_uid"] += 1
            child._parent = self._uid
            child._born = state["step"]
            births.append((state["step"], self._uid, child._uid,
                           float(self.x), float(self.y)))

    eti.Boid.reproduce = reproduce

    snaps, series = {}, []
    alive_before = {b._uid for b in boids}

    for step in range(steps):
        state["step"] = step
        in_window = window[0] <= step <= window[1]
        if (in_window and step % snap_every == 0) or step % coarse_every == 0:
            members = sorted(boids, key=lambda b: b._uid)
            snaps[step] = {
                "uid": np.array([b._uid for b in members], dtype=np.int64),
                "parent": np.array([b._parent for b in members], dtype=np.int64),
                "born": np.array([b._born for b in members], dtype=np.int64),
                "x": np.array([b.x for b in members]),
                "y": np.array([b.y for b in members]),
                "dx": np.array([b.dx for b in members]),
                "dy": np.array([b.dy for b in members]),
                "energy": np.array([b.energy for b in members]),
                "age": np.array([b.age for b in members]),
                "soldier": np.array([b.role == "soldier" for b in members]),
                "px": np.array([p.x for p in predators]),
                "py": np.array([p.y for p in predators]),
                "pdx": np.array([p.dx for p in predators]),
                "pdy": np.array([p.dy for p in predators]),
            }
        if step % 25 == 0:
            flocks = vitanova.flock_sets(
                eti.identify_flocks(boids, eti.FLOCK_THRESHOLD))
            series.append((step, len(boids), len(flocks),
                           max((len(f) for f in flocks), default=0),
                           float(np.mean([b.energy for b in boids])) if boids else 0.0,
                           sum(b.role == "soldier" for b in boids)))
        if not boids:
            break

        to_remove = [b for b in boids if b.should_die()]
        for b in to_remove:
            deaths.append((step, b._uid, "age_or_energy"))
            boids.remove(b)
        for b in boids:
            b.update(boids, predators)
        alive_after = {b._uid for b in boids}
        for uid in alive_before - alive_after:
            if not any(d[1] == uid and d[0] == step for d in deaths):
                deaths.append((step, uid, "eaten"))
        for p in predators:
            p.update(boids)
        alive_before = {b._uid for b in boids}
        if step % 500 == 0:
            print(f"  step {step:5d}  n={len(boids):4d}  uids={state['next_uid']}",
                  flush=True)

    eti.Boid.reproduce = original_reproduce
    return snaps, np.array(series), births, deaths


def save(path, snaps, series, births, deaths, meta):
    payload = {}
    for s, snap in snaps.items():
        for k, a in snap.items():
            payload[f"snap_{s}_{k}"] = a
    cols = ("step", "n_boids", "n_flocks", "largest", "mean_energy", "n_soldier")
    for i, name in enumerate(cols):
        payload[name] = series[:, i]
    payload["ring_score"] = np.zeros(len(series))   # keeps shared tools happy
    payload["births"] = np.array(births, dtype=float) if births else np.zeros((0, 5))
    payload["deaths"] = np.array([(s, u, 0 if c == "eaten" else 1)
                                  for s, u, c in deaths], dtype=float) \
        if deaths else np.zeros((0, 3))
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--steps", type=int, default=6500)
    ap.add_argument("--from-step", type=int, default=3500)
    ap.add_argument("--to-step", type=int, default=6500)
    ap.add_argument("--snap-every", type=int, default=25)
    ap.add_argument("--out", default="runs/instr_seed2024.npz")
    args = ap.parse_args()

    snaps, series, births, deaths = run(
        args.seed, args.steps, (args.from_step, args.to_step), args.snap_every)
    meta = {"seed": args.seed, "label": f"instrumented seed {args.seed}",
            "detect_every": 25, "window": [args.from_step, args.to_step],
            "n_births": len(births), "n_deaths": len(deaths)}
    save(args.out, snaps, series, births, deaths, meta)
    print(f"\nsaved {args.out}: {len(snaps)} snapshots, "
          f"{len(births)} births, {len(deaths)} deaths")


if __name__ == "__main__":
    main()
