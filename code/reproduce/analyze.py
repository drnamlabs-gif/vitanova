# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Apply one strict definition of collective reproduction to every stored run.

The loose test -- "the flock count went from 1 to 2" -- cannot tell reproduction
apart from disintegration. A ring that starves and falls into two pieces scores
the same as a ring that grows and divides. The difference is whether the split
is driven by growth.

A division event here requires all four:

  1. a single flock of at least MIN_PARENT members, held for HOLD samples;
  2. that flock is ring-shaped at the time (ring score >= RING) -- otherwise any
     crowded blob that happens to split would count;
  3. that flock grew over the window before the split (accumulation phase);
  4. afterwards, at least two flocks of at least MIN_CHILD members each, held
     for HOLD samples;
  5. the combined membership of the children is at least KEEP of the parent --
     a split, not a collapse.
"""

import argparse
import glob
import json
import os

import numpy as np

MIN_PARENT = 15
MIN_CHILD = 6
HOLD = 3
KEEP = 0.7
RING = 0.35


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return z, meta


def sustained(mask, hold):
    """Indices where `mask` has been true for `hold` consecutive samples."""
    out, run = [], 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= hold:
            out.append(i)
    return out


def verdict(z, meta):
    step = z["step"]
    largest = z["largest"]
    ring = z["ring_score"]
    n_boids = z["n_boids"]
    # big_flocks counts flocks >= min_flock; the ablation runs already filter.
    nflocks = z["big_flocks"] if "big_flocks" in z.files else z["n_flocks"]

    single = sustained((nflocks == 1) & (largest >= MIN_PARENT)
                       & (ring >= RING), HOLD)
    ringed = sustained((nflocks == 1) & (ring >= RING), HOLD)

    division_at, parent_size, child_total = None, None, None
    for i in single:
        # accumulation: the parent is larger than it was HOLD*2 samples back
        j = max(0, i - 2 * HOLD)
        if largest[i] <= largest[j]:
            continue
        after = sustained(nflocks[i:] >= 2, HOLD)
        if not after:
            continue
        k = i + after[0]
        # children must together retain most of the parent
        if n_boids[k] >= KEEP * largest[i] and largest[k] >= MIN_CHILD:
            division_at = int(step[k])
            parent_size = int(largest[i])
            child_total = int(n_boids[k])
            break

    return {
        "run": meta.get("label") or f"seed {meta.get('seed')}",
        "seed": meta.get("seed"),
        "steps": int(step[-1]) if len(step) else 0,
        "pop_start": int(n_boids[0]) if len(n_boids) else 0,
        "pop_max": int(n_boids.max()) if len(n_boids) else 0,
        "pop_end": int(n_boids[-1]) if len(n_boids) else 0,
        "population_grew": bool(len(n_boids) and n_boids.max() > n_boids[0]),
        "max_ring_score": round(float(ring.max()), 3) if len(ring) else 0.0,
        "ring_at": int(step[ringed[0]] - (HOLD - 1) * meta.get("detect_every", 25)) if ringed else None,
        "max_flock": int(largest.max()) if len(largest) else 0,
        "division_at": division_at,
        "parent_size_at_split": parent_size,
        "children_total": child_total,
        "collective_reproduction": division_at is not None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help=".npz files or globs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    rows = []
    for f in files:
        try:
            z, meta = load(f)
            v = verdict(z, meta)
        except Exception as exc:
            v = {"run": os.path.basename(f), "error": f"{type(exc).__name__}: {exc}"}
        v["file"] = os.path.basename(f)
        rows.append(v)

    hdr = f"{'run':<52}{'pop':>14}  {'ring':>5}  {'maxflk':>6}  {'division':>9}"
    print(hdr)
    print("-" * len(hdr))
    for v in rows:
        if "error" in v:
            print(f"{v['run']:<52}{v['error']}")
            continue
        pop = f"{v['pop_start']}->{v['pop_max']}->{v['pop_end']}"
        div = str(v["division_at"]) if v["collective_reproduction"] else "-"
        print(f"{str(v['run'])[:51]:<52}{pop:>14}  {v['max_ring_score']:>5.2f}  "
              f"{v['max_flock']:>6}  {div:>9}")

    hits = [v for v in rows if v.get("collective_reproduction")]
    print(f"\ncollective reproduction: {len(hits)}/{len(rows)} runs")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
