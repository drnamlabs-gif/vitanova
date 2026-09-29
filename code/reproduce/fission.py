"""Are the two daughter structures lineage-built, or a random split of the mother?

This is the fraternal / egalitarian question in the ETI literature, asked of one
concrete division event. Two independent tests:

  lineage   trace every member back through recorded parent links. If daughters
            are lineage-structured, members of the same daughter coalesce more
            recently with each other than with members of the other daughter.
            Tested against a permutation null that keeps the tree and reshuffles
            only the daughter labels.

  spatial   project members onto the axis joining the two eventual daughter
            centroids, and look backwards in time. If the split is prefigured,
            the projection is bimodal well before the structures separate.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ring_report as rr


def parent_map(z):
    """child uid -> parent uid, from the recorded birth events."""
    pm = {}
    for step, pu, cu, _, _ in z["births"]:
        pm[int(cu)] = int(pu)
    return pm


def path_to_root(uid, pm, cache):
    if uid in cache:
        return cache[uid]
    path, u = [], uid
    while True:
        path.append(u)
        if u not in pm:
            break
        u = pm[u]
    cache[uid] = path
    return path


def coalescent(a, b, pm, cache):
    """Generations back to the most recent common ancestor, or None."""
    pa = path_to_root(a, pm, cache)
    pb = path_to_root(b, pm, cache)
    ib = {u: i for i, u in enumerate(pb)}
    for i, u in enumerate(pa):
        if u in ib:
            return max(i, ib[u])
    return None


def daughters_at(z, step):
    f = rr.frame(z, step)
    comps = rr.components(f["x"], f["y"])
    comps = sorted(comps, key=len, reverse=True)[:2]
    return f, comps


def lineage_test(f, comps, pm, n_perm=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.concatenate(comps)
    label = np.concatenate([np.full(len(c), i) for i, c in enumerate(comps)])
    uids = f["uid"][idx].astype(int)

    cache = {}
    n = len(uids)
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            c = coalescent(uids[i], uids[j], pm, cache)
            D[i, j] = D[j, i] = np.nan if c is None else c

    iu = np.triu_indices(n, 1)
    same = label[iu[0]] == label[iu[1]]
    d = D[iu]
    ok = ~np.isnan(d)

    stat = lambda m: (np.nanmean(d[ok & m]), int((ok & m).sum()))
    w_mean, w_n = stat(same)
    b_mean, b_n = stat(~same)
    observed = b_mean - w_mean

    null = np.empty(n_perm)
    for k in range(n_perm):
        lab = rng.permutation(label)
        s = lab[iu[0]] == lab[iu[1]]
        null[k] = np.nanmean(d[ok & ~s]) - np.nanmean(d[ok & s])
    p = float((null >= observed).mean())

    founders = []
    for c in comps:
        fs = {path_to_root(int(u), pm, cache)[-1] for u in f["uid"][c].astype(int)}
        founders.append(fs)

    return dict(
        sizes=[len(c) for c in comps],
        within_mean=w_mean, within_pairs=w_n,
        between_mean=b_mean, between_pairs=b_n,
        observed=observed, p_value=p,
        null_mean=float(null.mean()), null_sd=float(null.std()),
        pairs_with_common_ancestor=float(ok.mean()),
        founders=[sorted(x) for x in founders],
        shared_founders=sorted(founders[0] & founders[1]),
    )


def spatial_precursor(z, steps, split_step, f_split, comps):
    """How early is the eventual split visible as a bimodal projection?"""
    cx = [f_split["x"][c].mean() for c in comps]
    cy = [f_split["y"][c].mean() for c in comps]
    ax = np.array([cx[1] - cx[0], cy[1] - cy[0]], float)
    ax /= max(np.linalg.norm(ax), 1e-9)

    member = {}
    for i, c in enumerate(comps):
        for u in f_split["uid"][c]:
            member[int(u)] = i

    rows = []
    for s in [t for t in steps if t <= split_step][-25:]:
        f = rr.frame(z, s)
        comps_s = rr.components(f["x"], f["y"])
        if not comps_s:
            continue
        big = comps_s[0]
        lab = np.array([member.get(int(u), -1) for u in f["uid"][big]])
        seen = lab >= 0
        if seen.sum() < 8 or len(set(lab[seen])) < 2:
            continue
        proj = (f["x"][big] - f["x"][big].mean()) * ax[0] + \
               (f["y"][big] - f["y"][big].mean()) * ax[1]
        g0, g1 = proj[seen][lab[seen] == 0], proj[seen][lab[seen] == 1]
        pooled = np.sqrt((g0.var() + g1.var()) / 2) if len(g0) > 1 and len(g1) > 1 else np.nan
        sep = abs(g0.mean() - g1.mean())
        rows.append((s, len(big), int(seen.sum()), sep,
                     sep / pooled if pooled and pooled > 0 else np.nan))
    return rows, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--split-step", type=int, default=None)
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    steps = sorted({int(k.split("_")[1]) for k in z.files
                    if k.startswith("snap_") and k.endswith("_uid")})
    pm = parent_map(z)

    split = a.split_step
    if split is None:
        prev = 1
        for s in steps:
            f = rr.frame(z, s)
            c = rr.components(f["x"], f["y"])
            if len(c) >= 2 and prev == 1:
                split = s
                break
            prev = len(c)
    f, comps = daughters_at(z, split)
    print(f"division at step {split}: daughters of {[len(c) for c in comps]} members\n")

    res = lineage_test(f, comps, pm)
    print("LINEAGE")
    print(f"  pairs with a common ancestor in the record: "
          f"{res['pairs_with_common_ancestor']*100:.0f}%")
    print(f"  mean coalescent depth within a daughter : {res['within_mean']:.2f} "
          f"generations   ({res['within_pairs']} pairs)")
    print(f"  mean coalescent depth between daughters : {res['between_mean']:.2f} "
          f"generations   ({res['between_pairs']} pairs)")
    print(f"  difference (between - within)           : {res['observed']:+.3f}")
    print(f"  permutation null                        : "
          f"{res['null_mean']:+.3f} +/- {res['null_sd']:.3f}")
    print(f"  p (one-sided, daughters more related than chance): {res['p_value']:.4f}")
    print(f"  founder lineages represented: daughter A {len(res['founders'][0])}, "
          f"daughter B {len(res['founders'][1])}, "
          f"shared {len(res['shared_founders'])}")

    rows, ax = spatial_precursor(z, steps, split, f, comps)
    print(f"\nSPATIAL PRECURSOR   (axis {ax.round(2).tolist()})")
    print(f"  {'step':>6}{'cluster':>9}{'labelled':>10}{'separation':>12}{'effect size':>13}")
    for s, nb, ns, sep, eff in rows:
        print(f"  {s:>6}{nb:>9}{ns:>10}{sep:>12.1f}{eff:>13.2f}")

    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        st = [r[0] for r in rows]
        axs[0].plot(st, [r[3] for r in rows], marker="o", ms=3)
        axs[0].axvline(split, ls="--", c="k", lw=.8)
        axs[0].set_title("centroid separation along the split axis (px)")
        axs[0].set_xlabel("step")
        axs[1].plot(st, [r[4] for r in rows], marker="o", ms=3, color="#b10dc9")
        axs[1].axvline(split, ls="--", c="k", lw=.8)
        axs[1].axhline(2.0, ls=":", c="grey", lw=.8)
        axs[1].set_title("effect size (separation / pooled sd)")
        axs[1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(f"{a.outdir}/fission.png", dpi=140)
        print(f"\nfigure: {a.outdir}/fission.png")


if __name__ == "__main__":
    main()
