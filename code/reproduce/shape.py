# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Radial-structure statistics for a flock.

The structure of interest is not simply "hollow". The flocks that divide have a
dense core *and* a dense outer annulus with a trough between them -- bimodal in
radius. Two earlier metrics both failed on this:

  histogram "hollow"  the innermost fixed-width bin has tiny area, so at n < 50
                      it is empty by chance and hollowness is overestimated.
  mean_r / max_r      a core pulls the mean radius down, so a core+annulus
                      scores lower than a uniform disk -- the wrong direction.

This uses equal-count radial bins (each bin holds the same number of members, so
the density estimate has the same sampling noise everywhere) and looks for an
interior trough between two peaks.
"""

import numpy as np

UNIFORM_DISK_MEAN_RATIO = 2.0 / 3.0


def radial_profile(x, y, bins=5):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < bins * 3:
        return None
    cx, cy = x.mean(), y.mean()
    r = np.sort(np.hypot(x - cx, y - cy))
    if r[-1] <= 0:
        return None
    edges = [0.0] + [float(r[min(n - 1, (i + 1) * n // bins - 1)])
                     for i in range(bins)]
    edges = np.array(edges)
    counts = np.histogram(r, bins=edges)[0].astype(float)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    dens = counts / np.maximum(areas, 1e-12)
    return dens, edges, r, (cx, cy)


def annularity(x, y, bins=5):
    """Score in [0,1] for a core+annulus structure, plus its parts.

    trough    depth of the interior density minimum relative to the peaks on
              both sides -- 0 when density is monotone, ->1 for a deep gap.
    coverage  fraction of 12 angular sectors occupied; separates a full ring
              from an arc.
    core_frac fraction of members inside the trough radius.
    """
    prof = radial_profile(x, y, bins)
    if prof is None:
        return None
    dens, edges, r, (cx, cy) = prof

    inner = np.arange(1, len(dens) - 1)
    if len(inner) == 0:
        return None
    t = int(inner[np.argmin(dens[inner])])
    left, right = dens[:t + 1].max(), dens[t:].max()
    trough = 1.0 - dens[t] / min(left, right) if min(left, right) > 0 else 0.0

    ang = np.arctan2(np.asarray(y, float) - cy, np.asarray(x, float) - cx)
    idx = np.floor((ang + np.pi) / (2 * np.pi) * 12).astype(int) % 12
    coverage = len(np.unique(idx)) / 12.0

    # A purely hollow ring has no interior peak, so `trough` is 0 for it.
    # Hollowness catches that case: density rising away from the centre.
    hollowness = 1.0 - dens[0] / dens.max() if dens.max() > 0 else 0.0

    r_trough = float(edges[t + 1])
    core_frac = float((r < r_trough).mean())

    return {
        "n": int(len(r)),
        "trough": float(max(trough, 0.0)),
        "coverage": float(coverage),
        "core_frac": core_frac,
        "r_trough": r_trough,
        "r_max": float(r[-1]),
        "mean_ratio": float(r.mean() / r[-1]),
        "hollowness": float(max(hollowness, 0.0)),
        "density": dens.tolist(),
        # a full ring of members with a real interior gap
        "annularity": float(max(trough, 0.0) * coverage),
    }


def largest_cluster(x, y, half=60):
    """Members of the densest square window -- the collective, without a
    flock-detector in the loop."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if not len(x):
        return x, y
    best, cx, cy = -1, x[0], y[0]
    for px, py in zip(x, y):
        n = int(((np.abs(x - px) <= half) & (np.abs(y - py) <= half)).sum())
        if n > best:
            best, cx, cy = n, px, py
    m = (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
    return x[m], y[m]
