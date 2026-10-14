# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Caste-free VitaNova with a density-dependent maintenance cost.

Motivation.  Without any cap, every run ends the same way: a collective reaches
the ring window (n ~ 20-55), keeps growing, outgrows the one-perception-radius
geometry that makes the ring possible, and turns into a permanently
unstructured blob.  21/21 runs did this.  The ring is a transient.

What the cost is.  Interference competition: the more competitors share your
foraging neighbourhood, the less you net.  In DEB terms this is the maintenance
term that scales with volume, and it is what makes growth self-limiting in
every real organism.  It is paid per individual from a locally sensed quantity,
so nothing at the collective level is being book-kept or rewarded.

Why the sensing radius is 40 px and not 15.  Measured over 4595 collectives:
at 15 px the ring CORE is the densest thing in the system (13.3 neighbours)
while the blob is loose (8.3), so a short-range cost dissolves the ring and
spares the blob -- exactly backwards.  At 40 px -- wider than an entire ring,
whose radius is one perception radius (25 px) -- the count saturates at the
collective's own size for a ring (rim 30.6) but keeps climbing with density and
area inside a blob (54.4).  It is the local expression of "can I see all of my
collective?", which is the quantity that actually separates the two states.

Everything else is nocaste.py untouched.
"""
from __future__ import annotations

import numpy as np

import nocaste as NC
import vitafast as VF

CROWD_RADIUS = 8 * VF.BOID_SIZE          # 40 px
CROWD_K = 40.0                           # free allowance, ~ring rim p90
CROWD_BETA = 0.02                        # energy per excess competitor per step


class Crowded(NC.NoCaste):
    def __init__(self, seed=2024, crowd_radius=CROWD_RADIUS, crowd_k=CROWD_K,
                 crowd_beta=CROWD_BETA, **kw):
        super().__init__(seed=seed, **kw)
        self.crowd_radius = crowd_radius
        self.crowd_k = crowd_k
        self.crowd_beta = crowd_beta

    def _energy(self):
        c = self.cfg
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        eye = np.eye(n, dtype=bool)

        close = (d < VF.PROXIMITY_THRESHOLD) & ~eye
        nw = close.sum(1)
        allee = np.where((nw > 1) & (nw < c["CRITICAL_GROUP_SIZE"]),
                         c["WORKER_TOGETHER_ENERGY"],
                         np.where(nw >= c["CRITICAL_GROUP_SIZE"],
                                  -c["WORKER_TOGETHER_ENERGY"], 0.0))

        n_wide = ((d < self.crowd_radius) & ~eye).sum(1)
        crowd = self.crowd_beta * np.maximum(0.0, n_wide - self.crowd_k)

        self.E += (c["WORKER_ENERGY_GAIN"] + allee
                   - VF.ENERGY_COST_PER_STEP - crowd)
