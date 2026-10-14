# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Reproduction gated on the behavioural state.

One change to the model: an agent in the soldier state cannot reproduce on that
step. It may reproduce as soon as its own network puts it back into the worker
state and it still has the energy.

Why this is a trade-off and not an assumed answer. The soldier/worker output
already exists, chosen by each agent's evolved network every step, but it has
almost no reproductive consequence: across sixteen replicates the correlation
between time spent as a soldier and offspring produced is +0.07, so selection
cannot see it. The gate gives that output a cost, which is what makes it
evolvable. Nothing here assigns castes, declares anyone sterile, or says
lineages should differ.

The prediction, if specialisation pays, is that the fraction of life an
individual spends as a soldier becomes **bimodal** across the population: some
lineages effectively never breed while others do. If it stays unimodal, agents
are simply switching out of the soldier state whenever they are ready to breed,
and no division of labour has evolved.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fast"))
import vitafast as VF  # noqa: E402


class Gated(VF.Fast):
    def __init__(self, seed=2024, gate=True, **kw):
        super().__init__(seed=seed, **kw)
        self.gate = bool(gate)
        self.sol_time = defaultdict(int)     # steps spent as a soldier, by uid
        self.seen = defaultdict(int)

    def _reproduce(self):
        c = self.cfg
        ready = (self.E > c["REPRODUCTION_ENERGY_THRESHOLD"])
        if self.gate:
            ready = ready & (~self.sol)      # not while defending
        idx = np.flatnonzero(ready & (self.n + 1 <= c["MAX_BOIDS"]))
        if not len(idx):
            return
        k = len(idx)
        g = self.G[idx].copy()
        m = self.rng.random(g.shape) < c["MUTATION_RATE"]
        g += m * self.rng.normal(0, c["MUTATION_STRENGTH"], g.shape)
        ang = self.rng.uniform(-1, 1, (k, 2))
        nrm = np.maximum(np.hypot(ang[:, 0], ang[:, 1]), 1e-12)
        ux, uy = ang[:, 0] / nrm, ang[:, 1] / nrm
        uids = np.arange(self._next, self._next + k, dtype=np.int64)
        self._next += k
        for pu, cu in zip(self.uid[idx], uids):
            self.births.append((self.step_i, int(pu), int(cu)))
        self.x = np.concatenate([self.x, self.x[idx] + ux * VF.REPRODUCTION_DISTANCE])
        self.y = np.concatenate([self.y, self.y[idx] + uy * VF.REPRODUCTION_DISTANCE])
        self.vx = np.concatenate([self.vx, self.rng.uniform(-1, 1, k)])
        self.vy = np.concatenate([self.vy, self.rng.uniform(-1, 1, k)])
        self.sol = np.concatenate([self.sol, self.rng.random(k) < 0.5])
        self.G = np.vstack([self.G, g])
        self.E = np.concatenate([self.E, np.full(k, VF.INITIAL_ENERGY)])
        self.age = np.concatenate([self.age, np.zeros(k, np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.par = np.concatenate([self.par, self.uid[idx]])
        self.E[idx] -= VF.REPRODUCTION_COST
        self.x = np.clip(self.x, 0, VF.WIDTH)
        self.y = np.clip(self.y, 0, VF.HEIGHT)

    def step(self):
        VF.Fast.step(self)
        for u, s in zip(self.uid, self.sol):
            self.seen[int(u)] += 1
            self.sol_time[int(u)] += int(bool(s))

    def soldier_fractions(self, min_seen=100):
        return np.array([self.sol_time[u] / self.seen[u]
                         for u in self.seen if self.seen[u] >= min_seen])
