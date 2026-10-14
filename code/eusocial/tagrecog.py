# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Nestmate recognition on a dedicated tag locus.

Recognition on the 150-weight control genome does not work, and the reason is
variance rather than calibration.  Measured: genetic distance within a
collective is 0.735 +/- 0.278, between collectives 1.110 +/- 0.043.  The means
separate cleanly (paired Wilcoxon p = 2.7e-19) but the distributions overlap,
so at the level of an individual pair -- which is where recognition has to act
-- nestmate and stranger are not distinguishable.  Every threshold tried left a
non-zero misfire rate (0.06-0.13) against nestmates while still failing to
block fusion.

Real eusocial insects do not compare their whole genome either.  Nestmate
recognition runs on a cuticular hydrocarbon profile: a small, low-dimensional,
faithfully inherited label that is decoupled from the genes for behaviour.

So this adds one: TAG_DIM heritable floats that do nothing except identify.
They mutate far more rarely and far more weakly than the control weights, so a
lineage keeps its label while its behaviour is free to drift.
"""
from __future__ import annotations

import numpy as np

import crowded as CR
import vitafast as VF

TAG_DIM = 4
TAG_MUT_RATE = 0.05          # vs MUTATION_RATE for the control genome
TAG_MUT_STRENGTH = 0.05
RECOG_RADIUS = VF.PERCEPTION_RADIUS


class TagRecog(CR.Crowded):
    def __init__(self, seed=2024, gamma=2.0, tag_d0=0.5, tag_dim=TAG_DIM,
                 tag_mut_rate=TAG_MUT_RATE, tag_mut_strength=TAG_MUT_STRENGTH,
                 recog_radius=RECOG_RADIUS, **kw):
        super().__init__(seed=seed, **kw)
        self.gamma = gamma
        self.tag_d0 = tag_d0
        self.tag_mut_rate = tag_mut_rate
        self.tag_mut_strength = tag_mut_strength
        self.recog_radius = recog_radius
        self.tag = self.rng.standard_normal((self.n, tag_dim))
        self._stranger = np.zeros(self.n)

    # keep the tag aligned with every other per-individual array
    def _keep(self, mask):
        self.tag = self.tag[mask]
        super()._keep(mask)

    def _reproduce(self):
        before = self.n
        parents = np.flatnonzero(
            (self.E > self.cfg["REPRODUCTION_ENERGY_THRESHOLD"])
            & (self.n + 1 <= self.cfg["MAX_BOIDS"]))
        super()._reproduce()
        k = self.n - before
        if k > 0:
            t = self.tag[parents[:k]].copy()
            m = self.rng.random(t.shape) < self.tag_mut_rate
            t += m * self.rng.normal(0, self.tag_mut_strength, t.shape)
            self.tag = np.vstack([self.tag, t])

    def _tag_dist(self):
        T = self.tag
        d = np.abs(T[:, None, :] - T[None, :, :]).mean(2)
        return d / max(T.std(), 1e-9)

    def step(self):
        if self.n == 0:
            return
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        near = (d < self.recog_radius) & ~np.eye(n, dtype=bool)
        if near.any():
            td = self._tag_dist()
            excess = np.where(near, np.maximum(0.0, td - self.tag_d0), 0.0)
            self._stranger = excess.sum(1) / np.maximum(near.sum(1), 1)
            dx = self.x[None, :] - self.x[:, None]
            dy = self.y[None, :] - self.y[:, None]
            w = np.where(near, excess / np.maximum(d, 1e-5), 0.0)
            rx, ry = -(dx * w).sum(1), -(dy * w).sum(1)
        else:
            self._stranger = np.zeros(n)
            rx = ry = np.zeros(n)
        super().step()
        k = min(self.n, len(rx))
        if k:
            self.vx[:k] += self.gamma * rx[:k]
            self.vy[:k] += self.gamma * ry[:k]
            mag = np.hypot(self.vx, self.vy)
            nz = mag != 0
            self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * VF.SPEED, 0.0)
            self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * VF.SPEED, 0.0)
