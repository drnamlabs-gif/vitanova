# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Nestmate recognition on top of the crowding cost.

In real eusocial insects colony boundaries are held by nestmate recognition:
cuticular hydrocarbon profiles are compared pairwise on contact, and
non-nestmates are avoided or attacked.  It is a purely local comparison, so it
adds no collective-level bookkeeping.

The precondition holds here.  Measured live over 107 two-collective frames from
6 seeds: genetic distance within a collective is 0.735, between collectives
1.110, a ratio of 1.51 in 100% of frames (paired Wilcoxon p = 2.7e-19).

Two ways of paying for a stranger, because they are not interchangeable:

  mode="energy"  an energy penalty, which is what "a cost" literally means.
                 Note that energy never kills in this model (death is by age
                 alone); energy only gates reproduction.  So this can suppress
                 a merged group's reproduction but cannot stop the merge.

  mode="force"   repulsion from dissimilar neighbours, which is what actually
                 keeps two collectives apart -- and what the insects do.

Both are driven by the same locally sensed quantity.
"""
from __future__ import annotations

import numpy as np

import crowded as CR
import vitafast as VF

RECOG_RADIUS = VF.PERCEPTION_RADIUS      # 25 px, contact range
RECOG_D0 = 0.92                          # midpoint of measured within (0.735)
                                         # and between (1.110) collectives


class Recog(CR.Crowded):
    def __init__(self, seed=2024, mode="force", gamma=1.0,
                 recog_radius=RECOG_RADIUS, recog_d0=RECOG_D0, **kw):
        super().__init__(seed=seed, **kw)
        self.mode = mode
        self.gamma = gamma
        self.recog_radius = recog_radius
        self.recog_d0 = recog_d0
        self._stranger = np.zeros(self.n)     # last sensed excess dissimilarity

    def _gene_dist(self):
        """Mean |per-gene difference|, in units of the genome's own spread.

        This must match probe_kin.py exactly, or the tolerance threshold is
        calibrated in one scale and applied in another: RMS distance runs a
        factor sqrt(pi/2) = 1.254 larger than mean-absolute for gaussian
        genomes, which silently puts d0 below the within-collective value and
        penalises every individual from step 0.
        """
        G = self.G
        d = np.abs(G[:, None, :] - G[None, :, :]).mean(2)
        return d / max(G.std(), 1e-9)

    def _recognise(self, d):
        """mean excess dissimilarity of each boid's contacts, and its direction"""
        n = self.n
        eye = np.eye(n, dtype=bool)
        near = (d < self.recog_radius) & ~eye
        if not near.any():
            self._stranger = np.zeros(n)
            return np.zeros(n), np.zeros(n)
        gd = self._gene_dist()
        excess = np.where(near, np.maximum(0.0, gd - self.recog_d0), 0.0)
        cnt = np.maximum(near.sum(1), 1)
        self._stranger = excess.sum(1) / cnt
        # push away from strangers, weighted by how strange they are
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        w = np.where(near, excess / np.maximum(d, 1e-5), 0.0)
        return -(dx * w).sum(1), -(dy * w).sum(1)

    def step(self):
        if self.n == 0:
            return
        if self.mode == "force":
            d = np.hypot(self.x[None, :] - self.x[:, None],
                         self.y[None, :] - self.y[:, None])
            rx, ry = self._recognise(d)
            super().step()
            if self.n >= len(rx):
                self.vx[:len(rx)] += self.gamma * rx
                self.vy[:len(ry)] += self.gamma * ry
                mag = np.hypot(self.vx, self.vy)
                nz = mag != 0
                self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * VF.SPEED, 0.0)
                self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * VF.SPEED, 0.0)
        else:
            super().step()

    def _energy(self):
        super()._energy()
        if self.mode == "energy":
            d = np.hypot(self.x[None, :] - self.x[:, None],
                         self.y[None, :] - self.y[:, None])
            self._recognise(d)
            k = min(len(self.E), len(self._stranger))
            self.E[:k] -= self.gamma * self._stranger[:k]
