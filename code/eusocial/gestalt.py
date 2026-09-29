"""Nestmate recognition on a gestalt badge exchanged by contact.

Heritable recognition cannot work here, and the reason is structural rather
than a matter of calibration: VitaNova collectives are chimeras.  Traced by
founder lineage over 132 collectives from 6 seeds, a collective holds a median
of 3 distinct founding lineages (effective 3.47), the largest lineage accounts
for 48% of members, and 0% of collectives are a single lineage.  There is no
kin to recognise, so no heritable tag at any dimension or mutation rate can
label one -- which is what the four failed sweeps were each rediscovering.

Ants in this exact situation -- polygynous, multi-lineage colonies -- do not
use a genetic label either.  They use a gestalt odour (Crozier & Dix 1979): a
blend continuously exchanged among nestmates, so the colony converges on a
shared badge no matter how unrelated its members are.

Here each individual carries a badge vector, relaxes it toward the mean badge
of whoever it is in contact with, and passes its own badge to its offspring.
Membership, not ancestry, is what the badge ends up encoding.
"""
from __future__ import annotations

import numpy as np

import crowded as CR
import vitafast as VF

BADGE_DIM = 16
BLEND = 0.02                 # per-step relaxation toward contacts' mean badge
MIX_SIGMA = 0.5              # similarity scale for who one exchanges badge with
RECOG_RADIUS = VF.PERCEPTION_RADIUS


class Gestalt(CR.Crowded):
    def __init__(self, seed=2024, gamma=2.0, badge_d0=0.5, badge_dim=BADGE_DIM,
                 blend=BLEND, mix_sigma=MIX_SIGMA, recog_radius=RECOG_RADIUS,
                 **kw):
        super().__init__(seed=seed, **kw)
        self.gamma = gamma
        self.badge_d0 = badge_d0
        self.blend = blend
        self.mix_sigma = mix_sigma
        self.recog_radius = recog_radius
        self.badge = self.rng.standard_normal((self.n, badge_dim))
        self._stranger = np.zeros(self.n)

    def _keep(self, mask):
        self.badge = self.badge[mask]
        super()._keep(mask)

    def _reproduce(self):
        before = self.n
        par = np.flatnonzero(
            (self.E > self.cfg["REPRODUCTION_ENERGY_THRESHOLD"])
            & (self.n + 1 <= self.cfg["MAX_BOIDS"]))
        super()._reproduce()
        k = self.n - before
        if k > 0:
            self.badge = np.vstack([self.badge, self.badge[par[:k]].copy()])

    def _badge_dist(self):
        B = self.badge
        d = np.abs(B[:, None, :] - B[None, :, :]).mean(2)
        return d / max(B.std(), 1e-9)

    def _mix(self, near):
        """Drift toward the mean badge of contacts ALREADY recognised as nestmates.

        Blending with everyone is self-defeating: two collectives in contact
        homogenise their badges within tens of steps, so the stranger signal
        decays exactly when it is needed.  Measured with indiscriminate
        blending, fusion events rose from 19 to 64 over 30 seeds (p=0.022)
        instead of falling.  Ants exchange hydrocarbons by trophallaxis and
        allogrooming among nestmates; they do not groom intruders.

        A hard nestmate-only gate does not work either: badges start random, so
        nobody is a nestmate, blending never starts and nothing ever converges
        (within-collective distance 0.864, 71% overlap).  The weight is
        therefore graded by similarity, which bootstraps from a random start
        and still concentrates exchange inside a collective once one forms.
        """
        cnt = near.sum(1)
        has = cnt > 0
        if not has.any():
            return
        mean = (near @ self.badge) / np.maximum(cnt, 1)[:, None]
        self.badge[has] += self.blend * (mean[has] - self.badge[has])

    def step(self):
        if self.n == 0:
            return
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        near = (d < self.recog_radius) & ~np.eye(n, dtype=bool)
        if near.any():
            bd = self._badge_dist()
            excess = np.where(near, np.maximum(0.0, bd - self.badge_d0), 0.0)
            self._stranger = excess.sum(1) / np.maximum(near.sum(1), 1)
            dx = self.x[None, :] - self.x[:, None]
            dy = self.y[None, :] - self.y[:, None]
            w = np.where(near, excess / np.maximum(d, 1e-5), 0.0)
            rx, ry = -(dx * w).sum(1), -(dy * w).sum(1)
            self._mix(np.where(near, np.exp(-bd / self.mix_sigma), 0.0))
        else:
            self._stranger = np.zeros(n)
            rx = ry = np.zeros(n)
        super().step()
        k = min(self.n, len(rx))
        if k and self.gamma:
            self.vx[:k] += self.gamma * rx[:k]
            self.vy[:k] += self.gamma * ry[:k]
            mag = np.hypot(self.vx, self.vy)
            nz = mag != 0
            self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * VF.SPEED, 0.0)
            self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * VF.SPEED, 0.0)
