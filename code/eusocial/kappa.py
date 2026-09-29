"""Reproductive division of labour, if it evolves.

In the model as it stands a soldier and a grouped worker earn exactly the same
net energy, +0.30 per step, and anyone above the threshold reproduces. There is
no trade-off, so there is nothing for a division of labour to divide, and the
measured correlation between time spent as a soldier and offspring produced is
+0.07 -- nothing.

This adds one heritable gene, kappa, the fraction of energy surplus an
individual puts into reproduction rather than into defence. It is a gene and
not a network output on purpose: a network output is recomputed every step, so
an agent would simply switch to reproducing whenever it had energy, which is
what castes already do here and why they never specialise. Kappa is fixed for
life and inherited, which is what a developmental commitment means.

What is assumed, and what is not:

  assumed      an individual cannot both reproduce at full rate and invest in
               defence. That is the basic life-history trade-off and it is not
               a claim about eusociality.
  not assumed  that anyone should be sterile, that defenders should be a
               distinct class, or that castes should exist at all. Writing
               "soldiers cannot reproduce" would put the answer in by hand,
               which is the mistake this project has been unpicking.

Defence is a public good: what an individual invests deters predators from its
neighbours as well as itself. That is the channel through which a non-reproducing
strategy could pay, and it replaces the hard-coded rule that a predator flees
three soldiers -- a rule the factorial experiment showed was not necessary.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fast"))
import vitafast as VF  # noqa: E402


class Eusocial(VF.Fast):
    def __init__(self, seed=2024, defence_pays=True, d_escape_half=5.0,
                 d_deter=8.0, d_decay=0.002, kappa_mut=0.05,
                 repro_threshold=150.0, init_kappa_gene=2.0, **kw):
        super().__init__(seed=seed, **kw)
        n = self.n
        # kappa lives on a real line and is squashed, so mutation is unbounded
        # while the fraction stays in [0, 1]
        # Start everyone as a reproducer. The experiment asks whether a
        # non-reproducing strategy can invade a population that has none, so the
        # initial state has to be the non-eusocial one. Starting at kappa = 0.5
        # instead doubles the effective time to first offspring and the
        # population dies before it can group at all.
        self.kap_gene = self.rng.normal(float(init_kappa_gene), 0.3, n)
        self.D = np.zeros(n)                     # standing defence investment
        self.R = np.zeros(n)                     # reproduction buffer
        self.defence_pays = bool(defence_pays)   # control arm turns this off
        self.d_escape_half = float(d_escape_half)
        self.d_deter = float(d_deter)
        self.d_decay = float(d_decay)
        self.kappa_mut = float(kappa_mut)
        # Calibrated so that an individual putting everything into reproduction
        # breeds on the original schedule: at +0.30 surplus per step it reaches
        # the threshold at about step 500, as in the unmodified model. Anyone
        # holding back then breeds later, or not at all within a lifespan --
        # which is the trade-off, not a penalty applied to a caste.
        self.repro_threshold = float(repro_threshold)
        self.escapes = 0
        self.deters = 0

    @property
    def kappa(self):
        return 1.0 / (1.0 + np.exp(-self.kap_gene))

    # ------------------------------------------------------------ allocation
    def _energy(self):
        c = self.cfg
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        close = (d < VF.PROXIMITY_THRESHOLD) & ~np.eye(n, dtype=bool)
        nw = (close & (~self.sol)[None, :]).sum(1)
        gain = np.where(self.sol, c["SOLDIER_ENERGY_GAIN"],
                        c["WORKER_ENERGY_GAIN"])
        allee = np.where((nw > 1) & (nw < c["CRITICAL_GROUP_SIZE"]),
                         c["WORKER_TOGETHER_ENERGY"],
                         np.where(nw >= c["CRITICAL_GROUP_SIZE"],
                                  -c["WORKER_TOGETHER_ENERGY"], 0.0))
        income = np.where(self.sol,
                          gain + np.where(nw > 0, c["NEAR_WORKER_GAIN"], 0.0),
                          gain + allee) - VF.ENERGY_COST_PER_STEP
        k = self.kappa
        surplus = np.maximum(income, 0.0)
        deficit = np.minimum(income, 0.0)
        self.R += k * surplus
        self.D += (1.0 - k) * surplus
        self.D *= (1.0 - self.d_decay)           # defence needs upkeep
        self.E += deficit                        # losses still hit the body
        self.E = np.maximum(self.E, 0.0)

    def _reproduce(self):
        c = self.cfg
        idx = np.flatnonzero((self.R > self.repro_threshold)
                             & (self.n + 1 <= c["MAX_BOIDS"]))
        if not len(idx):
            return
        k = len(idx)
        g = self.G[idx].copy()
        m = self.rng.random(g.shape) < c["MUTATION_RATE"]
        g += m * self.rng.normal(0, c["MUTATION_STRENGTH"], g.shape)
        kg = self.kap_gene[idx] + self.rng.normal(0, self.kappa_mut, k)
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
        self.kap_gene = np.concatenate([self.kap_gene, kg])
        self.D = np.concatenate([self.D, np.zeros(k)])
        self.R = np.concatenate([self.R, np.zeros(k)])
        self.E = np.concatenate([self.E, np.full(k, VF.INITIAL_ENERGY)])
        self.age = np.concatenate([self.age, np.zeros(k, np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.par = np.concatenate([self.par, self.uid[idx]])
        self.R[idx] -= self.repro_threshold
        self.x = np.clip(self.x, 0, VF.WIDTH)
        self.y = np.clip(self.y, 0, VF.HEIGHT)

    # -------------------------------------------------------------- predators
    def _predators(self):
        c = self.cfg
        for i in range(len(self.px)):
            if self.p_eat[i] > 0:
                self.p_eat[i] -= 1
                continue
            if self.n == 0:
                continue
            bx, by = self.x - self.px[i], self.y - self.py[i]
            dist = np.hypot(bx, by)
            head = math.atan2(self.pvy[i], self.pvx[i])
            diff = np.abs(np.arctan2(by, bx) - head)
            diff = np.where(diff > math.pi, 2 * math.pi - diff, diff)
            in_view = (dist <= VF.PRED_VISION) & (diff <= VF.PRED_FOV_HALF)

            # deterrence by invested defence, a public good, replacing the
            # hard-coded "three soldiers" rule
            if self.defence_pays and in_view.any() and \
                    self.D[in_view].sum() >= self.d_deter:
                self.pvx[i] *= -1
                self.pvy[i] *= -1
                self.p_rev[i] = VF.REVERSAL_DURATION
                self.deters += 1
            elif c["LARGE_GROUP_THRESHOLD"] is not None and \
                    (dist < VF.PRED_VISION).sum() > c["LARGE_GROUP_THRESHOLD"]:
                a = self.rng.uniform(0, 2 * math.pi)
                self.pvx[i] = math.cos(a) * VF.PREDATOR_SPEED
                self.pvy[i] = math.sin(a) * VF.PREDATOR_SPEED
            elif self.p_rev[i] > 0:
                self.p_rev[i] -= 1
            else:
                cand = np.flatnonzero(in_view)
                if len(cand):
                    j = int(cand[0])
                    if dist[j] <= VF.PRED_CATCH:
                        p_esc = (self.D[j] / (self.D[j] + self.d_escape_half)
                                 if self.defence_pays else 0.0)
                        if self.rng.random() < p_esc:
                            self.D[j] *= 0.5     # defence is spent escaping
                            self.escapes += 1
                        else:
                            self.deaths.append((self.step_i, int(self.uid[j]),
                                                "eaten"))
                            keep = np.ones(self.n, bool)
                            keep[j] = False
                            self._keep(keep)
                            self.p_eat[i] = VF.EATING_DURATION
                            continue
                    else:
                        self.pvx[i] = bx[j] / dist[j] * VF.PREDATOR_SPEED
                        self.pvy[i] = by[j] / dist[j] * VF.PREDATOR_SPEED
                        self.px[i] += self.pvx[i]
                        self.py[i] += self.pvy[i]
                else:
                    self.px[i] += self.pvx[i]
                    self.py[i] += self.pvy[i]
                    self.pvx[i] += self.rng.uniform(-0.5, 0.5)
                    self.pvy[i] += self.rng.uniform(-0.5, 0.5)
                    s = max(math.hypot(self.pvx[i], self.pvy[i]), 1e-12)
                    self.pvx[i] = self.pvx[i] / s * VF.PREDATOR_SPEED
                    self.pvy[i] = self.pvy[i] / s * VF.PREDATOR_SPEED
            if not 0 <= self.px[i] <= VF.WIDTH:
                self.pvx[i] *= -1
            if not 0 <= self.py[i] <= VF.HEIGHT:
                self.pvy[i] *= -1
            self.px[i] = min(max(self.px[i], 0.0), VF.WIDTH)
            self.py[i] = min(max(self.py[i], 0.0), VF.HEIGHT)

    def _keep(self, mask):
        for a in ("x", "y", "vx", "vy", "sol", "E", "age", "uid", "par",
                  "kap_gene", "D", "R"):
            setattr(self, a, getattr(self, a)[mask])
        self.G = self.G[mask]

    def snapshot(self):
        s = VF.Fast.snapshot(self)
        s.update(kappa=self.kappa.copy(), defence=self.D.copy(),
                 repro_buffer=self.R.copy())
        return s
