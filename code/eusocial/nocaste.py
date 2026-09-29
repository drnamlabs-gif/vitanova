"""The model with castes removed entirely.

Locking the role at birth changed nothing, and only 6% of individuals ever
switched roles in the first place, so the natural next question is whether the
caste machinery is doing anything at all. Here it is deleted:

  the network loses its two role outputs and keeps four (separation, alignment,
    cohesion, avoidance), so the genome is 150 weights instead of 170;
  every agent uses the worker energy rule;
  the soldier interception behaviour is gone;
  the predator rule that reverses on seeing three soldiers can never fire,
    which the factorial experiment had already shown to be unnecessary.

Everything else is evo_cooperative_flock.py untouched.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fast"))
import vitafast as VF  # noqa: E402

N_OUT = 4
GENOME = VF.N_IN * VF.N_HID + VF.N_HID * N_OUT      # 150


class NoCaste(VF.Fast):
    def __init__(self, seed=2024, **kw):
        super().__init__(seed=seed, **kw)
        self.G = self.rng.standard_normal((self.n, GENOME))
        self.sol = np.zeros(self.n, bool)           # nobody is ever a soldier

    def _energy(self):
        c = self.cfg
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        close = (d < VF.PROXIMITY_THRESHOLD) & ~np.eye(n, dtype=bool)
        nw = close.sum(1)                            # everyone is a worker
        allee = np.where((nw > 1) & (nw < c["CRITICAL_GROUP_SIZE"]),
                         c["WORKER_TOGETHER_ENERGY"],
                         np.where(nw >= c["CRITICAL_GROUP_SIZE"],
                                  -c["WORKER_TOGETHER_ENERGY"], 0.0))
        self.E += c["WORKER_ENERGY_GAIN"] + allee - VF.ENERGY_COST_PER_STEP

    def _reproduce(self):
        c = self.cfg
        idx = np.flatnonzero((self.E > c["REPRODUCTION_ENERGY_THRESHOLD"])
                             & (self.n + 1 <= c["MAX_BOIDS"]))
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
        self.sol = np.concatenate([self.sol, np.zeros(k, bool)])
        self.G = np.vstack([self.G, g])
        self.E = np.concatenate([self.E, np.full(k, VF.INITIAL_ENERGY)])
        self.age = np.concatenate([self.age, np.zeros(k, np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.par = np.concatenate([self.par, self.uid[idx]])
        self.E[idx] -= VF.REPRODUCTION_COST
        self.x = np.clip(self.x, 0, VF.WIDTH)
        self.y = np.clip(self.y, 0, VF.HEIGHT)

    def _social_weights(self, mall, d):
        """How strongly each neighbour counts for alignment and cohesion.

        A hook, so a subclass can make adhesion selective without duplicating
        step().  The default is the original model: every neighbour inside
        ALIGNMENT_RADIUS counts equally, self included.
        """
        return mall.astype(float)

    def step(self):
        if self.n == 0:
            return
        n = self.n
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        d = np.sqrt(dx * dx + dy * dy)
        eye = np.eye(n, dtype=bool)
        near = (d <= VF.PERCEPTION_RADIUS) & ~eye
        cols = np.where(near, np.arange(n)[None, :], n)
        if n > 3:
            first3 = np.partition(cols, 2, axis=1)[:, :3]
        else:
            first3 = np.sort(cols, axis=1)[:, :3]
            if first3.shape[1] < 3:
                first3 = np.pad(first3, ((0, 0), (0, 3 - first3.shape[1])),
                                constant_values=n)
        first3 = np.sort(first3, axis=1)
        ok = first3 < n
        idx = np.where(ok, first3, 0)
        rows = np.arange(n)[:, None]
        rel = np.zeros((n, 6))
        rel[:, 0::2] = np.where(ok, dx[rows, idx] / VF.PERCEPTION_RADIUS, 0.0)
        rel[:, 1::2] = np.where(ok, dy[rows, idx] / VF.PERCEPTION_RADIUS, 0.0)
        if len(self.px):
            qx = self.px[None, :] - self.x[:, None]
            qy = self.py[None, :] - self.y[:, None]
            q = np.sqrt(qx * qx + qy * qy)
            qm = np.where(q < VF.PERCEPTION_RADIUS, q, np.inf)
            k = np.argmin(qm, axis=1)
            has = np.isfinite(qm[np.arange(n), k])
            pdx = np.where(has, qx[np.arange(n), k], 0.0)
            pdy = np.where(has, qy[np.arange(n), k], 0.0)
            pdist = np.where(has, q[np.arange(n), k], 0.0)
            in_pd = np.where(has, pdist / VF.PERCEPTION_RADIUS, 0.0)
            in_pa = np.where(has, np.arctan2(pdy, pdx) / np.pi, 0.0)
        else:
            has = np.zeros(n, bool)
            pdx = pdy = pdist = in_pd = in_pa = np.zeros(n)
        density = near.sum(1) / 3.0 * 2 - 1
        X = np.column_stack([rel, in_pd, in_pa, self.vx / VF.SPEED,
                             self.vy / VF.SPEED, density])
        w1 = self.G[:, :VF.N_IN * VF.N_HID].reshape(-1, VF.N_IN, VF.N_HID)
        w2 = self.G[:, VF.N_IN * VF.N_HID:].reshape(-1, VF.N_HID, N_OUT)
        h = np.maximum(np.einsum("ni,nih->nh", X, w1), 0.0)
        out = np.tanh(np.einsum("nh,nho->no", h, w2))
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        msep = (d < VF.SEPARATION_RADIUS) & ~eye
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)
        mall = d < VF.ALIGNMENT_RADIUS
        f = self._social_weights(mall, d)
        cnt = np.maximum(f.sum(1), 1)
        ali_x, ali_y = f @ self.vx / cnt, f @ self.vy / cnt
        coh_x = f @ self.x / cnt - self.x
        coh_y = f @ self.y / cnt - self.y
        esc = np.where(has & (t_avo > 0) & (pdist > 0),
                       t_avo / np.maximum(pdist, 1e-12), 0.0)
        self.vx += sep_x * t_sep + ali_x * t_ali + coh_x * t_coh - pdx * esc
        self.vy += sep_y * t_sep + ali_y * t_ali + coh_y * t_coh - pdy * esc
        mag = np.hypot(self.vx, self.vy)
        nz = mag != 0
        self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * VF.SPEED, 0.0)
        self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * VF.SPEED, 0.0)
        self.x += self.vx
        self.y += self.vy
        ox = (self.x < 0) | (self.x > VF.WIDTH)
        oy = (self.y < 0) | (self.y > VF.HEIGHT)
        self.vx[ox] *= -1
        self.vy[oy] *= -1
        self.x = np.clip(self.x, 0, VF.WIDTH)
        self.y = np.clip(self.y, 0, VF.HEIGHT)

        self.age += 1
        self._energy()
        self._reproduce()
        self._predators()
        dead = self.age >= VF.LIFESPAN
        if dead.any():
            for u in self.uid[dead]:
                self.deaths.append((self.step_i, int(u), "age"))
            self._keep(~dead)
        self.step_i += 1
