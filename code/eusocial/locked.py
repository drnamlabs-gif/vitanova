"""Is within-lifetime role switching necessary for the transition?

The model recomputes each agent's caste every step from its own network, and the
published description calls this "adaptive plastic division of labor". Measured,
it is not plastic: 94% of individuals spend either almost all or almost none of
their life as a soldier, and only 6% ever switch.

Here the role is read from an agent's network on its first step and then fixed
for life. Nothing else changes.

The locking has to happen *inside* the step, before the caste is used. A first
attempt wrapped the original step and restored the role afterwards, which left
the soldier behaviour and the energy accounting of that step running on the
freshly recomputed role -- and produced results identical to the plastic arm to
three significant figures, which is how the mistake was caught.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fast"))
import vitafast as VF  # noqa: E402


class Locked(VF.Fast):
    def __init__(self, seed=2024, lock_role=True, **kw):
        super().__init__(seed=seed, **kw)
        self.lock_role = bool(lock_role)
        self.locked = np.zeros(self.n, bool)
        self.role_flips = 0

    def _keep(self, mask):
        self.locked = self.locked[mask]
        VF.Fast._keep(self, mask)

    def step(self):
        if self.n == 0:
            return
        c = self.cfg
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
        w2 = self.G[:, VF.N_IN * VF.N_HID:].reshape(-1, VF.N_HID, VF.N_OUT)
        h = np.maximum(np.einsum("ni,nih->nh", X, w1), 0.0)
        out = np.tanh(np.einsum("nh,nho->no", h, w2))
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        msep = (d < VF.SEPARATION_RADIUS) & ~eye
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)
        mall = d < VF.ALIGNMENT_RADIUS
        f = mall.astype(float)
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

        # ---- the caste, set here and used by everything below --------------
        proposed = out[:, -2] > out[:, -1]
        if self.lock_role:
            keep = self.locked[:n] if len(self.locked) >= n else \
                np.zeros(n, bool)
            new_sol = np.where(keep, self.sol, proposed)
            self.locked = np.ones(n, bool)
        else:
            new_sol = proposed
        self.role_flips += int((new_sol != self.sol).sum())
        self.sol = new_sol

        self._soldiers()
        self.age += 1
        self._energy()
        before = self.n
        self._reproduce()
        if self.n > before:
            self.locked = np.concatenate(
                [self.locked, np.zeros(self.n - before, bool)])
        self._predators()
        dead = self.age >= VF.LIFESPAN
        if dead.any():
            for u in self.uid[dead]:
                self.deaths.append((self.step_i, int(u), "age"))
            self._keep(~dead)
        self.step_i += 1
