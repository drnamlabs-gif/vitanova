"""VitaNova with the controller written as what it actually is.

The network in the original model, and the minimal NEAT genome that reproduced
its behaviour, both reduce to the same object: one squashed linear map from
sensors to steering tendencies,

    T_j = tanh( sum_i  w_ji * x_i ),

with j over {separation, alignment, cohesion, avoidance, soldier, worker} and i
over the eleven sensory inputs. The genome is therefore a 6 x 11 matrix of 66
coefficients and nothing else: no hidden layer, no activation choice, no
topology. Calling it a neural network adds vocabulary, not content.

This matters for two reasons. The controller becomes readable -- each
coefficient says how strongly one sensory quantity pushes one steering rule --
and the sensory basis becomes ablatable, so one can ask which inputs the
phenomenon actually needs.

Everything else is identical to evo_cooperative_flock.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fast"))
import vitafast as VF  # noqa: E402

N_IN, N_OUT = 11, 6
GENOME = N_IN * N_OUT          # 66, against 170 for the 11-10-6 network

INPUT_NAMES = ["nb1_dx", "nb1_dy", "nb2_dx", "nb2_dy", "nb3_dx", "nb3_dy",
               "pred_dist", "pred_angle", "own_vx", "own_vy", "density"]
OUTPUT_NAMES = ["separation", "alignment", "cohesion", "avoidance",
                "soldier", "worker"]


class LinearVita(VF.Fast):
    """Fast, with the hidden layer removed. Nothing else changes."""

    def __init__(self, seed=2024, mask=None, **kw):
        super().__init__(seed=seed, **kw)
        n = self.n
        self.G = self.rng.standard_normal((n, GENOME))
        # mask selects which sensory inputs are available, for ablation
        self.mask = (np.ones(N_IN, bool) if mask is None
                     else np.asarray(mask, bool))

    def step(self):
        if self.n == 0:
            return
        VF.Fast.step(self)

    def _outputs(self, X):
        W = self.G.reshape(-1, N_OUT, N_IN)
        return np.tanh(np.einsum("noi,ni->no", W, X * self.mask))


def _patch():
    """Replace the two-layer evaluation inside Fast.step with the linear map."""
    src = VF.Fast.step
    import types
    def step(self):
        if self.n == 0:
            return
        c = self.cfg
        n = self.n
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        d = np.sqrt(dx * dx + dy * dy)
        self_mask = np.eye(n, dtype=bool)
        near = (d <= VF.PERCEPTION_RADIUS) & ~self_mask
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
            qmask = np.where(q < VF.PERCEPTION_RADIUS, q, np.inf)
            k = np.argmin(qmask, axis=1)
            has = np.isfinite(qmask[np.arange(n), k])
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
        out = self._outputs(X)
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        msep = (d < VF.SEPARATION_RADIUS) & ~self_mask
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)
        mall = d < VF.ALIGNMENT_RADIUS
        f = mall.astype(float)
        cnt = mall.sum(1).astype(float)
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
        self.sol = out[:, -2] > out[:, -1]
        self._soldiers()
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
    LinearVita.step = step


_patch()
