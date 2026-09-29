"""The flocking substrate of evo_cooperative_flock.py, reproduced exactly.

Every constant and every formula is taken from that file so that the only thing
changed in this branch is the controller: a NEAT network with evolving topology
in place of the fixed 11-10-6 feed-forward network. Predators, energy, castes
and reproduction are absent at this step and are added one at a time later.

Points that are easy to get wrong and are preserved here:
  * all four radii are BOID_SIZE * 5 = 25, not separate values;
  * boundaries reflect, they do not wrap;
  * the separation, alignment and cohesion vectors are **not** normalised
    individually --- the raw sums are weighted by the tendencies and only the
    resulting velocity is renormalised to SPEED;
  * alignment and cohesion include the focal boid in their means, separation
    excludes it;
  * the three sensed neighbours are the first three **in list order** within
    the radius, not the three nearest;
  * there is no turning-rate limit and no steering noise.
"""
from __future__ import annotations

import numpy as np

BOID_SIZE = 5.0
WIDTH, HEIGHT = 160 * BOID_SIZE, 120 * BOID_SIZE          # 800 x 600
RADIUS = BOID_SIZE * 5                                     # 25, all four zones
SPEED = 1.0
BOID_COUNT = 100
N_IN, N_OUT = 9, 3        # VitaNova's 11 inputs minus the two predator ones


class VitaFlock:
    """controller(X) -> (n, 3) tendencies, or a constant triple."""

    def __init__(self, controller, n=BOID_COUNT, seed=0):
        self.ctl = controller
        self.const = None if callable(controller) else np.asarray(controller, float)
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(0, WIDTH, n)
        self.y = rng.uniform(0, HEIGHT, n)
        self.vx = rng.uniform(-1, 1, n)
        self.vy = rng.uniform(-1, 1, n)

    @property
    def n(self):
        return len(self.x)

    def _delta(self):
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        return dx, dy, np.sqrt(dx * dx + dy * dy)

    def sense(self, dx, dy, d, eye):
        n = self.n
        near = (d <= RADIUS) & ~eye                    # find_neighbors_within_radius
        cols = np.where(near, np.arange(n)[None, :], n)
        if n > 3:
            first3 = np.partition(cols, 2, axis=1)[:, :3]
        else:
            first3 = np.sort(cols, axis=1)[:, :3]
        first3 = np.sort(first3, axis=1)               # first three in list order
        ok = first3 < n
        idx = np.where(ok, first3, 0)
        rows = np.arange(n)[:, None]
        rel = np.zeros((n, 6))
        rel[:, 0::2] = np.where(ok, dx[rows, idx] / RADIUS, 0.0)
        rel[:, 1::2] = np.where(ok, dy[rows, idx] / RADIUS, 0.0)
        density = near.sum(1) / 3.0 * 2 - 1            # len(close)/max_density*2-1
        return np.column_stack([rel, self.vx / SPEED, self.vy / SPEED, density])

    def step(self):
        n = self.n
        dx, dy, d = self._delta()
        eye = np.eye(n, dtype=bool)
        X = self.sense(dx, dy, d, eye)
        T = (np.repeat(self.const[None, :], n, 0) if self.const is not None
             else self.ctl(X))

        msep = (d < RADIUS) & ~eye                     # separation excludes self
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)

        mall = d < RADIUS                              # alignment/cohesion include self
        f = mall.astype(float)
        cnt = np.maximum(f.sum(1), 1)
        ali_x, ali_y = f @ self.vx / cnt, f @ self.vy / cnt
        coh_x = f @ self.x / cnt - self.x
        coh_y = f @ self.y / cnt - self.y

        self.vx += sep_x * T[:, 0] + ali_x * T[:, 1] + coh_x * T[:, 2]
        self.vy += sep_y * T[:, 0] + ali_y * T[:, 1] + coh_y * T[:, 2]
        mag = np.hypot(self.vx, self.vy)
        nz = mag != 0
        self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * SPEED, 0.0)
        self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * SPEED, 0.0)
        self.x += self.vx
        self.y += self.vy
        ox = (self.x < 0) | (self.x > WIDTH)           # reflecting boundaries
        oy = (self.y < 0) | (self.y > HEIGHT)
        self.vx[ox] *= -1
        self.vy[oy] *= -1
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)

    # ------------------------------------------------------------- observables
    def clusters(self, link=RADIUS, min_size=4):
        _, _, d = self._delta()
        adj = d <= link
        seen = np.zeros(self.n, bool)
        out = []
        for i in range(self.n):
            if seen[i]:
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                j = stack.pop()
                comp.append(j)
                nxt = np.flatnonzero(adj[j] & ~seen)
                seen[nxt] = True
                stack.extend(nxt.tolist())
            if len(comp) >= min_size:
                out.append(np.array(comp))
        return sorted(out, key=len, reverse=True)

    def group_order(self, idx):
        if len(idx) < 3:
            return 0.0, 0.0
        vx, vy = self.vx[idx], self.vy[idx]
        s = np.maximum(np.hypot(vx, vy), 1e-9)
        ux, uy = vx / s, vy / s
        P = float(np.hypot(ux.mean(), uy.mean()))
        rx = self.x[idx] - self.x[idx].mean()
        ry = self.y[idx] - self.y[idx].mean()
        r = np.maximum(np.hypot(rx, ry), 1e-9)
        M = float(abs(np.mean((rx * uy - ry * ux) / r)))
        return P, M

    def measure(self):
        _, _, d = self._delta()
        np.fill_diagonal(d, np.inf)
        nnd = float(d.min(1).mean())
        cs = self.clusters()
        if not cs:
            return dict(P=0.0, M=0.0, group=0, n_clusters=0, nnd=nnd)
        P, M = self.group_order(cs[0])
        return dict(P=P, M=M, group=len(cs[0]), n_clusters=len(cs), nnd=nnd)
