# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 2: a NEAT network per boid, whose outputs are the steering tendencies.

The network does not move the boid. It emits one weight per Reynolds rule, and
those weights modulate the classical separation, alignment and cohesion vectors:

    dv = T_sep * Separation + T_ali * Alignment + T_coh * Cohesion

with each T in [-1, 1] through tanh, so a rule can also be inverted. This is the
same interface the fixed-topology controller uses in the main model; what NEAT
adds is that the mapping from context to tendencies can grow structure.

Inputs (9): the relative positions of the three nearest neighbours, the boid's
own velocity, the local neighbour count, and a constant reference.
"""
from __future__ import annotations

import numpy as np

from vecnet import VectorNetwork

N_IN, N_OUT = 9, 3


class NeatFlock:
    def __init__(self, network: VectorNetwork, n=40, seed=0, speed=1.0,
                 r_sep=15.0, r_ali=25.0, r_coh=60.0, perception=60.0,
                 turn=0.35, noise=0.03, wrap=True, width=None, height=None):
        # Arena size matters: at 40 boids in 800x600 the expected
        # nearest-neighbour distance is about 55 px against a perception radius
        # of 60, so agents barely meet and groups never grow. Default to the
        # arena that matches the density used for the fixed-rule sweep.
        self.W = float(width) if width else 462.0
        self.H = float(height) if height else 346.0
        self.net = network
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.uniform(0, self.W, n)
        self.y = self.rng.uniform(0, self.H, n)
        a = self.rng.uniform(0, 2 * np.pi, n)
        self.vx, self.vy = np.cos(a), np.sin(a)
        self.r_sep, self.r_ali, self.r_coh = r_sep, r_ali, r_coh
        self.perception, self.speed = perception, speed
        self.turn, self.noise, self.wrap = turn, noise, wrap

    @property
    def n(self):
        return len(self.x)

    def _delta(self):
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        if self.wrap:
            dx -= self.W * np.round(dx / self.W)
            dy -= self.H * np.round(dy / self.H)
        return dx, dy, np.sqrt(dx * dx + dy * dy)

    def sense(self, dx, dy, d, eye):
        n = self.n
        near = (d <= self.perception) & ~eye
        big = np.where(near, d, np.inf)
        k = min(3, max(1, n - 1))
        idx = np.argpartition(big, k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        ok = np.isfinite(big[rows, idx])
        rel = np.zeros((n, 6))
        for j in range(min(3, k)):
            col = idx[:, j]
            rel[:, 2 * j] = np.where(ok[:, j], dx[rows[:, 0], col] / self.perception, 0)
            rel[:, 2 * j + 1] = np.where(ok[:, j], dy[rows[:, 0], col] / self.perception, 0)
        dens = np.tanh(near.sum(1) / 5.0)
        return np.column_stack([rel, self.vx / self.speed, self.vy / self.speed,
                                dens]), near

    def step(self):
        n = self.n
        dx, dy, d = self._delta()
        eye = np.eye(n, dtype=bool)
        X, near = self.sense(dx, dy, d, eye)
        T = self.net(X)                      # (n, 3) tendencies in [-1, 1]

        msep = (d < self.r_sep) & ~eye
        w = np.where(msep, 1.0 / np.maximum(d, 1e-6), 0.0)
        sx, sy = -(dx * w).sum(1), -(dy * w).sum(1)
        mali = (d < self.r_ali) & ~eye
        f = mali.astype(float)
        c = np.maximum(f.sum(1), 1)
        ax, ay = f @ self.vx / c - self.vx, f @ self.vy / c - self.vy
        mcoh = (d < self.r_coh) & ~eye
        g = mcoh.astype(float)
        c2 = np.maximum(g.sum(1), 1)
        cx, cy = (dx * g).sum(1) / c2, (dy * g).sum(1) / c2

        nrm = lambda a, b: (a / np.maximum(np.hypot(a, b), 1e-9),
                            b / np.maximum(np.hypot(a, b), 1e-9))
        sx, sy = nrm(sx, sy)
        ax, ay = nrm(ax, ay)
        cx, cy = nrm(cx, cy)

        ux = T[:, 0] * sx + T[:, 1] * ax + T[:, 2] * cx
        uy = T[:, 0] * sy + T[:, 1] * ay + T[:, 2] * cy
        ux, uy = nrm(ux, uy)
        if self.noise:
            th = self.rng.normal(0, self.noise, n)
            ux, uy = ux * np.cos(th) - uy * np.sin(th), ux * np.sin(th) + uy * np.cos(th)

        self.vx += self.turn * ux
        self.vy += self.turn * uy
        sp = np.maximum(np.hypot(self.vx, self.vy), 1e-9)
        self.vx, self.vy = self.vx / sp * self.speed, self.vy / sp * self.speed
        self.x = (self.x + self.vx) % self.W if self.wrap else self.x + self.vx
        self.y = (self.y + self.vy) % self.H if self.wrap else self.y + self.vy
        self.last_T = T

    # ------------------------------------------------------------- observables
    def clusters(self, link=None, min_size=5):
        link = link if link is not None else self.r_coh * 0.6
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
        x, y = self.x[idx].copy(), self.y[idx].copy()
        if self.wrap:
            x = x[0] + ((x - x[0] + self.W / 2) % self.W - self.W / 2)
            y = y[0] + ((y - y[0] + self.H / 2) % self.H - self.H / 2)
        rx, ry = x - x.mean(), y - y.mean()
        r = np.maximum(np.hypot(rx, ry), 1e-9)
        M = float(abs(np.mean((rx * uy - ry * ux) / r)))
        return P, M

    def nnd(self):
        _, _, d = self._delta()
        np.fill_diagonal(d, np.inf)
        return float(d.min(1).mean())

    def measure(self):
        cs = self.clusters()
        if not cs:
            return dict(P=0.0, M=0.0, group=0, n_clusters=0, nnd=self.nnd())
        P, M = self.group_order(cs[0])
        return dict(P=P, M=M, group=len(cs[0]), n_clusters=len(cs),
                    nnd=self.nnd())
