# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step 1: classical Reynolds flocking, no evolution, as a reference.

Before putting an evolved controller in charge of a boid it is worth knowing
what the substrate does on its own. The states are characterised the way the
collective-behaviour literature does it, with two order parameters:

    polarisation      P = |mean unit velocity|      0 disordered, 1 aligned
    angular momentum  M = |mean of r_hat x v_hat|   0 no rotation, 1 milling

Couzin et al. (2002) show that a Reynolds-type model moves between a disordered
swarm (P low, M low), a rotating torus (P low, M high) and a polarised school
(P high, M low) as the zones of the three rules change. Reproducing that
structure is the test that the flocking substrate is behaving normally --- and
the torus is the state the ring structures in this project resemble.
"""
from __future__ import annotations

import numpy as np

W, H = 800.0, 600.0


class Flock:
    def __init__(self, n=100, seed=0, w_sep=1.0, w_ali=1.0, w_coh=1.0,
                 r_sep=10.0, r_ali=25.0, r_coh=50.0, speed=1.0,
                 noise=0.05, wrap=True, turn=0.35):
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.uniform(0, W, n)
        self.y = self.rng.uniform(0, H, n)
        a = self.rng.uniform(0, 2 * np.pi, n)
        self.vx, self.vy = np.cos(a), np.sin(a)
        self.p = dict(w_sep=w_sep, w_ali=w_ali, w_coh=w_coh, r_sep=r_sep,
                      r_ali=r_ali, r_coh=r_coh, speed=speed, noise=noise,
                      wrap=wrap, turn=turn)

    @property
    def n(self):
        return len(self.x)

    def _delta(self):
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        if self.p["wrap"]:
            dx -= W * np.round(dx / W)
            dy -= H * np.round(dy / H)
        return dx, dy, np.sqrt(dx * dx + dy * dy)

    def step(self):
        p = self.p
        n = self.n
        dx, dy, d = self._delta()
        eye = np.eye(n, dtype=bool)

        msep = (d < p["r_sep"]) & ~eye
        w = np.where(msep, 1.0 / np.maximum(d, 1e-6), 0.0)
        sx, sy = -(dx * w).sum(1), -(dy * w).sum(1)

        mali = (d < p["r_ali"]) & ~eye
        f = mali.astype(float)
        c = np.maximum(f.sum(1), 1)
        ax, ay = f @ self.vx / c - self.vx, f @ self.vy / c - self.vy

        mcoh = (d < p["r_coh"]) & ~eye
        g = mcoh.astype(float)
        c2 = np.maximum(g.sum(1), 1)
        cx, cy = (dx * g).sum(1) / c2, (dy * g).sum(1) / c2

        norm = lambda a, b: (a / np.maximum(np.hypot(a, b), 1e-9),
                             b / np.maximum(np.hypot(a, b), 1e-9))
        sx, sy = norm(sx, sy)
        ax, ay = norm(ax, ay)
        cx, cy = norm(cx, cy)

        ux = p["w_sep"] * sx + p["w_ali"] * ax + p["w_coh"] * cx
        uy = p["w_sep"] * sy + p["w_ali"] * ay + p["w_coh"] * cy
        ux, uy = norm(ux, uy)
        if p["noise"]:
            th = self.rng.normal(0, p["noise"], n)
            ux, uy = (ux * np.cos(th) - uy * np.sin(th),
                      ux * np.sin(th) + uy * np.cos(th))

        self.vx += p["turn"] * ux
        self.vy += p["turn"] * uy
        sp = np.maximum(np.hypot(self.vx, self.vy), 1e-9)
        self.vx, self.vy = self.vx / sp * p["speed"], self.vy / sp * p["speed"]
        self.x += self.vx
        self.y += self.vy
        if p["wrap"]:
            self.x %= W
            self.y %= H
        else:
            ox, oy = (self.x < 0) | (self.x > W), (self.y < 0) | (self.y > H)
            self.vx[ox] *= -1
            self.vy[oy] *= -1
            self.x = np.clip(self.x, 0, W)
            self.y = np.clip(self.y, 0, H)

    # ------------------------------------------------------------- observables
    def clusters(self, link=None, min_size=5):
        """Single-linkage groups. Order parameters are only meaningful within
        one group: angular momentum about the centroid of a fragmented
        population measures nothing."""
        link = link if link is not None else self.p["r_coh"] * 0.6
        dx, dy, d = self._delta()
        adj = d <= link
        n = self.n
        seen = np.zeros(n, bool)
        out = []
        for i in range(n):
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

    def group_order(self, idx=None):
        """Polarisation and angular momentum of one group, about its own
        centroid and with positions unwrapped relative to it."""
        idx = np.arange(self.n) if idx is None else idx
        if len(idx) < 3:
            return 0.0, 0.0
        vx, vy = self.vx[idx], self.vy[idx]
        s = np.maximum(np.hypot(vx, vy), 1e-9)
        ux, uy = vx / s, vy / s
        P = float(np.hypot(ux.mean(), uy.mean()))
        x, y = self.x[idx].copy(), self.y[idx].copy()
        if self.p["wrap"]:
            x = x[0] + ((x - x[0] + W / 2) % W - W / 2)
            y = y[0] + ((y - y[0] + H / 2) % H - H / 2)
        rx, ry = x - x.mean(), y - y.mean()
        r = np.maximum(np.hypot(rx, ry), 1e-9)
        M = float(abs(np.mean((rx * uy - ry * ux) / r)))
        return P, M

    def largest_group_order(self):
        cs = self.clusters()
        if not cs:
            return 0.0, 0.0, 0
        P, M = self.group_order(cs[0])
        return P, M, len(cs[0])

    def order_parameters(self):
        vx, vy = self.vx, self.vy
        s = np.maximum(np.hypot(vx, vy), 1e-9)
        ux, uy = vx / s, vy / s
        P = float(np.hypot(ux.mean(), uy.mean()))
        cx, cy = self.x.mean(), self.y.mean()
        rx, ry = self.x - cx, self.y - cy
        r = np.maximum(np.hypot(rx, ry), 1e-9)
        M = float(abs(np.mean((rx * uy - ry * ux) / r)))
        return P, M

    def nnd(self):
        _, _, d = self._delta()
        np.fill_diagonal(d, np.inf)
        return float(d.min(1).mean())

    def clark_evans(self):
        return self.nnd() / (0.5 / np.sqrt(self.n / (W * H)))

    def snapshot(self):
        return dict(x=self.x.copy(), y=self.y.copy(),
                    vx=self.vx.copy(), vy=self.vy.copy())
