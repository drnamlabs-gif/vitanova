# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Vectorised port of evo_cooperative_flock.py -- same model, ~100x faster.

The model is unchanged: same constants, same network shape, same Allee term,
same castes, same predator rules. Only the implementation differs. The original
runs six separate O(N^2) passes per step in pure Python (perceive, separation,
alignment, cohesion, count_nearby_workers, near_worker) plus three more per
predator; all of them become array operations here.

Three fidelity notes, since bit-identity is not the goal but faithfulness is:

1. **Update order.** The original iterates `for boid in boids: boid.update(...)`,
   so a boid sees neighbours that have already moved this step (Gauss-Seidel).
   This port updates every boid from the same snapshot (Jacobi). That is the one
   real semantic difference and it is why `validate.py` compares distributions
   rather than trajectories.
2. **List-order quirks are preserved.** `perceive` takes the first three
   neighbours *in list order*, not the three nearest, and a predator targets the
   first boid in list order within range and field of view, not the nearest.
   Both are reproduced here.
3. **Self-inclusion is preserved.** `alignment` and `cohesion` include the focal
   boid in their means; `separation` excludes it.
"""

import numpy as np

# ---- constants, copied from evo_cooperative_flock.py -----------------------
BOID_SIZE = 5.0
PREDATOR_SIZE = 2 * BOID_SIZE
WIDTH, HEIGHT = 160 * BOID_SIZE, 120 * BOID_SIZE
SPEED = 1.0
PREDATOR_SPEED = 1.2 * SPEED
SEPARATION_RADIUS = ALIGNMENT_RADIUS = COHESION_RADIUS = PERCEPTION_RADIUS = BOID_SIZE * 5

BOID_COUNT, PREDATOR_COUNT = 100, 10
INITIAL_ENERGY = 50.0
REPRODUCTION_ENERGY_THRESHOLD = 200.0
REPRODUCTION_COST = 100.0
MUTATION_RATE, MUTATION_STRENGTH = 0.1, 0.1
REPRODUCTION_DISTANCE = BOID_SIZE * 2

WORKER_ENERGY_GAIN = 1.1
SOLDIER_ENERGY_GAIN = 0.7
NEAR_WORKER_GAIN = 0.6
ENERGY_COST_PER_STEP = 1.0
WORKER_TOGETHER_ENERGY = 0.2
CRITICAL_GROUP_SIZE = 20
PROXIMITY_THRESHOLD = BOID_SIZE * 3
SOLDIER_PROTECTION_RADIUS = BOID_SIZE * 5
FLOCK_THRESHOLD = BOID_SIZE * 5
FLOCK_SIZE_THRESHOLD = 4
LIFESPAN = 1000

PRED_VISION = PREDATOR_SIZE * 5
PRED_CATCH = PREDATOR_SIZE
PRED_FOV_HALF = np.pi / 2
REVERSAL_DURATION = 30
EATING_DURATION = 60
LARGE_GROUP_THRESHOLD = 20
SOLDIER_REVERSAL_COUNT = 3

N_IN, N_HID, N_OUT = 11, 10, 6
GENOME = N_IN * N_HID + N_HID * N_OUT


class Fast:
    def __init__(self, seed=2024, **overrides):
        self.cfg = dict(
            WORKER_TOGETHER_ENERGY=WORKER_TOGETHER_ENERGY,
            CRITICAL_GROUP_SIZE=CRITICAL_GROUP_SIZE,
            NEAR_WORKER_GAIN=NEAR_WORKER_GAIN,
            SOLDIER_REVERSAL_COUNT=SOLDIER_REVERSAL_COUNT,
            LARGE_GROUP_THRESHOLD=LARGE_GROUP_THRESHOLD,
            PREDATOR_COUNT=PREDATOR_COUNT,
            MUTATION_RATE=MUTATION_RATE,
            MUTATION_STRENGTH=MUTATION_STRENGTH,
            REPRODUCTION_ENERGY_THRESHOLD=REPRODUCTION_ENERGY_THRESHOLD,
            WORKER_ENERGY_GAIN=WORKER_ENERGY_GAIN,
            SOLDIER_ENERGY_GAIN=SOLDIER_ENERGY_GAIN,
            MAX_BOIDS=5000,
        )
        for k, v in overrides.items():
            if k not in self.cfg:
                raise KeyError(f"unknown parameter {k}")
            self.cfg[k] = v

        self.rng = np.random.default_rng(seed)
        n = BOID_COUNT
        self.x = self.rng.uniform(0, WIDTH, n)
        self.y = self.rng.uniform(0, HEIGHT, n)
        self.vx = self.rng.uniform(-1, 1, n)
        self.vy = self.rng.uniform(-1, 1, n)
        self.sol = self.rng.random(n) < 0.5
        self.G = self.rng.standard_normal((n, GENOME))
        self.E = np.full(n, INITIAL_ENERGY)
        self.age = np.zeros(n, np.int64)
        self.uid = np.arange(n, dtype=np.int64)
        self.par = np.full(n, -1, np.int64)
        self._next = n

        p = int(self.cfg["PREDATOR_COUNT"])
        self.px = self.rng.uniform(0, WIDTH, p)
        self.py = self.rng.uniform(0, HEIGHT, p)
        self.pvx = self.rng.uniform(-1, 1, p)
        self.pvy = self.rng.uniform(-1, 1, p)
        self.p_eat = np.zeros(p, np.int64)
        self.p_rev = np.full(p, 20, np.int64)

        self.step_i = 0
        self.births, self.deaths = [], []

    @property
    def n(self):
        return len(self.x)

    # ------------------------------------------------------------------ step
    def step(self):
        if self.n == 0:
            return
        c = self.cfg
        n = self.n
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        d = np.sqrt(dx * dx + dy * dy)
        self_mask = np.eye(n, dtype=bool)

        # ---- perceive: first three neighbours in LIST order, radius inclusive
        near = (d <= PERCEPTION_RADIUS) & ~self_mask
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
        rel[:, 0::2] = np.where(ok, dx[rows, idx] / PERCEPTION_RADIUS, 0.0)
        rel[:, 1::2] = np.where(ok, dy[rows, idx] / PERCEPTION_RADIUS, 0.0)

        # ---- nearest visible predator: distance and angle, normalised
        if len(self.px):
            qx = self.px[None, :] - self.x[:, None]
            qy = self.py[None, :] - self.y[:, None]
            q = np.sqrt(qx * qx + qy * qy)
            vis = q < PERCEPTION_RADIUS
            qmask = np.where(vis, q, np.inf)
            k = np.argmin(qmask, axis=1)
            has = np.isfinite(qmask[np.arange(n), k])
            pdx = np.where(has, qx[np.arange(n), k], 0.0)
            pdy = np.where(has, qy[np.arange(n), k], 0.0)
            pdist = np.where(has, q[np.arange(n), k], 0.0)
            in_pd = np.where(has, pdist / PERCEPTION_RADIUS, 0.0)
            in_pa = np.where(has, np.arctan2(pdy, pdx) / np.pi, 0.0)
        else:
            has = np.zeros(n, bool)
            pdx = pdy = pdist = in_pd = in_pa = np.zeros(n)

        density = near.sum(1) / 3.0 * 2 - 1
        X = np.column_stack([rel, in_pd, in_pa, self.vx / SPEED,
                             self.vy / SPEED, density])

        w1 = self.G[:, :N_IN * N_HID].reshape(-1, N_IN, N_HID)
        w2 = self.G[:, N_IN * N_HID:].reshape(-1, N_HID, N_OUT)
        h = np.maximum(np.einsum("ni,nih->nh", X, w1), 0.0)
        out = np.tanh(np.einsum("nh,nho->no", h, w2))
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        # ---- steering, matching the originals exactly
        msep = (d < SEPARATION_RADIUS) & ~self_mask          # excludes self
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)

        mall = d < ALIGNMENT_RADIUS                          # includes self
        cnt = mall.sum(1).astype(float)
        f = mall.astype(float)
        ali_x, ali_y = f @ self.vx / cnt, f @ self.vy / cnt
        coh_x = f @ self.x / cnt - self.x
        coh_y = f @ self.y / cnt - self.y

        esc = np.where(has & (t_avo > 0) & (pdist > 0), t_avo / np.maximum(pdist, 1e-12), 0.0)
        esc_x, esc_y = -pdx * esc, -pdy * esc

        self.vx += sep_x * t_sep + ali_x * t_ali + coh_x * t_coh + esc_x
        self.vy += sep_y * t_sep + ali_y * t_ali + coh_y * t_coh + esc_y
        mag = np.hypot(self.vx, self.vy)
        nz = mag != 0
        self.vx = np.where(nz, self.vx / np.where(nz, mag, 1) * SPEED, 0.0)
        self.vy = np.where(nz, self.vy / np.where(nz, mag, 1) * SPEED, 0.0)
        self.x += self.vx
        self.y += self.vy
        ox = (self.x < 0) | (self.x > WIDTH)
        oy = (self.y < 0) | (self.y > HEIGHT)
        self.vx[ox] *= -1
        self.vy[oy] *= -1
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)

        # ---- role switch, then soldier behaviour
        self.sol = out[:, -2] > out[:, -1]
        self._soldiers()

        self.age += 1
        self._energy()
        self._reproduce()
        self._predators()

        dead = self.age >= LIFESPAN
        if dead.any():
            for u in self.uid[dead]:
                self.deaths.append((self.step_i, int(u), "age"))
            self._keep(~dead)
        self.step_i += 1

    # ------------------------------------------------------------- behaviours
    def _soldiers(self):
        s = np.flatnonzero(self.sol)
        if not len(s):
            return
        mv_x = self.rng.uniform(-1, 1, len(s))
        mv_y = self.rng.uniform(-1, 1, len(s))

        wk = np.flatnonzero(~self.sol)
        if len(wk):
            wd = np.hypot(self.x[wk][None, :] - self.x[s][:, None],
                          self.y[wk][None, :] - self.y[s][:, None])
            j = wk[np.argmin(wd, axis=1)]
            wx, wy = self.x[j] - self.x[s], self.y[j] - self.y[s]
            wn = np.hypot(wx, wy)
            wx, wy = np.divide(wx, wn, out=np.zeros_like(wx), where=wn != 0), \
                     np.divide(wy, wn, out=np.zeros_like(wy), where=wn != 0)
        else:
            wx = wy = None

        if len(self.px):
            pd = np.hypot(self.px[None, :] - self.x[s][:, None],
                          self.py[None, :] - self.y[s][:, None])
            k = np.argmin(pd, axis=1)
            near_pred = pd[np.arange(len(s)), k] < SOLDIER_PROTECTION_RADIUS
            ppx, ppy = self.px[k] - self.x[s], self.py[k] - self.y[s]
            pn = np.hypot(ppx, ppy)
            ppx, ppy = np.divide(ppx, pn, out=np.zeros_like(ppx), where=pn != 0), \
                       np.divide(ppy, pn, out=np.zeros_like(ppy), where=pn != 0)
        else:
            near_pred = np.zeros(len(s), bool)
            ppx = ppy = np.zeros(len(s))

        if wx is not None:
            inter = near_pred
            mv_x = np.where(inter, (wx - ppx) / 2, np.where(~near_pred, wx, mv_x))
            mv_y = np.where(inter, (wy - ppy) / 2, np.where(~near_pred, wy, mv_y))

        norm = np.hypot(mv_x, mv_y)
        good = norm != 0
        self.vx[s] += np.where(good, mv_x / np.where(good, norm, 1) * SPEED, 0.0)
        self.vy[s] += np.where(good, mv_y / np.where(good, norm, 1) * SPEED, 0.0)

    def _energy(self):
        c = self.cfg
        n = self.n
        d = np.hypot(self.x[None, :] - self.x[:, None],
                     self.y[None, :] - self.y[:, None])
        close = (d < PROXIMITY_THRESHOLD) & ~np.eye(n, dtype=bool)
        nw = (close & (~self.sol)[None, :]).sum(1)

        gain = np.where(self.sol, c["SOLDIER_ENERGY_GAIN"], c["WORKER_ENERGY_GAIN"])
        allee = np.where((nw > 1) & (nw < c["CRITICAL_GROUP_SIZE"]),
                         c["WORKER_TOGETHER_ENERGY"],
                         np.where(nw >= c["CRITICAL_GROUP_SIZE"],
                                  -c["WORKER_TOGETHER_ENERGY"], 0.0))
        near_w = nw > 0
        self.E += np.where(self.sol,
                           gain + np.where(near_w, c["NEAR_WORKER_GAIN"], 0.0),
                           gain + allee) - ENERGY_COST_PER_STEP

    def _reproduce(self):
        c = self.cfg
        can = (self.E > c["REPRODUCTION_ENERGY_THRESHOLD"]) & \
              (np.arange(self.n) < c["MAX_BOIDS"] - self.n)
        idx = np.flatnonzero(can)
        if not len(idx):
            return
        k = len(idx)
        g = self.G[idx].copy()
        m = self.rng.random(g.shape) < c["MUTATION_RATE"]
        g += m * self.rng.normal(0, c["MUTATION_STRENGTH"], g.shape)
        ang = self.rng.uniform(-1, 1, (k, 2))
        nrm = np.hypot(ang[:, 0], ang[:, 1])
        ux, uy = ang[:, 0] / nrm, ang[:, 1] / nrm
        uids = np.arange(self._next, self._next + k, dtype=np.int64)
        self._next += k
        for pu, cu in zip(self.uid[idx], uids):
            self.births.append((self.step_i, int(pu), int(cu)))

        self.x = np.concatenate([self.x, self.x[idx] + ux * REPRODUCTION_DISTANCE])
        self.y = np.concatenate([self.y, self.y[idx] + uy * REPRODUCTION_DISTANCE])
        self.vx = np.concatenate([self.vx, self.rng.uniform(-1, 1, k)])
        self.vy = np.concatenate([self.vy, self.rng.uniform(-1, 1, k)])
        self.sol = np.concatenate([self.sol, self.rng.random(k) < 0.5])
        self.G = np.vstack([self.G, g])
        self.E = np.concatenate([self.E, np.full(k, INITIAL_ENERGY)])
        self.age = np.concatenate([self.age, np.zeros(k, np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.par = np.concatenate([self.par, self.uid[idx]])
        self.E[idx] -= REPRODUCTION_COST

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
            head = np.arctan2(self.pvy[i], self.pvx[i])
            diff = np.abs(np.arctan2(by, bx) - head)
            diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
            in_view = (dist <= PRED_VISION) & (diff <= PRED_FOV_HALF)

            if c["SOLDIER_REVERSAL_COUNT"] is not None and \
                    (in_view & self.sol).sum() >= c["SOLDIER_REVERSAL_COUNT"]:
                self.pvx[i] *= -1
                self.pvy[i] *= -1
                self.p_rev[i] = REVERSAL_DURATION
            elif c["LARGE_GROUP_THRESHOLD"] is not None and \
                    (dist < PRED_VISION).sum() > c["LARGE_GROUP_THRESHOLD"]:
                a = self.rng.uniform(0, 2 * np.pi)
                self.pvx[i] = np.cos(a) * PREDATOR_SPEED
                self.pvy[i] = np.sin(a) * PREDATOR_SPEED
            elif self.p_rev[i] > 0:
                self.p_rev[i] -= 1
            else:
                cand = np.flatnonzero(in_view)          # first in list order
                if len(cand):
                    j = int(cand[0])
                    if dist[j] <= PRED_CATCH:
                        self.deaths.append((self.step_i, int(self.uid[j]), "eaten"))
                        keep = np.ones(self.n, bool)
                        keep[j] = False
                        self._keep(keep)
                        self.p_eat[i] = EATING_DURATION
                        continue
                    self.pvx[i] = bx[j] / dist[j] * PREDATOR_SPEED
                    self.pvy[i] = by[j] / dist[j] * PREDATOR_SPEED
                    self.px[i] += self.pvx[i]
                    self.py[i] += self.pvy[i]
                else:
                    self.px[i] += self.pvx[i]
                    self.py[i] += self.pvy[i]
                    self.pvx[i] += self.rng.uniform(-0.5, 0.5)
                    self.pvy[i] += self.rng.uniform(-0.5, 0.5)
                    s = np.hypot(self.pvx[i], self.pvy[i])
                    self.pvx[i] = self.pvx[i] / s * PREDATOR_SPEED
                    self.pvy[i] = self.pvy[i] / s * PREDATOR_SPEED
            if not 0 <= self.px[i] <= WIDTH:
                self.pvx[i] *= -1
            if not 0 <= self.py[i] <= HEIGHT:
                self.pvy[i] *= -1
            self.px[i] = min(max(self.px[i], 0.0), WIDTH)
            self.py[i] = min(max(self.py[i], 0.0), HEIGHT)

    def _keep(self, mask):
        for a in ("x", "y", "vx", "vy", "sol", "E", "age", "uid", "par"):
            setattr(self, a, getattr(self, a)[mask])
        self.G = self.G[mask]

    def snapshot(self):
        return dict(x=self.x.copy(), y=self.y.copy(), dx=self.vx.copy(),
                    dy=self.vy.copy(), energy=self.E.copy(), age=self.age.copy(),
                    soldier=self.sol.copy(), uid=self.uid.copy(),
                    parent=self.par.copy(), px=self.px.copy(), py=self.py.copy(),
                    pdx=self.pvx.copy(), pdy=self.pvy.copy())
