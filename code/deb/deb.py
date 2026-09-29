"""Minimal Dynamic Energy Budget rebuild of VitaNova.

Written from scratch rather than patched onto evo_cooperative_flock.py, so that
every hand-coded group-level term is gone. Nothing in an agent's rules reads a
group size, a flock, or any other aggregate; all an agent sees is what is within
its own perception radius.

Metabolism replaces the tuned energy constants. Each agent carries a structural
volume V and follows the standard DEB split:

    assimilation   pA = a * f * V^(2/3)      f = rho/(rho + K)   [surface]
    maintenance    pM = m * V                                     [volume]
    surplus        p  = pA - pM
        p >= 0:  V += kappa*p/Eg  ,  R += (1-kappa)*p
        p <  0:  V += p/Eg                       (shrink, then starve)

Because supply scales as V^(2/3) and maintenance as V, growth is automatically
positive when small and self-limiting when large -- the "positive but not too
positive" condition is a property of the equation, not a tuned number. The
individual asymptote is V* = (a*f/m)^3.

The two things that used to be written by hand now have to emerge:

  grouping benefit  predators chase the nearest visible agent, so interior
                    positions are safer purely by Hamilton's selfish-herd
                    geometry. No rule mentions group size.
  grouping cost     the resource field is local and depletable, so f falls
                    where agents crowd. No Allee band, no critical size.

Castes are removed entirely for this first test: they existed only to carry the
hand-coded energy bonuses. The network keeps the four steering outputs.
"""

import numpy as np

TWO_THIRDS = 2.0 / 3.0


class Params:
    # world
    WIDTH, HEIGHT = 800.0, 600.0
    BOID_SIZE = 5.0

    # perception / steering
    PERCEPTION = 25.0
    SEP_R = 25.0
    ALI_R = 25.0
    COH_R = 25.0
    SPEED = 1.0
    MAX_TURN_GAIN = 1.0

    # DEB
    A_MAX = 0.30        # {p_Am}  surface-specific assimilation
    K_HALF = 0.50       # half-saturation resource density
    P_M = 0.020         # [p_M]   volume-specific maintenance
    KAPPA = 0.7         # fraction of surplus to soma
    E_G = 1.0           # cost per unit structure
    V_BIRTH = 1.0
    R_REPRO = 3.0       # reproduction buffer threshold
    V_STARVE = 0.4      # die below this multiple of V_BIRTH
    MAX_AGE = 4000

    # resource field
    CELL = 25.0
    R_MAX = 1.0
    R_REGEN = 0.010     # per step, linear toward R_MAX
    R_YIELD = 1.0       # resource consumed per unit assimilated

    # predators
    N_PRED = 10
    PRED_SPEED = 1.2
    PRED_VISION = 50.0
    PRED_CATCH = 10.0
    PRED_HANDLING = 60   # steps spent after a catch

    # genetics
    N_IN, N_HID, N_OUT = 11, 10, 4
    MUT_RATE = 0.1
    MUT_SIGMA = 0.1

    # init
    N_BOIDS = 100
    MAX_BOIDS = 4000

    @classmethod
    def genome_len(cls):
        return cls.N_IN * cls.N_HID + cls.N_HID * cls.N_OUT


class World:
    def __init__(self, seed=2024, params=Params, **overrides):
        self.p = params
        for k, v in overrides.items():
            if not hasattr(params, k):
                raise KeyError(f"unknown parameter {k}")
            setattr(self, k, v)
        self._ov = overrides
        self.rng = np.random.default_rng(seed)
        p = self.par

        n = int(p("N_BOIDS"))
        self.x = self.rng.uniform(0, p("WIDTH"), n)
        self.y = self.rng.uniform(0, p("HEIGHT"), n)
        ang = self.rng.uniform(0, 2 * np.pi, n)
        self.vx, self.vy = np.cos(ang), np.sin(ang)
        self.V = np.full(n, p("V_BIRTH"))
        self.R = np.zeros(n)
        self.age = np.zeros(n, dtype=np.int64)
        self.uid = np.arange(n, dtype=np.int64)
        self.parent = np.full(n, -1, dtype=np.int64)
        self._next_uid = n
        self.G = self.rng.standard_normal((n, params.genome_len()))

        gw, gh = int(p("WIDTH") // p("CELL")), int(p("HEIGHT") // p("CELL"))
        self.grid_w, self.grid_h = gw, gh
        self.res = np.full((gh, gw), p("R_MAX"))

        self.px = self.rng.uniform(0, p("WIDTH"), int(p("N_PRED")))
        self.py = self.rng.uniform(0, p("HEIGHT"), int(p("N_PRED")))
        pang = self.rng.uniform(0, 2 * np.pi, int(p("N_PRED")))
        self.pvx, self.pvy = np.cos(pang), np.sin(pang)
        self.phandle = np.zeros(int(p("N_PRED")), dtype=np.int64)

        self.step_i = 0
        self.births = []       # (step, parent_uid, child_uid)
        self.deaths = []       # (step, uid, cause)

    def par(self, name):
        return getattr(self, name, getattr(self.p, name))

    # ---------------------------------------------------------------- helpers
    @property
    def n(self):
        return len(self.x)

    def cell_of(self, x, y):
        gx = np.clip((x // self.par("CELL")).astype(int), 0, self.grid_w - 1)
        gy = np.clip((y // self.par("CELL")).astype(int), 0, self.grid_h - 1)
        return gy, gx

    def brains(self):
        p = self.p
        w1 = self.G[:, : p.N_IN * p.N_HID].reshape(-1, p.N_IN, p.N_HID)
        w2 = self.G[:, p.N_IN * p.N_HID:].reshape(-1, p.N_HID, p.N_OUT)
        return w1, w2

    # ------------------------------------------------------------------ step
    def step(self):
        if self.n == 0:
            return
        p = self.par
        n = self.n

        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)

        near = d <= p("PERCEPTION")
        n_near = near.sum(1)

        # ---- sensory input: three nearest neighbours, nearest predator, state
        rows = np.arange(n)[:, None]
        if n > 4:
            part = np.argpartition(d, 3, axis=1)[:, :3]
            order = np.take_along_axis(
                part, np.argsort(np.take_along_axis(d, part, 1), axis=1), 1)
        else:
            # Fewer agents than neighbour slots: pad so the input layer keeps
            # its shape, and let `valid` zero the padding out.
            srt = np.argsort(d, axis=1)
            order = np.zeros((n, 3), dtype=int)
            order[:, :srt.shape[1]] = srt[:, :3]
        valid = d[rows, order] <= p("PERCEPTION")
        rel = np.zeros((n, 6))
        for j in range(3):
            idx = order[:, j]
            ok = valid[:, j]
            rel[:, 2 * j] = np.where(ok, dx[rows[:, 0], idx] / p("PERCEPTION"), 0)
            rel[:, 2 * j + 1] = np.where(ok, dy[rows[:, 0], idx] / p("PERCEPTION"), 0)

        if len(self.px):
            pdx = self.px[None, :] - self.x[:, None]
            pdy = self.py[None, :] - self.y[:, None]
            pd = np.sqrt(pdx ** 2 + pdy ** 2)
            k = np.argmin(pd, axis=1)
            pnear = pd[rows[:, 0], k] <= p("PERCEPTION") * 2
            prx = np.where(pnear, pdx[rows[:, 0], k] / (p("PERCEPTION") * 2), 0)
            pry = np.where(pnear, pdy[rows[:, 0], k] / (p("PERCEPTION") * 2), 0)
        else:
            prx = pry = np.zeros(n)

        gy, gx = self.cell_of(self.x, self.y)
        rho = self.res[gy, gx]

        X = np.column_stack([
            rel, prx, pry,
            np.tanh(self.V - 1.0),
            rho * 2 - 1,
            np.tanh(n_near / 5.0) * 2 - 1,
        ])

        w1, w2 = self.brains()
        h = np.maximum(np.einsum("ni,nih->nh", X, w1), 0.0)
        out = np.tanh(np.einsum("nh,nho->no", h, w2))
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        # ---- steering (Reynolds, weighted by the network)
        radii = {p("SEP_R"), p("ALI_R"), p("COH_R"), p("PERCEPTION")}
        mask = {r: (near if r == p("PERCEPTION") else (d <= r) & near)
                for r in radii}
        msep, mali, mcoh = (mask[p("SEP_R")], mask[p("ALI_R")], mask[p("COH_R")])

        inv = np.where(d > 1e-9, 1.0 / np.maximum(d, 1e-9), 0.0)
        wsep = np.where(msep, inv, 0.0)
        sep_x = -(dx * wsep).sum(1)
        sep_y = -(dy * wsep).sum(1)

        fali = mali.astype(np.float64)          # matrix-vector goes through BLAS
        na = fali.sum(1)
        cnt = np.maximum(na, 1)
        ali_x = fali @ self.vx / cnt - self.vx
        ali_y = fali @ self.vy / cnt - self.vy
        ali_x[na == 0] = 0
        ali_y[na == 0] = 0

        fcoh = mcoh.astype(np.float64)
        nc = fcoh.sum(1)
        cnt2 = np.maximum(nc, 1)
        coh_x = (dx * fcoh).sum(1) / cnt2
        coh_y = (dy * fcoh).sum(1) / cnt2
        coh_x[nc == 0] = 0
        coh_y[nc == 0] = 0

        norm = lambda a, b: (a / np.maximum(np.hypot(a, b), 1e-9),
                             b / np.maximum(np.hypot(a, b), 1e-9))
        sep_x, sep_y = norm(sep_x, sep_y)
        ali_x, ali_y = norm(ali_x, ali_y)
        coh_x, coh_y = norm(coh_x, coh_y)
        avo_x, avo_y = norm(-prx, -pry)

        self.vx += p("MAX_TURN_GAIN") * (t_sep * sep_x + t_ali * ali_x
                                         + t_coh * coh_x + t_avo * avo_x)
        self.vy += p("MAX_TURN_GAIN") * (t_sep * sep_y + t_ali * ali_y
                                         + t_coh * coh_y + t_avo * avo_y)
        sp = np.maximum(np.hypot(self.vx, self.vy), 1e-9)
        self.vx, self.vy = self.vx / sp * p("SPEED"), self.vy / sp * p("SPEED")
        self.x += self.vx
        self.y += self.vy

        out_x = (self.x < 0) | (self.x > p("WIDTH"))
        out_y = (self.y < 0) | (self.y > p("HEIGHT"))
        self.vx[out_x] *= -1
        self.vy[out_y] *= -1
        self.x = np.clip(self.x, 0, p("WIDTH"))
        self.y = np.clip(self.y, 0, p("HEIGHT"))

        # ---- DEB: assimilate, maintain, grow, fill the reproduction buffer
        gy, gx = self.cell_of(self.x, self.y)
        rho = self.res[gy, gx]
        f = rho / (rho + p("K_HALF"))
        pA = p("A_MAX") * f * self.V ** TWO_THIRDS
        pM = p("P_M") * self.V
        surplus = pA - pM

        np.subtract.at(self.res, (gy, gx), pA * p("R_YIELD") * (p("CELL") ** -2) * 100)
        np.clip(self.res, 0.0, p("R_MAX"), out=self.res)
        self.res += p("R_REGEN") * (p("R_MAX") - self.res)

        pos = surplus >= 0
        self.V += np.where(pos, p("KAPPA") * surplus, surplus) / p("E_G")
        self.R += np.where(pos, (1 - p("KAPPA")) * surplus, 0.0)
        self.age += 1

        # ---- reproduction
        can = (self.R >= p("R_REPRO")) & (self.n + 1 <= p("MAX_BOIDS"))
        if can.any():
            idx = np.flatnonzero(can)
            self.R[idx] -= p("R_REPRO")
            self._spawn(idx)

        # ---- predators: chase the nearest visible agent. No group rule.
        self._predators()

        # ---- death by starvation or age
        dead = (self.V < p("V_BIRTH") * p("V_STARVE")) | (self.age > p("MAX_AGE"))
        if dead.any():
            for u in self.uid[dead]:
                self.deaths.append((self.step_i, int(u), "starve_or_age"))
            self._keep(~dead)

        self.step_i += 1

    def _spawn(self, parents):
        p = self.par
        k = len(parents)
        ang = self.rng.uniform(0, 2 * np.pi, k)
        g = self.G[parents].copy()
        mask = self.rng.random(g.shape) < p("MUT_RATE")
        g += mask * self.rng.normal(0, p("MUT_SIGMA"), g.shape)
        uids = np.arange(self._next_uid, self._next_uid + k)
        self._next_uid += k
        for pu, cu in zip(self.uid[parents], uids):
            self.births.append((self.step_i, int(pu), int(cu)))

        self.x = np.concatenate([self.x, self.x[parents] + np.cos(ang) * p("BOID_SIZE")])
        self.y = np.concatenate([self.y, self.y[parents] + np.sin(ang) * p("BOID_SIZE")])
        self.vx = np.concatenate([self.vx, np.cos(ang)])
        self.vy = np.concatenate([self.vy, np.sin(ang)])
        self.V = np.concatenate([self.V, np.full(k, p("V_BIRTH"))])
        self.R = np.concatenate([self.R, np.zeros(k)])
        self.age = np.concatenate([self.age, np.zeros(k, dtype=np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.parent = np.concatenate([self.parent, self.uid[parents]])
        self.G = np.vstack([self.G, g])
        self.x = np.clip(self.x, 0, p("WIDTH"))
        self.y = np.clip(self.y, 0, p("HEIGHT"))

    def _predators(self):
        p = self.par
        if not len(self.px):
            return
        busy = self.phandle > 0
        self.phandle[busy] -= 1
        for i in range(len(self.px)):
            if busy[i] or self.n == 0:
                continue
            ddx = self.x - self.px[i]
            ddy = self.y - self.py[i]
            dd = np.hypot(ddx, ddy)
            j = int(np.argmin(dd))
            if dd[j] <= p("PRED_CATCH"):
                self.deaths.append((self.step_i, int(self.uid[j]), "eaten"))
                keep = np.ones(self.n, bool); keep[j] = False
                self._keep(keep)
                self.phandle[i] = p("PRED_HANDLING")
                continue
            if dd[j] <= p("PRED_VISION"):
                self.pvx[i], self.pvy[i] = ddx[j] / dd[j], ddy[j] / dd[j]
            else:
                a = self.rng.uniform(0, 2 * np.pi)
                self.pvx[i] += np.cos(a) * 0.3
                self.pvy[i] += np.sin(a) * 0.3
            s = max(np.hypot(self.pvx[i], self.pvy[i]), 1e-9)
            self.pvx[i], self.pvy[i] = self.pvx[i] / s, self.pvy[i] / s
            self.px[i] += self.pvx[i] * p("PRED_SPEED")
            self.py[i] += self.pvy[i] * p("PRED_SPEED")
            if not 0 <= self.px[i] <= p("WIDTH"):
                self.pvx[i] *= -1
            if not 0 <= self.py[i] <= p("HEIGHT"):
                self.pvy[i] *= -1
            self.px[i] = min(max(self.px[i], 0), p("WIDTH"))
            self.py[i] = min(max(self.py[i], 0), p("HEIGHT"))

    def _keep(self, mask):
        for a in ("x", "y", "vx", "vy", "V", "R", "age", "uid", "parent"):
            setattr(self, a, getattr(self, a)[mask])
        self.G = self.G[mask]

    def snapshot(self):
        return {
            "x": self.x.copy(), "y": self.y.copy(),
            "dx": self.vx.copy(), "dy": self.vy.copy(),
            "V": self.V.copy(), "R": self.R.copy(),
            "age": self.age.copy(), "uid": self.uid.copy(),
            "soldier": np.zeros(self.n, bool),
            "energy": self.R.copy(),
            "px": self.px.copy(), "py": self.py.copy(),
            "pdx": self.pvx.copy(), "pdy": self.pvy.copy(),
        }
