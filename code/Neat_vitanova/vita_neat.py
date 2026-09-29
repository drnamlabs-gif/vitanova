"""VitaNova with NEAT controllers.

Every element of evo_cooperative_flock.py is reproduced: 100 boids and 10
predators in an 800x600 arena with reflecting boundaries, all four interaction
radii at 25, the same energy accounting with its Allee term, the same worker and
soldier castes assigned each step from the network's last two outputs, the same
soldier interception behaviour, the same predator rules, the same reproduction
at an energy threshold with a lifespan of 1000 steps.

One thing differs. The controller is a NEAT genome whose **topology** evolves,
starting minimal rather than at a fixed 11-10-6, and each boid carries its own.
There is no fitness function, no generation, no speciation and no selection
operator: a boid reproduces when its energy crosses the threshold and the
offspring is a mutated copy of its genome. Selection is whatever survival and
reproduction in the world happen to produce.

Inputs (11), exactly as in the original: relative positions of the first three
neighbours in list order within the radius, the nearest visible predator's
normalised distance and angle, the boid's own velocity, and local density.
Outputs (6): separation, alignment, cohesion and avoidance tendencies, then two
role tendencies whose comparison sets the caste.
"""
from __future__ import annotations

import math
import random

import numpy as np

from neat import ACTIVATIONS, Config, Genome, Innovations

# ---- constants, from evo_cooperative_flock.py ------------------------------
BOID_SIZE = 5.0
PREDATOR_SIZE = 2 * BOID_SIZE
WIDTH, HEIGHT = 160 * BOID_SIZE, 120 * BOID_SIZE
RADIUS = BOID_SIZE * 5
SPEED = 1.0
PREDATOR_SPEED = 1.2 * SPEED
BOID_COUNT, PREDATOR_COUNT = 100, 10

INITIAL_ENERGY = 50.0
REPRODUCTION_ENERGY_THRESHOLD = 200.0
REPRODUCTION_COST = 100.0
REPRODUCTION_DISTANCE = BOID_SIZE * 2
WORKER_ENERGY_GAIN = 1.1
SOLDIER_ENERGY_GAIN = 0.7
NEAR_WORKER_GAIN = 0.6
ENERGY_COST_PER_STEP = 1.0
WORKER_TOGETHER_ENERGY = 0.2
CRITICAL_GROUP_SIZE = 20
PROXIMITY_THRESHOLD = BOID_SIZE * 3
SOLDIER_PROTECTION_RADIUS = BOID_SIZE * 5
LIFESPAN = 1000

PRED_VISION = PREDATOR_SIZE * 5
PRED_CATCH = PREDATOR_SIZE
PRED_FOV_HALF = math.pi / 2
REVERSAL_DURATION = 30
EATING_DURATION = 60
LARGE_GROUP_THRESHOLD = 20
SOLDIER_REVERSAL_COUNT = 3

N_IN, N_OUT = 11, 6


def neat_config(**kw):
    """Matched to evo_cooperative_flock.py wherever there is a counterpart.

    The original mutates each weight with probability MUTATION_RATE = 0.1 by
    N(0, MUTATION_STRENGTH = 0.1), has **no biases** at all (two weight
    matrices only), and uses **ReLU** in the hidden layer with tanh on the
    output. Earlier runs here used a 0.9 perturbation rate, per-node biases and
    tanh throughout, which at 170 connections meant 153 weights disturbed per
    birth instead of 17 -- enough to stop any lineage holding a behaviour.

    The structural rates are the only parameters with no counterpart in the
    original: they are what NEAT adds.
    """
    cfg = Config(n_inputs=N_IN, n_outputs=N_OUT, output_activation="tanh",
                 activation="relu",
                 weight_mutate_rate=1.0, weight_perturb_rate=0.1,
                 weight_perturb_sd=0.1, weight_replace_sd=1.0,
                 bias_mutate_rate=0.0, bias_perturb_sd=0.0,
                 add_connection_rate=0.05, add_node_rate=0.02,
                 toggle_rate=0.01, normalise_by_size=False)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class Brain:
    """Compiled genome: an evaluation order and per-node incoming lists."""

    __slots__ = ("order", "incoming", "bias", "outputs", "act", "out_act",
                 "n_hidden", "n_conn", "_ins")

    def __init__(self, g: Genome):
        self.outputs = sorted(g.outputs())
        ins = sorted(g.inputs())
        self.bias = {n: b for n, (t, b) in g.nodes.items()}
        self.incoming = {n: [] for n in g.nodes}
        for c in g.conns.values():
            if c.enabled and c.a in g.nodes and c.b in g.nodes:
                self.incoming[c.b].append((c.a, c.w))
        deg = {n: len(self.incoming[n]) for n in g.nodes}
        outgoing = {n: [] for n in g.nodes}
        for n, lst in self.incoming.items():
            for a, _ in lst:
                outgoing[a].append(n)
        ready = list(ins) + [n for n in g.nodes if deg[n] == 0 and n not in ins]
        order, seen = [], set()
        while ready:
            n = ready.pop(0)
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            for m in outgoing[n]:
                deg[m] -= 1
                if deg[m] == 0:
                    ready.append(m)
        order += [n for n in g.nodes if n not in seen]
        self.order = [n for n in order if n not in ins]
        self.act = ACTIVATIONS[g.cfg.activation]
        self.out_act = ACTIVATIONS[g.cfg.output_activation]
        self.n_hidden = len(g.hidden())
        self.n_conn = sum(1 for c in g.conns.values() if c.enabled)
        self._ins = ins

    def __call__(self, x):
        v = dict(zip(self._ins, x))
        for n in self.order:
            s = self.bias.get(n, 0.0)
            for a, w in self.incoming[n]:
                s += v.get(a, 0.0) * w
            v[n] = (self.out_act if n in self.outputs else self.act)(s)
        return [v.get(n, 0.0) for n in self.outputs]


class VitaNeat:
    def __init__(self, seed=2024, n=BOID_COUNT, n_pred=PREDATOR_COUNT,
                 max_boids=1200, cfg=None, init_hidden=0):
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.cfg = cfg or neat_config()
        self.innov = Innovations(start_node=N_IN + N_OUT)
        for i in range(N_IN):
            for j in range(N_OUT):
                self.innov.connection(i, N_IN + j)
        self.max_boids = max_boids

        self.x = self.rng.uniform(0, WIDTH, n)
        self.y = self.rng.uniform(0, HEIGHT, n)
        self.vx = self.rng.uniform(-1, 1, n)
        self.vy = self.rng.uniform(-1, 1, n)
        self.E = np.full(n, INITIAL_ENERGY)
        self.age = np.zeros(n, dtype=np.int64)
        self.sol = self.rng.random(n) < 0.5
        self.uid = np.arange(n, dtype=np.int64)
        self.par = np.full(n, -1, dtype=np.int64)
        self._next = n
        self.init_hidden = init_hidden
        self.genomes = [Genome.with_hidden(self.cfg, self.innov, self.pyrng,
                                           init_hidden)
                        for _ in range(n)]
        self.brains = [Brain(g) for g in self.genomes]

        self.px = self.rng.uniform(0, WIDTH, n_pred)
        self.py = self.rng.uniform(0, HEIGHT, n_pred)
        self.pvx = self.rng.uniform(-1, 1, n_pred)
        self.pvy = self.rng.uniform(-1, 1, n_pred)
        self.p_eat = np.zeros(n_pred, dtype=np.int64)
        self.p_rev = np.full(n_pred, 20, dtype=np.int64)

        self.step_i = 0
        self.births, self.deaths = [], []

    @property
    def n(self):
        return len(self.x)

    # ------------------------------------------------------------------ step
    def step(self):
        if self.n == 0:
            return
        n = self.n
        dx = self.x[None, :] - self.x[:, None]
        dy = self.y[None, :] - self.y[:, None]
        d = np.sqrt(dx * dx + dy * dy)
        eye = np.eye(n, dtype=bool)
        near = (d <= RADIUS) & ~eye

        cols = np.where(near, np.arange(n)[None, :], n)
        first3 = (np.partition(cols, 2, axis=1)[:, :3] if n > 3
                  else np.pad(np.sort(cols, axis=1)[:, :3],
                              ((0, 0), (0, max(0, 3 - n))), constant_values=n))
        first3 = np.sort(first3, axis=1)
        ok = first3 < n
        idx = np.where(ok, first3, 0)
        rows = np.arange(n)[:, None]
        rel = np.zeros((n, 6))
        rel[:, 0::2] = np.where(ok, dx[rows, idx] / RADIUS, 0.0)
        rel[:, 1::2] = np.where(ok, dy[rows, idx] / RADIUS, 0.0)

        if len(self.px):
            qx = self.px[None, :] - self.x[:, None]
            qy = self.py[None, :] - self.y[:, None]
            q = np.sqrt(qx * qx + qy * qy)
            vis = q < RADIUS
            qm = np.where(vis, q, np.inf)
            k = np.argmin(qm, axis=1)
            has = np.isfinite(qm[np.arange(n), k])
            pdx = np.where(has, qx[np.arange(n), k], 0.0)
            pdy = np.where(has, qy[np.arange(n), k], 0.0)
            pdist = np.where(has, q[np.arange(n), k], 0.0)
            in_pd = np.where(has, pdist / RADIUS, 0.0)
            in_pa = np.where(has, np.arctan2(pdy, pdx) / np.pi, 0.0)
        else:
            has = np.zeros(n, bool)
            pdx = pdy = pdist = in_pd = in_pa = np.zeros(n)

        density = near.sum(1) / 3.0 * 2 - 1
        X = np.column_stack([rel, in_pd, in_pa, self.vx / SPEED,
                             self.vy / SPEED, density])

        out = np.array([b(x) for b, x in zip(self.brains, X)])   # per-boid NEAT
        t_sep, t_ali, t_coh, t_avo = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        msep = (d < RADIUS) & ~eye
        w = np.where(msep, 1.0 / (d + 1e-5), 0.0)
        sep_x, sep_y = -(dx * w).sum(1), -(dy * w).sum(1)
        mall = d < RADIUS
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

        self.sol = out[:, -2] > out[:, -1]
        self._soldiers()
        self.age += 1
        self._energy(d, eye)
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
            wn = np.maximum(np.hypot(wx, wy), 1e-12)
            wx, wy = wx / wn, wy / wn
        else:
            wx = wy = None
        if len(self.px):
            pd = np.hypot(self.px[None, :] - self.x[s][:, None],
                          self.py[None, :] - self.y[s][:, None])
            k = np.argmin(pd, axis=1)
            near_pred = pd[np.arange(len(s)), k] < SOLDIER_PROTECTION_RADIUS
            ppx, ppy = self.px[k] - self.x[s], self.py[k] - self.y[s]
            pn = np.maximum(np.hypot(ppx, ppy), 1e-12)
            ppx, ppy = ppx / pn, ppy / pn
        else:
            near_pred = np.zeros(len(s), bool)
            ppx = ppy = np.zeros(len(s))
        if wx is not None:
            mv_x = np.where(near_pred, (wx - ppx) / 2, wx)
            mv_y = np.where(near_pred, (wy - ppy) / 2, wy)
        nrm = np.hypot(mv_x, mv_y)
        good = nrm != 0
        self.vx[s] += np.where(good, mv_x / np.where(good, nrm, 1) * SPEED, 0.0)
        self.vy[s] += np.where(good, mv_y / np.where(good, nrm, 1) * SPEED, 0.0)

    def _energy(self, d, eye):
        close = (d < PROXIMITY_THRESHOLD) & ~eye
        nw = (close & (~self.sol)[None, :]).sum(1)
        gain = np.where(self.sol, SOLDIER_ENERGY_GAIN, WORKER_ENERGY_GAIN)
        allee = np.where((nw > 1) & (nw < CRITICAL_GROUP_SIZE),
                         WORKER_TOGETHER_ENERGY,
                         np.where(nw >= CRITICAL_GROUP_SIZE,
                                  -WORKER_TOGETHER_ENERGY, 0.0))
        self.E += np.where(self.sol,
                           gain + np.where(nw > 0, NEAR_WORKER_GAIN, 0.0),
                           gain + allee) - ENERGY_COST_PER_STEP

    def _reproduce(self):
        idx = np.flatnonzero((self.E > REPRODUCTION_ENERGY_THRESHOLD)
                             & (self.n + 1 <= self.max_boids))
        if not len(idx):
            return
        k = len(idx)
        ang = self.rng.uniform(-1, 1, (k, 2))
        nrm = np.maximum(np.hypot(ang[:, 0], ang[:, 1]), 1e-12)
        ux, uy = ang[:, 0] / nrm, ang[:, 1] / nrm
        uids = np.arange(self._next, self._next + k, dtype=np.int64)
        self._next += k
        for pu, cu in zip(self.uid[idx], uids):
            self.births.append((self.step_i, int(pu), int(cu)))
        for i in idx:
            child = self.genomes[i].copy()
            child.mutate(self.innov, self.pyrng)       # weights *and* topology
            self.genomes.append(child)
            self.brains.append(Brain(child))
        self.x = np.concatenate([self.x, self.x[idx] + ux * REPRODUCTION_DISTANCE])
        self.y = np.concatenate([self.y, self.y[idx] + uy * REPRODUCTION_DISTANCE])
        self.vx = np.concatenate([self.vx, self.rng.uniform(-1, 1, k)])
        self.vy = np.concatenate([self.vy, self.rng.uniform(-1, 1, k)])
        self.sol = np.concatenate([self.sol, self.rng.random(k) < 0.5])
        self.E = np.concatenate([self.E, np.full(k, INITIAL_ENERGY)])
        self.age = np.concatenate([self.age, np.zeros(k, dtype=np.int64)])
        self.uid = np.concatenate([self.uid, uids])
        self.par = np.concatenate([self.par, self.uid[idx]])
        self.E[idx] -= REPRODUCTION_COST
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)

    def _predators(self):
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
            in_view = (dist <= PRED_VISION) & (diff <= PRED_FOV_HALF)
            if (in_view & self.sol).sum() >= SOLDIER_REVERSAL_COUNT:
                self.pvx[i] *= -1
                self.pvy[i] *= -1
                self.p_rev[i] = REVERSAL_DURATION
            elif (dist < PRED_VISION).sum() > LARGE_GROUP_THRESHOLD:
                a = self.rng.uniform(0, 2 * math.pi)
                self.pvx[i] = math.cos(a) * PREDATOR_SPEED
                self.pvy[i] = math.sin(a) * PREDATOR_SPEED
            elif self.p_rev[i] > 0:
                self.p_rev[i] -= 1
            else:
                cand = np.flatnonzero(in_view)
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
                    sp = max(math.hypot(self.pvx[i], self.pvy[i]), 1e-12)
                    self.pvx[i] = self.pvx[i] / sp * PREDATOR_SPEED
                    self.pvy[i] = self.pvy[i] / sp * PREDATOR_SPEED
            if not 0 <= self.px[i] <= WIDTH:
                self.pvx[i] *= -1
            if not 0 <= self.py[i] <= HEIGHT:
                self.pvy[i] *= -1
            self.px[i] = min(max(self.px[i], 0.0), WIDTH)
            self.py[i] = min(max(self.py[i], 0.0), HEIGHT)

    def _keep(self, mask):
        for a in ("x", "y", "vx", "vy", "sol", "E", "age", "uid", "par"):
            setattr(self, a, getattr(self, a)[mask])
        keep = np.flatnonzero(mask)
        self.genomes = [self.genomes[i] for i in keep]
        self.brains = [self.brains[i] for i in keep]

    # ------------------------------------------------------------ observables
    def complexity(self):
        if not self.brains:
            return 0.0, 0.0
        return (float(np.mean([b.n_hidden for b in self.brains])),
                float(np.mean([b.n_conn for b in self.brains])))

    def snapshot(self):
        return dict(x=self.x.copy(), y=self.y.copy(), dx=self.vx.copy(),
                    dy=self.vy.copy(), energy=self.E.copy(),
                    age=self.age.copy(), soldier=self.sol.copy(),
                    uid=self.uid.copy(), parent=self.par.copy(),
                    hidden=np.array([b.n_hidden for b in self.brains]),
                    conns=np.array([b.n_conn for b in self.brains]),
                    px=self.px.copy(), py=self.py.copy(),
                    pdx=self.pvx.copy(), pdy=self.pvy.copy())
