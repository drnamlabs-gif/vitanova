"""A minimal NEAT, written for this project rather than taken from a library.

Two reasons for writing it. First, the standard implementations assume
generational evolution against an explicit fitness function, while VitaNova
needs agents that reproduce asynchronously in the world with no fitness function
at all; that requires the genome and the mutation operators, but not the
population manager. Second, a component this easy to get subtly wrong should be
tested here rather than trusted.

Structure follows Stanley & Miikkulainen (2002): genomes carry node genes and
connection genes, structural mutations are stamped with innovation numbers so
that crossover can align them, and the population is divided into species by a
compatibility distance with fitness sharing inside each species.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

INPUT, HIDDEN, OUTPUT = "input", "hidden", "output"


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, 4.9 * x))))


def tanh(x):
    return math.tanh(max(-60.0, min(60.0, x)))


ACTIVATIONS = {"sigmoid": sigmoid, "tanh": tanh,
               "relu": lambda x: x if x > 0 else 0.0,
               "identity": lambda x: x}


@dataclass
class Config:
    n_inputs: int = 2
    n_outputs: int = 1
    activation: str = "sigmoid"
    output_activation: str = "sigmoid"

    weight_init_sd: float = 1.0
    weight_mutate_rate: float = 0.8      # chance a genome perturbs its weights
    weight_perturb_rate: float = 0.9     # of those, chance each gene is nudged
    weight_perturb_sd: float = 0.5
    weight_replace_sd: float = 1.0
    weight_cap: float = 8.0

    bias_mutate_rate: float = 0.7
    bias_perturb_sd: float = 0.3

    add_connection_rate: float = 0.08
    add_node_rate: float = 0.03
    toggle_rate: float = 0.01

    # compatibility distance
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_weight: float = 0.4
    compat_threshold: float = 3.0
    # Dividing the structural terms by genome size makes one added node worth
    # 2/N, which at N = 27 is far below ordinary weight variation -- so
    # speciation stops seeing structure exactly when genomes get big enough for
    # structure to matter. Off by default; the original paper only applied the
    # division above 20 genes, which has the same effect in reverse.
    normalise_by_size: bool = False
    # Species count is held near this by moving the threshold, rather than
    # fixing a threshold that means different things for different genome sizes.
    target_species: int = 8
    threshold_step: float = 0.05
    threshold_min: float = 0.05

    crossover_rate: float = 0.75
    interspecies_mating_rate: float = 0.001
    survival_threshold: float = 0.2      # top fraction of a species that breeds
    elitism: int = 1                     # copied unchanged per large species
    stagnation: int = 15
    min_species: int = 2


class Innovations:
    """Global bookkeeping so the same structural change gets the same number."""

    def __init__(self, start_node: int):
        self.conn = {}
        self.node_split = {}
        self._next_conn = 0
        self._next_node = start_node

    def connection(self, a: int, b: int) -> int:
        key = (a, b)
        if key not in self.conn:
            self.conn[key] = self._next_conn
            self._next_conn += 1
        return self.conn[key]

    def split(self, innov: int) -> int:
        """The new node id for splitting a given connection."""
        if innov not in self.node_split:
            self.node_split[innov] = self._next_node
            self._next_node += 1
        return self.node_split[innov]

    def new_node(self) -> int:
        self._next_node += 1
        return self._next_node - 1


@dataclass
class Conn:
    a: int
    b: int
    w: float
    enabled: bool = True
    innov: int = -1


@dataclass
class Genome:
    cfg: Config
    nodes: dict = field(default_factory=dict)      # id -> (type, bias)
    conns: dict = field(default_factory=dict)      # innov -> Conn
    fitness: float = 0.0
    adjusted: float = 0.0

    # ------------------------------------------------------------- construction
    @classmethod
    def minimal(cls, cfg: Config, innov: Innovations, rng: random.Random,
                connect: bool = True) -> "Genome":
        g = cls(cfg)
        for i in range(cfg.n_inputs):
            g.nodes[i] = (INPUT, 0.0)
        for j in range(cfg.n_outputs):
            g.nodes[cfg.n_inputs + j] = (OUTPUT, 0.0)
        if connect:
            for i in range(cfg.n_inputs):
                for j in range(cfg.n_outputs):
                    b = cfg.n_inputs + j
                    n = innov.connection(i, b)
                    g.conns[n] = Conn(i, b, rng.gauss(0, cfg.weight_init_sd),
                                      True, n)
        return g

    @classmethod
    def with_hidden(cls, cfg: Config, innov: Innovations, rng: random.Random,
                    n_hidden: int) -> "Genome":
        """Start from a layered network instead of a minimal one.

        NEAT normally begins with no hidden nodes, which in an ecology where
        reproduction requires already behaving well is a chicken-and-egg
        problem: a boid must group to gain energy, and must gain energy to
        leave any offspring at all. Seeding hidden nodes gives the same
        starting architecture the fixed network has, while leaving topology
        free to change afterwards. Connections are input->hidden->output with
        no direct input->output path, matching the original 11-10-6 network.
        """
        if n_hidden <= 0:
            return cls.minimal(cfg, innov, rng)
        g = cls(cfg)
        for i in range(cfg.n_inputs):
            g.nodes[i] = (INPUT, 0.0)
        for j in range(cfg.n_outputs):
            g.nodes[cfg.n_inputs + j] = (OUTPUT, 0.0)
        base = cfg.n_inputs + cfg.n_outputs
        hidden = list(range(base, base + n_hidden))
        for h in hidden:
            g.nodes[h] = (HIDDEN, 0.0)
        innov._next_node = max(innov._next_node, base + n_hidden)
        for i in range(cfg.n_inputs):
            for h in hidden:
                n = innov.connection(i, h)
                g.conns[n] = Conn(i, h, rng.gauss(0, cfg.weight_init_sd), True, n)
        for h in hidden:
            for j in range(cfg.n_outputs):
                b = cfg.n_inputs + j
                n = innov.connection(h, b)
                g.conns[n] = Conn(h, b, rng.gauss(0, cfg.weight_init_sd), True, n)
        return g

    def copy(self) -> "Genome":
        g = Genome(self.cfg, dict(self.nodes),
                   {k: Conn(c.a, c.b, c.w, c.enabled, c.innov)
                    for k, c in self.conns.items()})
        return g

    # ---------------------------------------------------------------- structure
    def inputs(self):
        return [i for i, (t, _) in self.nodes.items() if t == INPUT]

    def outputs(self):
        return [i for i, (t, _) in self.nodes.items() if t == OUTPUT]

    def hidden(self):
        return [i for i, (t, _) in self.nodes.items() if t == HIDDEN]

    def would_cycle(self, a: int, b: int) -> bool:
        """True if adding a->b closes a loop (feed-forward genomes only)."""
        if a == b:
            return True
        seen, stack = set(), [b]
        while stack:
            n = stack.pop()
            if n == a:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(c.b for c in self.conns.values()
                         if c.enabled and c.a == n)
        return False

    # ---------------------------------------------------------------- mutation
    def mutate(self, innov: Innovations, rng: random.Random):
        cfg = self.cfg
        if rng.random() < cfg.weight_mutate_rate:
            for c in self.conns.values():
                if rng.random() < cfg.weight_perturb_rate:
                    c.w += rng.gauss(0, cfg.weight_perturb_sd)
                else:
                    c.w = rng.gauss(0, cfg.weight_replace_sd)
                c.w = max(-cfg.weight_cap, min(cfg.weight_cap, c.w))
        if rng.random() < cfg.bias_mutate_rate:
            for nid, (t, b) in self.nodes.items():
                if t != INPUT:
                    self.nodes[nid] = (t, max(-cfg.weight_cap,
                                              min(cfg.weight_cap,
                                                  b + rng.gauss(0, cfg.bias_perturb_sd))))
        if rng.random() < cfg.toggle_rate and self.conns:
            c = rng.choice(list(self.conns.values()))
            c.enabled = not c.enabled
        if rng.random() < cfg.add_connection_rate:
            self.mutate_add_connection(innov, rng)
        if rng.random() < cfg.add_node_rate:
            self.mutate_add_node(innov, rng)

    def mutate_add_connection(self, innov: Innovations, rng: random.Random,
                              tries: int = 20):
        sources = self.inputs() + self.hidden()
        targets = self.hidden() + self.outputs()
        if not sources or not targets:
            return False
        existing = {(c.a, c.b) for c in self.conns.values()}
        for _ in range(tries):
            a, b = rng.choice(sources), rng.choice(targets)
            if (a, b) in existing or self.would_cycle(a, b):
                continue
            n = innov.connection(a, b)
            self.conns[n] = Conn(a, b, rng.gauss(0, self.cfg.weight_init_sd),
                                 True, n)
            return True
        return False

    def mutate_add_node(self, innov: Innovations, rng: random.Random):
        live = [c for c in self.conns.values() if c.enabled]
        if not live:
            return False
        c = rng.choice(live)
        c.enabled = False
        new = innov.split(c.innov)
        self.nodes[new] = (HIDDEN, 0.0)
        n1 = innov.connection(c.a, new)
        n2 = innov.connection(new, c.b)
        # weight 1 into the new node and the old weight out of it, so the
        # network's function is unchanged at the moment of the mutation
        self.conns[n1] = Conn(c.a, new, 1.0, True, n1)
        self.conns[n2] = Conn(new, c.b, c.w, True, n2)
        return True

    # --------------------------------------------------------------- crossover
    @staticmethod
    def crossover(fitter: "Genome", other: "Genome",
                  rng: random.Random, equal: bool = False) -> "Genome":
        cfg = fitter.cfg
        child = Genome(cfg)
        child.nodes = dict(fitter.nodes)
        for innov, c in fitter.conns.items():
            m = other.conns.get(innov)
            if m is None:
                pick = c
            else:
                pick = c if rng.random() < 0.5 else m
                if (not c.enabled or not m.enabled) and rng.random() < 0.75:
                    pick = Conn(pick.a, pick.b, pick.w, False, pick.innov)
                else:
                    pick = Conn(pick.a, pick.b, pick.w, True, pick.innov)
            child.conns[innov] = Conn(pick.a, pick.b, pick.w, pick.enabled, innov)
        if equal:
            for innov, c in other.conns.items():
                if innov not in child.conns:
                    child.conns[innov] = Conn(c.a, c.b, c.w, c.enabled, innov)
        for c in child.conns.values():
            for nid in (c.a, c.b):
                if nid not in child.nodes:
                    child.nodes[nid] = other.nodes.get(nid, (HIDDEN, 0.0))
        return child

    # ----------------------------------------------------------- compatibility
    def distance(self, other: "Genome") -> float:
        cfg = self.cfg
        a, b = self.conns, other.conns
        if not a and not b:
            return 0.0
        max_a = max(a) if a else -1
        max_b = max(b) if b else -1
        cutoff = min(max_a, max_b)
        excess = disjoint = matching = 0
        wdiff = 0.0
        for innov in set(a) | set(b):
            in_a, in_b = innov in a, innov in b
            if in_a and in_b:
                matching += 1
                wdiff += abs(a[innov].w - b[innov].w)
            elif innov > cutoff:
                excess += 1
            else:
                disjoint += 1
        n = max(len(a), len(b)) if cfg.normalise_by_size else 1
        n = 1 if n < 20 else n
        wbar = wdiff / matching if matching else 0.0
        return (cfg.c_excess * excess / n + cfg.c_disjoint * disjoint / n
                + cfg.c_weight * wbar)

    def complexity(self):
        return len(self.hidden()), sum(1 for c in self.conns.values() if c.enabled)


class Network:
    """Feed-forward evaluation of a genome, with nodes in topological order."""

    def __init__(self, genome: Genome):
        cfg = genome.cfg
        self.inputs = sorted(genome.inputs())
        self.outputs = sorted(genome.outputs())
        self.bias = {nid: b for nid, (t, b) in genome.nodes.items()}
        self.incoming = {nid: [] for nid in genome.nodes}
        for c in genome.conns.values():
            if c.enabled and c.a in genome.nodes and c.b in genome.nodes:
                self.incoming[c.b].append((c.a, c.w))
        self.act = ACTIVATIONS[cfg.activation]
        self.out_act = ACTIVATIONS[cfg.output_activation]
        self.order = self._topological(genome)

    def _topological(self, genome):
        indeg = {nid: 0 for nid in genome.nodes}
        for nid, ins in self.incoming.items():
            indeg[nid] = len(ins)
        ready = [n for n in self.inputs]
        for nid in genome.nodes:
            if indeg[nid] == 0 and nid not in self.inputs:
                ready.append(nid)
        order, seen = [], set()
        outgoing = {nid: [] for nid in genome.nodes}
        for nid, ins in self.incoming.items():
            for a, _ in ins:
                outgoing[a].append(nid)
        deg = dict(indeg)
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
        for nid in genome.nodes:              # unreachable nodes, evaluated last
            if nid not in seen:
                order.append(nid)
        return order

    def __call__(self, values):
        v = {}
        for nid, x in zip(self.inputs, values):
            v[nid] = x
        for nid in self.order:
            if nid in self.inputs:
                continue
            s = self.bias.get(nid, 0.0)
            for a, w in self.incoming[nid]:
                s += v.get(a, 0.0) * w
            v[nid] = (self.out_act if nid in self.outputs else self.act)(s)
        return [v.get(nid, 0.0) for nid in self.outputs]
