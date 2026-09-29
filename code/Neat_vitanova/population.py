"""Generational NEAT population: speciation, fitness sharing, reproduction.

Only used for benchmarking the core. VitaNova itself reproduces agents
asynchronously in the world and does not use this class.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from neat import Config, Genome, Innovations


@dataclass
class Species:
    representative: Genome
    members: list = field(default_factory=list)
    best: float = -1e18
    last_improved: int = 0
    adjusted_sum: float = 0.0


class Population:
    def __init__(self, cfg: Config, size: int = 150, seed: int = 0):
        self.cfg = cfg
        self.size = size
        self.rng = random.Random(seed)
        self.innov = Innovations(start_node=cfg.n_inputs + cfg.n_outputs)
        for i in range(cfg.n_inputs):
            for j in range(cfg.n_outputs):
                self.innov.connection(i, cfg.n_inputs + j)
        self.genomes = [Genome.minimal(cfg, self.innov, self.rng)
                        for _ in range(size)]
        self.species: list[Species] = []
        self.generation = 0
        self.history = []

    # ------------------------------------------------------------- speciation
    def speciate(self):
        """Assign genomes to species, then nudge the threshold towards the
        target species count. A fixed threshold cannot serve genomes of
        different sizes."""
        for s in self.species:
            s.members = []
        for g in self.genomes:
            placed = False
            for s in self.species:
                if g.distance(s.representative) < self.cfg.compat_threshold:
                    s.members.append(g)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(representative=g, members=[g]))
        self.species = [s for s in self.species if s.members]
        for s in self.species:
            s.representative = self.rng.choice(s.members)
        cfg = self.cfg
        if getattr(cfg, "target_species", 0):
            if len(self.species) < cfg.target_species:
                cfg.compat_threshold = max(cfg.threshold_min,
                                           cfg.compat_threshold - cfg.threshold_step)
            elif len(self.species) > cfg.target_species:
                cfg.compat_threshold += cfg.threshold_step

    # ----------------------------------------------------------- reproduction
    def reproduce(self):
        cfg = self.cfg
        for s in self.species:
            best = max(g.fitness for g in s.members)
            if best > s.best + 1e-12:
                s.best, s.last_improved = best, self.generation
            for g in s.members:
                g.adjusted = g.fitness / len(s.members)
            s.adjusted_sum = sum(g.adjusted for g in s.members)

        alive = [s for s in self.species
                 if self.generation - s.last_improved < cfg.stagnation]
        if len(alive) >= cfg.min_species:
            self.species = alive

        total = sum(s.adjusted_sum for s in self.species) or 1e-12
        offspring = [max(1, int(round(self.size * s.adjusted_sum / total)))
                     for s in self.species]

        children = []
        for s, n in zip(self.species, offspring):
            ranked = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            if len(ranked) > 5 and cfg.elitism:
                children.extend(g.copy() for g in ranked[:cfg.elitism])
                n -= cfg.elitism
            cut = max(1, int(len(ranked) * cfg.survival_threshold))
            pool = ranked[:cut]
            for _ in range(max(0, n)):
                if len(pool) > 1 and self.rng.random() < cfg.crossover_rate:
                    a, b = self.rng.sample(pool, 2)
                    if b.fitness > a.fitness:
                        a, b = b, a
                    child = Genome.crossover(a, b, self.rng,
                                             equal=abs(a.fitness - b.fitness) < 1e-12)
                else:
                    child = self.rng.choice(pool).copy()
                child.mutate(self.innov, self.rng)
                children.append(child)
        while len(children) < self.size:
            children.append(self.rng.choice(self.genomes).copy())
        self.genomes = children[:self.size]

    # ------------------------------------------------------------------- run
    def run(self, evaluate, generations: int, target=None, verbose=False):
        for _ in range(generations):
            for g in self.genomes:
                g.fitness = evaluate(g)
            best = max(self.genomes, key=lambda g: g.fitness)
            h, c = best.complexity()
            self.speciate()
            self.history.append(dict(
                generation=self.generation,
                best=best.fitness,
                mean=statistics.fmean(g.fitness for g in self.genomes),
                species=len(self.species),
                hidden=h, connections=c,
                mean_hidden=statistics.fmean(g.complexity()[0] for g in self.genomes),
                mean_conns=statistics.fmean(g.complexity()[1] for g in self.genomes)))
            if verbose and self.generation % 10 == 0:
                print(f"  gen {self.generation:4d}  best {best.fitness:.4f}  "
                      f"species {len(self.species):2d}  "
                      f"hidden {h} conns {c}", flush=True)
            if target is not None and best.fitness >= target:
                return best
            self.reproduce()
            self.generation += 1
        return max(self.genomes, key=lambda g: g.fitness)
