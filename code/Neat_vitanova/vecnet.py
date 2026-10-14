# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Evaluate one NEAT genome for many agents at once.

The genome is feed-forward, so its nodes can be put in topological order and
each node evaluated as a single vector operation across all agents. This is what
makes evolving controllers for a whole flock affordable: one numpy operation per
node per timestep instead of one Python call per agent.
"""
from __future__ import annotations

import numpy as np

from neat import ACTIVATIONS, Genome


class VectorNetwork:
    def __init__(self, genome: Genome):
        cfg = genome.cfg
        self.inputs = sorted(genome.inputs())
        self.outputs = sorted(genome.outputs())
        self.bias = {n: b for n, (t, b) in genome.nodes.items()}
        self.incoming = {n: [] for n in genome.nodes}
        for c in genome.conns.values():
            if c.enabled and c.a in genome.nodes and c.b in genome.nodes:
                self.incoming[c.b].append((c.a, c.w))
        self.act = ACTIVATIONS[cfg.activation]
        self.out_act = ACTIVATIONS[cfg.output_activation]
        self.order = self._order(genome)
        self._np_act = self._vector(cfg.activation)
        self._np_out = self._vector(cfg.output_activation)

    @staticmethod
    def _vector(name):
        if name == "sigmoid":
            return lambda z: 1.0 / (1.0 + np.exp(-np.clip(4.9 * z, -60, 60)))
        if name == "tanh":
            return lambda z: np.tanh(np.clip(z, -60, 60))
        if name == "relu":
            return lambda z: np.maximum(z, 0.0)
        return lambda z: z

    def _order(self, genome):
        outgoing = {n: [] for n in genome.nodes}
        deg = {n: len(self.incoming[n]) for n in genome.nodes}
        for n, ins in self.incoming.items():
            for a, _ in ins:
                outgoing[a].append(n)
        ready = [n for n in self.inputs]
        ready += [n for n in genome.nodes if deg[n] == 0 and n not in self.inputs]
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
        order += [n for n in genome.nodes if n not in seen]
        return order

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X is (n_agents, n_inputs); returns (n_agents, n_outputs)."""
        n = X.shape[0]
        v = {nid: X[:, k] for k, nid in enumerate(self.inputs)}
        for nid in self.order:
            if nid in v:
                continue
            s = np.full(n, self.bias.get(nid, 0.0))
            for a, w in self.incoming[nid]:
                src = v.get(a)
                if src is not None:
                    s = s + src * w
            v[nid] = (self._np_out if nid in self.outputs
                      else self._np_act)(s)
        return np.column_stack([v.get(nid, np.zeros(n)) for nid in self.outputs])
