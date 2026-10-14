# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Differential adhesion instead of repulsion.

Five recognition mechanisms failed to stop collectives merging, and they failed
the same way: each ADDED a repulsive force that had to overcome cohesion, at
the same 25 px range cohesion acts on.  The summed pull of ~30 members beats a
pairwise push, and a gamma large enough to win tears the collective apart.

Steinberg's differential adhesion hypothesis says tissues stay separate for a
different reason: cells adhere strongly to their own kind and weakly to others,
so the two never mix in the first place.  Nothing has to be overcome.

Here the gestalt badge modulates cohesion and alignment themselves rather than
fighting them.  A stranger simply exerts no pull.  The energetic brake of
crowded.py (contact inhibition of proliferation) is kept underneath.
"""
from __future__ import annotations

import numpy as np

import gestalt as GS
import vitafast as VF


class Adhesion(GS.Gestalt):
    def __init__(self, seed=2024, adh_sigma=0.35, **kw):
        # gamma=0: no repulsion at all, adhesion does the whole job
        kw.setdefault("gamma", 0.0)
        super().__init__(seed=seed, **kw)
        self.adh_sigma = adh_sigma

    def _social_weights(self, mall, d):
        w = mall.astype(float)
        if self.adh_sigma <= 0 or not len(self.badge):
            return w
        k = min(len(self.badge), w.shape[0])
        bd = self._badge_dist()
        sim = np.exp(-bd / self.adh_sigma)
        w[:k, :k] *= sim[:k, :k]
        np.fill_diagonal(w, 1.0)          # a boid always counts itself
        return w
