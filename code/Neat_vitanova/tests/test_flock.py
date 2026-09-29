"""Does the flocking substrate reproduce the known collective states?"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flock import Flock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIMES = {
    "disordered swarm":  dict(w_sep=1.0, w_ali=0.1, w_coh=0.6, r_ali=8.0),
    "rotating torus":    dict(w_sep=1.0, w_ali=0.5, w_coh=1.2, r_ali=15.0),
    "polarised school":  dict(w_sep=0.6, w_ali=2.5, w_coh=0.8, r_ali=60.0),
}


def run(params, steps=1200, seed=0, burn=600, snap_at=()):
    f = Flock(n=120, seed=seed, **params)
    P, M, nnd, snaps = [], [], [], {}
    for i in range(steps):
        if i in snap_at:
            snaps[i] = f.snapshot()
        f.step()
        if i >= burn:
            p, m = f.order_parameters()
            P.append(p); M.append(m); nnd.append(f.nnd())
    return dict(P=float(np.mean(P)), P_sd=float(np.std(P)),
                M=float(np.mean(M)), M_sd=float(np.std(M)),
                nnd=float(np.mean(nnd)),
                clark_evans=float(f.clark_evans())), snaps


if __name__ == "__main__":
    out = {}
    print(f"{'regime':<20}{'polarisation':>14}{'angular mom.':>14}{'NND':>8}{'C-E':>7}")
    for name, p in REGIMES.items():
        vals = [run(p, seed=s)[0] for s in range(5)]
        agg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        out[name] = dict(params=p, **agg)
        print(f"{name:<20}{agg['P']:>9.2f}±{agg['P_sd']:<4.2f}"
              f"{agg['M']:>9.2f}±{agg['M_sd']:<4.2f}"
              f"{agg['nnd']:>8.1f}{agg['clark_evans']:>7.2f}")
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "flock_regimes.json"), "w"),
              indent=2)
    ok = (out["polarised school"]["P"] > 0.7
          and out["rotating torus"]["M"] > out["polarised school"]["M"]
          and out["disordered swarm"]["P"] < 0.5)
    print("\nthree distinct states recovered:", ok)
    sys.exit(0 if ok else 1)
