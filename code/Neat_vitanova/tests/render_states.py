"""Pick the three canonical states from the phase diagram and draw them."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flock import Flock, W, H

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd = json.load(open(os.path.join(HERE, "data", "phase_diagram.json")))
ALI, COH = pd["alignment"], pd["cohesion"]
P, M, D, G = map(np.array, (pd["P"], pd["M"], pd["nnd"], pd["group"]))
mask = D > 5.0          # exclude parameter sets where the group collapses

def cell(i, j, note=""):
    return dict(w_ali=ALI[i], w_coh=COH[j], P=float(P[i, j]), M=float(M[i, j]),
                nnd=float(D[i, j]), group=float(G[i, j]), note=note)

# highest rotation among non-collapsed groups -- reported whatever it turns out
# to be, rather than assumed to be a torus
i, j = np.unravel_index(np.argmax(np.where(mask, M, -9)), M.shape)
rot = cell(i, j, "most rotation found")
k, l = np.unravel_index(np.argmax(np.where(mask, P * (G / G.max()), -9)), P.shape)
school = cell(k, l, "one cohesive polarised group")
a, b = np.unravel_index(np.argmin(np.where(mask, P, 9)), P.shape)
swarm = cell(a, b, "no alignment")
states = {"disordered swarm": swarm, "most rotation found": rot,
          "polarised school": school}
json.dump(states, open(os.path.join(HERE, "data", "flock_states.json"), "w"),
          indent=2)

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.4))
for a, (name, s) in zip(ax, states.items()):
    f = Flock(n=120, seed=3, w_sep=2.0, w_ali=s["w_ali"], w_coh=s["w_coh"],
              r_sep=15.0, r_ali=25.0, r_coh=60.0, noise=0.05)
    for _ in range(1400):
        f.step()
    a.set_facecolor("#05070a")
    a.quiver(f.x, f.y, f.vx, f.vy, color="#38bdf8", scale=32, width=.006,
             headwidth=3)
    a.set_xlim(0, W); a.set_ylim(H, 0); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{name}\n$P$={s['P']:.2f}  $M$={s['M']:.2f}  "
                f"NND={s['nnd']:.1f}  group={s['group']:.0f}/120\n"
                f"$w_{{ali}}$={s['w_ali']}, $w_{{coh}}$={s['w_coh']}", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "s1_states.pdf"))
fig.savefig(os.path.join(HERE, "figures", "s1_states.png"), dpi=170)
print(json.dumps(states, indent=2))
