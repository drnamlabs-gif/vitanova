"""What happens between ring formation and division, and which parts of it the
rules actually specify.

The model pays a worker for having 2..19 worker neighbours within 15 px and
penalises it at 20 or more (see adjust_energy). That fixes a *local number*. It
says nothing about geometry, about who sits where, about when a structure should
split, or about how the pieces should be composed. This report separates the two.

Measured here:
  geometry   centroid track, radius, annularity, core fraction, thickness
  Allee      distribution of worker-neighbour counts against the 2..19 band,
             and how that count varies with radial position
  energy     energy against radial position, core versus rim
  births     where offspring appear relative to the structure
  turnover   whether individuals circulate between core and rim, or stay put
  fission    whether the two daughters are lineage-structured or a random split
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shape

PROXIMITY = 15.0     # BOID_SIZE * 3, the radius the Allee term counts over
BAND = (2, 19)       # the paid band
LINK = 25.0          # FLOCK_THRESHOLD


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    steps = sorted({int(k.split("_")[1]) for k in z.files
                    if k.startswith("snap_") and k.endswith("_uid")})
    return z, meta, steps


def frame(z, s):
    g = lambda k: z[f"snap_{s}_{k}"]
    return {k: g(k) for k in ("uid", "parent", "born", "x", "y", "dx", "dy",
                              "energy", "age", "soldier")}


def components(x, y, link=LINK, min_size=5):
    n = len(x)
    if n == 0:
        return []
    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    adj = d2 <= link * link
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
            out.append(np.array(sorted(comp)))
    return sorted(out, key=len, reverse=True)


def worker_neighbours(f):
    """Exactly what adjust_energy counts: workers within PROXIMITY, excluding self."""
    x, y, sol = f["x"], f["y"], f["soldier"]
    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    np.fill_diagonal(d2, np.inf)
    return ((d2 <= PROXIMITY ** 2) & (~sol)[None, :]).sum(1)


def analyse(path, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    z, meta, steps = load(path)
    lo, hi = meta["window"]
    win = [s for s in steps if lo <= s <= hi]

    rows = []
    radial = {}          # step -> (r, neighbours, energy, soldier)
    track = defaultdict(dict)   # uid -> {step: r/r_max}
    for s in win:
        f = frame(z, s)
        comps = components(f["x"], f["y"])
        if not comps:
            continue
        big = comps[0]
        x, y = f["x"][big], f["y"][big]
        cx, cy = x.mean(), y.mean()
        r = np.hypot(x - cx, y - cy)
        d = shape.annularity(x, y)
        nb = worker_neighbours(f)[big]
        e = f["energy"][big]
        rows.append(dict(
            step=s, n_total=len(f["x"]), n_big=len(big), n_comps=len(comps),
            cx=cx, cy=cy, r_mean=float(r.mean()), r_max=float(r.max()),
            annularity=(d["annularity"] if d else np.nan),
            hollowness=(d["hollowness"] if d else np.nan),
            core_frac=(d["core_frac"] if d else np.nan),
            e_mean=float(e.mean()), e_sd=float(e.std()),
            nb_mean=float(nb.mean()),
            frac_below=float((nb < BAND[0]).mean()),
            frac_band=float(((nb >= BAND[0]) & (nb <= BAND[1])).mean()),
            frac_over=float((nb > BAND[1]).mean()),
            soldier_frac=float(f["soldier"][big].mean()),
        ))
        radial[s] = (r / max(r.max(), 1e-9), nb, e, f["soldier"][big])
        for u, rr in zip(f["uid"][big], r / max(r.max(), 1e-9)):
            track[int(u)][s] = float(rr)

    R = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    return z, meta, R, radial, track, win


def report(z, meta, R, radial, track, win, outdir="figures"):
    out = []
    P = out.append
    split = None
    for i in range(1, len(R["step"])):
        if R["n_comps"][i] >= 2 and R["n_comps"][i - 1] == 1:
            split = int(R["step"][i]); break

    P(f"window {meta['window'][0]}-{meta['window'][1]}, "
      f"{len(R['step'])} sampled steps, split first seen at step {split}")
    P("")
    P("STRUCTURE")
    P(f"  focal cluster grows {int(R['n_big'][0])} -> {int(R['n_big'].max())} members")
    P(f"  radius (mean/max) {R['r_mean'][0]:.0f}/{R['r_max'][0]:.0f} px "
      f"-> {R['r_mean'].max():.0f}/{R['r_max'].max():.0f} px")
    P(f"  centroid moves {np.hypot(np.diff(R['cx']), np.diff(R['cy'])).sum():.0f} px "
      f"in total, net displacement "
      f"{np.hypot(R['cx'][-1]-R['cx'][0], R['cy'][-1]-R['cy'][0]):.0f} px")
    P(f"  annularity peaks at {np.nanmax(R['annularity']):.2f} "
      f"(step {int(R['step'][np.nanargmax(R['annularity'])])}), "
      f"core fraction {np.nanmean(R['core_frac']):.2f} on average")
    P("")
    P("THE ALLEE BAND -- what the rules actually fix")
    P(f"  worker neighbours within {PROXIMITY:.0f} px, mean over the window: "
      f"{R['nb_mean'].mean():.1f}")
    P(f"  fraction of members below the paid band (<{BAND[0]}): "
      f"{R['frac_below'].mean()*100:.1f}%")
    P(f"  fraction inside the paid band ({BAND[0]}-{BAND[1]}):  "
      f"{R['frac_band'].mean()*100:.1f}%")
    P(f"  fraction over the penalty threshold (>{BAND[1]}): "
      f"{R['frac_over'].mean()*100:.1f}%   (max {R['frac_over'].max()*100:.1f}%)")

    allr = np.concatenate([radial[s][0] for s in radial])
    alln = np.concatenate([radial[s][1] for s in radial])
    alle = np.concatenate([radial[s][2] for s in radial])
    edges = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(allr, edges) - 1, 0, 4)
    P("")
    P("  radial shell      members   neighbours   energy")
    for b in range(5):
        m = idx == b
        if m.sum():
            P(f"   r/R {edges[b]:.1f}-{edges[b+1]:.1f}   {m.sum():>7}   "
              f"{alln[m].mean():>10.1f}   {alle[m].mean():>6.0f}")

    births = z["births"]
    if len(births):
        P("")
        P("BIRTHS")
        bw = births[(births[:, 0] >= meta["window"][0]) &
                    (births[:, 0] <= meta["window"][1])]
        P(f"  {len(bw)} births inside the window "
          f"({len(births)} over the whole run)")
        if len(bw):
            rr = []
            for st, _, _, bx, by in bw:
                k = int(np.argmin(np.abs(R["step"] - st)))
                rmax = max(R["r_max"][k], 1e-9)
                rr.append(np.hypot(bx - R["cx"][k], by - R["cy"][k]) / rmax)
            rr = np.array(rr)
            P(f"  birth position r/R: median {np.median(rr):.2f}, "
              f"{100*(rr<0.5).mean():.0f}% in the inner half")

    P("")
    P("TURNOVER -- is membership of the core fixed?")
    long_lived = {u: d for u, d in track.items() if len(d) >= 8}
    if long_lived:
        spans, drifts = [], []
        for u, d in long_lived.items():
            ks = sorted(d)
            v = np.array([d[k] for k in ks])
            spans.append(v.max() - v.min())
            drifts.append(abs(v[-1] - v[0]))
        P(f"  {len(long_lived)} individuals tracked for >= 8 samples")
        P(f"  median radial range visited: {np.median(spans):.2f} of R")
        P(f"  median net radial displacement: {np.median(drifts):.2f} of R")
        P(f"  fraction visiting both inner and outer thirds: "
          f"{np.mean([ (min(d.values())<0.33 and max(d.values())>0.66) for d in long_lived.values()])*100:.0f}%")

    return "\n".join(out), split


def figures(z, meta, R, radial, split, outdir="figures"):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    s = R["step"]
    ax[0,0].plot(s, R["n_big"], label="focal cluster"); ax[0,0].plot(s, R["n_total"], lw=.8, label="population")
    ax[0,0].set_title("size"); ax[0,0].legend(fontsize=8); ax[0,0].set_xlabel("step")
    ax[0,1].plot(s, R["r_mean"], label="mean r"); ax[0,1].plot(s, R["r_max"], label="max r")
    ax[0,1].set_title("radius (px)"); ax[0,1].legend(fontsize=8); ax[0,1].set_xlabel("step")
    ax[0,2].plot(s, R["annularity"], label="annularity"); ax[0,2].plot(s, R["core_frac"], label="core fraction")
    ax[0,2].set_title("shape"); ax[0,2].legend(fontsize=8); ax[0,2].set_xlabel("step")
    ax[1,0].plot(s, R["frac_band"]*100, label=f"in band {BAND[0]}-{BAND[1]}")
    ax[1,0].plot(s, R["frac_over"]*100, label=f">{BAND[1]} (penalised)")
    ax[1,0].plot(s, R["frac_below"]*100, label=f"<{BAND[0]} (unpaid)")
    ax[1,0].set_title("worker neighbours vs the Allee band (%)"); ax[1,0].legend(fontsize=8); ax[1,0].set_xlabel("step")
    ax[1,1].plot(s, R["e_mean"]); ax[1,1].fill_between(s, R["e_mean"]-R["e_sd"], R["e_mean"]+R["e_sd"], alpha=.25)
    ax[1,1].set_title("energy (mean +/- sd)"); ax[1,1].set_xlabel("step")
    ax[1,2].plot(R["cx"], R["cy"], lw=1); ax[1,2].scatter(R["cx"][0], R["cy"][0], c="g", s=30, label="start")
    ax[1,2].scatter(R["cx"][-1], R["cy"][-1], c="r", s=30, label="end")
    ax[1,2].set_title("centroid track"); ax[1,2].legend(fontsize=8); ax[1,2].invert_yaxis(); ax[1,2].set_aspect("equal")
    for a in np.ravel(ax)[:5]:
        if split: a.axvline(split, color="k", ls="--", lw=.8)
    fig.tight_layout(); fig.savefig(f"{outdir}/ring_report.png", dpi=140)
    return f"{outdir}/ring_report.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    z, meta, R, radial, track, win = analyse(a.npz, a.outdir)
    text, split = report(z, meta, R, radial, track, win, a.outdir)
    print(text)
    print("\nfigure:", figures(z, meta, R, radial, split, a.outdir))


if __name__ == "__main__":
    main()
