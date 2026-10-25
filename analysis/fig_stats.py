# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Two statistics figures for the draft, from the stored 32-seed census.

fig_population.pdf  (a) deepest lineage per seed; (b) parent-offspring
                    regression of size at fission (the heredity result).
fig_selection.pdf   (a) offspring-number distribution; (b) standardised
                    selection differential per seed; (c) P(child divides)
                    against parent size at fission; (d) size at fission along
                    the lineage (the response).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

CENSUS = os.path.join(_HERE, "..", "results", "selection_census.json")
FIGDIR = os.path.join(_HERE, "..", "paper", "figures")
STEPS = 12000


def load():
    rows = json.load(open(CENSUS))
    by_id = {(r["seed"], r["cid"]): r for r in rows}
    depth = {}

    def get_depth(key):
        if key in depth:
            return depth[key]
        r = by_id[key]
        d = 0 if r["parent"] is None else get_depth((r["seed"], r["parent"])) + 1
        depth[key] = d
        return d

    for k in by_id:
        get_depth(k)
    waits = [r["fission"] - r["birth"] for r in rows if r["fission"] is not None]
    t95 = float(np.percentile(waits, 95))
    score = [r for r in rows
             if r["fission"] is not None or r["birth"] <= STEPS - t95]
    return rows, by_id, depth, score


def save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(FIGDIR, name + ext))
    print("saved:", name)


def fig_population(rows, by_id, depth):
    style.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                               gridspec_kw=dict(wspace=0.3))
    # (a) deepest lineage per seed
    seeds = sorted(set(r["seed"] for r in rows))
    dmax = {s: 0 for s in seeds}
    for (s, c), d in depth.items():
        dmax[s] = max(dmax[s], d)
    vals = sorted(dmax.values(), reverse=True)
    a.bar(range(len(vals)), vals, width=0.7, color=style.ACC, lw=0)
    a.set_xlabel("run (sorted)")
    a.set_ylabel("deepest lineage\n(collective generations)")
    a.axhline(3, color=style.MUTED, lw=0.7, ls=":")
    a.text(len(vals) - 0.5, 3.4, "3 generations", color=style.MUTED,
           fontsize=7, ha="right")
    style.panel_label(a, "a")

    # (b) heredity: parent vs child size at fission
    px, cx = [], []
    for r in rows:
        if r["fission"] is None or r["parent"] is None:
            continue
        p = by_id.get((r["seed"], r["parent"]))
        if p is None or p["fission_size"] is None:
            continue
        px.append(p["fission_size"])
        cx.append(r["fission_size"])
    px, cx = np.array(px, float), np.array(cx, float)
    b.scatter(px, cx, s=6, c=style.ACC, alpha=0.45, lw=0)
    lo, hi = min(px.min(), cx.min()) * 0.9, max(px.max(), cx.max()) * 1.1
    b.plot([lo, hi], [lo, hi], color=style.GRID, lw=0.8, zorder=0)
    k, m = np.polyfit(px, cx, 1)
    xs = np.array([px.min(), px.max()])
    b.plot(xs, k * xs + m, color=style.ACC2, lw=1.3)
    b.text(0.04, 0.93, f"slope = {k:+.2f}   n = {len(px)}",
           transform=b.transAxes, fontsize=8, color=style.ACC2)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("parent size at its fission")
    b.set_ylabel("offspring size at\nits own fission")
    style.panel_label(b, "b")
    save(fig, "fig_population")


def fig_selection(rows, by_id, depth, score):
    style.use()
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.6),
                             gridspec_kw=dict(wspace=0.45))
    a, b, c, d = axes

    # (a) fitness distribution
    w = np.array([r["n_children"] for r in score])
    ks = sorted(set(w.tolist()))
    a.bar([str(k) for k in ks], [int((w == k).sum()) for k in ks],
          color=style.ACC, width=0.6, lw=0)
    a.set_xlabel("offspring collectives")
    a.set_ylabel("collectives")
    style.panel_label(a, "a")

    # (b) per-seed standardised selection differential
    def sdiff(rr, trait):
        z = np.array([r[trait] for r in rr], float)
        ww = np.array([r["n_children"] for r in rr], float)
        if len(z) < 5 or z.std() == 0 or ww.mean() == 0:
            return None
        return float(np.cov((z - z.mean()) / z.std(), ww / ww.mean())[0, 1])

    per = {}
    for r in score:
        per.setdefault(r["seed"], []).append(r)
    for j, (trait, col) in enumerate((("birth_size", style.ACC),
                                      ("max_size", style.ACC3))):
        vals = [v for rr in per.values() if (v := sdiff(rr, trait)) is not None]
        xj = j + np.random.default_rng(j).uniform(-0.1, 0.1, len(vals))
        b.scatter(xj, vals, s=10, c=col, alpha=0.7, lw=0)
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        b.errorbar([j], [m], yerr=[1.96 * se], fmt="_", color=style.INK,
                   ms=16, capsize=3, lw=1.2, zorder=3)
    b.axhline(0, color=style.MUTED, lw=0.7)
    b.set_xticks([0, 1], ["size at\nbirth", "peak\nsize"])
    b.set_ylabel("selection differential $S'$\n(per seed, standardised)")
    b.set_xlim(-0.5, 1.5)
    style.panel_label(b, "b")

    # (c) P(child divides) by parent size at fission (quintiles)
    xx, yy = [], []
    for r in score:
        if r["parent"] is None:
            continue
        p = by_id.get((r["seed"], r["parent"]))
        if p is None or p["fission_size"] is None:
            continue
        xx.append(p["fission_size"])
        yy.append(1.0 if r["fission"] is not None else 0.0)
    xx, yy = np.array(xx, float), np.array(yy, float)
    qs = np.quantile(xx, np.linspace(0, 1, 6))
    mids, ps, ns = [], [], []
    for i in range(5):
        m = (xx >= qs[i]) & (xx <= qs[i + 1] if i == 4 else xx < qs[i + 1])
        mids.append(np.median(xx[m]))
        ps.append(yy[m].mean())
        ns.append(int(m.sum()))
    err = [1.96 * np.sqrt(p_ * (1 - p_) / n_) for p_, n_ in zip(ps, ns)]
    c.errorbar(mids, ps, yerr=err, fmt="o-", color=style.ACC, ms=4,
               capsize=2.5, lw=1.2)
    c.set_xscale("log")
    c.set_ylim(0, 1)
    c.set_xlabel("parent size at fission")
    c.set_ylabel("P(offspring divides)")
    c.text(0.04, 0.92, "within-run r = +0.18\nperm. p < 0.0002",
           transform=c.transAxes, fontsize=7.5, va="top")
    style.panel_label(c, "c")

    # (d) response: size at fission along the lineage
    by_gen = {}
    for r in rows:
        if r["fission_size"] is None:
            continue
        g = depth[(r["seed"], r["cid"])]
        by_gen.setdefault(g, []).append(r["fission_size"])
    gens = sorted(g for g in by_gen if len(by_gen[g]) >= 5)
    med = [np.median(by_gen[g]) for g in gens]
    q1 = [np.percentile(by_gen[g], 25) for g in gens]
    q3 = [np.percentile(by_gen[g], 75) for g in gens]
    d.fill_between(gens, q1, q3, color=style.FILL, lw=0)
    d.plot(gens, med, color=style.ACC, lw=1.3)
    d.set_xlabel("collective generation")
    d.set_ylabel("size at fission\n(median, IQR)")
    d.text(0.04, 0.97, "raw rise tracks the run's own\nexpansion; no residual trend\n"
           "after time control (p = 0.40)",
           transform=d.transAxes, fontsize=7, va="top")
    style.panel_label(d, "d")
    save(fig, "fig_selection")


def main():
    rows, by_id, depth, score = load()
    fig_population(rows, by_id, depth)
    fig_selection(rows, by_id, depth, score)


if __name__ == "__main__":
    main()
