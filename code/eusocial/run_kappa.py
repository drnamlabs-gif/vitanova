"""Does a non-reproducing strategy invade a population of reproducers?

Two arms, identical except that in the control the defence investment buys
nothing: predators ignore it and it confers no escape. If kappa falls and its
distribution becomes bimodal only when defence pays, reproductive division of
labour has evolved. If kappa behaves the same in both arms, it has not.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "fast"))
import kappa as K  # noqa: E402


def dip(v):
    """Crude bimodality: variance explained by a two-cluster split."""
    v = np.sort(np.asarray(v, float))
    if len(v) < 10 or v.std() == 0:
        return 0.0
    best = 0.0
    tot = v.var()
    for i in range(3, len(v) - 3):
        a, b = v[:i], v[i:]
        within = (len(a) * a.var() + len(b) * b.var()) / len(v)
        best = max(best, 1 - within / max(tot, 1e-12))
    return float(best)


def one(a):
    pays, seed, steps = a
    w = K.Eusocial(seed=seed, MAX_BOIDS=600, defence_pays=pays)
    traj = []
    t0 = time.time()
    for i in range(steps):
        if i % 500 == 0 and w.n:
            traj.append((i, w.n, float(w.kappa.mean()), float(w.D.mean())))
        if w.n == 0:
            break
        w.step()
    if w.n < 10:
        return dict(pays=pays, seed=seed, survived=False,
                    wall=round(time.time() - t0, 1))
    k = w.kappa
    off = {}
    for _, p, _ in w.births:
        off[int(p)] = off.get(int(p), 0) + 1
    kids = np.array([off.get(int(u), 0) for u in w.uid], float)
    r = float(np.corrcoef(k, kids)[0, 1]) if k.std() > 0 and kids.std() > 0 else 0.0
    return dict(pays=pays, seed=seed, survived=True, n=int(w.n),
                kappa_mean=round(float(k.mean()), 3),
                kappa_sd=round(float(k.std()), 3),
                kappa_min=round(float(k.min()), 3),
                bimodality=round(dip(k), 3),
                corr_kappa_offspring=round(r, 3),
                D_mean=round(float(w.D.mean()), 1),
                deters=w.deters, escapes=w.escapes,
                traj=traj, wall=round(time.time() - t0, 1))


if __name__ == "__main__":
    steps = int(os.environ.get("STEPS", 8000))
    seeds = list(range(1, int(os.environ.get("NSEED", 16)) + 1))
    jobs = [(p, s, steps) for p in (True, False) for s in seeds]
    print(f"{len(jobs)} runs, initial kappa 0.87 (everyone a reproducer)",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            rows.append(f.result())
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "kappa_runs.json"), "w"), indent=2)
    print(f"\n{'defence pays':>13}{'survived':>11}{'kappa mean':>12}"
          f"{'kappa sd':>10}{'min kappa':>11}{'bimodality':>12}{'corr':>8}")
    print("-" * 78)
    for p in (True, False):
        g = [r for r in rows if r["pays"] == p]
        s = [r for r in g if r["survived"]]
        if not s:
            print(f"{str(p):>13}{f'0/{len(g)}':>11}")
            continue
        f_ = lambda key: np.mean([r[key] for r in s])
        print(f"{str(p):>13}{f'{len(s)}/{len(g)}':>11}{f_('kappa_mean'):>12.3f}"
              f"{f_('kappa_sd'):>10.3f}{f_('kappa_min'):>11.3f}"
              f"{f_('bimodality'):>12.3f}{f_('corr_kappa_offspring'):>8.2f}")
    print("\nstarting value 0.870;  a fall plus bimodality would mean castes")
