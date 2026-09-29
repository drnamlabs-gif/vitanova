"""Do the NEAT runs show group-level reproduction, or only ring structure?

Same detector used elsewhere in this project: a sustained single cluster that is
growing, followed by two or more sustained clusters that together keep most of
the parent's membership. Cluster series are recomputed from the stored
snapshots so that runs saved with different keys are treated identically."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "reproduce"))
import shape                                  # noqa: E402
from run_vita_neat import clusters            # noqa: E402

MIN_PARENT, MIN_CHILD, HOLD, KEEP = 15, 6, 2, 0.6


def series_from_snaps(z):
    steps = sorted({int(k.split("_")[1]) for k in z.files
                    if k.startswith("snap_") and k.endswith("_x")})
    S, N, NC, LG, AN = [], [], [], [], []
    for s in steps:
        x, y = z[f"snap_{s}_x"], z[f"snap_{s}_y"]
        cs = clusters(x, y, min_size=10)
        big = cs[0] if cs else None
        d = shape.annularity(x[big], y[big]) if big is not None else None
        S.append(s); N.append(len(x)); NC.append(len(cs))
        LG.append(len(big) if big is not None else 0)
        AN.append(d["annularity"] if d else 0.0)
    return (np.array(S), np.array(N), np.array(NC), np.array(LG), np.array(AN))


def sustained(mask, hold=HOLD):
    out, run = [], 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= hold:
            out.append(i)
    return out


def detect(S, N, NC, LG):
    single = sustained((NC == 1) & (LG >= MIN_PARENT))
    for i in single:
        j = max(0, i - 2 * HOLD)
        if LG[i] <= LG[j]:
            continue
        after = sustained(NC[i:] >= 2)
        if not after:
            continue
        k = i + after[0]
        if N[k] >= KEEP * LG[i] and LG[k] >= MIN_CHILD:
            return dict(division_at=int(S[k]), parent=int(LG[i]),
                        parent_at=int(S[i]), children_total=int(N[k]))
    return None


if __name__ == "__main__":
    files = (sorted(glob.glob(os.path.join(HERE, "data", "neat_runs", "*.npz")))
             + sorted(glob.glob(os.path.join(HERE, "data", "hidden_runs", "*.npz"))))
    rows = []
    print(f"{'run':<26}{'snaps':>6}{'peak n':>8}{'largest':>8}"
          f"{'annular':>9}{'division':>10}{'parent':>8}")
    print("-" * 76)
    for p in files:
        z = np.load(p)
        S, N, NC, LG, AN = series_from_snaps(z)
        if not len(S) or N.max() < 150:
            continue
        d = detect(S, N, NC, LG)
        name = os.path.basename(p)[:-4]
        rows.append(dict(run=name, peak_n=int(N.max()), largest=int(LG.max()),
                         annular=round(float(AN.max()), 2),
                         division=d, best_annular_step=int(S[int(np.argmax(AN))])))
        print(f"{name:<26}{len(S):>6}{N.max():>8}{LG.max():>8}{AN.max():>9.2f}"
              f"{(str(d['division_at']) if d else '—'):>10}"
              f"{(str(d['parent']) if d else '—'):>8}")
    n_div = sum(1 for r in rows if r["division"])
    print(f"\ngroup-level division detected in {n_div}/{len(rows)} NEAT runs")
    if n_div:
        ps = [r["division"]["parent"] for r in rows if r["division"]]
        print(f"parent size at division: {np.mean(ps):.0f} "
              f"(range {min(ps)}-{max(ps)})")
    json.dump(rows, open(os.path.join(HERE, "data", "neat_divisions.json"), "w"),
              indent=2, default=str)
    print("saved data/neat_divisions.json")
