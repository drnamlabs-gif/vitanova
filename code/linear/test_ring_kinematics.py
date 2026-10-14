# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Is the ring a kinematic structure -- an orbit -- or a static arrangement?

If some members sit in the middle while others circle, then velocity should be
mostly tangential at the rim and small or unstructured at the core. Each member
is decomposed into its radial and tangential components about the cluster
centroid, and binned by radial position.

    tangential fraction = |v . t| / |v|      1 = pure circling
    radial fraction     = |v . r| / |v|      1 = moving straight in or out
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("..", "../fast", "../reproduce"):
    sys.path.insert(0, os.path.join(HERE, d))
import shape                          # noqa: E402
from runfast import clusters          # noqa: E402

SHELLS = np.linspace(0, 1, 6)


def profile(files, xk="_x", yk="_y", vxk="_dx", vyk="_dy", min_annular=0.4):
    tan = [[] for _ in range(5)]
    rad = [[] for _ in range(5)]
    spd = [[] for _ in range(5)]
    n_frames = 0
    for p in files:
        z = np.load(p)
        steps = sorted({int(k.split("_")[1]) for k in z.files
                        if k.startswith("snap_") and k.endswith(xk)})
        for s in steps:
            if f"snap_{s}{vxk}" not in z.files:
                continue
            x, y = z[f"snap_{s}{xk}"], z[f"snap_{s}{yk}"]
            vx, vy = z[f"snap_{s}{vxk}"], z[f"snap_{s}{vyk}"]
            cs = clusters(x, y, min_size=15)
            if not cs:
                continue
            c = cs[0]
            d = shape.annularity(x[c], y[c])
            if d is None or d["annularity"] < min_annular:
                continue          # only look at frames that really are rings
            n_frames += 1
            xb, yb, vxb, vyb = x[c], y[c], vx[c], vy[c]
            cx, cy = xb.mean(), yb.mean()
            rx, ry = xb - cx, yb - cy
            r = np.hypot(rx, ry)
            ok = r > 1e-6
            rhx, rhy = rx[ok] / r[ok], ry[ok] / r[ok]     # radial unit
            thx, thy = -rhy, rhx                          # tangential unit
            v = np.hypot(vxb[ok], vyb[ok])
            vn = np.maximum(v, 1e-9)
            tfrac = np.abs(vxb[ok] * thx + vyb[ok] * thy) / vn
            rfrac = np.abs(vxb[ok] * rhx + vyb[ok] * rhy) / vn
            rn = r[ok] / max(r.max(), 1e-9)
            idx = np.clip(np.digitize(rn, SHELLS) - 1, 0, 4)
            for b in range(5):
                m = idx == b
                if m.any():
                    tan[b].append(float(tfrac[m].mean()))
                    rad[b].append(float(rfrac[m].mean()))
                    spd[b].append(float(v[m].mean()))
    return tan, rad, spd, n_frames


if __name__ == "__main__":
    sets = {
        "linear-tanh controller": sorted(glob.glob(os.path.join(HERE, "data",
                                                                "lin_seed*.npz"))),
        "fixed 11-10-6 network": sorted(glob.glob(os.path.join(
            HERE, "..", "..", "papers", "shared", "data", "seed*.npz"))),
    }
    out = {}
    for name, files in sets.items():
        vxk = "_dx" if name.startswith("linear") else "_dx"
        tan, rad, spd, nf = profile(files, vxk=vxk)
        if nf == 0:
            print(f"{name}: no ring frames with velocity stored")
            continue
        print(f"\n{name}  ({nf} ring frames)")
        print(f"{'shell r/R':<12}{'tangential':>12}{'radial':>9}{'speed':>8}")
        print("-" * 42)
        rows = []
        for b in range(5):
            if not tan[b]:
                continue
            t, r_, s_ = np.mean(tan[b]), np.mean(rad[b]), np.mean(spd[b])
            rows.append(dict(shell=f"{SHELLS[b]:.1f}-{SHELLS[b+1]:.1f}",
                             tangential=round(float(t), 3),
                             radial=round(float(r_), 3), speed=round(float(s_), 3)))
            print(f"{SHELLS[b]:.1f}-{SHELLS[b+1]:.1f}   {t:>12.3f}{r_:>9.3f}{s_:>8.3f}")
        out[name] = rows
        if len(rows) >= 2:
            print(f"  core to rim, tangential fraction: "
                  f"{rows[0]['tangential']:.3f} -> {rows[-1]['tangential']:.3f}")
    json.dump(out, open(os.path.join(HERE, "ring_kinematics.json"), "w"), indent=2)
    print("\nan unstructured velocity field gives 0.64 tangential by chance "
          "(mean |cos| over random angles = 2/pi)")
