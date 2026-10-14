# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""What happens to a collective's size after it absorbs another one?

The crowding cost is an energetic brake on reproduction, not a barrier to
merging.  So the claim that size stays bounded has to be checked at the merge
itself: does the merged group stay merged and keep growing, or does it decay
back into the ring window?
"""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "fast"), os.path.join(HERE, "..", "reproduce")):
    sys.path.insert(0, p)
import crowded as CR, nocaste as NC          # noqa: E402
from runfast import clusters                 # noqa: E402

SAMPLE, HORIZON = 25, 60      # every 25 steps, follow 60 samples = 1500 steps


def trace(cost, seeds, steps=30000):
    events = []
    for seed in seeds:
        w = (CR.Crowded(seed=seed, MAX_BOIDS=1200, crowd_beta=0.08, crowd_k=26)
             if cost else NC.NoCaste(seed=seed, MAX_BOIDS=1200))
        hist, prev = [], None
        for i in range(steps):
            if w.n == 0:
                break
            if i % SAMPLE == 0:
                cs = clusters(w.x, w.y, min_size=10)
                sets = [set(w.uid[c].tolist()) for c in cs]
                sizes = [len(c) for c in cs]
                hist.append((i, sets, sizes))
                if prev is not None:
                    for si, c in enumerate(sets):
                        src = [k for k, p in enumerate(prev[1])
                               if len(p & c) >= 0.5 * len(p)]
                        if len(src) >= 2:
                            events.append((len(hist) - 1, seed,
                                           sum(prev[2][k] for k in src),
                                           sizes[si], c))
                prev = (i, sets, sizes)
            w.step()
        # follow each merged group forward by tracking its uid set
        for (hi, sd, pre, post, uids) in [e for e in events if e[1] == seed]:
            traj = []
            for h in range(hi, min(hi + HORIZON, len(hist))):
                _, sets, sizes = hist[h]
                best = max((len(s & uids), sz) for s, sz in zip(sets, sizes)) \
                    if sets else (0, 0)
                traj.append(best[1])
            yield sd, pre, post, traj


for cost, name in ((True, "with crowding cost"),):
    rows = list(trace(cost, [1, 3, 4, 5, 6, 7, 9, 12, 14, 15]))
    if not rows:
        print(f"{name}: no fusion events"); continue
    pre = np.array([r[1] for r in rows], float)
    post = np.array([r[2] for r in rows], float)
    later = np.array([r[3][-1] if r[3] else 0 for r in rows], float)
    peak = np.array([max(r[3]) if r[3] else 0 for r in rows], float)
    print(f"\n{name}: {len(rows)} fusion events over 10 seeds x 30000 steps")
    print(f"  combined size at the merge          : {np.median(post):.0f} "
          f"(range {post.min():.0f}-{post.max():.0f})")
    print(f"  peak size within 1500 steps after   : {np.median(peak):.0f} "
          f"(range {peak.min():.0f}-{peak.max():.0f})")
    print(f"  size 1500 steps after the merge     : {np.median(later):.0f} "
          f"(range {later.min():.0f}-{later.max():.0f})")
    print(f"  merged groups back inside n<=55     : {(later <= 55).mean():.0%}")
    print(f"  merged groups that kept growing >100: {(peak > 100).mean():.0%}")
