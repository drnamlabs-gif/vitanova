# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Build a standalone HTML viewer from a run's stored snapshots.

Replaces watching pygame: the simulation has already run headless, so this is
just a scrubber over saved state. Opens offline, no server, no dependencies.
Controls: play/pause, frame slider, speed, and a zoom that follows the densest
cluster so the ring is actually visible.
"""

import argparse
import json
import os

import numpy as np

HTML = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0b0d10;color:#d8dee9;
      font:13px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
 header{padding:10px 16px;border-bottom:1px solid #1e242c;display:flex;
        gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:14px;margin:0;font-weight:600}
 .meta{color:#7a8794}
 main{padding:12px 16px;max-width:1200px}
 canvas{background:#000;border:1px solid #1e242c;border-radius:4px;
        width:100%;height:auto;display:block}
 .controls{display:flex;gap:12px;align-items:center;margin:12px 0;flex-wrap:wrap}
 button{background:#1b2129;border:1px solid #2b3440;color:#d8dee9;
        padding:5px 12px;border-radius:4px;cursor:pointer;font:inherit}
 button:hover{background:#243040}
 input[type=range]{flex:1;min-width:200px}
 label{display:flex;gap:6px;align-items:center;color:#9aa7b4}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
        gap:8px;margin-top:12px}
 .stat{background:#12161c;border:1px solid #1e242c;border-radius:4px;padding:8px 10px}
 .stat b{display:block;font-size:17px;font-weight:600;color:#e8eef5}
 .stat span{color:#7a8794;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 .legend{display:flex;gap:14px;color:#9aa7b4;font-size:12px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;
     vertical-align:middle;margin-right:4px}
</style>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
  <div class="legend">
    <span><i class="sw" style="background:#2ecc40"></i>worker</span>
    <span><i class="sw" style="background:#0074d9"></i>soldier</span>
    <span><i class="sw" style="background:#ff4136"></i>predator</span>
  </div>
</header>
<main>
  <canvas id="c" width="1200" height="900"></canvas>
  <div class="controls">
    <button id="play">play</button>
    <input type="range" id="t" min="0" max="0" value="0">
    <label>speed <input type="range" id="sp" min="1" max="30" value="8"
           style="width:90px;min-width:0;flex:none"></label>
    <label><input type="checkbox" id="zoom"> follow cluster</label>
  </div>
  <div class="stats">
    <div class="stat"><span>step</span><b id="s-step">-</b></div>
    <div class="stat"><span>boids</span><b id="s-n">-</b></div>
    <div class="stat"><span>soldier %</span><b id="s-sol">-</b></div>
    <div class="stat"><span>flocks</span><b id="s-fl">-</b></div>
    <div class="stat"><span>largest flock</span><b id="s-lg">-</b></div>
    <div class="stat"><span>ring score</span><b id="s-ring">-</b></div>
  </div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const W = D.width, H = D.height;
const el = id => document.getElementById(id);
let i = 0, playing = false, acc = 0, last = 0;

el('t').max = D.frames.length - 1;

function series(name, step){
  const s = D.series[name]; if(!s) return null;
  let lo = 0, hi = D.series.step.length - 1, best = 0;
  while(lo <= hi){ const m = (lo+hi)>>1;
    if(D.series.step[m] <= step){ best = m; lo = m+1 } else hi = m-1 }
  return s[best];
}

function densest(f, half){
  let best = -1, cx = W/2, cy = H/2;
  for(let k = 0; k < f.x.length; k++){
    let n = 0;
    for(let j = 0; j < f.x.length; j++)
      if(Math.abs(f.x[j]-f.x[k]) <= half && Math.abs(f.y[j]-f.y[k]) <= half) n++;
    if(n > best){ best = n; cx = f.x[k]; cy = f.y[k] }
  }
  return [cx, cy];
}

function draw(){
  const f = D.frames[i];
  const zoom = el('zoom').checked && f.x.length;
  let ox = 0, oy = 0, sc = cv.width / W;
  if(zoom){
    const half = 70, [cx, cy] = densest(f, half);
    sc = cv.width / (2*half); ox = cx - half; oy = cy - half;
    cv.height = Math.round(2*half*sc);
  } else {
    cv.height = Math.round(H * sc);
  }
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cv.width, cv.height);
  const tri = (x, y, dx, dy, size, col) => {
    const a = Math.atan2(dy, dx);
    ctx.fillStyle = col; ctx.beginPath();
    ctx.moveTo((x-ox+Math.cos(a)*size)*sc,       (y-oy+Math.sin(a)*size)*sc);
    ctx.lineTo((x-ox+Math.cos(a+2.5)*size)*sc,   (y-oy+Math.sin(a+2.5)*size)*sc);
    ctx.lineTo((x-ox+Math.cos(a-2.5)*size)*sc,   (y-oy+Math.sin(a-2.5)*size)*sc);
    ctx.closePath(); ctx.fill();
  };
  for(let k = 0; k < f.x.length; k++)
    tri(f.x[k], f.y[k], f.dx[k], f.dy[k], D.boid_size,
        f.sol[k] ? '#0074d9' : '#2ecc40');
  for(let k = 0; k < f.px.length; k++)
    tri(f.px[k], f.py[k], f.pdx[k], f.pdy[k], D.pred_size, '#ff4136');

  el('s-step').textContent = f.step;
  el('s-n').textContent = f.x.length;
  const sol = f.sol.reduce((a,b) => a+b, 0);
  el('s-sol').textContent = f.x.length ? Math.round(100*sol/f.x.length)+'%' : '-';
  const fl = series('n_flocks', f.step), lg = series('largest', f.step),
        rg = series('ring_score', f.step);
  el('s-fl').textContent = fl ?? '-';
  el('s-lg').textContent = lg ?? '-';
  el('s-ring').textContent = rg == null ? '-' : rg.toFixed(2);
  el('t').value = i;
}

function loop(ts){
  if(playing){
    if(!last) last = ts;
    acc += (ts - last) * el('sp').value / 1000; last = ts;
    while(acc >= 1){ acc -= 1; i = (i + 1) % D.frames.length }
    draw();
  } else last = 0;
  requestAnimationFrame(loop);
}
el('play').onclick = () => { playing = !playing;
  el('play').textContent = playing ? 'pause' : 'play' };
el('t').oninput = e => { i = +e.target.value; draw() };
el('zoom').onchange = draw;
addEventListener('keydown', e => {
  if(e.key === 'ArrowRight'){ i = Math.min(i+1, D.frames.length-1); draw() }
  if(e.key === 'ArrowLeft'){ i = Math.max(i-1, 0); draw() }
  if(e.key === ' '){ e.preventDefault(); el('play').click() }
});
draw(); requestAnimationFrame(loop);
</script>
"""


def build(npz_path, out_path, every=1, decimals=1):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    steps = sorted({int(k.split("_")[1]) for k in z.files
                    if k.startswith("snap_") and k.endswith("_x")})
    if not steps:
        raise SystemExit(f"{npz_path} has no snapshots; rerun with --snapshot-every")
    steps = steps[::every]

    r = lambda a: np.round(np.asarray(a, dtype=float), decimals).tolist()
    frames = []
    for s in steps:
        g = lambda k: z[f"snap_{s}_{k}"]
        frames.append({
            "step": s,
            "x": r(g("x")), "y": r(g("y")),
            "dx": r(g("dx")), "dy": r(g("dy")),
            "sol": [int(v) for v in g("soldier")],
            "px": r(g("px")), "py": r(g("py")),
            "pdx": r(g("pdx")), "pdy": r(g("pdy")),
        })

    series = {"step": [int(v) for v in z["step"]]}
    for k in ("n_flocks", "largest", "ring_score", "n_boids"):
        if k in z.files:
            series[k] = np.round(z[k].astype(float), 3).tolist()

    tag = os.path.splitext(os.path.basename(npz_path))[0]
    label = meta.get("label") or f"seed {meta.get('seed')}"
    data = {
        "width": 800, "height": 600, "boid_size": 5, "pred_size": 10,
        "frames": frames, "series": series,
    }
    html = (HTML
            .replace("__TITLE__", f"{tag} — {label}")
            .replace("__META__", f"{len(frames)} frames · steps "
                                 f"{steps[0]}–{steps[-1]}")
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path, len(frames), os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth stored snapshot")
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.npz)[0] + ".html"
    path, n, size = build(args.npz, out, args.every)
    print(f"wrote {path}  ({n} frames, {size/1e6:.1f} MB)")
    print(f"open:  open '{os.path.abspath(path)}'")


if __name__ == "__main__":
    main()
