"""Shader-variant compiles during a scored scene-list run, headless (EGL or a standalone context).

PYTHONPATH=.:<scenes dir with scenekit.py> python tools/benchmark_variants.py \
    --scenes scenes.json --score score.json --audio mix.wav [--seconds 192] [--prewarm none|scenes|take]

Builds one patch from the scene files (as a Studio live set does), plays the score's lanes and scene cues
through the tracker at ``--fps`` with Studio's live settle, and prints one JSON line per variant compiled
after the prewarm (frame, time, output, milliseconds), then a summary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--scenes', required=True, help='JSON list of {name, path} (or of paths)')
ap.add_argument('--score', required=True)
ap.add_argument('--audio', required=True, help='an audio file standing in for the live input')
ap.add_argument('--seconds', type=float, default=None, help='default: the whole score cycle')
ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--size', default='480x270')
ap.add_argument('--prewarm', choices=['none', 'scenes', 'take'], default='scenes',
                help="scenes: what Studio does on load; take: also GLRenderer.prewarm_take")
ap.add_argument('--settle', type=int, default=None, help='default: LIVE_SETTLE')
ap.add_argument('--quiet', action='store_true')
ap.add_argument('--save', type=int, nargs='*', default=(), help='frames to save as .npy beside --digests')
ap.add_argument('--digests', help='write each frame\'s MD5 here, to compare the images of two runs')
ap.add_argument("--no-hold", dest="hold", action="store_false", help="take: keep settle after the prewarm")
ap.add_argument('--why', action='store_true', help='name the gates that differ from the nearest compiled variant')
args = ap.parse_args()

import hydrapy as h
import scenekit
from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer
from hydrapy.core import default_patch
from hydrapy.studio_tracker import Tracker, cycle_beats, scene_fades

h.set_bpm(json.loads(Path(args.score).read_text()).get('bpm', 120))
ctx = scenekit.Ctx(args.audio, bpm=h.default_patch().bpm, studio=True)
for i, item in enumerate(json.loads(Path(args.scenes).read_text())):
    path = item['path'] if isinstance(item, dict) else item
    mod = scenekit.load_scene(path)
    ctx.seed = 1000 + i
    chain = mod.build(ctx)
    h.scene(item['name'] if isinstance(item, dict) else mod.SCENE['name'])(lambda chain=chain: chain)
h.render(h.o0)

patch = default_patch()
tracker = Tracker(json.loads(Path(args.score).read_text()))
tracker.seek(0., 0.)
tracker.enabled = True
tracker.apply(patch, 0.)
seconds = args.seconds or cycle_beats(tracker.document) * 60 / tracker.document['bpm']
w, hgt = map(int, args.size.split('x'))
r = GLRenderer(patch, size=(w, hgt), fps=args.fps)
r.settle = LIVE_SETTLE if args.settle is None else args.settle

log, frame = [], [0]
compiled = r._compiled


def timed(prog, key):                  # a new variant: generating, compiling and linking its shaders
    new = (prog.output.name, key) not in r._variants
    start = time.perf_counter()
    out = compiled(prog, key)
    if new:
        log.append(dict(frame=frame[0], time=round(frame[0] / args.fps, 3), output=prog.output.name,
                        skipped=len(key or ()), **({'gates': _why(prog.output.name, key)} if args.why else {}), ms=round((time.perf_counter() - start) * 1000, 1)))
    return out


r._compiled = timed


def _why(name, key):
    key = key or frozenset()
    near = min((k or frozenset() for n, k in r._variants if n == name and (k or frozenset()) != key),
               key=lambda k: len(k ^ key))
    return sorted(f"{'+' if id(tr) in key else '-'}{tr.op.name}({v!r})"[:120]
                  for tr, v, _ in r._gates[name] if id(tr) in key ^ near)
t0 = time.perf_counter()
fades = scene_fades(tracker.document, patch.scenes)
warmed = r.prewarm_scenes(fades) if args.prewarm != 'none' else 0
if args.prewarm == 'take':
    warmed += r.prewarm_take(tracker.document, seconds, hold=args.hold)
warm_ms = (time.perf_counter() - t0) * 1000
log.clear()

frames = int(round(seconds * args.fps))
steps, digests = [], []
for frame[0] in range(frames):
    t = frame[0] / args.fps
    tracker.apply(patch, t)
    start = time.perf_counter()
    r.step(t)
    r.ctx.finish()                    # drivers link lazily: the first draw pays for a new program
    steps.append((time.perf_counter() - start) * 1000)
    if args.digests:
        import hashlib
        image = r.read()
        digests.append(hashlib.md5(image.tobytes()).hexdigest())
        if frame[0] in args.save:
            import numpy as np
            np.save(f'{args.digests}.{frame[0]}.npy', image)
    if log and log[-1]['frame'] == frame[0]:
        log[-1]['step_ms'] = round(steps[-1], 1)
        if not args.quiet:
            print(json.dumps(log[-1]), flush=True)
r.ctx.finish()
if args.digests:
    Path(args.digests).write_text('\n'.join(digests) + '\n')
slowest = max(range(len(steps)), key=steps.__getitem__)
steps.sort()
print(json.dumps(dict(prewarm=args.prewarm, settle=r.settle, hold=getattr(r, "hold", False), frames=frames, fades=len(fades),
                      prewarm_compiled=warmed, prewarm_ms=round(warm_ms), variants=len(r._variants),
                      mid_run_compiles=len(log), mid_run_compile_ms=round(sum(e['ms'] for e in log), 1),
                      worst_compile_step_ms=max((e['step_ms'] for e in log), default=0),
                      step_ms_p50=round(steps[len(steps) // 2], 2), step_ms_max=round(steps[-1], 1), slowest_frame=slowest)))
