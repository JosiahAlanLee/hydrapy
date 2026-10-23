"""Command line:

    hydrapy render patch.py -o out.gif --seconds 6 --fps 30 --size 320x180 [--gpu]
    hydrapy render patch.py -o out.mp4 --seconds 30 --size 1280x720 --gpu
    hydrapy html   patch.py -o patch.html
    hydrapy hydra  patch.py            # print the Hydra JS
    hydrapy glsl   patch.py            # print the fragment shader(s)
    hydrapy live   patch.py            # window (needs moderngl + glfw)
    hydrapy ansi   patch.py            # play it in the terminal
    hydrapy import sketch.js -o patch.py   # Hydra JS -> Python
    hydrapy fxhash patch.py --sdk fxhash.min.js   # an fxhash project folder + upload.zip
    hydrapy render patch.py --fxhash oo…  # any command: the hash (and --fxparam id=value) a Mint previews
    hydrapy ops                        # list every op

A patch file is ordinary Python that uses ``from hydrapy import *`` and calls
``.out()``; it may also be a Hydra ``.js`` sketch, which is imported first.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time


def load_patch(path: str, strict: bool = True):
    """Run a patch file (or import a Hydra .js sketch) into a fresh default patch."""
    from . import default_patch, reset_patch
    patch = reset_patch()
    if path.endswith(".js"):
        from .hydra_import import from_hydra
        with open(path, encoding="utf-8") as f:
            from_hydra(f.read(), patch)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        ns = runpy.run_path(path, run_name="__hydrapy_patch__")
        if not default_patch().active_outputs():
            # maybe the file built an explicit Patch object
            for v in ns.values():
                from .core import Patch
                if isinstance(v, Patch) and v.active_outputs():
                    return v
    if not patch.active_outputs():
        if not strict:
            raise ValueError("the patch assigns nothing to an output: call .out() on a chain")
        sys.exit("the patch assigns nothing to an output: call .out() on a chain")
    return patch


def _mint_env(a):
    """--fxhash / --fxparam become FXHASH / FXPARAMS, which a Mint reads when the patch runs."""
    import json
    if getattr(a, "fxhash", None):
        os.environ["FXHASH"] = a.fxhash
    params = {}
    for kv in getattr(a, "fxparam", None) or []:
        k, _, v = kv.partition("=")
        try:
            params[k] = json.loads(v)
        except ValueError:
            params[k] = v
    if params:
        os.environ["FXPARAMS"] = json.dumps(params)


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hydrapy", description="Hydra video synth in Python")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render to gif/mp4/png frames")
    r.add_argument("patch")
    r.add_argument("-o", "--out", default="out.gif")
    r.add_argument("--seconds", type=float, default=4.0)
    r.add_argument("--fps", type=float, default=30.0)
    r.add_argument("--size", type=parse_size, default=None)
    r.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    r.add_argument("--warmup", type=int, default=0, help="frames to run before recording (feedback patches)")
    r.add_argument("--scale", type=int, default=1, help="nearest-neighbour upscale factor")
    r.add_argument("--gpu", action="store_true", help="render with moderngl instead of NumPy")
    r.add_argument("--nearest", action="store_true", help="nearest-neighbour texture sampling (CPU)")

    h = sub.add_parser("html", help="export a standalone WebGL page")
    h.add_argument("patch")
    h.add_argument("-o", "--out", default=None)
    h.add_argument("--size", type=parse_size, default=None, help="fixed canvas size (default: fills the window)")
    h.add_argument("--pixel-ratio", type=float, default=1.0)

    for name, help_ in (("hydra", "print Hydra JavaScript"), ("glsl", "print the fragment shaders")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("patch")
        s.add_argument("--loose", action="store_true", help="drop ops Hydra lacks instead of failing")

    lv = sub.add_parser("live", help="run in a window (moderngl + glfw)")
    lv.add_argument("patch")
    lv.add_argument("--size", type=parse_size, default=(1280, 720))
    lv.add_argument("--fullscreen", action="store_true")
    lv.add_argument("--no-watch", action="store_true", help="don't reload the patch when the file changes")

    an = sub.add_parser("ansi", help="play in the terminal")
    an.add_argument("patch")
    an.add_argument("--cols", type=int, default=80)
    an.add_argument("--fps", type=float, default=15.0)
    an.add_argument("--seconds", type=float, default=0, help="0 = until interrupted")

    im = sub.add_parser("import", help="convert a Hydra sketch to a Python patch file")
    im.add_argument("sketch")
    im.add_argument("-o", "--out", default=None)

    fh = sub.add_parser("fxhash", help="export an fxhash project folder (index.html, fxhash.min.js, upload.zip)")
    fh.add_argument("patch")
    fh.add_argument("-o", "--out", default=None, help="project folder (default: <patch>_fxhash)")
    fh.add_argument("--sdk", default=None, help="the SDK's fxhash.min.js (npx fxhash create, or npm pack @fxhash/project-sdk)")
    fh.add_argument("--preview", type=float, default=2.0, help="seconds of running before $fx.preview() (the capture)")
    fh.add_argument("--size", type=parse_size, default=None, help="fixed canvas size (default: fills the window)")
    fh.add_argument("--pixel-ratio", type=float, default=1.0)
    fh.add_argument("--no-zip", action="store_true")

    for sp in (r, h, lv, an, fh):
        sp.add_argument("--fxhash", default=None, help="the hash a Mint previews (sets FXHASH)")
        sp.add_argument("--fxparam", action="append", default=[], metavar="ID=VALUE",
                        help="an fx(params) value for a Mint's preview (repeatable; sets FXPARAMS)")

    sub.add_parser("ops", help="list ops")
    sub.add_parser("devices", help="list audio devices and MIDI ports for CvIn/CvOut/Midi")
    nc = sub.add_parser("nca", help="neural cellular automata: train a texture rule from one image (needs torch)")
    nc.add_argument("action", choices=["train"])
    nc.add_argument("image")
    nc.add_argument("-o", "--out", default=None)
    nc.add_argument("--scale", type=int, default=96, help="the image's shorter side in cells: feature size (default 96)")
    nc.add_argument("--iters", type=int, default=2000)
    nc.add_argument("--loss", default="gram", choices=["gram", "sw", "both"])
    nc.add_argument("--hidden", type=int, default=96)
    nc.add_argument("--seed", type=int, default=0)
    nc.add_argument("--steps", type=int, nargs=2, default=(32, 64), metavar=("MIN", "MAX"), help="rollout length range")
    nc.add_argument("--lr", type=float, default=1e-3)
    nc.add_argument("--init", default=None, help="a trained .npz to fine-tune (e.g. with longer --steps 64 128 --lr 3e-4)")

    a = ap.parse_args(argv)

    if a.cmd == "nca":
        from .nca_train import train
        out = a.out or os.path.splitext(os.path.basename(a.image))[0] + ".npz"
        train(a.image, out, scale=a.scale, iters=a.iters, loss=a.loss, hidden=a.hidden, seed=a.seed,
              steps=tuple(a.steps), lr=a.lr, init=a.init)
        print("wrote", out)
        return

    if a.cmd == "devices":
        try:
            from .live import audio_devices
            print("audio devices (sounddevice):"); print(audio_devices())
        except Exception as e:
            print("audio: sounddevice not usable here:", e)
        try:
            import mido
            print("MIDI inputs:", mido.get_input_names())
        except Exception as e:
            print("MIDI: mido/python-rtmidi not usable here:", e)
        return

    if a.cmd == "ops":
        from . import REGISTRY, describe
        for name in REGISTRY:
            print(describe(name))
        return

    if a.cmd == "import":
        from .hydra_import import from_hydra
        from .export import patch_to_python
        from . import reset_patch
        with open(a.sketch, encoding="utf-8") as f:
            patch = from_hydra(f.read(), reset_patch())
        code = patch_to_python(patch)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(code)
            print("wrote", a.out)
        else:
            print(code)
        return

    _mint_env(a)
    patch = load_patch(a.patch)

    if a.cmd == "render":
        ext = os.path.splitext(a.out)[1].lower()
        size = a.size or patch.size
        t0 = time.time()
        if a.gpu:
            from .backend_gl import iter_frames_gl
            frames = iter_frames_gl(patch, seconds=a.seconds, fps=a.fps, size=size, start=a.start, warmup=a.warmup)
        else:
            from .backend_np import iter_frames
            frames = iter_frames(patch, seconds=a.seconds, fps=a.fps, size=size, start=a.start, warmup=a.warmup,
                                   filter="nearest" if a.nearest else "linear")
        from .export import save_frames, save_gif, save_mp4, save_png
        if ext == ".gif":
            save_gif(frames, a.out, fps=a.fps, scale=a.scale)
        elif ext in (".mp4", ".mov", ".webm", ".mkv"):
            track = next((pr for pr in patch.probes if hasattr(pr, "chunks")), None)
            if track is not None:
                track.record = True
                track.keep = max(track.keep, a.seconds + 1)
            save_mp4(frames, a.out, fps=a.fps, scale=a.scale, audio=track)
        elif ext == ".png":
            last = None
            for last in frames:
                pass
            if last is None:
                sys.exit("render duration must contain at least one frame")
            save_png(last, a.out)
        elif ext == "":
            save_frames(frames, a.out)
        else:
            sys.exit(f"unknown output type {ext}")
        print(f"wrote {a.out}: {round(a.seconds * a.fps)} frames at {size[0]}x{size[1]} in {time.time() - t0:.1f}s")

    elif a.cmd == "html":
        from .export import to_html
        out = a.out or os.path.splitext(os.path.basename(a.patch))[0] + ".html"
        to_html(patch, out, title=os.path.basename(a.patch), size=a.size, pixel_ratio=a.pixel_ratio)
        print("wrote", out)

    elif a.cmd == "fxhash":
        from .export import to_fxhash
        folder = a.out or os.path.splitext(os.path.basename(a.patch))[0] + "_fxhash"
        out = to_fxhash(patch, folder, sdk=a.sdk, preview=a.preview, title=os.path.basename(a.patch),
                        size=a.size, pixel_ratio=a.pixel_ratio, zip=not a.no_zip)
        m = out["mint"]
        print("wrote", out["index"] + (" and " + out["zip"] if out["zip"] else ""))
        if m:
            print(f"mint: {m['draws']} draws, {len(m['params'])} params, {len(m['features'])} features; preview hash {m['hash']}")

    elif a.cmd == "hydra":
        from .export import patch_to_hydra
        print(patch_to_hydra(patch, strict=not a.loose))

    elif a.cmd == "glsl":
        from .glsl import compile_patch
        for prog in compile_patch(patch):
            for p in prog.passes:
                print(f"// ---- {prog.output.name}: pass -> {p.target}")
                print(p.source)

    elif a.cmd == "live":
        from .backend_gl import live
        live(patch, size=a.size, fullscreen=a.fullscreen, watch=None if a.no_watch else a.patch,
             loader=lambda: load_patch(a.patch, strict=False))

    elif a.cmd == "ansi":
        from .backend_np import Renderer
        from .export import to_ansi
        import shutil
        cols = min(a.cols, shutil.get_terminal_size().columns)
        rows = max(int(cols * patch.size[1] / patch.size[0] / 2), 1)
        r = Renderer(patch, size=(cols, rows * 2), fps=a.fps)
        sys.stdout.write("\x1b[?25l\x1b[2J")
        start = time.time()
        try:
            while True:
                t = time.time() - start
                if a.seconds and t > a.seconds:
                    break
                frame = r.step(t)
                sys.stdout.write("\x1b[H" + to_ansi(frame, cols=cols, rows=rows, composite=patch.composite_alpha) + "\n")
                sys.stdout.flush()
                time.sleep(max(0.0, 1.0 / a.fps - (time.time() - start - t)))
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\x1b[?25h\x1b[0m\n")


if __name__ == "__main__":
    main()
