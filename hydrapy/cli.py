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
    hydrapy encoders                   # test hardware/software video encoders

A patch file is ordinary Python that uses ``from hydrapy import *`` and calls
``.out()``; it may also be a Hydra ``.js`` sketch, which is imported first.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time


def load_patch(path: str, strict: bool = True, source=None):
    """Run a patch file (or import a Hydra .js sketch) into a fresh default patch."""
    from . import default_patch
    from . import core
    # Private outputs keep a failed candidate from mutating the running patch.
    patch = core.Patch()
    core._default_patch = patch
    if path.endswith(".js"):
        from .hydra_import import from_hydra
        if source is None:
            with open(path, encoding="utf-8") as f:
                source = f.read()
        from_hydra(source, patch)
    else:
        previous_path = list(sys.path)
        previous_isolation = core._isolate_patch_outputs
        core._isolate_patch_outputs = True
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        try:
            if source is None:
                ns = runpy.run_path(path, run_name="__hydrapy_patch__")
            else:
                ns = {'__file__': path, '__name__': '__hydrapy_patch__'}
                exec(compile(source, path, 'exec'), ns)
        finally:
            sys.path[:] = previous_path
            core._isolate_patch_outputs = previous_isolation
        patch = default_patch()
        if not default_patch().active_outputs():
            # maybe the file built an explicit Patch object
            for v in ns.values():
                from .core import Patch
                if isinstance(v, Patch) and v.active_outputs():
                    patch = core._default_patch = v
                    break
    if not patch.active_outputs():
        if not strict:
            raise ValueError("the patch assigns nothing to an output: call .out() on a chain")
        sys.exit("the patch assigns nothing to an output: call .out() on a chain")
    import json
    from .controls import apply_controls
    patch.declared_display = patch.display.index      # what render() in the file chose, for reloads
    apply_controls(patch, json.loads(os.environ.get("HYDRAPY_PARAMS", "{}")))
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
    r.add_argument("--compute", action="store_true", help="try experimental rd/bleed compute kernels (implies --gpu; falls back on older GL)")
    r.add_argument("--nearest", action="store_true", help="nearest-neighbour texture sampling (CPU)")
    r.add_argument("--encoder", default="auto", help="H.264 encoder: auto, software, nvenc, videotoolbox, qsv, amf, vaapi (or FFmpeg name)")

    e = sub.add_parser("encoders", help="test H.264 encoders on this machine with a real frame")
    e.add_argument("--size", type=parse_size, default=(1280, 720))
    e.add_argument("--fps", type=float, default=30.0)
    for parser in (r, e):
        parser.add_argument("--quality", type=int, default=8, help="video quality 1..10, higher is better (default: 8)")
        parser.add_argument("--bitrate", help="target video bitrate, e.g. 8M; overrides quality control")
        parser.add_argument("--ffmpeg", help="FFmpeg executable (otherwise checks system and imageio FFmpeg)")
        parser.add_argument("--device", help="VAAPI render node, e.g. /dev/dri/renderD128")

    h = sub.add_parser("html", help="export a standalone WebGL page")
    h.add_argument("patch")
    h.add_argument("-o", "--out", default=None)
    h.add_argument("--size", type=parse_size, default=None, help="fixed canvas size (default: fills the window)")
    h.add_argument("--pixel-ratio", type=float, default=1.0)
    h.add_argument("--phone", action="store_true",
                   help="for phones: --precision mixed and --adaptive (add --fast-noise and --8bit for more)")
    h.add_argument("--precision", choices=("highp", "mixed", "mediump"), default="highp",
                   help="mixed: colour math in mediump, coordinates/time/noise in highp")
    h.add_argument("--adaptive", nargs="?", type=float, const=0.35, default=None, metavar="MIN",
                   help="render smaller while the page can't keep up with the display (down to MIN, default 0.35)")
    h.add_argument("--8bit", dest="eight_bit", action="store_true", help="8-bit output buffers, as Hydra (no half floats)")

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

    for sp in (r, h, lv, fh):
        sp.add_argument("--fast-noise", action="store_true",
                        help="GPU noise() from a texture lookup: cheaper, similar but not identical (set_fast_noise)")
    for sp in (r, h, lv, an, fh):
        sp.add_argument("--fxhash", default=None, help="the hash a Mint previews (sets FXHASH)")
        sp.add_argument("--fxparam", action="append", default=[], metavar="ID=VALUE",
                        help="an fx(params) value for a Mint's preview (repeatable; sets FXPARAMS)")

    op = sub.add_parser("ops", help="search the operator reference")
    op.add_argument("query", nargs="?", default="")
    op.add_argument("--kind", choices=["src", "coord", "color", "combine", "combineCoord", "texture"])
    check = sub.add_parser("check", help="inspect patch compatibility before export")
    check.add_argument("patch")
    check.add_argument("--target", choices=["cpu", "gl", "html", "hydra"], default="html")
    check.add_argument("--json", action="store_true")
    studio = sub.add_parser("studio", help="open the local patch workspace")
    studio.add_argument("patch", nargs="?")
    studio.add_argument("--root", default=".")
    studio.add_argument("--backend", choices=["auto", "cpu", "gl"], default="auto")
    studio.add_argument("--size", type=parse_size, default=(640, 360))
    studio.add_argument("--fps", type=float, default=30.)
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument("--no-browser", action="store_true")
    studio.add_argument("--tracker", action="store_true", help="open the Gorgon tracker workspace")
    cv_output = studio.add_mutually_exclusive_group()
    cv_output.add_argument("--cv-device", help="send direct CV on this audio device (see `hydrapy devices`)")
    cv_output.add_argument("--cv-profile", help="JSON routing profile for ES-40 / ESX-8CV units and direct CV")
    studio.add_argument("--cv-map", default="0", help="first channel for the default eight signals, or e.g. 8=clock,9=reset,10=gate1")
    studio.add_argument("--cv-ppqn", type=int, default=4, help="clock pulses per quarter note (default 4: sixteenths)")
    studio.add_argument("--cv-offset-ms", type=float, default=0., help="send CV earlier (+) or later (-) than the picture")
    studio.add_argument("--cv-pitch-zero", type=int, default=60, help="MIDI note at 0 V (default 60 = C4); pitch is 1 V/oct")
    diagnostic = sub.add_parser("es40-check", help="write an encoded S/PDIF diagnostic WAV for an ES-40 + ESX-8CV")
    diagnostic.add_argument("-o", "--out", required=True, help="24-bit stereo WAV destination; data, not listening audio")
    diagnostic.add_argument("--header", type=int, choices=range(1, 6), default=1, help="ES-40 expansion header (1–5)")
    diagnostic.add_argument("--samplerate", type=int, choices=[44100, 48000], default=48000)
    session = sub.add_parser("session", help="play or inspect a beat-timed OSC session")
    from .session import add_arguments as session_arguments
    session_arguments(session)
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

    if a.cmd == "es40-check":
        from .es40 import diagnostic_wav
        diagnostic_wav(a.out, header=a.header, samplerate=a.samplerate)
        print(f'{a.out}: 18 s, {a.samplerate} Hz / 24-bit S/PDIF data. Route only to ES-40 at unity, without mixing or resampling.\n'
              'Zero, then 1 V on each ESX-8CV jack in turn for 1 s, with 1 s zero between steps; ends at zero.')
        return

    if a.cmd == "session":
        from .session import execute
        try:
            execute(a)
        except KeyboardInterrupt:
            raise SystemExit(130)
        except (ValueError, OSError, RuntimeError) as exc:
            ap.exit(1, f"session: {exc}\n")
        return

    if a.cmd == "studio":
        from .studio import serve
        cv = None
        try:
            options = dict(ppqn=a.cv_ppqn, offset_ms=a.cv_offset_ms, pitch_zero=a.cv_pitch_zero)
            if a.cv_profile:
                from .es40 import load_profile
                cv = load_profile(a.cv_profile, **options)
            elif a.cv_device is not None:
                from .tracker_cv import TrackerCV, parse_map
                device = int(a.cv_device) if a.cv_device.isdigit() else a.cv_device
                cv = TrackerCV(parse_map(a.cv_map), device=device, **options)
        except Exception as exc:
            ap.exit(1, f'Tracker CV: {exc}\n')
        if cv is not None:
            import json
            print('Tracker CV: ' + json.dumps(cv.describe()), flush=True)
        serve(a.root, a.patch, a.backend, a.size, a.fps, a.port, not a.no_browser, tracker=a.tracker, cv=cv)
        return

    if a.cmd == "check":
        import json
        from .inspect import capabilities
        from .live import ResourceStage
        with ResourceStage():
            result = capabilities(load_patch(a.patch, strict=False), a.target)
        if a.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{a.target}: {'ready' if result['ready'] else 'needs changes'}")
            for item in result['findings']:
                print(f"  {item['status']}: {item['path']} — {item['message']}")
        if not result['ready']:
            raise SystemExit(1)
        return

    if a.cmd == "nca":
        from .nca_train import train
        out = a.out or os.path.splitext(os.path.basename(a.image))[0] + ".npz"
        train(a.image, out, scale=a.scale, iters=a.iters, loss=a.loss, hidden=a.hidden, seed=a.seed,
              steps=tuple(a.steps), lr=a.lr, init=a.init)
        print("wrote", out)
        return

    if a.cmd == "encoders":
        from .encoding import video_encoders
        try:
            results = video_encoders(a.size, a.fps, a.quality, a.bitrate, ffmpeg=a.ffmpeg, device=a.device)
        except (ValueError, RuntimeError, OSError) as exc:
            ap.exit(1, f"encoder check failed: {exc}\n")
        for result in results:
            node = f" ({result.device})" if result.device else ""
            print(f"{'OK' if result.available else '--'} {result.name}: {result.ffmpeg}{node}")
            if result.reason:
                print("   " + " ".join(result.reason.split()))
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
        for name, op in REGISTRY.items():
            text = describe(name)
            if a.query.lower() in text.lower() and (not a.kind or op.kind == a.kind):
                print(text)
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

    def loaded(strict=True):
        p = load_patch(a.patch, strict=strict)
        if getattr(a, "fast_noise", False):
            p.fast_noise = True
        return p

    patch = loaded()

    if a.cmd == "render":
        ext = os.path.splitext(a.out)[1].lower()
        size = a.size or patch.size
        t0 = time.time()
        if a.gpu or a.compute:
            from .backend_gl import iter_frames_gl
            frames = iter_frames_gl(patch, seconds=a.seconds, fps=a.fps, size=size, start=a.start,
                                    warmup=a.warmup, compute=a.compute)
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
            def report_encoder(selected):
                kind = "hardware" if selected.hardware else "software"
                print(f"encoding with {selected.name} ({kind}): {selected.ffmpeg}", flush=True)

            try:
                save_mp4(frames, a.out, fps=a.fps, scale=a.scale, audio=track, encoder=a.encoder,
                         quality=a.quality, bitrate=a.bitrate, ffmpeg=a.ffmpeg, device=a.device,
                         on_encoder=report_encoder)
            except (ValueError, RuntimeError, OSError) as exc:
                ap.exit(1, f"video export failed: {exc}\n")
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
        to_html(patch, out, title=os.path.basename(a.patch), size=a.size, pixel_ratio=a.pixel_ratio,
                precision=a.precision, adaptive=a.adaptive or False, float_buffers=not a.eight_bit, phone=a.phone)
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
             loader=lambda: loaded(strict=False))

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
