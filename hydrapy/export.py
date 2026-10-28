"""Exports: images / GIF / MP4 from the CPU renderer, ANSI for the terminal,
Hydra JavaScript, and a self-contained WebGL HTML page that runs the patch live."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Optional

import numpy as np

from .core import Camera, Chain, DelayRef, FrameSource, Image, Output, Patch, Texture, Video, default_patch
from .glsl import TmpRef, compile_patch
from .noise import NOISE_TEXTURE_JS
from .signals import JS_HELPERS, Audio, Fn, Seq, Signal, as_signal


# ---------------------------------------------------------------------------
# Images and movies
# ---------------------------------------------------------------------------

def to_image(arr: np.ndarray, composite: bool = False):
    """float (H, W, 4) or uint8 (H, W, 3) -> PIL image (``composite``: RGB x alpha over black)."""
    from PIL import Image as PILImage
    if arr.dtype != np.uint8:
        from .backend_np import to_uint8
        arr = to_uint8(arr, composite)
    return PILImage.fromarray(arr[..., :3], "RGB")


def save_png(arr: np.ndarray, path: str):
    to_image(arr).save(path)
    return path


def save_gif(frames, path: str, fps: float = 30.0, scale: int = 1):
    """Write uint8 RGB frames to an animated GIF (optionally nearest-neighbour upscaled)."""
    import imageio.v3 as iio
    frames = [np.repeat(np.repeat(f, scale, 0), scale, 1) if scale > 1 else f for f in frames]
    iio.imwrite(path, frames, duration=1000.0 / fps, loop=0)
    return path


def save_mp4(frames, path: str, fps: float = 30.0, quality: int = 8, scale: int = 1, audio=None,
             *, encoder="auto", bitrate=None, ffmpeg=None, device=None, on_encoder=None):
    """Stream uint8 RGB/RGBA frames to H.264 in MP4, MOV, or MKV (alpha is discarded).

    ``encoder='auto'`` tries hardware with an actual one-frame encode, then libx264.
    Explicit choices (``nvenc``, ``videotoolbox``, ``qsv``, ``amf``, ``vaapi``, ``software``)
    fail if unavailable. ``quality`` is 1..10; ``bitrate='8M'`` overrides quality control.
    ``ffmpeg`` selects an executable; ``device`` selects a VAAPI render node.
    ``on_encoder``, if supplied, receives the selected ``VideoEncoder`` before streaming.

    Frames are consumed once and closed on success or failure. Odd dimensions are cropped
    to even after upscaling. The destination is replaced only after a successful export.
    ``audio`` may be a ``Soundtrack`` rendered along with the frames, a WAV path, or
    ``(samples, sr)``; it is muxed in as AAC without re-encoding the video.
    Video duration (frame count / fps) is authoritative: shorter audio is padded
    with silence and longer audio is cut at the video end.
    """
    import tempfile
    from .encoding import _select_encoder, _validate, _write_video

    iterator = None
    temporary = []
    closed = False

    def close_frames():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            if iterator is not None and iterator is not frames and hasattr(iterator, "close"):
                iterator.close()
        finally:
            if hasattr(frames, "close"):
                frames.close()

    try:
        destination = os.fspath(path)
        suffix = os.path.splitext(destination)[1].lower()
        if suffix not in (".mp4", ".mov", ".mkv"):
            raise ValueError("H.264 export requires .mp4, .mov, or .mkv; WebM requires a different codec")
        if not isinstance(scale, (int, np.integer)) or scale < 1:
            raise ValueError("scale must be a positive integer")
        _validate((2, 2), fps, quality, bitrate)

        def prepare(frame):
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise ValueError("video frames must be uint8 RGB or RGBA arrays")
            frame = frame[..., :3]
            if scale > 1:
                frame = np.repeat(np.repeat(frame, scale, 0), scale, 1)
            h, w = frame.shape[:2]
            return frame[:h-h % 2, :w-w % 2]

        iterator = iter(frames)
        try:
            first = prepare(next(iterator))
        except StopIteration:
            raise ValueError("video requires at least one frame") from None
        size = (first.shape[1], first.shape[0])
        selected = _select_encoder(encoder, size, fps, quality, bitrate, ffmpeg, device)
        if on_encoder is not None:
            on_encoder(selected)

        frame_count = 0

        def prepared_frames():
            nonlocal frame_count
            frame_count += 1
            yield first
            for frame in iterator:
                frame = prepare(frame)
                if frame.shape != first.shape:
                    raise ValueError("all video frames must have the same dimensions after scaling/cropping")
                frame_count += 1
                yield frame

        def temporary_video():
            fd, name = tempfile.mkstemp(prefix=".hydrapy-", suffix=suffix,
                                        dir=os.path.dirname(os.path.abspath(destination)))
            os.close(fd)
            temporary.append(name)
            return name

        video = temporary_video()
        _write_video(prepared_frames(), video, size, fps, selected)
        close_frames()
        if audio is not None:
            muxed = temporary_video()
            _mux_audio(video, audio, muxed, duration=frame_count / fps, ffmpeg=selected.ffmpeg)
            video = muxed
        os.replace(video, destination)
        temporary.remove(video)
        return path
    finally:
        try:
            for name in temporary:
                try:
                    os.remove(name)
                except FileNotFoundError:
                    pass
        finally:
            close_frames()


def _mux_audio(video: str, audio, path: str, *, duration: float, ffmpeg=None):
    """Pad/cut AAC to the rendered video duration, avoiding shortest/DTS truncation."""
    import subprocess
    import tempfile
    import wave
    from .encoding import _ffmpeg_candidates
    wav = os.fspath(audio) if isinstance(audio, (str, os.PathLike)) else None
    tmp = None
    try:
        if wav is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
            if hasattr(audio, "wav"):
                audio.wav(tmp.name)
            else:
                samples, sr = audio
                samples = np.asarray(samples, dtype=np.float64)
                with wave.open(tmp.name, "wb") as wf:                     # (n,) mono or (n, k) interleaved
                    wf.setnchannels(1 if samples.ndim == 1 else samples.shape[1]); wf.setsampwidth(2); wf.setframerate(int(sr))
                    wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
            wav = tmp.name
        exe = ffmpeg or _ffmpeg_candidates()[0]
        cmd = [exe, "-y", "-loglevel", "error", "-i", video, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
               "-af", "apad", "-t", str(float(duration))]
        if os.path.splitext(path)[1].lower() in (".mp4", ".mov"):
            cmd += ["-movflags", "+faststart"]
        result = subprocess.run([*cmd, path], capture_output=True)
        if result.returncode:
            detail = result.stderr.decode("utf-8", "replace")[-4000:]
            raise RuntimeError(f"FFmpeg audio mux failed (exit {result.returncode}):\n{detail}")
    finally:
        if tmp is not None:
            os.remove(tmp.name)


def save_frames(frames, directory: str, prefix: str = "frame"):
    os.makedirs(directory, exist_ok=True)
    for i, f in enumerate(frames):
        to_image(f).save(os.path.join(directory, f"{prefix}_{i:05d}.png"))
    return directory


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def to_ansi(frame: np.ndarray, cols: int = 80, rows: Optional[int] = None, composite: bool = False) -> str:
    """Render a uint8 RGB frame as 24-bit ANSI half-block art."""
    if frame.dtype != np.uint8:
        from .backend_np import to_uint8
        frame = to_uint8(frame, composite)
    H, W = frame.shape[:2]
    rows = rows or max(int(round(cols * H / W / 2)), 1)
    ys = (np.arange(rows * 2) * H / (rows * 2)).astype(int)
    xs = (np.arange(cols) * W / cols).astype(int)
    small = frame[ys][:, xs]
    out = []
    for r in range(rows):
        top, bot = small[2 * r], small[2 * r + 1]
        line = []
        for x in range(cols):
            t, b = top[x], bot[x]
            line.append(f"\x1b[38;2;{t[0]};{t[1]};{t[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀")
        out.append("".join(line) + "\x1b[0m")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Hydra JavaScript
# ---------------------------------------------------------------------------

def _hydra_signal(sig) -> str:
    sig = as_signal(sig)
    if isinstance(sig, Seq) or hasattr(sig, "hydra_array"):
        return sig.hydra_array()
    if isinstance(sig, Fn):
        if getattr(sig, "js_source", None):   # came from a Hydra sketch: give it back verbatim
            return f"({{time, bpm, width, height, mouse}}) => {sig.js_source}"
        tbl = sig.bake()
        return ("[" + ",".join(f"{v:.4g}" for v in tbl) + f"].fast({sig.bake_rate * 60 / 30:g}).smooth()")
    js = sig.js().replace("__mx", "(mouse.x/width)").replace("__my", "(1-mouse.y/height)").replace("__fft", "a.fft").replace("__vol", "a.vol").replace("__flux", "a.vol")
    return f"({{time, bpm, width, height, mouse}}) => {js}"


def _hydra_arg(a, strict: bool) -> str:
    if isinstance(a, FrameSource):
        raise ValueError("Python FrameSource animations cannot run in Hydra JS; render a video or bake frames into a Bank")
    if isinstance(a, bool):
        return "1" if a else "0"
    if isinstance(a, (int, float)):
        return f"{a:g}"
    if isinstance(a, Chain):
        return chain_to_hydra(a, strict=strict)
    if isinstance(a, Output):
        return a.name
    if isinstance(a, Image):
        return "s0 /* s0.initImage('%s') */" % a.path
    if isinstance(a, Video):
        return "s0 /* s0.initVideo('%s') */" % a.path
    if isinstance(a, Camera):
        return "s0 /* s0.initCam() */"
    if type(a).__name__ == "External":
        return f"s0 /* {a.note} */"
    if hasattr(a, "render") and not hasattr(a, "array_at"):
        return "s0 /* s0.initImage(...): export this %s with Image.fromarray(x.render()).save(...) */" % type(a).__name__
    if hasattr(a, "array_at"):
        return "s0 /* frame bank: s0.initImage(...) one of %d images */" % a.count
    if isinstance(a, Signal) or callable(a):
        return _hydra_signal(a)
    raise TypeError(f"cannot export {a!r} to Hydra")


def chain_to_hydra(chain: Chain, strict: bool = True) -> str:
    """Hydra JS for a chain. Non-Hydra ops raise (strict) or are dropped with a comment.

    Per-pixel chain parameters (``osc(noise())``) have no Hydra equivalent and raise.
    """
    parts, dropped = [], []
    for tr in chain.transforms:
        op = tr.op
        if not op.hydra:
            if strict:
                raise ValueError(f".{op.name}() has no Hydra equivalent (export with strict=False to drop it)")
            if op.kind == "src":   # keep the chain valid: stand in a grey source
                parts.append(f"solid(0.5, 0.5, 0.5) /* was {op.name}() */")
            else:
                dropped.append(f"/* dropped .{op.name}() */")
            continue
        pairs = list(zip(tr.args, op.inputs))
        # trim trailing default numbers for readability (the chain argument of combine ops always stays)
        keep = 1 if op.kind in ("combine", "combineCoord") else 0
        while len(pairs) > keep:
            a, inp = pairs[-1]
            if isinstance(a, (int, float)) and not isinstance(a, bool) and inp.default is not None \
                    and float(a) == float(inp.default):
                pairs.pop()
            else:
                break
        args = []
        for a, inp in pairs:
            if isinstance(a, Chain) and inp.type != "chain":
                if strict:
                    raise ValueError(f"Hydra cannot modulate {op.name}({inp.name}) with a chain per pixel")
                args.append(f"{inp.default:g} /* was a per-pixel chain */")
                continue
            args.append(_hydra_arg(a, strict))
        parts.append(f"{op.name}({', '.join(args)})")
    return ".".join(parts) + "".join(" " + d for d in dropped)


def patch_to_hydra(patch: Optional[Patch] = None, strict: bool = True) -> str:
    """Whole patch as a Hydra sketch, ready to paste into hydra.ojack.xyz."""
    patch = patch or default_patch()
    lines = []
    uses_js = any(_needs_helpers(o.chain) for o in patch.active_outputs())
    if uses_js:
        lines.append("// helpers for hydrapy signals")
        lines.append(JS_HELPERS.strip())
        lines.append("")
    if patch.bpm != 30:
        lines.append(f"bpm = {patch.bpm:g}")
    if patch.speed != 1:
        lines.append(f"speed = {patch.speed:g}")
    for o in patch.active_outputs():
        lines.append(f"{chain_to_hydra(o.chain, strict=strict)}.out({o.name})")
    lines.append(f"render({patch.display.name})")
    return "\n".join(lines) + "\n"


def _needs_helpers(chain: Chain) -> bool:
    for tr in chain.transforms:
        for a in tr.args:
            if isinstance(a, Chain) and _needs_helpers(a):
                return True
            if isinstance(a, Signal) and not isinstance(a, Seq):
                if any(h in a.js() for h in ("__fract", "__hash11", "__smoothstep", "__drift", "__env", "__table", "__seq", "__fx")):
                    return True
            elif callable(a) and not isinstance(a, (Signal, Chain, Texture)):
                return False
    return False


# ---------------------------------------------------------------------------
# Python source (for `hydrapy import sketch.js`)
# ---------------------------------------------------------------------------

_PY_NOTES: list = []


def _py_arg(a) -> str:
    from .signals import Fft, Seq, Fn
    from .hydra_import import js_expr_to_python
    from .cellular import Cellular
    if isinstance(a, Cellular):
        names = ("rules", "width", "height", "rate", "seed", "stripes", "density", "perturb", "wrap", "chars", "cell")
        return "Cellular(" + ", ".join(f"{name}={getattr(a, name)!r}" for name in names) + ")"
    if isinstance(a, FrameSource):
        raise ValueError("FrameSource Python export needs the original draw callable; keep its source patch")
    if isinstance(a, bool):
        return repr(a)
    if isinstance(a, (int, float)):
        return f"{a:g}"
    if isinstance(a, Chain):
        return chain_to_python(a)
    if isinstance(a, Output):
        return a.name
    if isinstance(a, Image):
        return f"Image({a.path!r})"
    if isinstance(a, Video):
        return f"Video({a.path!r})"
    if isinstance(a, Camera):
        return f"Camera({a.device})"
    if type(a).__name__ == "External":
        return a.name
    if hasattr(a, "render") and not hasattr(a, "array_at"):
        return f"{type(a).__name__}(...)"
    if isinstance(a, Seq):
        s = f"Seq({a.values})"
        if a._speed != 1: s += f".fast({a._speed:g})"
        if a._smooth: s += f".smooth({a._smooth:g})"
        if a._offset: s += f".offset({a._offset:g})"
        if a._ease != "linear": s += f".ease({a._ease!r})"
        return s
    if isinstance(a, Fft):
        return f"a.fft[{a.band}]"
    native = _py_native_signal(a)
    if native is not None:
        return native
    if isinstance(a, Fn) and getattr(a, "js_source", None):
        py = js_expr_to_python(a.js_source).replace("_fft(", "a.band_at(_ctx.time, ")
        return f"Fn(lambda _ctx: {py})"
    if isinstance(a, Signal):
        _PY_NOTES.append(type(a).__name__)      # a live or opaque signal (LiveValue, Fx, Audio...): frozen
        return "Const(0)"
    raise TypeError(f"cannot write {a!r} as Python")


def _py_native_signal(a):
    """hydrapy's own signals written back as their constructors (None for anything else)."""
    from .signals import SH, Clock, Const, Decay, Drift, Env, Expr, Fn1, Lfo, Mouse, Time
    from .timeline import Track
    g = lambda v: f"{float(v):g}"
    if type(a).__name__ in ("FxDraw", "FxParam"):           # a Mint draw: this hash's value
        return g(a.preview)
    if isinstance(a, Track):
        return repr(a)
    if isinstance(a, Const):
        return g(a.v)
    if isinstance(a, Time):
        return "Time()"
    if isinstance(a, Decay):
        return f"Decay({g(a.half_life)})"
    if isinstance(a, Mouse):
        return f"Mouse({a.axis!r})"
    if isinstance(a, Lfo):
        return f"Lfo({g(a.freq)}, {a.shape!r}, {g(a.lo)}, {g(a.hi)}, {g(a.phase)}, {g(a.width)})"
    if isinstance(a, SH):
        return f"SH({g(a.rate)}, {g(a.lo)}, {g(a.hi)}, {g(a.seed)})"
    if isinstance(a, Drift):
        return f"Drift({g(a.rate)}, {g(a.lo)}, {g(a.hi)}, {g(a.seed)})"
    if isinstance(a, Clock):
        bpm = "None" if a.bpm is None else g(a.bpm)
        return f"Clock({g(a.div)}, {bpm}, {g(a.phase)})"
    if isinstance(a, Env):
        return f"Env({_py_arg(a.clock)}, {g(a.attack)}, {g(a.decay)}, {g(a.curve)}, {g(a.lo)}, {g(a.hi)})"
    if isinstance(a, Expr):
        return f"({_py_arg(a.a)} {a.op} {_py_arg(a.b)})"
    if isinstance(a, Fn1):
        extra = "".join(f", {g(e)}" for e in a.extra)
        return f"as_signal({_py_arg(a.a)}).{a.fn}({extra[2:]})"
    return None


def chain_to_python(chain: Chain) -> str:
    parts = []
    for tr in chain.transforms:
        op = tr.op
        if op.python_source is not None:
            parts.append(op.python_source(tr.args, _py_arg))
            continue
        pairs = list(zip(tr.args, op.inputs))
        keep = 1 if op.kind in ("combine", "combineCoord") else 0
        while len(pairs) > keep:
            a, inp = pairs[-1]
            if isinstance(a, (int, float)) and not isinstance(a, bool) and inp.default is not None \
                    and float(a) == float(inp.default):
                pairs.pop()
            else:
                break
        parts.append(f"{op.name}({', '.join(_py_arg(a) for a, _ in pairs)})")
    return ".".join(parts)


def _externals(chain, found):
    for tr in chain.transforms:
        for a in tr.args:
            if type(a).__name__ == "External":
                found[a.name] = a
            elif isinstance(a, Chain):
                _externals(a, found)


def patch_to_python(patch: Optional[Patch] = None) -> str:
    patch = patch or default_patch()
    body = []
    ext = {}
    del _PY_NOTES[:]
    for o in patch.active_outputs():
        _externals(o.chain, ext)
        scale = getattr(patch, "scales", {}).get(o.index)
        body.append(f"{chain_to_python(o.chain)}.out({o.name}" + (f", scale={scale:g})" if scale else ")"))
    src = "\n".join(body)
    head = ["import math", "from hydrapy import *", ""]
    if _PY_NOTES:
        head.append(f"# signals that only exist live were frozen to Const(0): {', '.join(sorted(set(_PY_NOTES)))}")
    for name, e in ext.items():
        head.append(f"# {name} was `{e.note}`: a source hydrapy can't read offline. Swap in an Image, Video, Bank or")
        head.append(f"# Camera; for the p5 'hydra canvas on a plane' loop use src(oN).plane3d(...) in its place.")
        head.append(f"{name} = External({name!r})")
    if "a.fft" in src or "a.band_at" in src:
        head.append("a = Audio()  # microphone in the HTML export; use Audio.from_file('track.wav') offline")
    if patch.bpm != 30:
        head.append(f"set_bpm({patch.bpm:g})")
    if patch.speed != 1:
        head.append(f"speed({patch.speed:g})")
    if patch.frame_rate:
        head.append(f"set_frame_rate({patch.frame_rate:g})")
    if patch.composite_alpha:
        head.append("set_composite_alpha(True)")
    if getattr(patch, "fast_noise", False):
        head.append("set_fast_noise(True)")
    head.append("")
    tail = ["", f"render({patch.display.name})", ""]
    return "\n".join(head) + src + "\n".join(tail)


# ---------------------------------------------------------------------------
# HTML / WebGL export
# ---------------------------------------------------------------------------

def _data_url(path: str, mime: Optional[str] = None) -> str:
    mime = mime or mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _png_data_url(arr_uint8) -> str:
    import io
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(arr_uint8).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _nca_specs(programs, precision="highp"):
    from .nca import NCA, VIEW_GLSL, shader_header, step_glsl
    seen, out = set(), []
    for prog in programs:
        for u in prog.sampler_uniforms:
            ref = u.source
            if isinstance(ref, NCA) and ref.name not in seen:
                seen.add(ref.name)
                head = shader_header(precision)
                out.append({"name": ref.name, "w": ref.size[0], "h": ref.size[1], "textures": ref.model.textures,
                            "speed": ref.speed, "maxSteps": ref.max_steps, "seed": ref.seed, "filter": ref.filter,
                            "steps": [head + step_glsl(ref.model, k) for k in range(ref.model.textures)],
                            "view": head + VIEW_GLSL,
                            "damage": ref.damage.name if ref.damage is not None else None,
                            "paint": ref.paint.name if ref.paint is not None else None,
                            "rate": ref.rate.js(), "perturb": ref.perturb.js(), "paintAmount": ref.paint_amount.js()})
    return out


def _collect_media(programs):
    images, videos, cameras, audios, banks = {}, {}, set(), [], {}

    def visit_sig(s):
        from .signals import Expr, Fft, Fn1
        if isinstance(s, Fft):
            if s.audio not in audios:
                audios.append(s.audio)
        elif isinstance(s, Expr):
            visit_sig(s.a); visit_sig(s.b)
        elif isinstance(s, Fn1):
            visit_sig(s.a)

    for prog in programs:
        for u in prog.uniforms:
            if u.type == "sampler2D":
                ref = u.source
                if isinstance(ref, FrameSource):
                    raise ValueError("Python FrameSource animations cannot run in HTML; render a video or bake frames into a Bank")
                if isinstance(ref, Image):
                    images[ref.name] = _data_url(ref.path)
                elif hasattr(ref, "render") and not hasattr(ref, "array_at"):      # Text, Lines
                    images[ref.name] = _png_data_url(ref.render())
                elif isinstance(ref, Video):
                    videos[ref.name] = _data_url(ref.path)
                elif isinstance(ref, Camera):
                    cameras.add(ref.name)
                elif hasattr(ref, "array_at") and ref.name not in banks:   # Bank / Quilt
                    banks[ref.name] = {"frames": [_png_data_url(f) for f in ref.frames()], "rate": ref.rate,
                                       "hold": ref.hold, "mode": ref.mode, "seed": float(ref.seed)}
            else:
                visit_sig(u.source)
    return images, videos, sorted(cameras), audios, banks


def _mint_spec(programs, mint=None, preview_at=None):
    from .mint import find_mints
    objs = [mint] if mint is not None else []
    for prog in programs:
        objs += [u.source for u in prog.float_uniforms]
        objs += [u.source for u in prog.sampler_uniforms if not isinstance(u.source, DelayRef)]
    mints = find_mints(objs)
    if len(mints) > 1:
        raise ValueError("one Mint per patch: the page has one hash")
    if not mints:
        return None
    spec = mints[0].spec()
    spec["previewAt"] = preview_at
    return spec


def build_spec(patch: Optional[Patch] = None, pixel_ratio: float = 1.0, size: Optional[tuple] = None,
               precision: str = "highp", mint=None, preview_at: Optional[float] = None,
               adaptive: bool | float = False, float_buffers: bool = True) -> dict:
    """Everything the JS runtime needs, as plain data (what the HTML export embeds as ``SPEC``).

    ``adaptive``: render the outputs smaller while frames arrive slower than the display refreshes
    (True, or the smallest scale to allow; True means 0.35). ``float_buffers=False``: 8-bit outputs
    even where half-float targets exist, as Hydra does — half the memory traffic on a phone."""
    patch = patch or default_patch()
    programs = compile_patch(patch, precision=precision)
    images, videos, cameras, audios, banks = _collect_media(programs)
    if len(audios) > 1:
        raise ValueError("the HTML export supports one Audio analyser per patch")
    audio = None
    if audios:
        a = audios[0]
        audio = {"src": a.embed_b64() if a.path else None, "mic": a.path is None, "bins": a.bins,
                 "smooth": a.smooth, "scale": a.scale, "cutoff": a.cutoff}
    outs = []
    for prog in programs:
        outs.append({
            "name": prog.output.name,
            "passes": [dict(target=p.target, source=p.source, **({"mipmap": True} if p.mipmap else {}))
                       for p in prog.passes],
            "floats": [{"name": u.name, "js": as_signal(u.source).js()} for u in prog.float_uniforms],
            "samplers": [({"name": u.name, "ref": u.source.output.name, "delay": u.source.frames}
                          if isinstance(u.source, DelayRef) else {"name": u.name, "ref": u.source.name})
                         for u in prog.sampler_uniforms],
        })
    history = {}
    for prog in programs:
        for k, v in prog.history.items():
            history[k] = max(history.get(k, 0), v)
    ncas = _nca_specs(programs, precision)
    sims = []
    for prog in programs:
        for u in prog.sampler_uniforms:
            if hasattr(u.source, "gl_step") and hasattr(u.source, "spec") and u.source.name not in [x["name"] for x in sims]:
                sims.append(u.source.spec(precision))
    return {
        "display": patch.display.name,
        "bpm": patch.bpm,
        "speed": patch.speed,
        "frameRate": patch.frame_rate,
        "compositeAlpha": patch.composite_alpha,
        "fastNoise": bool(getattr(patch, "fast_noise", False)),
        "soundtrack": next((pr.spec() for pr in patch.probes if hasattr(pr, "chunks")), None),
        "outputs": outs,
        "history": history,
        "images": images,
        "videos": videos,
        "banks": banks,
        "ncas": ncas,
        "physarum": sims,
        "cameras": cameras,
        "audio": audio,
        "pixelRatio": pixel_ratio,
        "size": list(size) if size else None,
        "mint": _mint_spec(programs, mint, preview_at),
        "scales": {f"o{i}": s for i, s in getattr(patch, "scales", {}).items() if s != 1.0},
        "adaptive": ({"min": 0.35 if adaptive is True else float(adaptive), "step": 0.85} if adaptive else None),
        "floatBuffers": bool(float_buffers),
        "hydra": _try_hydra(patch),
    }


def _try_hydra(patch):
    try:
        return patch_to_hydra(patch, strict=False)
    except Exception as e:  # pragma: no cover
        return f"// not exportable to Hydra: {e}"


def spec_json(spec: dict) -> str:
    return json.dumps(spec).replace("</", "<\\/")


_HUD = '<div id="hud">hydrapy &middot; 1-4: output &middot; s: save png &middot; f: fullscreen &middot; space: pause</div>'


def _page(spec: dict, title: str, head: str = "", hud: bool = True) -> str:
    from .mint import MINT_JS
    return (_HTML_TEMPLATE.replace("__TITLE__", title).replace("__HEAD__", head).replace("__HUD__", _HUD if hud else "")
            .replace("__HELPERS__", JS_HELPERS).replace("__SPEC__", spec_json(spec)).replace("__MINT__", MINT_JS)
            .replace("__RUNTIME__", RUNTIME_JS))


def to_html(patch: Optional[Patch] = None, path: Optional[str] = None, title: str = "hydrapy",
            pixel_ratio: float = 1.0, size: Optional[tuple] = None, precision: str = "highp",
            adaptive: bool | float = False, float_buffers: bool = True, phone: bool = False) -> str:
    """Compile the patch to a standalone HTML page (WebGL2 with WebGL1 fallback) and optionally write it.

    A patch with a ``Mint`` replays its draws from ``?fxhash=`` in the page's URL (default: the
    preview hash); press n for another mint.

    ``phone=True`` is ``precision="mixed"`` (colour math in mediump, coordinates, time and noise in
    highp) plus ``adaptive=True`` (outputs render smaller while the page can't keep up). See
    ``build_spec`` for ``adaptive`` and ``float_buffers``."""
    if phone:
        precision = "mixed" if precision == "highp" else precision
        adaptive = adaptive or True
    spec = build_spec(patch, pixel_ratio=pixel_ratio, size=size, precision=precision, adaptive=adaptive,
                      float_buffers=float_buffers)
    html = _page(spec, title)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


def to_fxhash(patch: Optional[Patch] = None, folder: str = "fxhash", sdk: Optional[str] = None,
              preview: Optional[float] = 2.0, title: str = "hydrapy", pixel_ratio: float = 1.0,
              size: Optional[tuple] = None, precision: str = "highp", zip: bool = True, mint=None) -> dict:
    """An fxhash project: ``folder/index.html`` (+ ``fxhash.min.js``) and ``folder/upload.zip``.

    The patch's ``Mint`` (if any) draws from ``$fx.hash``, declares its fx(params) and sets its
    features at load; ``$fx.preview()`` fires once the page has run ``preview`` seconds.  ``sdk`` is
    a path to the SDK's ``fxhash.min.js`` (from ``npx fxhash create``, or ``npm pack
    @fxhash/project-sdk`` → ``package/dist/fxhash.min.js``); without it the page uses a stand-in
    that reads ``?fxhash=``, fine for looking but not for fxlens or uploading.  Everything else is
    inlined, as fxhash requires (no network, relative paths only).
    """
    import shutil
    import warnings
    import zipfile
    spec = build_spec(patch, pixel_ratio=pixel_ratio, size=size, precision=precision, mint=mint, preview_at=preview)
    if spec["audio"] or spec["soundtrack"] or spec["cameras"]:
        warnings.warn("this patch waits for a click (audio, soundtrack or camera): fxhash's capture can't click, "
                      "so previews will show the start overlay", stacklevel=2)
    if spec["mint"] is None:
        warnings.warn("no Mint in this patch: every edition will look the same", stacklevel=2)
    os.makedirs(folder, exist_ok=True)
    sdk_dst = os.path.join(folder, "fxhash.min.js")
    if sdk:
        shutil.copyfile(sdk, sdk_dst)
    elif not os.path.exists(sdk_dst):
        warnings.warn("no fxhash.min.js: the page runs on a stand-in for $fx; pass sdk= (or --sdk) before "
                      "using fxlens or uploading", stacklevel=2)
    from .mint import FX_SHIM_JS
    head = '<script src="./fxhash.min.js"></script>\n<script>' + FX_SHIM_JS + "</script>"
    index = os.path.join(folder, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(_page(spec, title, head=head, hud=False))
    out = {"index": index, "zip": None, "mint": spec["mint"]}
    if zip:
        zpath = os.path.join(folder, "upload.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(index, "index.html")
            if os.path.exists(sdk_dst):
                z.write(sdk_dst, "fxhash.min.js")
        if os.path.getsize(zpath) > 15 * 1024 * 1024:
            warnings.warn("upload.zip is over fxhash's 15 MB limit", stacklevel=2)
        out["zip"] = zpath
    return out


_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__HEAD__
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; }
  #overlay { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
             color: #ddd; font: 16px/1.4 system-ui, sans-serif; background: rgba(0,0,0,.85); cursor: pointer; }
  #err { position: fixed; left: 0; top: 0; right: 0; max-height: 50%; overflow: auto; margin: 0; padding: 12px;
         color: #f88; background: rgba(0,0,0,.9); font: 12px/1.4 ui-monospace, monospace; white-space: pre-wrap; display: none; }
  #hud { position: fixed; right: 8px; bottom: 6px; color: rgba(255,255,255,.5); font: 11px system-ui, sans-serif; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<pre id="err"></pre>
<div id="overlay" style="display:none">click to start</div>
__HUD__
<script>
__HELPERS__
var SPEC = __SPEC__;
__MINT__
__RUNTIME__
</script>
</body>
</html>
"""


# The WebGL runtime shared by to_html() and anything else that embeds a SPEC (expects #c, #err, #overlay in the page)
AUDIO_HELPERS_JS = r"""
function hydrapySampleClock(sr, fps) {
  var carry = 0;
  return function() { carry += sr / fps; var n = Math.floor(carry + 1e-9); carry -= n; return n; };
}
function hydrapyAudioQueue(size) {
  var fifo = [], last = new Float32Array(Math.max(size, 1)), pos = 0;
  return {
    length: function(){ return fifo.length; },
    push: function(chunk){
      if (!chunk.length) return;
      fifo.push(chunk);
      if (fifo.length > 6) { fifo.splice(0, fifo.length - 6); pos = 0; }
    },
    pull: function(out){
      var i = 0;
      while (i < out.length) {
        if (!fifo.length) { fifo.push(last); pos = 0; }
        var c = fifo[0], take = Math.min(out.length - i, c.length - pos);
        out.set(c.subarray(pos, pos + take), i); i += take; pos += take;
        if (pos === c.length) { last = fifo.shift(); pos = 0; }
      }
    }
  };
}
"""

RUNTIME_JS = AUDIO_HELPERS_JS + NOISE_TEXTURE_JS + r"""
(function(){
  var canvas = document.getElementById('c');
  var errEl = document.getElementById('err');
  function fail(msg){ errEl.style.display = 'block'; errEl.textContent += msg + "\n"; }
  var gl = canvas.getContext('webgl2', {preserveDrawingBuffer: true, antialias: false, alpha: false});
  var isGL2 = !!gl;
  if (!gl) gl = canvas.getContext('webgl', {preserveDrawingBuffer: true, antialias: false, alpha: false});
  if (!gl) { fail('WebGL is not available'); return; }
  var floatOK = isGL2 ? !!gl.getExtension('EXT_color_buffer_float') : false;

  var VERT = 'attribute vec2 position;\nvoid main(){ gl_Position = vec4(position, 0.0, 1.0); }';
  var VERT3 = '#version 300 es\nin vec2 position;\nvoid main(){ gl_Position = vec4(position, 0.0, 1.0); }';
  // passes that pick mip levels (_texLod) need GLSL ES 3.00; the rest stay 1.00
  function es3(src){ return '#version 300 es\n#define texture2D texture\n#define _texLod textureLod\nout highp vec4 _fragColor;\n#define gl_FragColor _fragColor\n' + src; }
  function shader(type, src){
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { fail(gl.getShaderInfoLog(s) + "\n" + src.split("\n").map(function(l,i){return (i+1)+': '+l;}).join("\n")); return null; }
    return s;
  }
  function program(fs){
    var p = gl.createProgram();
    var v3 = fs.indexOf('#version 300 es') === 0;
    var v = shader(gl.VERTEX_SHADER, v3 ? VERT3 : VERT), f = shader(gl.FRAGMENT_SHADER, fs);
    if (!v || !f) return null;
    gl.attachShader(p, v); gl.attachShader(p, f); gl.bindAttribLocation(p, 0, 'position'); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { fail(gl.getProgramInfoLog(p)); return null; }
    return p;
  }
  var quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  function makeTex(w, h){
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (isGL2 && floatOK && SPEC.floatBuffers !== false) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    var wrap = isGL2 ? gl.REPEAT : gl.CLAMP_TO_EDGE;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
    return t;
  }
  function makeFBO(w, h){
    var tex = makeTex(w, h), fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);   // transparent black, as the GL and NumPy backends
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return {tex: tex, fb: fb};
  }
  function mediaTex(){
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function uploadMedia(t, el){
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  // set_fast_noise(): noise() reads this lattice instead of computing simplex noise per pixel
  var noiseTex = null;
  if (SPEC.fastNoise) {
    noiseTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, noiseTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 256, 0, gl.RGBA, gl.UNSIGNED_BYTE, noiseTexture());
    [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(function(k){ gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST); });
    [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(function(k){ gl.texParameteri(gl.TEXTURE_2D, k, gl.REPEAT); });
  }

  // media sources
  var media = {};   // name -> {tex, el, kind}
  Object.keys(SPEC.images).forEach(function(name){
    var t = mediaTex(), img = new Image();
    img.onload = function(){ uploadMedia(t, img); };
    img.src = SPEC.images[name];
    media[name] = {tex: t, kind: 'image'};
  });
  Object.keys(SPEC.videos).forEach(function(name){
    var t = mediaTex(), v = document.createElement('video');
    v.src = SPEC.videos[name]; v.loop = true; v.muted = true; v.playsInline = true; v.autoplay = true;
    v.play().catch(function(){});
    media[name] = {tex: t, el: v, kind: 'video'};
  });
  Object.keys(SPEC.banks || {}).forEach(function(name){
    var cfg = SPEC.banks[name];
    var texs = cfg.frames.map(function(url){
      var t = mediaTex(), img = new Image();
      img.onload = function(){ uploadMedia(t, img); };
      img.src = url;
      return t;
    });
    media[name] = {texs: texs, kind: 'bank', cfg: cfg};
  });
  SPEC.cameras.forEach(function(name){
    var t = mediaTex(), v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.autoplay = true;
    media[name] = {tex: t, el: v, kind: 'video', cam: true};
  });
  function startCameras(){
    SPEC.cameras.forEach(function(name){
      navigator.mediaDevices.getUserMedia({video: true}).then(function(stream){
        media[name].el.srcObject = stream; media[name].el.play();
      }).catch(function(e){ fail('camera: ' + e); });
    });
  }

  // audio
  var fft = [0,0,0,0,0,0,0,0], vol = 0, flux = 0;
  var audio = null;
  function startAudio(){
    var cfg = SPEC.audio; if (!cfg) return;
    var AC = window.AudioContext || window.webkitAudioContext;
    var actx = new AC();
    var an = actx.createAnalyser(); an.fftSize = 1024; an.smoothingTimeConstant = 0;
    var buf = new Float32Array(an.frequencyBinCount);
    var prev = new Float64Array(24), prevRaw = new Float64Array(24), fluxSm = 0;
    var edges = null;
    function setup(){
      edges = new Int32Array(an.frequencyBinCount);
      for (var i = 0; i < an.frequencyBinCount; i++) {
        var f = i * actx.sampleRate / an.fftSize;
        var bark = 13*Math.atan(0.00076*f) + 3.5*Math.atan((f/7500)*(f/7500));
        edges[i] = Math.min(23, Math.max(0, Math.floor(bark)));
      }
    }
    if (cfg.src) {
      var el = new Audio(cfg.src); el.loop = true; el.crossOrigin = 'anonymous';
      var src = actx.createMediaElementSource(el); src.connect(an); an.connect(actx.destination);
      el.play().catch(function(e){ fail('audio: ' + e); });
    } else {
      navigator.mediaDevices.getUserMedia({audio: true}).then(function(stream){
        actx.createMediaStreamSource(stream).connect(an);
      }).catch(function(e){ fail('mic: ' + e); });
    }
    setup();
    audio = { update: function(){
      an.getFloatFrequencyData(buf);
      var spec = new Float64Array(24);
      for (var i = 0; i < buf.length; i++) { var amp = Math.pow(10, buf[i]/20) * an.fftSize; spec[edges[i]] += amp*amp; }
      var fl = 0, tot = 0;
      for (var b = 0; b < 24; b++) { var v = Math.pow(spec[b], 0.23); fl += Math.max(0, v - prevRaw[b]); prevRaw[b] = v;
        prev[b] = prev[b]*cfg.smooth + v*(1-cfg.smooth); tot += prev[b]; }
      fluxSm = fluxSm*cfg.smooth + (fl/24)*(1-cfg.smooth);
      flux = fluxSm / cfg.scale;
      var per = Math.floor(24 / cfg.bins); vol = 0;
      for (var k = 0; k < cfg.bins; k++) {
        var s = 0; for (var j = 0; j < per; j++) s += prev[k*per + j];
        fft[k] = Math.max(0, (s/per - cfg.cutoff) / cfg.scale); vol += fft[k] / cfg.bins;
      }
    }};
  }

  // compile passes
  var outputs = SPEC.outputs.map(function(o){
    return {
      name: o.name,
      passes: o.passes.map(function(p){
        var src = (SPEC.fastNoise ? '#define _NOISE_TEX\n' : '') + p.source;
        var prog = program(isGL2 && p.source.indexOf('_texLod') >= 0 ? es3(src) : src);
        var locs = {};
        if (prog) {
          ['time','resolution','mouse','_noiseTex'].forEach(function(n){ locs[n] = gl.getUniformLocation(prog, n); });
          o.floats.forEach(function(f){ locs[f.name] = gl.getUniformLocation(prog, f.name); });
          o.samplers.forEach(function(s){ locs[s.name] = gl.getUniformLocation(prog, s.name); });
        }
        return {target: p.target, prog: prog, locs: locs, mipmap: !!p.mipmap && isGL2};
      }),
      floats: o.floats.map(function(f){
        try { return {name: f.name, fn: new Function('time','bpm','__mx','__my','__fft','__vol','__flux','dt', 'return (' + f.js + ');')}; }
        catch (e) { fail('signal ' + f.name + ': ' + e); return {name: f.name, fn: function(){ return 0; }}; }
      }),
      samplers: o.samplers
    };
  });
  var copyProg = program('precision mediump float; uniform sampler2D tex; uniform vec2 resolution;\nvoid main(){ vec4 c = texture2D(tex, gl_FragCoord.xy / resolution); gl_FragColor = vec4(c.rgb * ' + (SPEC.compositeAlpha ? 'clamp(c.a, 0.0, 1.0)' : '1.0') + ', 1.0); }');
  var copyLocs = {tex: gl.getUniformLocation(copyProg, 'tex'), resolution: gl.getUniformLocation(copyProg, 'resolution')};

  var W = 0, H = 0, outs = {}, tmps = {}, lastRenderTime = null;
  var dyn = 1;                       // adaptive render scale (SPEC.adaptive); 1 = full size
  var rawCopy = program('precision mediump float; uniform sampler2D tex; uniform vec2 resolution;\nvoid main(){ gl_FragColor = texture2D(tex, gl_FragCoord.xy / resolution); }');
  function freeTarget(t){ gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.tex); }
  function allocate(keep){
    var old = outs, oldTmps = tmps;
    outs = {}; tmps = {};
    ['o0','o1','o2','o3'].forEach(function(n){
      var depth = (SPEC.history && SPEC.history[n]) || 0, ring = [], sz = osz(n);
      for (var k = 0; k < Math.max(2, depth + 2); k++) ring.push(makeFBO(sz[0], sz[1]));
      outs[n] = {ring: ring, w: sz[0], h: sz[1]};   // ring[0] = latest completed frame, ring[ring.length-1] = written next
      if (keep && old[n]) old[n].ring.forEach(function(fb, k){   // a new size keeps the feedback, resampled
        if (k >= ring.length) return;
        gl.bindFramebuffer(gl.FRAMEBUFFER, ring[k].fb); gl.viewport(0, 0, sz[0], sz[1]);
        gl.useProgram(rawCopy); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fb.tex);
        gl.uniform1i(gl.getUniformLocation(rawCopy, 'tex'), 0);
        gl.uniform2f(gl.getUniformLocation(rawCopy, 'resolution'), sz[0], sz[1]);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      });
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    Object.keys(old).forEach(function(n){ old[n].ring.forEach(freeTarget); });
    Object.keys(oldTmps).forEach(function(k){ freeTarget(oldTmps[k]); });
    // intermediate targets belong to one output and share its size (Chain.out(scale=...))
    outputs.forEach(function(o){ var sz = osz(o.name); o.passes.forEach(function(p){
      var key = o.name + ':' + p.target;
      if (p.target.indexOf('tmp') === 0 && !tmps[key]) tmps[key] = makeFBO(sz[0], sz[1]); }); });
  }
  function resize(){
    var pr = SPEC.pixelRatio || 1;
    var w = SPEC.size ? SPEC.size[0] : Math.floor(window.innerWidth * pr);
    var h = SPEC.size ? SPEC.size[1] : Math.floor(window.innerHeight * pr);
    if (w === W && h === H) return;
    W = w; H = h; canvas.width = W; canvas.height = H;
    if (SPEC.size) canvas.style.objectFit = 'contain';     // a fixed size keeps its aspect, letterboxed
    lastRenderTime = null;
    allocate(false);
  }
  function osz(n){
    var s = ((SPEC.scales && SPEC.scales[n]) || 1) * dyn;
    return [Math.max(1, Math.round(W * s)), Math.max(1, Math.round(H * s))];
  }
  // SPEC.adaptive: when frames come slower than the display refreshes, render the outputs smaller
  // (the canvas keeps its size and upsamples); when there is headroom again, grow back
  var adaptive = SPEC.adaptive || null, ivals = [], refresh = 1000 / 60, lastChange = 0, lastDown = -1e9, ceiling = 1, prevNow = null;
  function adapt(now){
    if (prevNow !== null) { var iv = now - prevNow; if (iv > 0 && iv < 1000) ivals.push(iv); }
    prevNow = now;
    var span = 0; ivals.forEach(function(v){ span += v; });
    if (ivals.length < 45 && (span < 1500 || ivals.length < 4)) return;   // 45 frames, or 1.5 s of slow ones
    while (ivals.length > 45) ivals.shift();
    var med = ivals.slice().sort(function(a, b){ return a - b; })[ivals.length >> 1];
    refresh = Math.min(refresh, Math.max(med, 1000 / 240));   // a 90/120 Hz display shows itself
    if (now - lastChange < 1500) return;
    var next = dyn;
    if (med > refresh * 1.45 && dyn > adaptive.min) { next = Math.max(adaptive.min, dyn * adaptive.step); ceiling = dyn; lastDown = now; }
    else if (med < refresh * 1.12 && dyn < 1 && (dyn / adaptive.step < ceiling - 1e-6 || now - lastDown > 20000))
      next = Math.min(1, dyn / adaptive.step);
    if (next === dyn) return;
    dyn = next; lastChange = now; ivals = [];
    canvas.dataset.scale = dyn.toFixed(3);
    allocate(true);
  }
  window.addEventListener('resize', resize);
  resize();

  var mx = 0.5, my = 0.5;
  window.addEventListener('mousemove', function(e){ mx = e.clientX / window.innerWidth; my = 1 - e.clientY / window.innerHeight; });
  var paused = false, t0 = null, pausedAt = 0;
  window.addEventListener('keydown', function(e){
    if (e.key === ' ') { paused = !paused; if (paused) pausedAt = performance.now(); else t0 += performance.now() - pausedAt; e.preventDefault(); }
    if (e.key === 'f') { try { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(function(){}); } catch (err) {} }
    if (e.key === 's') { var a = document.createElement('a'); a.download = 'hydrapy.png'; a.href = canvas.toDataURL('image/png'); a.click(); }
    if (e.key >= '1' && e.key <= '4') SPEC.display = 'o' + (e.key - 1);
  });

  var curTime = 0;
  function bankIndex(cfg, n, time){
    var k = Math.floor(time * cfg.rate / Math.max(cfg.hold, 1));
    if (cfg.mode === 'random') return ((Math.floor(__hash11(k + cfg.seed * 131.0) * n)) % n + n) % n;
    return ((k % n) + n) % n;
  }
  // neural cellular automata: fixed-size float state, stepped before the outputs when a frame is drawn
  var ncas = {};
  (SPEC.ncas || []).forEach(function(cfg){
    if (!(isGL2 && floatOK)) { fail('NCA ' + cfg.name + ' needs WebGL2 float render targets'); return; }
    function stateFBO(){
      var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, cfg.w, cfg.h, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      var fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return {tex: t, fb: fb};
    }
    function fn(js){ try { return new Function('time','bpm','__mx','__my','__fft','__vol','__flux','dt', 'return (' + js + ');'); } catch (e) { fail('nca signal: ' + e); return function(){ return 0; }; } }
    var names = ['res','stepIndex','rate','perturb','seed','paintAmount','damageOn','paintOn','damage','paint'];
    for (var t = 0; t < cfg.textures; t++) names.push('s' + t);
    var steps = cfg.steps.map(function(src){
      var prog = program(src), locs = {};
      if (prog) names.forEach(function(n){ locs[n] = gl.getUniformLocation(prog, n); });
      return {prog: prog, locs: locs};
    });
    var view = program(cfg.view);
    var sets = [0, 1].map(function(){ var a = []; for (var t = 0; t < cfg.textures; t++) a.push(stateFBO()); return a; });
    var out = makeFBO(cfg.w, cfg.h);
    if (cfg.filter === 'nearest') { gl.bindTexture(gl.TEXTURE_2D, out.tex); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); }
    ncas[cfg.name] = {cfg: cfg, steps: steps, view: view, viewLocs: view ? {s0: gl.getUniformLocation(view, 's0'), res: gl.getUniformLocation(view, 'res')} : {},
                      sets: sets, cur: 0, out: out, stepIndex: 0, target: null,
                      rate: fn(cfg.rate), perturb: fn(cfg.perturb), paintAmount: fn(cfg.paintAmount)};
  });
  function clearSets(n){ n.sets.forEach(function(a){ a.forEach(function(s){ gl.bindFramebuffer(gl.FRAMEBUFFER, s.fb); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); }); }); }
  function stepNcas(time, dt){
    Object.keys(ncas).forEach(function(name){
      var n = ncas[name], cfg = n.cfg, k = Math.floor(time * cfg.speed + 1e-9), count;
      if (n.target === null || k < n.target) {             // first frame, or time went back: start over
        if (n.target !== null) { clearSets(n); n.stepIndex = 0; n.cur = 0; }
        n.target = k; count = n.stepIndex === 0 ? 1 : 0;
      } else { count = Math.min(k - n.target, cfg.maxSteps); n.target = k; }
      var args = [time, SPEC.bpm, mx, my, fft, vol, flux, dt];
      var rate = n.rate.apply(null, args), perturb = n.perturb.apply(null, args), amount = n.paintAmount.apply(null, args);
      var T = cfg.textures;
      var dmg = cfg.damage ? outs[cfg.damage].ring[0].tex : null, pnt = cfg.paint ? outs[cfg.paint].ring[0].tex : null;
      gl.viewport(0, 0, cfg.w, cfg.h);
      for (var i = 0; i < count; i++) {
        var cur = n.sets[n.cur], nxt = n.sets[1 - n.cur];
        n.steps.forEach(function(st, kk){
          if (!st.prog) return;
          gl.bindFramebuffer(gl.FRAMEBUFFER, nxt[kk].fb);
          gl.useProgram(st.prog);
          for (var t = 0; t < T; t++) { gl.activeTexture(gl.TEXTURE0 + t); gl.bindTexture(gl.TEXTURE_2D, cur[t].tex); gl.uniform1i(st.locs['s' + t], t); }
          gl.activeTexture(gl.TEXTURE0 + T); gl.bindTexture(gl.TEXTURE_2D, dmg); gl.uniform1i(st.locs.damage, T);
          gl.activeTexture(gl.TEXTURE0 + T + 1); gl.bindTexture(gl.TEXTURE_2D, pnt); gl.uniform1i(st.locs.paint, T + 1);
          gl.uniform2f(st.locs.res, cfg.w, cfg.h);
          gl.uniform1f(st.locs.stepIndex, n.stepIndex);
          gl.uniform1f(st.locs.rate, rate); gl.uniform1f(st.locs.perturb, perturb); gl.uniform1f(st.locs.seed, cfg.seed);
          gl.uniform1f(st.locs.paintAmount, amount);
          gl.uniform1f(st.locs.damageOn, dmg ? 1 : 0); gl.uniform1f(st.locs.paintOn, pnt ? 1 : 0);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
        });
        n.cur = 1 - n.cur; n.stepIndex++;
      }
      if (n.view) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, n.out.fb);
        gl.useProgram(n.view);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, n.sets[n.cur][0].tex);
        gl.uniform1i(n.viewLocs.s0, 0); gl.uniform2f(n.viewLocs.res, cfg.w, cfg.h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    });
    gl.viewport(0, 0, W, H);
  }

  // physarum: agents in a float texture, deposited as points (a WebGL2 / GLSL ES 3.00 pass), trail diffused
  var sims = {};
  (SPEC.physarum || []).forEach(function(cfg){
    if (!(isGL2 && floatOK)) { fail('Physarum ' + cfg.name + ' needs WebGL2 float render targets'); return; }
    function fbo32(w, h){
      var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
      [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(function(p){ gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST); });
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      var fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return {tex: t, fb: fb};
    }
    function prog2(vs, fs){
      var p = gl.createProgram(), v = shader(gl.VERTEX_SHADER, vs), f = shader(gl.FRAGMENT_SHADER, fs);
      if (!v || !f) return null;
      gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { fail(gl.getProgramInfoLog(p)); return null; }
      return p;
    }
    function locs(p, names){ var o = {}; names.forEach(function(n){ o[n] = p ? gl.getUniformLocation(p, n) : null; }); return o; }
    var params = {};
    Object.keys(cfg.params).forEach(function(k){ try { params[k] = new Function('time','bpm','__mx','__my','__fft','__vol','__flux','dt', 'return (' + cfg.params[k] + ');'); } catch (e) { fail('physarum ' + k + ': ' + e); params[k] = function(){ return 0; }; } });
    var s = {cfg: cfg, params: params, init: program(cfg.init), update: program(cfg.update), diffuse: program(cfg.diffuse),
             view: program(cfg.view), deposit: prog2(cfg.depositVert, cfg.depositFrag),
             agents: [fbo32(cfg.aw, cfg.ah), fbo32(cfg.aw, cfg.ah)], trail: [makeFBO(cfg.w, cfg.h), makeFBO(cfg.w, cfg.h)],
             out: makeFBO(cfg.w, cfg.h), a: 0, t: 0, stepIndex: 0, target: null, fresh: true};
    s.uInit = locs(s.init, ['seed']);
    s.uUpdate = locs(s.update, ['agents','trail','food','stepIndex','seed','sensorAngle','sensorDistance','turn','stepSize','foodWeight','foodOn']);
    s.uDiffuse = locs(s.diffuse, ['trail','decay','diffuse']);
    s.uView = locs(s.view, ['trail','exposure']);
    s.uDeposit = locs(s.deposit, ['agents','amount']);
    sims[cfg.name] = s;
  });
  function stepSims(time, dt){
    Object.keys(sims).forEach(function(name){
      var s = sims[name], cfg = s.cfg, k = Math.floor(time * cfg.speed + 1e-9), count;
      if (s.target === null || k < s.target) {
        if (s.target !== null) { s.stepIndex = 0; s.fresh = true; }
        s.target = k; count = s.stepIndex === 0 ? 1 : 0;
      } else { count = Math.min(k - s.target, cfg.maxSteps); s.target = k; }
      var args = [time, SPEC.bpm, mx, my, fft, vol, flux, dt], p = {};
      Object.keys(s.params).forEach(function(n){ p[n] = s.params[n].apply(null, args); });
      if (s.fresh) {
        s.trail.forEach(function(t){ gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); });
        gl.bindFramebuffer(gl.FRAMEBUFFER, s.agents[s.a].fb); gl.viewport(0, 0, cfg.aw, cfg.ah);
        gl.useProgram(s.init); gl.uniform1f(s.uInit.seed, cfg.seed); gl.drawArrays(gl.TRIANGLES, 0, 3);
        s.fresh = false;
      }
      var food = cfg.food ? outs[cfg.food].ring[0].tex : null;
      for (var i = 0; i < count; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, s.agents[1 - s.a].fb); gl.viewport(0, 0, cfg.aw, cfg.ah);
        gl.useProgram(s.update);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, s.agents[s.a].tex); gl.uniform1i(s.uUpdate.agents, 0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, s.trail[s.t].tex); gl.uniform1i(s.uUpdate.trail, 1);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, food); gl.uniform1i(s.uUpdate.food, 2);
        gl.uniform1f(s.uUpdate.stepIndex, s.stepIndex); gl.uniform1f(s.uUpdate.seed, cfg.seed);
        gl.uniform1f(s.uUpdate.sensorAngle, p.sensor_angle); gl.uniform1f(s.uUpdate.sensorDistance, p.sensor_distance);
        gl.uniform1f(s.uUpdate.turn, p.turn); gl.uniform1f(s.uUpdate.stepSize, p.step);
        gl.uniform1f(s.uUpdate.foodWeight, p.food_weight); gl.uniform1f(s.uUpdate.foodOn, food ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        s.a = 1 - s.a;
        if (s.deposit) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, s.trail[s.t].fb); gl.viewport(0, 0, cfg.w, cfg.h);
          gl.useProgram(s.deposit);
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, s.agents[s.a].tex); gl.uniform1i(s.uDeposit.agents, 0);
          gl.uniform1f(s.uDeposit.amount, p.deposit);
          gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
          gl.disableVertexAttribArray(0);
          gl.drawArrays(gl.POINTS, 0, cfg.aw * cfg.ah);
          gl.enableVertexAttribArray(0);
          gl.disable(gl.BLEND);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, s.trail[1 - s.t].fb); gl.viewport(0, 0, cfg.w, cfg.h);
        gl.useProgram(s.diffuse);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, s.trail[s.t].tex); gl.uniform1i(s.uDiffuse.trail, 0);
        gl.uniform1f(s.uDiffuse.decay, p.decay); gl.uniform1f(s.uDiffuse.diffuse, p.diffuse);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        s.t = 1 - s.t; s.stepIndex++;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, s.out.fb); gl.viewport(0, 0, cfg.w, cfg.h);
      gl.useProgram(s.view);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, s.trail[s.t].tex); gl.uniform1i(s.uView.trail, 0);
      gl.uniform1f(s.uView.exposure, p.exposure);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    });
    gl.viewport(0, 0, W, H);
  }

  function bindRef(ref, delay, owner){
    if (outs[ref]) { var ring = outs[ref].ring; return ring[Math.min(delay || 0, ring.length - 2)].tex; }
    if (owner && tmps[owner + ':' + ref]) return tmps[owner + ':' + ref].tex;
    if (ncas[ref]) return ncas[ref].out.tex;
    if (sims[ref]) return sims[ref].out.tex;
    if (media[ref]) {
      var m = media[ref];
      if (m.kind === 'bank') return m.texs[bankIndex(m.cfg, m.texs.length, curTime)];
      if (m.el && m.el.readyState >= 2) uploadMedia(m.tex, m.el);
      return m.tex;
    }
    return null;
  }

  function frame(now){
    requestAnimationFrame(frame);
    if (t0 === null) t0 = now;
    if (paused) return;
    if (adaptive) adapt(now);
    var time = ((now - t0) / 1000) * SPEC.speed;
    if (SPEC.frameRate) time = Math.floor(time * SPEC.frameRate + 1e-9) / SPEC.frameRate;
    var draw = !SPEC.frameRate || time !== lastRenderTime;
    var dt = lastRenderTime === null ? 1 / (SPEC.frameRate || 60) : Math.max(0, time - lastRenderTime);
    curTime = time;
    if (audio) audio.update();
    // only the displayed output and what it samples (transitively) need rendering this frame
    var needed = {}; var stack = [SPEC.display];
    if (SPEC.soundtrack) stack.push(SPEC.soundtrack.output);
    (SPEC.ncas || []).forEach(function(c){ if (c.damage) stack.push(c.damage); if (c.paint) stack.push(c.paint); });
    (SPEC.physarum || []).forEach(function(c){ if (c.food) stack.push(c.food); });
    while (stack.length) { var nm = stack.pop(); if (needed[nm]) continue; needed[nm] = true;
      outputs.forEach(function(o){ if (o.name === nm) o.samplers.forEach(function(s){ if (/^o[0-3]$/.test(s.ref) && !needed[s.ref]) stack.push(s.ref); }); }); }
    if (draw) { stepNcas(time, dt); stepSims(time, dt); }
    if (draw) outputs.forEach(function(o){
      if (!needed[o.name]) return;
      var vals = o.floats.map(function(f){ var v = f.fn(time, SPEC.bpm, mx, my, fft, vol, flux, dt); return isFinite(v) ? v : 0; });
      var OW = outs[o.name].w, OH = outs[o.name].h;
      o.passes.forEach(function(p){
        if (!p.prog) return;
        var target = p.target.indexOf('tmp') === 0 ? tmps[o.name + ':' + p.target] : outs[o.name].ring[outs[o.name].ring.length - 1];
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
        gl.viewport(0, 0, OW, OH);
        gl.useProgram(p.prog);
        gl.uniform1f(p.locs.time, time);
        gl.uniform2f(p.locs.resolution, OW, OH);
        gl.uniform2f(p.locs.mouse, mx * OW, my * OH);
        o.floats.forEach(function(f, i){ gl.uniform1f(p.locs[f.name], vals[i]); });
        o.samplers.forEach(function(s, i){
          var tex = bindRef(s.ref, s.delay, o.name);
          gl.activeTexture(gl.TEXTURE0 + i);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.uniform1i(p.locs[s.name], i);
        });
        if (noiseTex && p.locs._noiseTex) {
          gl.activeTexture(gl.TEXTURE0 + o.samplers.length);
          gl.bindTexture(gl.TEXTURE_2D, noiseTex);
          gl.uniform1i(p.locs._noiseTex, o.samplers.length);
        }
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        if (p.mipmap) {                    // a mipblur reads this target's levels
          gl.bindTexture(gl.TEXTURE_2D, target.tex);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          gl.generateMipmap(gl.TEXTURE_2D);
        }
      });
      outs[o.name].ring.unshift(outs[o.name].ring.pop());   // the frame just written becomes ring[0]
    });
    if (draw) lastRenderTime = time;
    if (track) track.read(now, draw);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(copyProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, outs[SPEC.display].ring[0].tex);
    gl.uniform1i(copyLocs.tex, 0);
    gl.uniform2f(copyLocs.resolution, W, H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (draw && typeof hydrapyAfterFrame === 'function') hydrapyAfterFrame(time);
  }

  // the optical soundtrack: a strip of an output, read top to bottom once per 1/fps, is the waveform
  var track = null;
  function startSoundtrack(){
    var cfg = SPEC.soundtrack; if (!cfg) return;
    var AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    var actx = new AC();
    var sr = actx.sampleRate, nextCount = hydrapySampleClock(sr, cfg.fps);
    var queue = hydrapyAudioQueue(Math.round(sr / cfg.fps));
    var nextT = null, lastCol = null, dirty = true, hpx = 0, hpy = 0, hpa = Math.exp(-2 * Math.PI * 20 / sr);
    var useFloat = isGL2 && floatOK;
    var node = actx.createScriptProcessor(2048, 0, 1);
    node.onaudioprocess = function(e){
      queue.pull(e.outputBuffer.getChannelData(0));
    };
    node.connect(actx.destination);
    var dbg = window.__hydrapySoundtrack = {chunks: 0, samples: 0, sr: sr, state: function(){ return actx.state; }};
    track = { read: function(now, changed){
      dirty = dirty || changed;
      if (nextT === null) nextT = now;
      if (now < nextT - 1e-6) return;
      var period = 1000 / cfg.fps;
      // Bound catch-up after a suspended tab. Audio keeps holding the last complete strip.
      if (now - nextT > period * 6) nextT = now - period * 5;
      var col = lastCol;
      var TW = outs[cfg.output].w, TH = outs[cfg.output].h;
      if (dirty || !col || col.length !== TH) {
        var w = Math.min(Math.max(Math.round(cfg.width * TW), 1), TW);
        var x0 = Math.min(Math.max(Math.floor(cfg.x * TW - w / 2), 0), TW - w);
        var pix = useFloat ? new Float32Array(w * TH * 4) : new Uint8Array(w * TH * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, outs[cfg.output].ring[0].fb);
        gl.readPixels(x0, 0, w, TH, gl.RGBA, useFloat ? gl.FLOAT : gl.UNSIGNED_BYTE, pix);
        col = new Float64Array(TH);
        var scale = useFloat ? 1 : 1 / 255;
        for (var y = 0; y < TH; y++) {
          var s = 0;
          for (var xx = 0; xx < w; xx++) {
            var o = ((y * w) + xx) * 4;
            s += (pix[o] * 0.2125 + pix[o+1] * 0.7154 + pix[o+2] * 0.0721) * scale * (SPEC.compositeAlpha ? Math.max(0, Math.min(1, pix[o+3] * scale)) : 1);
          }
          col[TH - 1 - y] = s / w;
        }
        dirty = false;
      }
      while (nextT <= now + 1e-6) {
        var n = nextCount(), chunk = new Float32Array(n);
        var column = lastCol && nextT < now - 1e-6 ? lastCol : col, height = column.length;
        for (var j = 0; j < n; j++) {
          var p = (j + 0.5) * height / n - 0.5, p0 = Math.max(0, Math.min(height - 1, Math.floor(p)));
          var p1 = Math.min(height - 1, p0 + 1), f = Math.max(0, Math.min(1, p - p0));
          var v = ((column[p0] * (1-f) + column[p1] * f) - 0.5) * 2 * cfg.gain;
          if (cfg.ac) { hpy = v - hpx + hpa * hpy; hpx = v; v = hpy; }
          chunk[j] = Math.max(-1, Math.min(1, v));
        }
        queue.push(chunk); nextT += period; dbg.chunks++; dbg.samples += n;
      }
      lastCol = col;
    }};
  }

  var needsGesture = !!SPEC.audio || !!SPEC.soundtrack || SPEC.cameras.length > 0;
  var overlay = document.getElementById('overlay');
  function start(){ overlay.style.display = 'none'; startAudio(); startSoundtrack(); startCameras(); requestAnimationFrame(frame); }
  if (needsGesture) { overlay.style.display = 'flex'; overlay.addEventListener('click', start, {once: true}); }
  else start();
})();"""
