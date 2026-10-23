"""hydrapy — Hydra's video synth as a Python library, with a modular-synth twist.

    from hydrapy import *
    osc(10, 0.1, 0.8).rotate(Lfo(0.1, lo=0, hi=6.28)).kaleid(5).out(o0)
    save_gif(render_frames(seconds=4), "out.gif")

Sources:  osc noise voronoi shape gradient solid src prev  +  fbm rings grid white
Coords:   rotate scale pixelate repeat repeatX repeatY kaleid scroll scrollX scrollY
          +  polar unpolar twist fisheye mirror wobble glitch
Colour:   posterize shift invert contrast brightness luma thresh color saturate hue
          colorama sum r g b a  +  fold gamma levels dither palette grain vignette
Combine:  add sub layer blend mult diff mask  +  lighten darken screen overlay wipe
Modulate: modulate modulateRepeat modulateRepeatX modulateRepeatY modulateKaleid
          modulateScrollX modulateScrollY modulateScale modulatePixelate modulateRotate
          modulateHue  +  displace
Texture:  blur sharpen edge emboss dilate erode life rd chroma bleed bake plane3d sphere3d cylinder3d pixelsort(CPU)
Paint:    stain alpha drops crackle veins scratches dust pan  (hydrapy.recipes: handpainted mothlight film)
Textures: src(o0) prev() delay(o0, k) Image Video Camera Bank Quilt Text Lines
Signals:  Lfo Seq Clock Env SH Drift Mouse Time Fn Fft Audio
Live:     CvIn CvOut (ES-6/ES-3 or any DC-coupled interface)  OscIn OscOut  Midi  Link  Probe
Glitch:   Fx (fxhash sfc32, entropy locking)  EntropyLock  tear  shred
"""
from __future__ import annotations

from . import ops as _ops  # noqa: F401  (populates the registry)
from . import ops_paint as _ops_paint  # noqa: F401  (paint-on-film ops)
from . import ops_3d as _ops_3d  # noqa: F401  (plane/sphere/cylinder)
from . import ops_glitch as _ops_glitch  # noqa: F401  (tear/shred)
from .core import (REGISTRY, Bank, Camera, Chain, Ctx, External, Image, Input, Op, Output, Patch, Texture, Transform, Video,
                   default_patch, normalize_args, reset_patch)
from .synth import Quilt, quilt
from .drawing import Lines, Text, meadow, plant, to_svg
from .live import CvIn, CvOut, Link, LiveValue, Midi, MidiClock, OscIn, OscOut, Probe, Transport, TrigPhase, audio_devices, register_hook
from .signals import SH, Audio, Clock, Const, Drift, EntropyLock, Env, Expr, Fft, Fn, Fx, Lfo, Mouse, Seq, Signal, Time, as_signal

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Build the chainable API from the registry
# ---------------------------------------------------------------------------

def _make_source(op: Op):
    def source(*args, **kwargs) -> Chain:
        return Chain([Transform(op, normalize_args(op, args, dict(kwargs)))])
    source.__name__ = op.name
    source.__doc__ = f"{op.doc}\n\nargs: " + ", ".join(f"{i.name}={i.default}" for i in op.inputs)
    return source


def _make_method(op: Op):
    def method(self: Chain, *args, **kwargs) -> Chain:
        return self._with(op, normalize_args(op, args, dict(kwargs)))
    method.__name__ = op.name
    method.__doc__ = f"{op.doc}\n\nargs: " + ", ".join(f"{i.name}={i.default}" for i in op.inputs)
    return method


_sources: dict[str, callable] = {}
for _name, _op in REGISTRY.items():
    if _op.kind == "src":
        _sources[_name] = _make_source(_op)
        globals()[_name] = _sources[_name]
    else:
        setattr(Chain, _name, _make_method(_op))

# snake_case aliases for the camelCase Hydra names
for _name, _op in list(REGISTRY.items()):
    import re as _re
    _snake = _re.sub(r"(?<!^)(?=[A-Z])", "_", _name).lower()
    if _snake != _name:
        if _op.kind == "src":
            globals()[_snake] = globals()[_name]
        else:
            setattr(Chain, _snake, getattr(Chain, _name))


# ---------------------------------------------------------------------------
# Default patch and Hydra-style globals
# ---------------------------------------------------------------------------

_patch = default_patch()
o0, o1, o2, o3 = _patch.outputs


def render(output: Output | None = None) -> Patch:
    """Choose which output is displayed (Hydra's ``render(o1)``)."""
    return default_patch().render(output)


def hush():
    default_patch().hush()


def set_bpm(bpm: float):
    default_patch().bpm = float(bpm)


def set_resolution(width: int, height: int):
    default_patch().size = (int(width), int(height))


def speed(s: float):
    default_patch().speed = float(s)


def patch() -> Patch:
    return default_patch()


# ---------------------------------------------------------------------------
# Rendering / export shortcuts
# ---------------------------------------------------------------------------

from .backend_np import Renderer, render_chain, render_frames  # noqa: E402
from .export import (chain_to_hydra, patch_to_hydra, save_gif, save_mp4, save_png, to_ansi, to_html,  # noqa: E402
                     to_image)
from .glsl import compile_chain, compile_output, compile_patch  # noqa: E402
from .hydra_import import from_hydra  # noqa: E402


def add_op(op: Op):
    """Register a user-defined op and attach it to Chain (or expose it as a source)."""
    from .core import register
    register(op)
    if op.kind == "src":
        _sources[op.name] = _make_source(op)
        globals()[op.name] = _sources[op.name]
    else:
        setattr(Chain, op.name, _make_method(op))
    return op


def ops() -> list[str]:
    """Names of every available op."""
    return list(REGISTRY)


def describe(name: str) -> str:
    op = REGISTRY[name]
    args = ", ".join(f"{i.name}={i.default}" for i in op.inputs)
    extra = " (Hydra)" if op.hydra else " (new)"
    cpu = " [CPU only]" if op.cpu_only else ""
    return f"{op.kind:<12} {name}({args}){extra}{cpu}\n    {op.doc}"


__all__ = (
    list(_sources) + [
        "o0", "o1", "o2", "o3", "render", "hush", "set_bpm", "set_resolution", "speed", "patch",
        "Chain", "Patch", "Output", "Image", "Video", "Camera", "Bank", "Quilt", "quilt", "Text", "Lines", "plant", "meadow", "to_svg",
        "CvIn", "CvOut", "Link", "LiveValue", "Midi", "MidiClock", "OscIn", "OscOut", "Probe", "Transport", "TrigPhase", "audio_devices", "register_hook", "Ctx", "reset_patch", "default_patch",
        "Signal", "Lfo", "Seq", "Clock", "Env", "SH", "Drift", "Mouse", "Time", "Fn", "Fx", "EntropyLock", "Fft", "Audio", "Const", "Expr", "External",
        "Renderer", "render_chain", "render_frames", "save_gif", "save_mp4", "save_png", "to_image", "to_ansi",
        "to_html", "chain_to_hydra", "patch_to_hydra", "from_hydra",
        "compile_chain", "compile_output", "compile_patch", "ops", "describe", "REGISTRY", "add_op", "Op", "Input",
    ]
)
