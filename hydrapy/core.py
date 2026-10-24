"""Chains, ops, outputs and patches.

A ``Chain`` is an immutable list of transforms, exactly like a Hydra chain:
``osc(10).rotate(0.5).out(o0)``.  Every op is declared once in ``ops.py`` with
both a GLSL body and a NumPy implementation; the backends walk the same chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Op declarations
# ---------------------------------------------------------------------------

KINDS = ("src", "coord", "color", "combine", "combineCoord", "texture")


@dataclass
class Input:
    name: str
    type: str = "float"       # float | vec2 | vec4 | sampler2D | int
    default: Any = 0.0
    scalar: bool = False      # True: must be a per-frame scalar (no per-pixel chains)
    doc: str = ""
    minimum: Optional[float] = None   # suggested UI range, never an evaluation clamp
    maximum: Optional[float] = None
    unit: str = ""
    scale: str = "linear"


@dataclass
class Op:
    name: str
    kind: str
    inputs: list[Input]
    glsl: str                      # function body (GLSL ES 1.00)
    np: Optional[Callable] = None  # NumPy implementation
    cpu_only: bool = False         # no GLSL implementation (pixelsort etc.)
    iterations: bool = False       # texture op that may run several passes per frame
    doc: str = ""
    hydra: bool = False            # part of Hydra's own op set (exportable to Hydra JS)
    glsl_helpers: str = ""         # optional op-specific functions before the body
    python_source: Optional[Callable] = None  # (args, formatter) -> constructor expression
    compute: Optional[str] = None  # opt-in desktop kernel for a whole, unwarped texture pass
    identity: Optional[tuple] = None    # (input name, value): the op does nothing at that value (see glsl.gates)

    @property
    def returns(self) -> str:
        return {"src": "vec4", "coord": "vec2", "color": "vec4", "combine": "vec4",
                "combineCoord": "vec2", "texture": "vec4"}[self.kind]

    @property
    def fixed_args(self) -> str:
        return {"src": "vec2 _st", "coord": "vec2 _st", "color": "vec4 _c0",
                "combine": "vec4 _c0, vec4 _c1", "combineCoord": "vec2 _st, vec4 _c0",
                "texture": "sampler2D _tex, vec2 _st"}[self.kind]


REGISTRY: dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.kind not in KINDS:
        raise ValueError(f"unknown op kind {op.kind!r}")
    REGISTRY[op.name] = op
    return op


# ---------------------------------------------------------------------------
# Textures: things ``src()`` can read
# ---------------------------------------------------------------------------

class Texture:
    """Base class for anything a shader can sample: outputs, images, cameras."""
    name: str = "tex"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"


class FrameSource(Texture):
    """A Python ``draw(seconds) -> uint8 RGB/RGBA`` animation, with a one-frame cache.

    Works in the NumPy and desktop GL renderers. Unlike Bank, GL updates one
    texture instead of keeping every past frame. The callable must produce a
    deterministic result for the requested time, including a backwards seek.
    Live HTML/Hydra JS export requires baking a finite sequence into a Bank.
    """
    _counter = 0

    def __init__(self, draw):
        FrameSource._counter += 1
        self.name = f"dynamic{FrameSource._counter}"
        self.draw = draw
        self._time = None
        self._frame = self._array = None

    def frame_at(self, t):
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("frame time must be finite")
        if self._time != t:
            frame = np.asarray(self.draw(t))
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4) or min(frame.shape[:2]) < 1:
                raise ValueError("FrameSource.draw must return a nonempty uint8 RGB/RGBA image")
            if frame.shape[2] == 3:
                frame = np.concatenate([frame, np.full(frame.shape[:2] + (1,), 255, np.uint8)], axis=-1)
            self._frame = np.array(frame, copy=True, order="C")
            self._time, self._array = t, None
        return self._frame

    def array_at(self, t):
        frame = self.frame_at(t)
        if self._array is None:
            self._array = (frame[::-1].astype(np.float32) / 255).copy()
        return self._array


class Output(Texture):
    """One of the four render targets o0..o3 (double-buffered for feedback)."""

    def __init__(self, index: int):
        self.index = index
        self.name = f"o{index}"
        self.chain: Optional[Chain] = None

    def __repr__(self):
        return f"<Output {self.name}>"


class DelayRef(Texture):
    """``delay(oN, k)``: the frame of output oN from k frames ago (k = 0 is what src(oN) gives)."""

    def __init__(self, output: "Output", frames: int):
        self.output = output
        self.frames = int(frames)
        self.name = f"{output.name}_d{self.frames}"


class Image(Texture):
    """A still image used as a source: ``src(Image('photo.png'))``."""

    _counter = 0

    def __init__(self, path: str, flip: bool = True):
        self.path = path
        self.flip = flip
        Image._counter += 1
        self.name = f"img{Image._counter}"
        self._array = None

    def array(self):
        """Float32 (H, W, 4) array, row 0 = bottom (GL convention)."""
        if self._array is None:
            import numpy as np
            from PIL import Image as PILImage
            im = PILImage.open(self.path).convert("RGBA")
            a = np.asarray(im).astype(np.float32) / 255.0
            self._array = a[::-1].copy() if self.flip else a
        return self._array


class External(Texture):
    """A source hydrapy can't read offline (a p5 canvas, the screen, a stream): black until replaced."""

    def __init__(self, name: str = "s0", note: str = ""):
        self.name = name
        self.note = note

    def array(self):
        return np.zeros((2, 2, 4), dtype=np.float32)

    def render(self):
        return np.zeros((2, 2, 4), dtype=np.uint8)

    def __repr__(self):
        return f"<External {self.name}: {self.note}>"


class Camera(Texture):
    """Webcam source. Live in the HTML export and the GL backend (via OpenCV); not available offline."""

    def __init__(self, device: int = 0):
        self.device = device
        self.name = f"cam{device}"


class Bank(Texture):
    """A set of still images shown one per frame: Mothlight from your own light-table photos.

    ``items``: a folder, a glob, or a list whose entries are image paths, numpy arrays, PIL images
    or static textures (``Image``, ``Text``, ``Lines``).  ``rate`` images per second, each held for
    ``hold`` frames, in ``mode`` 'cycle' or 'random' order.
    ``src(Bank("wings/*.jpg", rate=24, mode="random"))``.
    """

    _counter = 0

    def __init__(self, items, rate: float = 24.0, hold: int = 1, mode: str = "cycle", seed: float = 0.0,
                 max_size: int = 1280):
        import glob, os
        if isinstance(items, str):
            if os.path.isdir(items):
                found = sorted(p for p in glob.glob(os.path.join(items, "*"))
                               if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")))
            else:
                found = sorted(glob.glob(items))
            if not found:
                raise ValueError(f"Bank: no images found in {items!r}")
        else:
            found = list(items)
            if not found:
                raise ValueError("Bank: no items")
        self.items = found
        self.rate, self.hold, self.mode, self.seed, self.max_size = float(rate), int(hold), mode, float(seed), int(max_size)
        Bank._counter += 1
        self.name = f"bank{Bank._counter}"
        self._frames: dict[int, np.ndarray] = {}
        self._rgba: dict[int, np.ndarray] = {}

    @property
    def count(self) -> int:
        return len(self.items)

    def frame(self, i: int):
        """uint8 RGBA (H, W, 4), row 0 = top, downscaled to max_size."""
        i = int(i) % self.count
        if i not in self._frames:
            self._frames[i] = _to_rgba8(self.items[i], self.max_size)
        return self._frames[i]

    def frames(self):
        return [self.frame(i) for i in range(self.count)]

    def index_at(self, t: float) -> int:
        from .synth import bank_index
        return bank_index(t, self.count, self.rate, self.hold, self.mode, self.seed)

    def array_at(self, t: float):
        i = self.index_at(t)
        if i not in self._rgba:
            self._rgba[i] = (self.frame(i).astype(np.float32) / 255.0)[::-1].copy()
        return self._rgba[i]


def _to_rgba8(item, max_size: int = 0) -> np.ndarray:
    """Anything image-like -> uint8 RGBA (H, W, 4), row 0 = top."""
    from PIL import Image as PILImage
    if isinstance(item, str):
        im = PILImage.open(item).convert("RGBA")
        if max_size and max(im.size) > max_size:
            im.thumbnail((max_size, max_size))
        return np.asarray(im, dtype=np.uint8)
    if hasattr(item, "render"):                 # Text, Lines: uint8 RGBA row 0 = top
        return np.asarray(item.render(), dtype=np.uint8)
    if hasattr(item, "array") and callable(item.array):   # Image-like: float RGBA, row 0 = bottom
        a = item.array()
        return (np.clip(a[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    if hasattr(item, "convert"):
        return np.asarray(item.convert("RGBA"), dtype=np.uint8)
    a = np.asarray(item)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    return a


class Video(Texture):
    """A video file used as a source (decoded with imageio/ffmpeg)."""

    _counter = 0

    def __init__(self, path: str, loop: bool = True, fps: float | None = None):
        self.path = path
        self.loop = loop
        self.fps = fps
        Video._counter += 1
        self.name = f"vid{Video._counter}"
        self._reader = None
        self._frames = None

    def frame_at(self, t: float):
        import numpy as np
        if self._frames is None:
            try:
                import imageio.v2 as iio2
                reader = iio2.get_reader(self.path)
                meta = reader.get_meta_data()
                self.fps = self.fps or meta.get("fps", 30)
                self._frames = [f for f in reader]
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"could not decode video {self.path}: {e}")
        n = len(self._frames)
        i = int(t * self.fps)
        i = i % n if self.loop else min(i, n - 1)
        f = self._frames[i].astype(np.float32) / 255.0
        if f.shape[-1] == 3:
            f = np.concatenate([f, np.ones_like(f[..., :1])], axis=-1)
        return f[::-1]


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    op: Op
    args: list        # resolved to Input order; entries are numbers / Signals / Chains / Textures / callables


class Chain:
    """An immutable chain of transforms. Methods are attached from the op registry at import time."""

    def __init__(self, transforms: Optional[list[Transform]] = None):
        self.transforms: list[Transform] = list(transforms or [])

    # -- construction -------------------------------------------------------
    def _with(self, op: Op, args: list) -> "Chain":
        return Chain(self.transforms + [Transform(op, args)])

    def out(self, output: "Output | None" = None, patch: "Patch | None" = None) -> "Chain":
        """Assign this chain to an output (default o0) of a patch (default the global patch)."""
        p = patch or default_patch()
        p.assign(output or p.o0, self)
        return self

    # -- python operators (sugar) ----------------------------------------------
    def __add__(self, other):
        return self.add(_as_chain(other))

    def __sub__(self, other):
        return self.sub(_as_chain(other))

    def __mul__(self, other):
        return self.mult(_as_chain(other))

    def __or__(self, other):
        return self.layer(_as_chain(other))

    def __xor__(self, other):
        return self.diff(_as_chain(other))

    def __repr__(self):
        return "Chain(" + ".".join(
            f"{t.op.name}({', '.join(_fmt_arg(a) for a in t.args)})" for t in self.transforms) + ")"

    # -- conveniences -------------------------------------------------------
    def glsl(self, **kw) -> str:
        """Fragment shader for this chain alone (the last pass if it needs several)."""
        from .glsl import compile_chain
        return compile_chain(self, **kw).passes[-1].source

    def hydra(self) -> str:
        """This chain as Hydra JavaScript."""
        from .export import chain_to_hydra
        return chain_to_hydra(self)

    def render(self, t: float = 0.0, size=(320, 180), **kw):
        """Render a single frame with the NumPy backend -> float32 (H, W, 4), row 0 = top."""
        from .backend_np import render_chain
        return render_chain(self, t=t, size=size, **kw)

    def show(self, t: float = 0.0, size=(320, 180), **kw):
        """Render one frame and return a PIL image."""
        from .export import to_image
        return to_image(self.render(t, size, **kw))


def _fmt_arg(a) -> str:
    if isinstance(a, float):
        return f"{a:g}"
    if isinstance(a, Chain):
        return repr(a)
    if isinstance(a, Texture):
        return a.name
    if callable(a) and not hasattr(a, "js"):
        return "<fn>"
    return repr(a)


def _as_chain(x) -> "Chain":
    if isinstance(x, Chain):
        return x
    if isinstance(x, (int, float)):
        return Chain([Transform(REGISTRY["solid"], [float(x), float(x), float(x), 1.0])])
    raise TypeError(f"cannot combine a chain with {type(x).__name__}")


# ---------------------------------------------------------------------------
# Argument normalisation
# ---------------------------------------------------------------------------

def normalize_args(op: Op, args: tuple, kwargs: dict) -> list:
    """Map positional + keyword arguments onto the op's inputs, filling defaults."""
    inputs = op.inputs
    if len(args) > len(inputs):
        raise TypeError(f"{op.name}() takes at most {len(inputs)} arguments ({len(args)} given)")
    out = []
    for i, inp in enumerate(inputs):
        if i < len(args):
            v = args[i]
        elif inp.name in kwargs:
            v = kwargs.pop(inp.name)
        else:
            v = inp.default
        out.append(coerce_arg(op, inp, v))
    if kwargs:
        bad = ", ".join(kwargs)
        raise TypeError(f"{op.name}() got unexpected keyword argument(s): {bad}")
    return out


def coerce_arg(op: Op, inp: Input, v):
    from .signals import Signal, Seq
    if inp.type == "sampler2D":
        if not isinstance(v, Texture):
            raise TypeError(f"{op.name}({inp.name}=...) expects an output/image/camera, got {type(v).__name__}")
        return v
    if inp.type == "chain":
        if isinstance(v, (int, float)):
            return _as_chain(v)
        if isinstance(v, Texture):   # Hydra sugar: .modulate(o1) means .modulate(src(o1))
            return Chain([Transform(REGISTRY["src"], [v])])
        if not isinstance(v, Chain):
            raise TypeError(f"{op.name}({inp.name}=...) expects a chain, got {type(v).__name__}")
        return v
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        if inp.type == "int":
            return int(v)
        return float(v)
    if isinstance(v, (list, tuple)):
        return Seq(list(v))
    if isinstance(v, Signal):
        return v
    if isinstance(v, Chain):
        if inp.scalar:
            raise TypeError(f"{op.name}({inp.name}=...) must be a number or signal, not a per-pixel chain")
        return v
    if callable(v):
        return v
    raise TypeError(f"{op.name}({inp.name}=...): unsupported argument {v!r}")


# ---------------------------------------------------------------------------
# Patch: a set of outputs, like one Hydra instance
# ---------------------------------------------------------------------------

class Patch:
    """Four outputs plus settings. ``hydrapy.o0`` etc. belong to the default patch."""

    def __init__(self, bpm: float = 30.0, size=(320, 180), outputs: Optional[list] = None):
        self.outputs = list(outputs) if outputs is not None else [Output(i) for i in range(4)]
        for o in self.outputs:
            o.chain = None
        self.o0, self.o1, self.o2, self.o3 = self.outputs
        self.bpm = bpm
        self.size = tuple(size)
        self.display: Output = self.o0
        self.speed = 1.0
        self.frame_rate = None          # e.g. 18: update feedback and pixels at most 18 times a second
        self.composite_alpha = False   # Hydra's display: RGB, alpha is a fourth channel. True: show RGB x alpha over black
        self.controls: dict = {}        # explicitly exposed Param signals
        self.probes: list = []          # live.Probe instances, updated after every frame
        self.transport = None           # live.Transport (Link, MidiClock) for `hydrapy live`; None = wall clock

    def assign(self, output: Output, chain: Chain):
        # outputs are matched by index, so the global o0..o3 work with any patch
        self.outputs[output.index].chain = chain

    def render(self, output: Output | None = None):
        """Choose which output is displayed (Hydra's ``render(o1)``)."""
        if output is not None:
            self.display = self.outputs[output.index]
        return self

    def hush(self):
        for o in self.outputs:
            o.chain = None
        self.display = self.o0

    def active_outputs(self) -> list[Output]:
        return [o for o in self.outputs if o.chain is not None]

    def __repr__(self):
        return "Patch(" + ", ".join(f"{o.name}={o.chain!r}" for o in self.active_outputs()) + ")"


_default_patch: Optional[Patch] = None
_isolate_patch_outputs = False  # CLI loads must not reuse a running patch's outputs.
GLOBAL_OUTPUTS = [Output(i) for i in range(4)]   # the o0..o3 exported by hydrapy


def default_patch() -> Patch:
    global _default_patch
    if _default_patch is None:
        _default_patch = Patch(outputs=GLOBAL_OUTPUTS)
    return _default_patch


def reset_patch(bpm: float = 30.0, size=(320, 180)) -> Patch:
    """Start a fresh default patch; the global o0..o3 are cleared and reused."""
    global _default_patch
    _default_patch = Patch(bpm=bpm, size=size, outputs=None if _isolate_patch_outputs else GLOBAL_OUTPUTS)
    return _default_patch


# ---------------------------------------------------------------------------
# Per-frame context
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    time: float = 0.0
    frame: int = 0
    width: int = 320
    height: int = 180
    bpm: float = 30.0
    mouse: tuple = (0.0, 0.0)
    dt: float = 1 / 60
    fps: float = 60.0

    @property
    def resolution(self):
        return (self.width, self.height)

    # Hydra passes ({time, bpm, width, height, mouse}) to parameter functions
    def __getitem__(self, k):
        return getattr(self, k)


class RenderClock:
    """Presentation calls and simulation updates have separate counters."""

    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.fps = float(fps)
        self.calls = self.updates = 0
        self.last_key = self.last_time = None

    def next(self, patch, size, mouse, t):
        wall_time = self.calls / self.fps if t is None else float(t)
        time = wall_time * patch.speed
        rate = patch.frame_rate
        if rate is not None and (not math.isfinite(rate) or rate <= 0):
            raise ValueError("frame_rate must be positive and finite, or None")
        if rate:
            time = math.floor(time * rate + 1e-9) / rate
        key = (time, rate, size)
        draw = not rate or key != self.last_key
        dt = 1.0 / (rate or self.fps) if self.last_time is None else max(0.0, time - self.last_time)
        ctx = Ctx(time=time, frame=self.updates, width=size[0], height=size[1],
                  bpm=patch.bpm, mouse=mouse, dt=dt if draw else 0.0, fps=rate or self.fps)
        ctx.presentation_time, ctx.presentation_dt = wall_time, 1.0 / self.fps
        self.calls += 1
        if draw:
            self.last_key, self.last_time = key, time
            self.updates += 1
        return ctx, draw, wall_time
