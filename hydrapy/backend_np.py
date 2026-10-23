"""CPU renderer: evaluates chains on a NumPy coordinate grid.

Every op's NumPy twin is applied in the same nesting order the GLSL compiler
uses, so both backends produce the same picture (up to float precision and
texture filtering).  Slow compared with a GPU but it runs anywhere, it is
deterministic, and it is where the CPU-only ops (``pixelsort``) live.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .core import Camera, Chain, Ctx, DelayRef, Image, Output, Patch, RenderClock, Texture, Transform, Video, default_patch
from .signals import Signal


class Env:
    """Per-render-session state handed to every NumPy op."""

    def __init__(self, width: int, height: int, bpm: float = 30.0, filter: str = "linear"):
        self.w, self.h = int(width), int(height)
        self.res = np.array([self.w, self.h], dtype=np.float32)
        self.bpm = bpm
        self.ctx = Ctx(width=self.w, height=self.h, bpm=bpm)
        self.filter = filter
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        # gl_FragCoord is the pixel centre; st = gl_FragCoord / resolution
        self.px = np.stack([xs + 0.5, ys + 0.5], axis=-1)
        self.base_st = self.px / self.res
        self.buffers: dict[int, np.ndarray] = {}   # keyed by output index: latest completed frame
        self.rings: dict[int, list] = {}            # output index -> [latest, one before, ...] for delay()
        self.current_output: Optional[Output] = None
        self.pass_index = 0          # iteration index inside an iterated texture op (GLSL: _pass)
        self._frames_cache: dict = {}

    @property
    def time(self) -> float:
        return self.ctx.time

    # -- textures ----------------------------------------------------------
    def history(self, ref: Output, frames: int) -> np.ndarray:
        """Output ``ref`` as it was ``frames`` frames before what src() gives (0 = same as src)."""
        ring = self.rings.get(ref.index, [])
        if frames < len(ring):
            return ring[frames]
        return self.texture(ref) if not ring else ring[-1]

    def push(self, index: int, frame: np.ndarray, depth: int):
        self.buffers[index] = frame
        ring = self.rings.setdefault(index, [])
        ring.insert(0, frame)
        del ring[depth + 1:]

    def texture(self, ref: Texture) -> np.ndarray:
        if hasattr(ref, "advance_np"):              # simulations: NCA, Physarum (stepped to this frame's time)
            return ref.advance_np(self, self.time)
        if hasattr(ref, "dynamic_array"):           # vector.XY phosphor preview, anything computed per frame in Python
            return ref.dynamic_array(self.time)
        if isinstance(ref, DelayRef):
            return self.history(ref.output, ref.frames)
        if isinstance(ref, Output):
            buf = self.buffers.get(ref.index)
            if buf is None:
                buf = np.zeros((self.h, self.w, 4), dtype=np.float32)
                self.buffers[ref.index] = buf
            return buf
        if isinstance(ref, Image) or (hasattr(ref, "array") and not hasattr(ref, "array_at")):   # Image, Text, Lines
            return ref.array()
        if isinstance(ref, Video):
            return ref.frame_at(self.time)
        if hasattr(ref, "array_at"):          # Bank, Quilt
            return ref.array_at(self.time)
        if isinstance(ref, Camera):
            raise RuntimeError("Camera sources are only available in the HTML export / GL backend")
        raise TypeError(f"cannot sample {ref!r}")

    def sample(self, tex: np.ndarray, st: np.ndarray) -> np.ndarray:
        """Sample a (h, w, 4) texture at st (…, 2) with wrap-around, bilinear by default."""
        H, W = tex.shape[:2]
        st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.mod(st[..., 0], 1.0) * W - 0.5
        y = np.mod(st[..., 1], 1.0) * H - 0.5
        if self.filter == "nearest":
            xi = np.mod(np.floor(x + 0.5), W).astype(np.intp)
            yi = np.mod(np.floor(y + 0.5), H).astype(np.intp)
            return tex[yi, xi]
        x0f = np.floor(x)
        y0f = np.floor(y)
        fx = (x - x0f)[..., None].astype(np.float32)
        fy = (y - y0f)[..., None].astype(np.float32)
        x0 = x0f.astype(np.intp)
        y0 = y0f.astype(np.intp)
        x1 = np.mod(x0 + 1, W)
        y1 = np.mod(y0 + 1, H)
        x0 = np.mod(x0, W)
        y0 = np.mod(y0, H)
        return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
                + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


# ---------------------------------------------------------------------------
# Compilation of a chain into a closure f(st) -> rgba
# ---------------------------------------------------------------------------

def _arg_fn(a, env: Env, inp) -> Callable[[np.ndarray], object]:
    """Return a function of st giving this argument's value for the current frame."""
    if isinstance(a, (int, float)):
        v = a
        return lambda st: v
    if isinstance(a, Texture):
        return lambda st: a
    if isinstance(a, Signal):
        return lambda st: a(env.ctx)
    if isinstance(a, Chain):
        f = build(a, env)
        return lambda st: f(st)[..., 0]
    if callable(a):
        def fn(st):
            try:
                return float(a(env.ctx))
            except TypeError:
                return float(a())
        return fn
    raise TypeError(f"unsupported argument {a!r}")


def build(chain: Chain, env: Env) -> Callable[[np.ndarray], np.ndarray]:
    f: Optional[Callable] = None
    for tr in chain.transforms:
        op, args = tr.op, tr.args
        if op.np is None:
            raise NotImplementedError(f"op {op.name!r} has no CPU implementation")
        if op.kind != "src" and f is None:
            raise ValueError(f"a chain must start with a source, not .{op.name}()")

        if op.kind in ("combine", "combineCoord"):
            other = build(args[0], env)
            argfns = [_arg_fn(a, env, inp) for a, inp in zip(args[1:], op.inputs[1:])]
        else:
            other = None
            argfns = [_arg_fn(a, env, inp) for a, inp in zip(args, op.inputs)]

        f = _make_stage(op, f, other, argfns, env)
    if f is None:
        raise ValueError("empty chain")
    return f


def _make_stage(op, f0, other, argfns, env):
    npf = op.np
    kind = op.kind

    def vals(st):
        return [a(st) for a in argfns]

    f32 = lambda a: np.asarray(a, dtype=np.float32)   # keep everything float32 like the GPU
    if kind == "src":
        return lambda st: f32(npf(st, *vals(st), env))
    if kind == "coord":
        return lambda st: f0(f32(npf(st, *vals(st), env)))
    if kind == "color":
        return lambda st: f32(npf(f0(st), *vals(st), env))
    if kind == "combine":
        return lambda st: f32(npf(f0(st), other(st), *vals(st), env))
    if kind == "combineCoord":
        return lambda st: f0(f32(npf(st, other(st), *vals(st), env)))
    if kind == "texture":
        names = [i.name for i in op.inputs]
        it_idx = names.index("iterations") if "iterations" in names else None

        def stage(st):
            v = vals(st)
            iters = 1
            if it_idx is not None:
                iters = max(int(v[it_idx]), 1)
                v = v[:it_idx] + v[it_idx + 1:]
            tex = f0(env.base_st)
            for k in range(iters - 1):
                env.pass_index = k
                tex = f32(npf(tex, env.base_st, *v, env))
            env.pass_index = iters - 1
            out = f32(npf(tex, st, *v, env))
            env.pass_index = 0
            return out
        return stage
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class Renderer:
    """Stateful CPU renderer for a patch: keeps the output buffers between frames."""

    def __init__(self, patch: Optional[Patch] = None, size=None, fps: float = 30.0, filter: str = "linear"):
        self.patch = patch or default_patch()
        self.size = tuple(size or self.patch.size)
        self.fps = float(fps)
        self.env = Env(self.size[0], self.size[1], bpm=self.patch.bpm, filter=filter)
        self.frame = 0
        self.clock = RenderClock(self.fps)
        self._compiled: dict[Output, Callable] = {}
        self.depth = _history_depth(self.patch)

    def _compile(self):
        for o in self.patch.active_outputs():
            if o not in self._compiled or self._compiled[o][0] is not o.chain:
                self._compiled[o] = (o.chain, build(o.chain, self.env))

    def step(self, t: Optional[float] = None) -> np.ndarray:
        """Render one frame at time ``t`` (default: frame / fps) and return the displayed output, row 0 = top."""
        self._compile()
        ctx, draw, wall_time = self.clock.next(self.patch, self.size, self.env.ctx.mouse, t)
        self.env.ctx = ctx
        if draw:
            for o in self.patch.active_outputs():
                self.env.current_output = o
                self.env.push(o.index, self._compiled[o][1](self.env.base_st).astype(np.float32), self.depth.get(o.index, 0))
        self.frame += 1
        for pr in self.patch.probes:
            if hasattr(pr, "advance"):
                if not draw and pr.advance(wall_time, 1.0 / self.fps):
                    continue
                if not pr.needs_update(wall_time, 1.0 / self.fps):
                    continue
            elif not draw:
                continue
            x0, y0, w, h = pr.region(self.env.w, self.env.h)
            rgba = self.env.texture(pr.output)[y0:y0 + h, x0:x0 + w]
            if hasattr(pr, "advance"):
                pr.update(rgba, wall_time, 1.0 / self.fps)
            else:
                pr.update(rgba, ctx.time)
        from .live import run_hooks
        run_hooks(ctx, self)
        return self.displayed()

    def displayed(self) -> np.ndarray:
        out = self.env.texture(self.patch.display)
        return out[::-1]

    def frames(self, n: int, start: float = 0.0):
        """Yield n uint8 RGB frames (H, W, 3), row 0 = top."""
        for i in range(n):
            rgba = self.step(start + i / self.fps)
            yield to_uint8(rgba, self.patch.composite_alpha)


def _history_depth(patch: Patch) -> dict:
    """Deepest delay() per output index, found by walking the chains."""
    depth: dict = {}

    def walk(chain):
        for tr in chain.transforms:
            if tr.op.name == "delay" and isinstance(tr.args[0], Output):
                k = int(tr.args[1]) if isinstance(tr.args[1], (int, float)) else 0
                depth[tr.args[0].index] = max(depth.get(tr.args[0].index, 0), k)
            for a in tr.args:
                if isinstance(a, Chain):
                    walk(a)
    for o in patch.active_outputs():
        walk(o.chain)
    return depth


def to_uint8(rgba: np.ndarray, composite: bool = False) -> np.ndarray:
    """float RGB(A) -> uint8 RGB. ``composite``: RGB x alpha over black (``Patch.composite_alpha``);
    otherwise alpha is ignored, as Hydra's display ignores it."""
    rgb = rgba[..., :3]
    if composite and rgba.shape[-1] == 4:
        rgb = rgb * np.clip(rgba[..., 3:4], 0, 1)
    return (np.clip(np.nan_to_num(rgb), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def render_chain(chain: Chain, t: float = 0.0, size=(320, 180), bpm: float = 30.0, filter: str = "linear") -> np.ndarray:
    """Render one frame of a chain by itself (no feedback state). Returns float32 (H, W, 4), row 0 = top."""
    env = Env(size[0], size[1], bpm=bpm, filter=filter)
    env.ctx.time = t
    env.ctx.frame = int(t * 60)
    env.current_output = Output(0)
    out = build(chain, env)(env.base_st)
    return np.asarray(out, dtype=np.float32)[::-1]


def iter_frames(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                start: float = 0.0, warmup: int = 0, filter: str = "linear"):
    """Yield RGB frames without retaining the movie in memory."""
    r = Renderer(patch, size=size, fps=fps, filter=filter)
    for i in range(warmup):
        r.step(start - (warmup - i) / fps)
    from .live import clear_recordings
    clear_recordings()
    for pr in r.patch.probes:
        if hasattr(pr, "clear"):
            pr.clear()                      # a Soundtrack starts at `start`, not at the warm-up
    yield from r.frames(int(round(seconds * fps)), start=start)


def render_frames(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                  start: float = 0.0, warmup: int = 0, filter: str = "linear") -> list[np.ndarray]:
    """Collect RGB frames. Use ``iter_frames`` for bounded-memory video exports."""
    return list(iter_frames(patch, seconds, fps, size, start, warmup, filter))
