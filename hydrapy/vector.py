"""Vectors out: an oscilloscope in XY mode, a laser, a pen plotter.

The same paths can go to all three.  A path is an (N, 2) array in the square -1..1 (y up); a
frame is a list of paths.  Sources of paths:

* parametric curves that move with time and take signals for every parameter:
  ``Harmonograph`` (Lissajous with damped pendulums), ``Lissajous``, ``Rose``;
* drawings: ``Paths(polylines)``, ``Paths.svg("drawing.svg")``, ``Paths.lines(meadow())``;
* the picture itself: ``Contours(o0, levels=4)`` traces isolines of an output every frame (the
  image as vector art, the way a scan converter or a Vectrex would show it).

**XY** turns a frame of paths into audio-rate x/y (and a blanking z) for two or three DC-coupled
outputs: the beam traces the frame ``rate`` times a second, travels between paths blanked, and
keeps its phase across frames so changing shapes don't click.  ``CvOut().send(0, xy.x).send(1, xy.y)``
puts it on the ES-3 and into a scope's X and Y inputs (or a laser DAC's); because the beam's path
*is* a waveform, it is also oscilloscope music — ``xy.stereo()`` is the left/right pair, and
``save_mp4(..., audio=xy)`` muxes it.  ``src(xy)`` is a phosphor preview of what the scope shows.

**Plotter**: ``to_plotter_svg(paths, "plot.svg", size_mm=(297, 210))`` writes an SVG in millimetres
with the paths ordered to cut pen-up travel; ``chain_contours(chain)`` and ``streamlines(chain)``
turn any hydrapy field into line work for it.
"""
from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Sequence

import numpy as np

from .core import Output, Texture
from .signals import as_signal

Path = np.ndarray      # (N, 2) float, -1..1, y up


# -----------------------------------------------------------------------------
# path utilities
# -----------------------------------------------------------------------------

def as_paths(polylines) -> list[np.ndarray]:
    out = []
    for p in polylines:
        a = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        if len(a) >= 2:
            out.append(a)
    return out


def path_length(p: np.ndarray) -> float:
    return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(-1)).sum()) if len(p) > 1 else 0.0


def resample(p: np.ndarray, n: int) -> np.ndarray:
    """``n`` points evenly spaced along the path (by arc length)."""
    n = max(int(n), 2)
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(-1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], axis=-1)


def simplify(p: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker."""
    if len(p) < 3 or tolerance <= 0:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = p[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return p[keep]


def order_paths(paths: Sequence[np.ndarray], start=(0.0, 0.0)) -> list[np.ndarray]:
    """Greedy nearest-neighbour order (reversing paths where that is shorter): less travel for a
    plotter's pen and a beam's blanked jumps."""
    rest = list(paths)
    out = []
    pos = np.asarray(start, dtype=np.float64)
    while rest:
        heads = np.array([p[0] for p in rest])
        tails = np.array([p[-1] for p in rest])
        dh = ((heads - pos) ** 2).sum(-1)
        dt = ((tails - pos) ** 2).sum(-1)
        i = int(np.argmin(np.minimum(dh, dt)))
        p = rest.pop(i)
        if dt[i] < dh[i]:
            p = p[::-1]
        out.append(p)
        pos = p[-1]
    return out


def fit(paths: Sequence[np.ndarray], margin: float = 0.05) -> list[np.ndarray]:
    """Scale and centre paths into the square -1..1 (keeping their proportions)."""
    paths = as_paths(paths)
    if not paths:
        return []
    allp = np.concatenate(paths)
    lo, hi = allp.min(0), allp.max(0)
    c = (lo + hi) / 2
    k = (2.0 - 2 * margin) / max(float((hi - lo).max()), 1e-9)
    return [(p - c) * k for p in paths]


# -----------------------------------------------------------------------------
# sources of paths
# -----------------------------------------------------------------------------

class Curve:
    """A function of the frame context returning a list of paths."""

    def __call__(self, ctx) -> list[np.ndarray]:
        raise NotImplementedError

    def at(self, t: float = 0.0) -> list[np.ndarray]:
        from .core import Ctx
        return self(Ctx(time=t))


class Harmonograph(Curve):
    """x = sin(f1 s + phase) (+ sin(f3 s)), y = sin(f2 s) (+ sin(f4 s)), damped by ``decay`` over
    ``length`` turns — the same curve as the ``harmonograph`` op. Every argument takes a signal."""

    def __init__(self, f1=3, f2=2, f3=0, f4=0, phase=1.5708, decay=0, length=1, size=0.9, speed=0.1, points=1024):
        self.p = {k: as_signal(v) for k, v in dict(f1=f1, f2=f2, f3=f3, f4=f4, phase=phase, decay=decay,
                                                   length=length, size=size, speed=speed).items()}
        self.points = int(points)

    def __call__(self, ctx):
        v = {k: float(s(ctx)) for k, s in self.p.items()}
        t = np.linspace(0.0, 1.0, self.points)
        s = t * v["length"] * 2 * math.pi
        e = np.exp(-v["decay"] * t * v["length"])
        ph = v["phase"] + ctx.time * v["speed"]
        x = np.sin(v["f1"] * s + ph)
        y = np.sin(v["f2"] * s)
        if v["f3"] > 0:
            x = 0.5 * (x + np.sin(v["f3"] * s))
        if v["f4"] > 0:
            y = 0.5 * (y + np.sin(v["f4"] * s))
        return [np.stack([x, y], axis=-1) * (v["size"] * e)[:, None]]


def Lissajous(a=3, b=2, phase=1.5708, size=0.9, speed=0.1, points=1024) -> Harmonograph:
    return Harmonograph(a, b, 0, 0, phase, 0, 1, size, speed, points)


class Rose(Curve):
    """r = cos(k θ): petals. ``k`` = n/d gives n petals (d=1) or nested roses."""

    def __init__(self, k=5, d=1, size=0.9, speed=0.1, points=1024):
        self.k, self.d, self.size, self.speed = as_signal(k), as_signal(d), as_signal(size), as_signal(speed)
        self.points = int(points)

    def __call__(self, ctx):
        k, d = float(self.k(ctx)), max(float(self.d(ctx)), 1e-6)
        th = np.linspace(0, 2 * math.pi * d, self.points) + ctx.time * float(self.speed(ctx))
        r = np.cos(k / d * th) * float(self.size(ctx))
        return [np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)]


class Paths(Curve):
    """Fixed line work (fitted into -1..1). ``transform``: optional f(paths, ctx) -> paths to animate it."""

    def __init__(self, polylines, fit_square: bool = True, transform: Optional[Callable] = None):
        self.paths = fit(polylines) if fit_square else as_paths(polylines)
        self.transform = transform

    @classmethod
    def svg(cls, path_or_text: str, **kw) -> "Paths":
        from .drawing import parse_svg
        text = open(path_or_text).read() if not path_or_text.lstrip().startswith("<") else path_or_text
        polys = [[(x, -y) for x, y in p] for p in parse_svg(text)]          # SVG y runs down
        return cls(polys, **kw)

    @classmethod
    def lines(cls, polylines, **kw) -> "Paths":
        """Polylines in 0..1 with y up (``drawing.plant()`` / ``meadow()`` / ``Lines`` input)."""
        return cls(polylines, **kw)

    def __call__(self, ctx):
        return self.transform(self.paths, ctx) if self.transform else self.paths


def rotate(paths, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    m = np.array([[c, -s], [s, c]])
    return [p @ m.T for p in paths]


def field_contours(field: np.ndarray, levels=4, simplify_px: float = 0.75, min_points: int = 4) -> list[np.ndarray]:
    """Isolines of a (H, W) or (H, W, ≥3) field (GL-oriented: row 0 = bottom), in -1..1 keeping aspect."""
    from skimage import measure
    a = np.asarray(field, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0] * 0.2125 + a[..., 1] * 0.7154 + a[..., 2] * 0.0721
    H, W = a.shape
    lv = np.linspace(0, 1, int(levels) + 2)[1:-1] if np.isscalar(levels) else list(levels)
    k = 2.0 / max(H - 1, W - 1, 1)
    out = []
    for level in lv:
        for c in measure.find_contours(a, level):
            c = simplify(c, simplify_px)
            if len(c) >= min_points // 2 + 1:
                out.append(np.stack([(c[:, 1] - (W - 1) / 2) * k, (c[:, 0] - (H - 1) / 2) * k], axis=-1))
    return out


class Contours(Curve):
    """Isolines of an output's luminance, traced every frame: the picture as vector line work.
    Registered as a probe, so it reads the output after each drawn frame (one frame late)."""

    def __init__(self, output: Optional[Output] = None, levels=4, simplify_px: float = 0.75, max_paths: int = 64,
                 patch=None):
        from .core import default_patch
        from .live import Probe

        p = patch or default_patch()
        self.output = output or p.o0
        self.levels, self.simplify_px, self.max_paths = levels, simplify_px, int(max_paths)
        self.paths: list[np.ndarray] = []
        outer = self

        class _Reader(Probe):
            def region(self, width, height):
                return 0, 0, width, height

            def update(self, rgba, t=None):
                outer._update(rgba)

        self.probe = _Reader(self.output, patch=p)

    def _update(self, rgba):
        paths = field_contours(rgba, self.levels, self.simplify_px)
        paths.sort(key=path_length, reverse=True)
        self.paths = order_paths(paths[: self.max_paths])

    def __call__(self, ctx):
        return self.paths


# -----------------------------------------------------------------------------
# XY out
# -----------------------------------------------------------------------------

def _trace(trav, phase, step, n, start, gain):
    """``n`` samples of the beam over a traversal table from ``phase``, ``step`` traversals a sample (the exact
    pitch), the gain ramping from ``start`` to reach ``gain`` at the end: (xy (n, 2), z (n,), the phase after)."""
    per = len(trav)
    pos = (phase + np.arange(n) * step) % 1.0
    fi = pos * per
    i0 = np.floor(fi).astype(int) % per
    i1 = (i0 + 1) % per
    f = (fi - np.floor(fi))[:, None]
    s = trav[i0] * (1 - f) + trav[i1] * f
    ramp = start + (gain - start) * np.arange(1, n + 1) / n
    xy = np.clip(s[:, :2] * ramp[:, None], -1.0, 1.0).astype(np.float32)
    return xy, (s[:, 2] >= 0.5).astype(np.float32), float((phase + n * step) % 1.0)


class XYVoice:
    """An XY played by the audio thread, as an instrument: each pull continues the beam from the latest frame's
    traversal, pitch and gain, so its phase never breaks however unevenly frames arrive (a frame-pushed
    AudioQueue has to repeat or drop whole chunks, which clicks on a pure tone).  Pitch and gain move at the
    pull's block boundaries.  ``channels`` are x and y; pulling one generates both, kept for the other."""

    def __init__(self, xy, samplerate):
        self.xy, self.samplerate = xy, int(samplerate)
        self._lock = threading.Lock()
        self.channels = [_VoiceChannel(self, 0), _VoiceChannel(self, 1)]
        self._reset()

    def _reset(self):
        self.phase, self._gain = 0.0, None
        self._buffers = [np.zeros(0, np.float32), np.zeros(0, np.float32)]

    def _pull(self, k, n):
        if len(self._buffers[k]) < n:
            m = n - len(self._buffers[k])
            live = self.xy.live
            if live is None:
                block = np.zeros((m, 2), np.float32)
            else:
                trav, rate, gain = live
                start = gain if self._gain is None else self._gain
                self._gain = gain
                block, _, self.phase = _trace(trav, self.phase, rate / self.samplerate, m, start, gain)
            for c in range(2):          # a channel nobody pulls keeps at most a second
                self._buffers[c] = np.concatenate([self._buffers[c], block[:, c]])[-max(self.samplerate, n):]
        out, self._buffers[k] = self._buffers[k][:n], self._buffers[k][n:]
        return out


class _VoiceChannel:
    """One side of an XYVoice, with the AudioQueue interface Studio's audio producer pulls from."""

    def __init__(self, voice, k):
        self.voice, self.k = voice, k

    @property
    def samplerate(self):
        return self.voice.samplerate

    @samplerate.setter
    def samplerate(self, value):
        self.voice.samplerate = int(value)

    @property
    def _lock(self):
        return self.voice._lock

    def _reset(self):
        self.voice._reset()

    def pull(self, n: int) -> np.ndarray:
        with self.voice._lock:
            return self.voice._pull(self.k, n)

    def close(self):
        pass


class _Channel:
    """One audio-rate channel of an XY stream, routable with ``CvOut.send``."""

    def __init__(self, parent, k):
        self.parent, self.k = parent, k
        self.label = f"{parent.name} {'xyz'[k]}"
        self.listeners: list[Callable] = []
        self.chunks: list[np.ndarray] = []
        self._generation = 0          # bumped on clear(): an AudioQueue drops what it had queued

    @property
    def sr(self):
        return self.parent.sr

    @property
    def rate(self):
        return self.parent.fps

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    def live_source(self, samplerate):
        """For Studio's audio producer: this channel of the XY's voice (x or y), pulled by the audio thread."""
        if self.k > 1:
            return None
        voice = self.parent._voice
        if voice is None or voice.samplerate != int(samplerate):
            voice = self.parent._voice = XYVoice(self.parent, samplerate)
        return voice.channels[self.k]


class XY(Texture):
    """A frame of paths as an audio-rate beam: ``x``, ``y`` and ``z`` (1 drawing, 0 blanked).

    ``rate``: traversals of the whole frame per second (the refresh of the scope, and the pitch
    you hear; under ~40 it flickers).  The beam advances by exactly ``rate`` traversals a second,
    so the pitch is exact even when a period isn't a whole number of samples.  ``gain``: ±full
    scale (±10 V out of an ES-3 at 1.0; 0.5 is ±5 V), ramped across each frame's samples so an
    envelope doesn't step.  ``jump``: fraction of each traversal spent on blanked travel between
    paths.

    ``x`` and ``y`` join the patch's audio streams, so Studio plays them on its optical ports
    (x left, y right): the picture as an oscillator another instrument can filter.
    """

    _counter = 0

    def __init__(self, source, rate=80.0, sr: int = 48000, fps: float = 60.0, gain=0.5, jump: float = 0.03,
                 record: bool = True, keep: float = 1200.0, preview=(320, 180), persistence: float = 0.6,
                 patch=None):
        from .core import default_patch
        from .live import register_hook
        self.source = source if callable(source) else Paths(source)
        self.rate, self.gain = as_signal(rate), as_signal(gain)
        self.sr, self.fps, self.jump = int(sr), float(fps), float(jump)
        self.record, self.keep = bool(record), float(keep)
        XY._counter += 1
        self.name = f"xy{XY._counter}"
        self.x, self.y, self.z = _Channel(self, 0), _Channel(self, 1), _Channel(self, 2)
        self.phase = 0.0
        self._carry = 0.0
        self._last_t = None
        self._gain = None
        self.live = None              # (traversal, rate, gain) of the latest frame
        self._voice = None
        self.preview_size, self.persistence = tuple(preview), float(persistence)
        self.phosphor = np.zeros((self.preview_size[1], self.preview_size[0]), np.float32)
        self.patch = patch or default_patch()
        self.patch.audio_streams += [self.x, self.y]
        register_hook(self.tick)
        from .live import _RESOURCES
        _RESOURCES.append(self)

    # -- the beam ------------------------------------------------------------
    def traversal(self, paths: Sequence[np.ndarray], n: int) -> np.ndarray:
        """One pass over ``paths`` as ``n`` samples of (x, y, z)."""
        paths = [p for p in as_paths(paths)]
        if not paths:
            return np.zeros((max(n, 1), 3))
        lengths = np.array([max(path_length(p), 1e-6) for p in paths])
        n_jump = max(int(round(n * self.jump)), len(paths))
        n_draw = max(n - n_jump, 2 * len(paths))
        alloc = np.maximum((lengths / lengths.sum() * n_draw).astype(int), 2)
        parts = []
        jumps_each = max(n_jump // len(paths), 1)
        for i, p in enumerate(paths):
            d = resample(p, alloc[i])
            parts.append(np.concatenate([d, np.ones((len(d), 1))], axis=1))
            nxt = paths[(i + 1) % len(paths)][0]
            j = np.linspace(d[-1], nxt, jumps_each + 2)[1:-1]            # blanked travel to the next path
            parts.append(np.concatenate([j, np.zeros((len(j), 1))], axis=1))
        trav = np.concatenate(parts)
        if len(trav) != n:                                                  # exact length: stretch by index
            idx = np.linspace(0, len(trav) - 1, n)
            i0 = np.floor(idx).astype(int)
            i1 = np.minimum(i0 + 1, len(trav) - 1)
            f = (idx - i0)[:, None]
            out = trav[i0] * (1 - f) + trav[i1] * f
            out[:, 2] = np.where(f[:, 0] < 0.5, trav[i0, 2], trav[i1, 2])     # z stays a gate
            trav = out
        return trav

    def tick(self, ctx, renderer=None):
        t = getattr(ctx, "presentation_time", ctx.time)
        dt = getattr(ctx, "presentation_dt", 1.0 / self.fps)
        if self._last_t is not None and t < self._last_t - 1e-9:
            self.clear()
        self._last_t = t
        self._carry += self.sr * dt
        n = int(math.floor(self._carry + 1e-9))
        self._carry -= n
        if n <= 0:
            return
        rate = max(float(self.rate(ctx)), 1e-3)
        gain = float(self.gain(ctx))
        start = gain if self._gain is None else self._gain
        self._gain = gain
        per = max(int(round(self.sr / rate)), 8)                          # points in the traversal table
        trav = self.traversal(self.source(ctx), per)
        self.live = (trav, rate, gain)                                     # what an XYVoice plays from now on
        xy, z, self.phase = _trace(trav, self.phase, rate / self.sr, n, start, gain)
        for ch, data in ((self.x, xy[:, 0]), (self.y, xy[:, 1]), (self.z, z)):
            if self.record:
                ch.chunks.append(data)
                limit = int(self.keep * self.fps)
                if len(ch.chunks) > limit:
                    del ch.chunks[: len(ch.chunks) - limit]
            for fn in ch.listeners:
                fn(data)
        self._draw_phosphor(xy, z)

    def _draw_phosphor(self, xy, z):
        W, H = self.preview_size
        self.phosphor *= self.persistence
        on = z > 0.5
        if not on.any():
            return
        p = xy[on]
        # subdivide so fast segments stay continuous; brightness falls with beam speed, like phosphor
        a, b = p[:-1], p[1:]
        if len(a) == 0:
            return
        sub = 4
        u = (np.arange(sub) / sub)[None, :, None]
        pts = (a[:, None, :] * (1 - u) + b[:, None, :] * u).reshape(-1, 2)
        speed = np.repeat(np.sqrt(((b - a) ** 2).sum(-1)), sub) + 1e-3
        sc = (min(W, H) - 1) / 2.0                     # the scope's square screen, centred, in proportion
        px = (pts[:, 0] * sc + (W - 1) / 2).round().astype(int)
        py = (pts[:, 1] * sc + (H - 1) / 2).round().astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.add.at(self.phosphor, (py[ok], px[ok]), (0.02 / speed[ok]).astype(np.float32))

    def dynamic_array(self, t: Optional[float] = None) -> np.ndarray:
        """The phosphor preview, (H, W, 4) GL-oriented, green on black."""
        g = 1.0 - np.exp(-self.phosphor * 0.6)
        g = np.clip(g + 0.25 * np.roll(g, 1, 0) + 0.25 * np.roll(g, -1, 0) + 0.25 * np.roll(g, 1, 1) + 0.25 * np.roll(g, -1, 1), 0, 1)
        return np.stack([g * 0.45, g, g * 0.55, np.ones_like(g)], axis=-1).astype(np.float32)

    # -- export ------------------------------------------------------------------
    def stereo(self) -> np.ndarray:
        """(n, 2) float32: x left, y right — the oscilloscope-music convention."""
        x, y = self.x.samples, self.y.samples
        n = min(len(x), len(y))
        return np.stack([x[:n], y[:n]], axis=-1)

    def wav(self, path: str, channels=("x", "y")) -> str:
        import wave
        data = np.stack([getattr(self, c).samples for c in channels], axis=-1)
        with wave.open(path, "wb") as w:
            w.setnchannels(len(channels)); w.setsampwidth(2); w.setframerate(self.sr)
            w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
        return path

    def clear(self):
        for ch in (self.x, self.y, self.z):
            ch.chunks = []
            ch._generation += 1
        self.phase, self._carry, self._last_t, self._gain = 0.0, 0.0, None, None
        self.phosphor[:] = 0.0

    def close(self):
        from .live import _HOOKS
        if self.tick in _HOOKS:
            _HOOKS.remove(self.tick)
        self.patch.audio_streams[:] = [s for s in self.patch.audio_streams if s not in (self.x, self.y)]


# -----------------------------------------------------------------------------
# plotter
# -----------------------------------------------------------------------------

def chain_contours(chain, levels=6, size=(400, 400), t: float = 0.0, simplify_px: float = 0.6) -> list[np.ndarray]:
    """Isolines of a chain's luminance, rendered at ``size`` with the NumPy backend."""
    rgba = chain.render(t, size=size)[::-1]               # render() is row 0 = top; contours want GL rows
    return field_contours(rgba, levels, simplify_px)


def streamlines(chain, n: int = 400, steps: int = 80, step: float = 0.01, size=(200, 200), t: float = 0.0,
                mode: str = "angle", seed: int = 0, min_points: int = 6) -> list[np.ndarray]:
    """Flow-field lines through a chain: ``mode='angle'`` reads red as an angle (0..1 = one turn),
    ``'vector'`` reads red/green as a direction (0.5 = still).  ``n`` seeds, ``steps`` of ``step``
    each, bilinear field lookups.  The longer side of ``size`` spans -1..1, as in ``field_contours``."""
    rgba = chain.render(t, size=size)[::-1].astype(np.float64)
    H, W = rgba.shape[:2]
    if mode == "angle":
        ang = rgba[..., 0] * 2 * math.pi
        vx, vy = np.cos(ang), np.sin(ang)
    else:
        vx, vy = rgba[..., 0] - 0.5, rgba[..., 1] - 0.5
        nrm = np.hypot(vx, vy) + 1e-9
        vx, vy = vx / nrm, vy / nrm

    k = 2.0 / max(H - 1, W - 1, 1)                       # path units per pixel
    ext = np.array([(W - 1) * k / 2, (H - 1) * k / 2])    # half-extent of the field in path units

    def look(f, p):
        x = p[0] / k + (W - 1) / 2
        y = p[1] / k + (H - 1) / 2
        x0, y0 = int(np.clip(np.floor(x), 0, W - 2)), int(np.clip(np.floor(y), 0, H - 2))
        fx, fy = x - x0, y - y0
        return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy)
                + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)

    rng = np.random.default_rng(seed)
    out = []
    for p in rng.uniform(-ext, ext, (n, 2)):
        pts = [p.copy()]
        for _ in range(steps):
            d = np.array([look(vx, p), look(vy, p)])
            p = p + d * step
            if np.any(np.abs(p) > ext):
                break
            pts.append(p.copy())
        if len(pts) >= min_points:
            out.append(np.array(pts))
    return out


def to_plotter_svg(paths: Sequence[np.ndarray], path: Optional[str] = None, size_mm=(297, 210), margin_mm: float = 15,
                   stroke_mm: float = 0.35, order: bool = True, layers: Optional[Sequence[Sequence[np.ndarray]]] = None) -> str:
    """Paths in -1..1 -> an SVG in millimetres, fitted (proportions kept) inside the margins and
    ordered to shorten pen-up travel.  ``layers``: several path lists, one per pen (Inkscape layers,
    which vpype and AxiDraw both understand)."""
    W, H = size_mm
    groups = [as_paths(g) for g in (layers if layers is not None else [paths])]
    allp = [p for g in groups for p in g]
    if allp:
        a = np.concatenate(allp)
        lo, hi = a.min(0), a.max(0)
    else:
        lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    k = min((W - 2 * margin_mm) / max(hi[0] - lo[0], 1e-9), (H - 2 * margin_mm) / max(hi[1] - lo[1], 1e-9))
    cx, cy = (lo + hi) / 2
    body = []
    for li, g in enumerate(groups):
        g = order_paths(g, start=(lo[0], hi[1])) if order else g
        lines = []
        for p in g:
            pts = " ".join(f"{W / 2 + (x - cx) * k:.3f},{H / 2 - (y - cy) * k:.3f}" for x, y in p)
            lines.append(f'    <polyline points="{pts}" />')
        body.append(f'  <g id="layer{li + 1}" inkscape:groupmode="layer" inkscape:label="{li + 1}" fill="none" '
                    f'stroke="black" stroke-width="{stroke_mm}" stroke-linecap="round" stroke-linejoin="round">\n'
                    + "\n".join(lines) + "\n  </g>")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           f'width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">\n' + "\n".join(body) + "\n</svg>\n")
    if path:
        with open(path, "w") as f:
            f.write(svg)
    return svg


def travel(paths: Sequence[np.ndarray]) -> tuple[float, float]:
    """(pen-down length, pen-up travel) of an ordered list of paths, in path units."""
    down = sum(path_length(p) for p in paths)
    up = sum(float(np.hypot(*(b[0] - a[-1]))) for a, b in zip(paths, paths[1:]))
    return down, up
