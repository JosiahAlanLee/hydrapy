"""Control signals: things you can plug into any parameter.

Every signal is a pure function of the frame context (time, bpm, mouse, audio),
evaluated once per frame on the CPU and sent to the shader as a uniform —
the same way Hydra evaluates ``() => ...`` parameters.  Because they are
declarative they can also be emitted as JavaScript for the HTML export and
for Hydra itself.

    osc(Lfo(0.2, lo=5, hi=40)).rotate(Seq([0, 1, 2]).fast(2).smooth())

Arithmetic works on signals: ``Lfo(1) * 2 + 0.5``, ``Fft(a, 0) ** 2``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

_EASINGS = {
    "linear": lambda t: t,
    "easeInQuad": lambda t: t * t,
    "easeOutQuad": lambda t: t * (2 - t),
    "easeInOutQuad": lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t,
    "easeInCubic": lambda t: t ** 3,
    "easeOutCubic": lambda t: (t - 1) ** 3 + 1,
    "easeInOutCubic": lambda t: 4 * t ** 3 if t < 0.5 else (t - 1) * (2 * t - 2) ** 2 + 1,
    "easeInQuart": lambda t: t ** 4,
    "easeOutQuart": lambda t: 1 - (t - 1) ** 4,
    "easeInOutQuart": lambda t: 8 * t ** 4 if t < 0.5 else 1 - 8 * (t - 1) ** 4,
    "easeInQuint": lambda t: t ** 5,
    "easeOutQuint": lambda t: 1 + (t - 1) ** 5,
    "easeInOutQuint": lambda t: 16 * t ** 5 if t < 0.5 else 1 + 16 * (t - 1) ** 5,
    "sin": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
}

_EASINGS_JS = {
    "linear": "(t)",
    "easeInQuad": "(t*t)",
    "easeOutQuad": "(t*(2-t))",
    "easeInOutQuad": "(t<0.5?2*t*t:-1+(4-2*t)*t)",
    "easeInCubic": "(t*t*t)",
    "easeOutCubic": "(Math.pow(t-1,3)+1)",
    "easeInOutCubic": "(t<0.5?4*t*t*t:(t-1)*(2*t-2)*(2*t-2)+1)",
    "easeInQuart": "(t*t*t*t)",
    "easeOutQuart": "(1-Math.pow(t-1,4))",
    "easeInOutQuart": "(t<0.5?8*t*t*t*t:1-8*Math.pow(t-1,4))",
    "easeInQuint": "(Math.pow(t,5))",
    "easeOutQuint": "(1+Math.pow(t-1,5))",
    "easeInOutQuint": "(t<0.5?16*Math.pow(t,5):1+16*Math.pow(t-1,5))",
    "sin": "(0.5-0.5*Math.cos(Math.PI*t))",
}


def _fract(x: float) -> float:
    return x - math.floor(x)


def _hash11(x: float) -> float:
    return _fract(math.sin(x * 12.9898 + 78.233) * 43758.5453)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Signal:
    """Base class. Subclasses implement ``value(ctx)`` and ``js()``."""

    def value(self, ctx) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, ctx) -> float:
        return float(self.value(ctx))

    def js(self) -> str:
        """JavaScript expression using ``time``, ``bpm``, ``__mx``, ``__my``, ``__fft``."""
        raise NotImplementedError

    # arithmetic --------------------------------------------------------
    def __add__(self, o): return Expr("+", self, o)
    def __radd__(self, o): return Expr("+", o, self)
    def __sub__(self, o): return Expr("-", self, o)
    def __rsub__(self, o): return Expr("-", o, self)
    def __mul__(self, o): return Expr("*", self, o)
    def __rmul__(self, o): return Expr("*", o, self)
    def __truediv__(self, o): return Expr("/", self, o)
    def __rtruediv__(self, o): return Expr("/", o, self)
    def __pow__(self, o): return Expr("**", self, o)
    def __mod__(self, o): return Expr("%", self, o)
    def __neg__(self): return Expr("*", self, -1.0)

    # helpers -----------------------------------------------------------
    def map(self, lo: float, hi: float) -> "Signal":
        """Rescale a 0..1 signal to lo..hi."""
        return self * (hi - lo) + lo

    def clip(self, lo: float = 0.0, hi: float = 1.0) -> "Signal":
        return Fn1("clip", self, lo, hi)

    def abs(self) -> "Signal": return Fn1("abs", self)
    def sin(self) -> "Signal": return Fn1("sin", self)
    def cos(self) -> "Signal": return Fn1("cos", self)
    def floor(self) -> "Signal": return Fn1("floor", self)
    def fract(self) -> "Signal": return Fn1("fract", self)
    def sqrt(self) -> "Signal": return Fn1("sqrt", self)
    def smoothstep(self) -> "Signal": return Fn1("smoothstep", self)
    def quantize(self, steps: float) -> "Signal":
        return Fn1("quantize", self, steps)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.js()}>"


def as_signal(x) -> "Signal":
    if isinstance(x, Signal):
        return x
    if isinstance(x, (int, float)):
        return Const(float(x))
    if isinstance(x, (list, tuple)):
        return Seq(list(x))
    if callable(x):
        return Fn(x)
    raise TypeError(f"cannot treat {x!r} as a signal")


def _num(v: float) -> str:
    s = repr(float(v))
    if s in ("inf", "-inf", "nan"):
        raise ValueError("non-finite constant in signal")
    return s


class Const(Signal):
    def __init__(self, v: float):
        self.v = float(v)

    def value(self, ctx): return self.v
    def js(self): return _num(self.v)


class Time(Signal):
    """Seconds since start (Hydra's ``time``)."""
    def value(self, ctx): return ctx.time
    def js(self): return "time"


class Mouse(Signal):
    """Normalised mouse position 0..1 (``axis`` 'x' or 'y'). Zero when rendering offline."""
    def __init__(self, axis: str = "x"):
        self.axis = axis

    def value(self, ctx):
        return ctx.mouse[0] if self.axis == "x" else ctx.mouse[1]

    def js(self): return "__mx" if self.axis == "x" else "__my"


class Expr(Signal):
    OPS = {
        "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b != 0 else 0.0, "**": lambda a, b: a ** b if a >= 0 or float(b).is_integer() else 0.0,
        "%": lambda a, b: math.fmod(a, b) if b != 0 else 0.0,
    }

    def __init__(self, op: str, a, b):
        self.op = op
        self.a = as_signal(a)
        self.b = as_signal(b)

    def value(self, ctx):
        return self.OPS[self.op](self.a(ctx), self.b(ctx))

    def js(self):
        a, b = self.a.js(), self.b.js()
        if self.op == "**":
            return f"Math.pow({a},{b})"
        return f"({a}{self.op}{b})"


class Fn1(Signal):
    """Unary helpers (abs, sin, clip, ...)."""

    _PY = {
        "abs": lambda x, *a: abs(x), "sin": lambda x, *a: math.sin(x), "cos": lambda x, *a: math.cos(x),
        "floor": lambda x, *a: math.floor(x), "fract": lambda x, *a: _fract(x),
        "sqrt": lambda x, *a: math.sqrt(max(x, 0.0)), "smoothstep": lambda x, *a: _smoothstep(x),
        "clip": lambda x, lo, hi: min(max(x, lo), hi),
        "quantize": lambda x, n: math.floor(x * n) / n if n else x,
    }

    def __init__(self, fn: str, a, *extra):
        self.fn = fn
        self.a = as_signal(a)
        self.extra = [float(e) for e in extra]

    def value(self, ctx):
        return self._PY[self.fn](self.a(ctx), *self.extra)

    def js(self):
        a = self.a.js()
        e = self.extra
        if self.fn == "abs": return f"Math.abs({a})"
        if self.fn == "sin": return f"Math.sin({a})"
        if self.fn == "cos": return f"Math.cos({a})"
        if self.fn == "floor": return f"Math.floor({a})"
        if self.fn == "fract": return f"__fract({a})"
        if self.fn == "sqrt": return f"Math.sqrt(Math.max({a},0))"
        if self.fn == "smoothstep": return f"__smoothstep({a})"
        if self.fn == "clip": return f"Math.min(Math.max({a},{_num(e[0])}),{_num(e[1])})"
        if self.fn == "quantize": return f"(Math.floor(({a})*{_num(e[0])})/{_num(e[0])})"
        raise ValueError(self.fn)


class Lfo(Signal):
    """Low-frequency oscillator. ``shape``: sine | tri | saw | square | pulse. Output in [lo, hi]."""

    def __init__(self, freq: float = 1.0, shape: str = "sine", lo: float = 0.0, hi: float = 1.0,
                 phase: float = 0.0, width: float = 0.25):
        self.freq, self.shape, self.lo, self.hi, self.phase, self.width = (
            float(freq), shape, float(lo), float(hi), float(phase), float(width))
        if shape not in ("sine", "tri", "saw", "square", "pulse"):
            raise ValueError(f"unknown LFO shape {shape!r}")

    def value(self, ctx):
        p = _fract(ctx.time * self.freq + self.phase)
        if self.shape == "sine":
            u = 0.5 + 0.5 * math.sin(2 * math.pi * p)
        elif self.shape == "tri":
            u = 1.0 - abs(2.0 * p - 1.0)
        elif self.shape == "saw":
            u = p
        elif self.shape == "square":
            u = 1.0 if p < 0.5 else 0.0
        else:
            u = 1.0 if p < self.width else 0.0
        return self.lo + (self.hi - self.lo) * u

    def js(self):
        p = f"__fract(time*{_num(self.freq)}+{_num(self.phase)})"
        u = {
            "sine": f"(0.5+0.5*Math.sin(6.283185307179586*{p}))",
            "tri": f"(1-Math.abs(2*{p}-1))",
            "saw": p,
            "square": f"({p}<0.5?1:0)",
            "pulse": f"({p}<{_num(self.width)}?1:0)",
        }[self.shape]
        return f"({_num(self.lo)}+{_num(self.hi - self.lo)}*{u})"


class Seq(Signal):
    """Hydra array sequencing: ``Seq([1, 2, 3]).fast(2).smooth().ease('easeInOutCubic')``.

    Steps at ``bpm / 60`` values per second (bpm from the patch, Hydra's default 30).
    """

    def __init__(self, values: Sequence[float], speed: float = 1.0, smooth: float = 0.0,
                 offset: float = 0.0, ease: str = "linear"):
        if not values:
            raise ValueError("Seq needs at least one value")
        self.values = [float(v) for v in values]
        self._speed, self._smooth, self._offset, self._ease = float(speed), float(smooth), float(offset), ease

    def fast(self, speed: float = 1.0) -> "Seq":
        return Seq(self.values, speed, self._smooth, self._offset, self._ease)

    def smooth(self, smooth: float = 1.0) -> "Seq":
        return Seq(self.values, self._speed, smooth, self._offset, self._ease)

    def offset(self, offset: float = 0.5) -> "Seq":
        return Seq(self.values, self._speed, self._smooth, offset % 1.0, self._ease)

    def ease(self, ease: str = "linear") -> "Seq":
        if ease not in _EASINGS:
            raise ValueError(f"unknown easing {ease!r}; one of {', '.join(_EASINGS)}")
        return Seq(self.values, self._speed, self._smooth, self._offset, ease)

    def fit(self, lo: float = 0.0, hi: float = 1.0) -> "Seq":
        mn, mx = min(self.values), max(self.values)
        span = (mx - mn) or 1.0
        return Seq([lo + (v - mn) / span * (hi - lo) for v in self.values],
                   self._speed, self._smooth, self._offset, self._ease)

    def value(self, ctx):
        n = len(self.values)
        index = ctx.time * self._speed * (ctx.bpm / 60.0) + self._offset
        if self._smooth != 0:
            ease = _EASINGS[self._ease]
            _index = index - self._smooth / 2
            cur = self.values[int(math.floor(_index % n))]
            nxt = self.values[int(math.floor((_index + 1) % n))]
            t = min((_index % 1) / self._smooth, 1.0)
            return ease(t) * (nxt - cur) + cur
        return self.values[int(math.floor(index % n))]

    def js(self):
        arr = "[" + ",".join(_num(v) for v in self.values) + "]"
        return (f"__seq({arr},{_num(self._speed)},{_num(self._smooth)},{_num(self._offset)},"
                f"{self._ease!r},time,bpm)")

    def hydra_array(self) -> str:
        """Hydra-native array literal with modifiers."""
        s = "[" + ",".join(f"{v:g}" for v in self.values) + "]"
        if self._speed != 1:
            s += f".fast({self._speed:g})"
        if self._smooth:
            s += f".smooth({self._smooth:g})"
        if self._offset:
            s += f".offset({self._offset:g})"
        if self._ease != "linear":
            s += f".ease({self._ease!r})"
        return s


class Clock(Signal):
    """Phase 0..1 cycling ``div`` times per beat (beat = 60/bpm seconds)."""

    def __init__(self, div: float = 1.0, bpm: Optional[float] = None, phase: float = 0.0):
        self.div, self.bpm, self.phase = float(div), bpm, float(phase)

    def value(self, ctx):
        bpm = self.bpm if self.bpm is not None else ctx.bpm
        return _fract(ctx.time * (bpm / 60.0) * self.div + self.phase)

    def js(self):
        bpm = _num(self.bpm) if self.bpm is not None else "bpm"
        return f"__fract(time*({bpm}/60)*{_num(self.div)}+{_num(self.phase)})"


class Env(Signal):
    """Attack/decay envelope retriggered by a phase signal (a ``Clock``). ``attack``/``decay`` are fractions of the cycle."""

    def __init__(self, clock: Signal | None = None, attack: float = 0.05, decay: float = 0.5,
                 curve: float = 2.0, lo: float = 0.0, hi: float = 1.0):
        self.clock = clock or Clock()
        self.attack, self.decay, self.curve, self.lo, self.hi = (
            max(float(attack), 1e-6), max(float(decay), 1e-6), float(curve), float(lo), float(hi))

    def value(self, ctx):
        p = self.clock(ctx)
        if p < self.attack:
            u = p / self.attack
        else:
            u = max(0.0, 1.0 - (p - self.attack) / self.decay) ** self.curve
        return self.lo + (self.hi - self.lo) * u

    def js(self):
        return (f"__env({self.clock.js()},{_num(self.attack)},{_num(self.decay)},{_num(self.curve)},"
                f"{_num(self.lo)},{_num(self.hi)})")


class SH(Signal):
    """Sample-and-hold noise: a new random value in [lo, hi] ``rate`` times per second."""

    def __init__(self, rate: float = 2.0, lo: float = 0.0, hi: float = 1.0, seed: float = 0.0):
        self.rate, self.lo, self.hi, self.seed = float(rate), float(lo), float(hi), float(seed)

    def value(self, ctx):
        return self.lo + (self.hi - self.lo) * _hash11(math.floor(ctx.time * self.rate) + self.seed * 131.0)

    def js(self):
        return f"({_num(self.lo)}+{_num(self.hi - self.lo)}*__hash11(Math.floor(time*{_num(self.rate)})+{_num(self.seed * 131.0)}))"


class Drift(Signal):
    """Smooth random wander in [lo, hi] (1D value noise), ``rate`` turning points per second."""

    def __init__(self, rate: float = 0.5, lo: float = 0.0, hi: float = 1.0, seed: float = 0.0):
        self.rate, self.lo, self.hi, self.seed = float(rate), float(lo), float(hi), float(seed)

    def value(self, ctx):
        x = ctx.time * self.rate
        i = math.floor(x)
        a = _hash11(i + self.seed * 131.0)
        b = _hash11(i + 1 + self.seed * 131.0)
        u = a + (b - a) * _smoothstep(x - i)
        return self.lo + (self.hi - self.lo) * u

    def js(self):
        return f"({_num(self.lo)}+{_num(self.hi - self.lo)}*__drift(time*{_num(self.rate)},{_num(self.seed * 131.0)}))"


class Fn(Signal):
    """An opaque Python callable ``f(ctx) -> float`` (Hydra's ``({time}) => ...``).

    Exports can't run Python, so for HTML/Hydra it is *baked*: sampled over
    ``bake_seconds`` at ``bake_rate`` Hz and looped.
    """

    def __init__(self, fn: Callable, bake_seconds: float = 30.0, bake_rate: float = 20.0):
        self.fn = fn
        self.bake_seconds, self.bake_rate = float(bake_seconds), float(bake_rate)

    def value(self, ctx):
        try:
            return float(self.fn(ctx))
        except TypeError:
            return float(self.fn())

    def bake(self, bpm: float = 30.0) -> list[float]:
        from .core import Ctx
        n = int(self.bake_seconds * self.bake_rate)
        out = []
        for i in range(n):
            t = i / self.bake_rate
            out.append(self.value(Ctx(time=t, frame=int(t * 60), bpm=bpm)))
        return out

    def js(self):
        tbl = "[" + ",".join(f"{v:.5g}" for v in self.bake()) + "]"
        return f"__table({tbl},{_num(self.bake_rate)},time)"

    def __repr__(self):
        src = getattr(self, "js_source", None)
        return f"<Fn () => {src}>" if src else "<Fn python>"


class Fft(Signal):
    """One audio band of an ``Audio`` analyser, like Hydra's ``a.fft[i]``.

    ``kind`` may also be ``'vol'`` (overall loudness, ``a.vol``) or ``'flux'``
    (onset strength: positive spectral flux, ``a.flux``) — handy as a kick envelope.
    """

    def __init__(self, audio: "Audio", band: int = 0, kind: str = "band"):
        self.audio, self.band, self.kind = audio, int(band), kind

    def value(self, ctx):
        if self.kind == "vol":
            return self.audio.vol_at(ctx.time)
        if self.kind == "flux":
            return self.audio.flux_at(ctx.time)
        return self.audio.band_at(ctx.time, self.band)

    def js(self):
        if self.kind == "vol":
            return "__vol"
        if self.kind == "flux":
            return "__flux"
        return f"__fft[{self.band}]"


# ---------------------------------------------------------------------------
# Audio analysis (offline from a file; live input via sounddevice if present)
# ---------------------------------------------------------------------------

class Audio:
    """Audio-reactive signals.

    ``a = Audio.from_file('track.wav'); osc(10, 0.1, a.fft[0]).out()``

    Bands follow Hydra's pipeline loosely: Bark-scale specific loudness
    (power ** 0.23), averaged into ``bins`` bands, then
    ``max(0, (smoothed - cutoff) / scale)`` — tune with ``set_scale`` /
    ``set_cutoff`` / ``set_smooth`` exactly as you would ``a.setScale()`` in Hydra.
    """

    def __init__(self, samples=None, sr: int = 44100, bins: int = 4, smooth: float = 0.4,
                 scale: float = 10.0, cutoff: float = 2.0, path: Optional[str] = None):
        self.samples = samples
        self.sr = int(sr)
        self.bins = int(bins)
        self.smooth = float(smooth)
        self.scale = float(scale)
        self.cutoff = float(cutoff)
        self.path = path
        self.fft_size = 1024
        self._cache: dict[int, tuple] = {}
        self._prev_t: Optional[float] = None
        self._prev: list[float] = [0.0] * 24
        self._prev_raw: list[float] = [0.0] * 24
        self._flux_sm = 0.0
        self.fft = [Fft(self, i) for i in range(self.bins)]
        self.vol = Fft(self, 0, "vol")
        self.flux = Fft(self, 0, "flux")
        self._bark_edges = None

    # Hydra-compatible setters -------------------------------------------
    def set_bins(self, n: int):
        self.bins = int(n)
        self.fft = [Fft(self, i) for i in range(self.bins)]
        return self

    def set_smooth(self, s: float):
        self.smooth = float(s); return self

    def set_scale(self, s: float):
        self.scale = float(s); return self

    def set_cutoff(self, c: float):
        self.cutoff = float(c); return self

    setBins, setSmooth, setScale, setCutoff = set_bins, set_smooth, set_scale, set_cutoff

    @classmethod
    def from_file(cls, path: str, **kw) -> "Audio":
        samples, sr = _load_audio(path)
        return cls(samples, sr, path=path, **kw)

    def duration(self) -> float:
        return 0.0 if self.samples is None else len(self.samples) / self.sr

    # analysis ------------------------------------------------------------
    def _bands_raw(self, t: float) -> list[float]:
        import numpy as np
        if self.samples is None:
            return [0.0] * 24
        n = self.fft_size
        centre = int(t * self.sr)
        start = centre - n // 2
        seg = np.zeros(n, dtype=np.float32)
        s0, s1 = max(start, 0), min(start + n, len(self.samples))
        if s1 > s0:
            seg[s0 - start:s1 - start] = self.samples[s0:s1]
        seg *= np.hanning(n).astype(np.float32)
        mag = np.abs(np.fft.rfft(seg))
        power = mag * mag
        if self._bark_edges is None:
            freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
            bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)
            self._bark_edges = np.clip(np.floor(bark).astype(int), 0, 23)
        specific = np.zeros(24, dtype=np.float64)
        np.add.at(specific, self._bark_edges, power)
        return list(np.power(specific, 0.23))

    def _analyse(self, t: float) -> tuple:
        key = int(round(t * 1000))
        if key not in self._cache:
            raw = self._bands_raw(t)
            sequential = self._prev_t is not None and t > self._prev_t and (t - self._prev_t) < 0.5
            if sequential:
                sm = [p * self.smooth + r * (1 - self.smooth) for p, r in zip(self._prev, raw)]
                flux = sum(max(0.0, r - q) for r, q in zip(raw, self._prev_raw)) / 24.0
                self._flux_sm = self._flux_sm * self.smooth + flux * (1 - self.smooth)
            else:
                sm = raw
                self._flux_sm = 0.0
            self._prev, self._prev_raw, self._prev_t = sm, raw, t
            per = 24 // max(self.bins, 1)
            vals = []
            for i in range(self.bins):
                chunk = sm[i * per:(i + 1) * per] or [0.0]
                vals.append(max(0.0, (sum(chunk) / len(chunk) - self.cutoff) / self.scale))
            vol = sum(vals) / max(len(vals), 1)
            self._cache[key] = ([float(v) for v in vals], float(vol), float(self._flux_sm / self.scale))
        return self._cache[key]

    def band_at(self, t: float, band: int) -> float:
        vals = self._analyse(t)[0]
        return vals[band] if band < len(vals) else 0.0

    def vol_at(self, t: float) -> float:
        return self._analyse(t)[1]

    def flux_at(self, t: float) -> float:
        return self._analyse(t)[2]

    def embed_b64(self) -> Optional[str]:
        """Base64 of the source file for the HTML export (None if there is no file)."""
        if not self.path:
            return None
        import base64, mimetypes
        mime = mimetypes.guess_type(self.path)[0] or "audio/wav"
        with open(self.path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _load_audio(path: str):
    """Decode to mono float32 using the wave module or ffmpeg."""
    import numpy as np
    import subprocess, wave, shutil
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 1:
            a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise ValueError("unsupported WAV sample width")
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        return a, sr
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg is needed to decode non-WAV audio")
    sr = 44100
    out = subprocess.run([ff, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy(), sr


# JS runtime helpers used by the expressions above (included in the HTML export)
JS_HELPERS = r"""
function __fract(x){return x-Math.floor(x);}
function __hash11(x){return __fract(Math.sin(x*12.9898+78.233)*43758.5453);}
function __smoothstep(t){t=Math.min(Math.max(t,0),1);return t*t*(3-2*t);}
function __drift(x,seed){var i=Math.floor(x),a=__hash11(i+seed),b=__hash11(i+1+seed);return a+(b-a)*__smoothstep(x-i);}
function __env(p,a,d,c,lo,hi){var u=p<a?p/a:Math.pow(Math.max(0,1-(p-a)/d),c);return lo+(hi-lo)*u;}
function __table(tbl,rate,t){var x=t*rate,n=tbl.length,i=Math.floor(x)%n,j=(i+1)%n,f=x-Math.floor(x);return tbl[i]+(tbl[j]-tbl[i])*f;}
var __EASE={linear:function(t){return t;},easeInQuad:function(t){return t*t;},easeOutQuad:function(t){return t*(2-t);},
easeInOutQuad:function(t){return t<0.5?2*t*t:-1+(4-2*t)*t;},easeInCubic:function(t){return t*t*t;},easeOutCubic:function(t){return Math.pow(t-1,3)+1;},
easeInOutCubic:function(t){return t<0.5?4*t*t*t:(t-1)*(2*t-2)*(2*t-2)+1;},easeInQuart:function(t){return t*t*t*t;},easeOutQuart:function(t){return 1-Math.pow(t-1,4);},
easeInOutQuart:function(t){return t<0.5?8*t*t*t*t:1-8*Math.pow(t-1,4);},easeInQuint:function(t){return Math.pow(t,5);},easeOutQuint:function(t){return 1+Math.pow(t-1,5);},
easeInOutQuint:function(t){return t<0.5?16*Math.pow(t,5):1+16*Math.pow(t-1,5);},sin:function(t){return 0.5-0.5*Math.cos(Math.PI*t);}};
function __seq(arr,speed,smooth,offset,ease,time,bpm){var n=arr.length;var index=time*speed*(bpm/60)+offset;
 if(smooth!==0){var e=__EASE[ease]||__EASE.linear;var _i=index-smooth/2;var cur=arr[Math.floor(((_i%n)+n)%n)];var nxt=arr[Math.floor((((_i+1)%n)+n)%n)];
  var t=Math.min(((_i%1)+1)%1/smooth,1);return e(t)*(nxt-cur)+cur;}
 return arr[Math.floor(((index%n)+n)%n)];}
"""
