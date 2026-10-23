"""Neural cellular automata: a texture that grows itself, and heals.

After Mordvintsev, Niklasson, Randazzo and Levin, *Self-Organising Textures* (Distill, 2021).
Every cell of a small grid carries ``channels`` numbers (the first three are its colour); each
step it looks at its 3x3 neighbourhood through four fixed filters (identity, Sobel x, Sobel y,
Laplacian), passes those through a two-layer network (a few thousand weights) and adds the result
to itself — at a random half of the cells each step.  Trained on one photograph
(``hydrapy nca train bark.jpg``, needs PyTorch: see ``hydrapy.nca_train``) the rule grows that
texture from nothing, keeps it alive, and repairs it when it is damaged.

    lichen = NCA("lichen.npz", size=(160, 90), speed=30)
    src(lichen).out(o0)

    shape(4, 0.2).scroll(Lfo(0.1, lo=-0.4, hi=0.4), 0).out(o1)
    moss = NCA("moss.npz", damage=o1)            # wherever o1 is bright the cells are wiped; they regrow
    src(moss).out(o0)

Controls (numbers or signals, so CV works): ``rate`` the fraction of cells updating each step
(0.5 trained; lower is slower and looser), ``perturb`` noise injected into the hidden channels
(the texture mutates, then settles), ``paint`` an output whose colour is written into the cells
(``paint_amount``), ``speed`` steps per second.  Runs on the NumPy and GL backends and in the HTML
export (WebGL2 float targets).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .core import Output, Texture
from .noise import h31
from .signals import as_signal

# filters in image coordinates: weight on x[row + dy, col + dx], rows going *down* the picture
_IDENT = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
_SOBEL_Y = _SOBEL_X.T.copy()
_LAP = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], np.float32)
FILTERS = np.stack([_IDENT, _SOBEL_X, _SOBEL_Y, _LAP])          # (4, 3, 3), as in the Distill PyTorch notebook
CLAMP = 3.0                  # the state is clamped to ±CLAMP after each update, in training and at run time


class NCAModel:
    """Weights of a texture NCA: features (per channel: ident, sobel x, sobel y, laplacian) ->
    ``hidden`` ReLU units -> ``channels`` increments.  Feature index = channel * 4 + filter."""

    def __init__(self, w1, b1, w2, meta: Optional[dict] = None):
        self.w1 = np.asarray(w1, np.float32)                     # (hidden, 4 * channels)
        self.b1 = np.asarray(b1, np.float32)                     # (hidden,)
        self.w2 = np.asarray(w2, np.float32)                     # (channels, hidden)
        self.hidden, nf = self.w1.shape
        self.channels = self.w2.shape[0]
        if nf != 4 * self.channels or self.w2.shape[1] != self.hidden or self.b1.shape != (self.hidden,):
            raise ValueError("inconsistent NCA weights")
        if self.channels % 4:
            raise ValueError("NCA channels must be a multiple of 4 (they are packed into RGBA textures)")
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: str) -> "NCAModel":
        d = np.load(path, allow_pickle=False)
        meta = {k[5:]: d[k].item() if d[k].shape == () else d[k].tolist() for k in d.files if k.startswith("meta_")}
        return cls(d["w1"], d["b1"], d["w2"], meta)

    def save(self, path: str) -> str:
        extra = {f"meta_{k}": np.asarray(v) for k, v in self.meta.items()}
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, **extra)
        return path

    @classmethod
    def random(cls, channels: int = 12, hidden: int = 96, seed: int = 0, scale: float = 0.1) -> "NCAModel":
        """An untrained rule (for tests and happy accidents: random rules make their own textures)."""
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0, 1.0 / math.sqrt(4 * channels), (hidden, 4 * channels))
        b1 = rng.normal(0, 0.05, hidden)
        w2 = rng.normal(0, scale / math.sqrt(hidden), (channels, hidden))
        return cls(w1, b1, w2, {"source": "random", "seed": seed})

    @property
    def textures(self) -> int:
        return self.channels // 4


# -----------------------------------------------------------------------------
# NumPy step (state arrays are GL-oriented: row 0 = bottom, like every other hydrapy buffer)
# -----------------------------------------------------------------------------

def perceive(state: np.ndarray) -> np.ndarray:
    """(h, w, C) -> (h, w, 4C) with feature index c * 4 + f, circular boundaries."""
    h, w, c = state.shape
    feats = np.zeros((h, w, c, 4), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # image row r + dy is GL row r' - dy; np.roll(a, k)[i] == a[i - k]
            nb = np.roll(np.roll(state, dy, axis=0), -dx, axis=1)
            k = FILTERS[:, dy + 1, dx + 1]
            if np.any(k):
                feats += nb[..., None] * k[None, None, None, :]
    return feats.reshape(h, w, 4 * c)


def step_np(model: NCAModel, state: np.ndarray, step_index: int, rate: float = 0.5, perturb: float = 0.0,
            seed: float = 0.0, damage: Optional[np.ndarray] = None, paint: Optional[np.ndarray] = None,
            paint_amount: float = 0.5) -> np.ndarray:
    h, w, c = state.shape
    f = perceive(state)
    hid = np.maximum(f @ model.w1.T + model.b1, 0.0)
    dx = hid @ model.w2.T
    iy, ix = np.mgrid[0:h, 0:w].astype(np.float32)
    s = np.float32(step_index) + np.float32(seed) * np.float32(131.0)
    fire = (h31(np.stack([ix, iy, np.full_like(ix, s)], axis=-1)) < rate).astype(np.float32)
    new = np.clip(state + dx * fire[..., None], -CLAMP, CLAMP)
    if perturb:
        for ch in range(3, c):
            n = h31(np.stack([ix, iy, np.full_like(ix, s * np.float32(3.1) + np.float32(ch + 0.37))], axis=-1))
            new[..., ch] += np.float32(perturb) * (n - np.float32(0.5))
    if paint is not None:
        cov = np.clip(paint[..., :3].max(-1), 0.0, 1.0) * np.float32(paint_amount)
        new[..., :3] = new[..., :3] * (1 - cov[..., None]) + (paint[..., :3] - 0.5) * cov[..., None]
    if damage is not None:
        d = damage[..., 0] * 0.2125 + damage[..., 1] * 0.7154 + damage[..., 2] * 0.0721
        t = np.clip((d - 0.4) / 0.2, 0.0, 1.0)
        new = new * (1.0 - t * t * (3.0 - 2.0 * t))[..., None]
    return new.astype(np.float32)


def visible(state: np.ndarray) -> np.ndarray:
    rgb = np.clip(state[..., :3] + 0.5, 0.0, 1.0)
    return np.concatenate([rgb, np.ones(state.shape[:2] + (1,), np.float32)], axis=-1).astype(np.float32)


# -----------------------------------------------------------------------------
# GLSL (ES 1.00; the weights are compiled into the shader as constants)
# -----------------------------------------------------------------------------

def _f(v: float) -> str:
    s = f"{float(v):.7g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def _v4(vals) -> str:
    return "vec4(" + ", ".join(_f(v) for v in vals) + ")"


def step_glsl(model: NCAModel, k: int) -> str:
    """Fragment shader computing state texture ``k`` (channels 4k..4k+3) for one step."""
    T = model.textures
    lines = [
        "uniform vec2 res;",
        "uniform float stepIndex, rate, perturb, seed, paintAmount, damageOn, paintOn;",
        "uniform sampler2D damage, paint;",
    ] + [f"uniform sampler2D s{t};" for t in range(T)] + [
        "void main() {",
        "  vec2 px = 1.0 / res;",
        "  vec2 uv = gl_FragCoord.xy * px;",
    ]
    # taps: image offset (dy, dx) is GL offset (dx, -dy)
    for t in range(T):
        lines.append(f"  vec4 F{t}_0 = vec4(0.0), F{t}_1 = vec4(0.0), F{t}_2 = vec4(0.0), F{t}_3 = vec4(0.0);")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ks = FILTERS[:, dy + 1, dx + 1]
                if not np.any(ks):
                    continue
                v = f"texture2D(s{t}, fract(uv + vec2({_f(dx)}, {_f(-dy)}) * px))"
                lines.append(f"  {{ vec4 n = {v};")
                for fi in range(4):
                    if ks[fi]:
                        lines.append(f"    F{t}_{fi} += {_f(ks[fi])} * n;")
                lines.append("  }")
    lines.append("  vec4 o = vec4(0.0);")
    for j in range(model.hidden):
        terms = []
        for t in range(T):
            for fi in range(4):
                wv = [model.w1[j, (4 * t + i) * 4 + fi] for i in range(4)]
                if any(wv):
                    terms.append(f"dot(F{t}_{fi}, {_v4(wv)})")
        out_w = [model.w2[4 * k + i, j] for i in range(4)]
        if not any(out_w):
            continue
        lines.append(f"  o += max(0.0, {_f(model.b1[j])} + {' + '.join(terms) or '0.0'}) * {_v4(out_w)};")
    lines += [
        "  vec2 cell = floor(gl_FragCoord.xy);",
        "  float s = stepIndex + seed * 131.0;",
        "  float fire = _h31(vec3(cell, s)) < rate ? 1.0 : 0.0;",
        f"  vec4 x = clamp(texture2D(s{k}, uv) + o * fire, {_f(-CLAMP)}, {_f(CLAMP)});",
        "  if (perturb != 0.0) {",
        "    vec4 n = vec4(" + ", ".join(f"_h31(vec3(cell, s * 3.1 + {_f(4 * k + i + 0.37)}))" for i in range(4)) + ") - 0.5;",
        "    x += perturb * n" + (" * vec4(0.0, 0.0, 0.0, 1.0)" if k == 0 else "") + ";",
        "  }",
    ]
    if k == 0:
        lines += [
            "  if (paintOn > 0.5) {",
            "    vec4 p = texture2D(paint, uv);",
            "    float cov = clamp(max(p.r, max(p.g, p.b)), 0.0, 1.0) * paintAmount;",
            "    x.rgb = x.rgb * (1.0 - cov) + (p.rgb - 0.5) * cov;",
            "  }",
        ]
    lines += [
        "  if (damageOn > 0.5) {",
        "    float d = _luminance(texture2D(damage, uv).rgb);",
        "    x *= 1.0 - smoothstep(0.4, 0.6, d);",
        "  }",
        "  gl_FragColor = x;",
        "}",
    ]
    return "\n".join(lines) + "\n"


VIEW_GLSL = """uniform vec2 res;
uniform sampler2D s0;
void main() {
  vec4 x = texture2D(s0, gl_FragCoord.xy / res);
  gl_FragColor = vec4(clamp(x.rgb + 0.5, 0.0, 1.0), 1.0);
}
"""


def shader_header(precision: str = "highp") -> str:
    from .noise import GLSL_NOISE
    return f"precision {precision} float;\n" + GLSL_NOISE + "\n"


# -----------------------------------------------------------------------------
# The texture
# -----------------------------------------------------------------------------

class NCA(Texture):
    """A neural-CA texture for ``src()``.  ``model``: an ``NCAModel`` or a path to a ``.npz``."""

    _counter = 0

    def __init__(self, model, size=(160, 90), speed: float = 30.0, rate=0.5, perturb=0.0,
                 damage: Optional[Output] = None, paint: Optional[Output] = None, paint_amount=0.5,
                 seed: float = 0.0, max_steps: int = 8, filter: str = "linear"):
        self.model = model if isinstance(model, NCAModel) else NCAModel.load(model)
        self.size = (int(size[0]), int(size[1]))
        self.speed, self.max_steps, self.seed = float(speed), int(max_steps), float(seed)
        self.rate, self.perturb, self.paint_amount = as_signal(rate), as_signal(perturb), as_signal(paint_amount)
        self.damage, self.paint, self.filter = damage, paint, filter
        NCA._counter += 1
        self.name = f"nca{NCA._counter}"
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size[1], self.size[0], self.model.channels), np.float32)
        self.step_index = 0
        self._target = None          # steps are a function of time: floor(t * speed)
        self._visible = None
        self.gpu_reset = True        # tells a GL renderer to clear its copy of the state

    def steps_due(self, t: float) -> int:
        """How many steps to run to reach time ``t`` (bounded by ``max_steps``); restarts if time goes back."""
        target = math.floor(t * self.speed + 1e-9)
        if self._target is None or target < self._target:
            if self._target is not None:
                self.reset()
            self._target = target
            return 1 if self.step_index == 0 else 0
        n = min(target - self._target, self.max_steps)
        self._target = target
        return n

    # NumPy backend
    def advance_np(self, env, t: float) -> np.ndarray:
        n = self.steps_due(t)
        if n:
            ctx = env.ctx
            w, h = self.size
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            st = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], axis=-1)          # cell centres, as the shader samples
            dmg = env.sample(env.texture(self.damage), st) if self.damage is not None else None
            pnt = env.sample(env.texture(self.paint), st) if self.paint is not None else None
            for _ in range(n):
                self.state = step_np(self.model, self.state, self.step_index, float(self.rate(ctx)), float(self.perturb(ctx)),
                                     self.seed, dmg, pnt, float(self.paint_amount(ctx)))
                self.step_index += 1
            self._visible = None
        if getattr(self, "_visible", None) is None:
            self._visible = visible(self.state)
        return self._visible
