"""Hyperbolic geometry: the Poincaré disk and band models as coordinate ops.

``hyperkaleid(p, q, ...)`` folds the frame into the fundamental triangle of the (2, p, q) triangle group, so
whatever the chain draws there tiles the hyperbolic plane the way Escher's *Circle Limit* prints do, with the
tiles shrinking towards the boundary. ``mobius(shift, direction, turn)`` is an isometry of the disk, the way to
scroll or turn such a picture without breaking it. ``hyperfade(fade, model)`` is a grey source, 1 in the middle
and 0 at the boundary: multiplied into a picture it fades it to black at the edge, which hides the sub-pixel tiles
near the boundary (they shimmer under motion, which is a photosensitivity problem on a screen, not only an
aliasing one).

Two models of the plane, chosen with ``model``:

- 0, the Poincaré disk: a circle whose diameter is the frame's height, centred on the frame;
- 1, the band: the strip |Im w| < pi/2 across the frame's height, w = log((1 + z) / (1 - z)). It fills a
  rectangular frame, and a hyperbolic translation along its axis is a plain horizontal shift.

In both, ``shift`` is a hyperbolic distance along the geodesic through the centre at angle ``direction``, so a
scroll is the same move whichever model shows it.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Op, register
from .ops import F, bc

PI = math.pi
ASP = "vec2 asp = vec2(resolution.x / resolution.y, 1.0);"
MAX_FOLDS = 24


def _asp(env):
    return np.array([env.res[0] / env.res[1], 1.0], np.float32)


def triangle(p: float, q: float):
    """The (2, p, q) right triangle at the origin: (wedge angle, rho_C, circle centre x0, circle radius r, rho_B).

    Angle pi/p at the origin, pi/2 at C on the positive real axis, pi/q at B on the ray at angle pi/p.
    The side BC is the geodesic through C orthogonal to the real axis: a circle of radius ``r`` about
    ``(x0, 0)``, orthogonal to the unit circle. ``q`` at or above 100 is treated as infinity: B sits on the
    boundary (the modular group is p=3, q=infinity). Hyperbolic when (p - 2)(q - 2) > 4.
    """
    p = max(float(p), 2.0001)
    alpha = PI / p
    cos_b = 1.0 if q >= 100 else math.cos(PI / max(float(q), 1.0001))
    cosh_oc = max(cos_b / math.sin(alpha), 1.0 + 1e-6)                # the leg OC
    rho_c = math.sqrt((cosh_oc - 1.0) / (cosh_oc + 1.0))              # tanh(OC / 2)
    x0 = (1.0 + rho_c * rho_c) / (2.0 * rho_c)
    r = x0 - rho_c
    cosh_ob = max((1.0 / math.tan(alpha)) / math.tan(PI / q), 1.0) if q < 100 else 1e9   # the hypotenuse OB
    rho_b = 1.0 if cosh_ob > 1e8 else math.sqrt((cosh_ob - 1.0) / (cosh_ob + 1.0))
    return alpha, rho_c, x0, r, rho_b


def _to_disk(st, model, asp):
    """Frame st -> complex disk coordinate (x, y) arrays, for either model."""
    x = (st[..., 0] - 0.5) * asp[0] * 2.0
    y = (st[..., 1] - 0.5) * 2.0
    if model < 0.5:
        return x, y
    u, v = x * (PI / 2.0), y * (PI / 2.0)                              # the band: |v| < pi/2 across the frame
    # z = tanh(w / 2), w = u + i v
    d = np.cosh(u) + np.cos(v)
    return np.sinh(u) / d, np.sin(v) / d


def _from_disk(x, y, model, asp):
    if model >= 0.5:
        # w = log((1 + z) / (1 - z)) = 2 atanh(z)
        den = (1.0 - x) ** 2 + y * y + 1e-12
        u = 0.5 * np.log(((1.0 + x) ** 2 + y * y) / den)
        v = np.arctan2(2.0 * y, 1.0 - x * x - y * y)
        x, y = u / (PI / 2.0), v / (PI / 2.0)
    return np.stack([x / (2.0 * asp[0]) + 0.5, y / 2.0 + 0.5], axis=-1).astype(np.float32)


def _mobius(x, y, shift, direction, turn):
    """Sample at T^-1 z: translate by `shift` along `direction`, then turn, so the picture moves the other way."""
    t = np.tanh(np.clip(shift / 2.0, -20.0, 20.0))
    ax, ay = t * np.cos(direction), t * np.sin(direction)
    # (z - a) / (1 - conj(a) z)
    nx, ny = x - ax, y - ay
    dx, dy = 1.0 - (ax * x + ay * y), -(ax * y - ay * x)
    dd = dx * dx + dy * dy + 1e-12
    rx, ry = (nx * dx + ny * dy) / dd, (ny * dx - nx * dy) / dd
    c, s = np.cos(-turn), np.sin(-turn)
    return rx * c - ry * s, rx * s + ry * c


def _valid_group(p, q):
    return (math.isfinite(p) and math.isfinite(q) and p > 2 and q > 2
            and (p - 2) * (q - 2) > 4)


def _validate_hyperkaleid(args):
    p, q = args[:2]
    if isinstance(p, (int, float)) and isinstance(q, (int, float)) and not _valid_group(p, q):
        raise ValueError(f"hyperkaleid({p:g}, {q:g}): (2,p,q) needs 1/p + 1/q < 1/2 "
                         "with finite p,q > 2; try (4, 5) or (3, 7)")


def _np_hyperkaleid(st, p, q, shift, direction, turn, model, zoom, env):
    if not _valid_group(float(p), float(q)):
        return st
    asp = _asp(env)
    alpha, rho_c, x0, r, rho_b = triangle(float(p), float(q))
    x, y = _to_disk(st, float(model), asp)
    x, y = _mobius(x, y, bc(shift, st), bc(direction, st), bc(turn, st))
    rr = x * x + y * y
    out = rr >= 1.0                                                    # the outside: inverted in the unit circle
    inv = np.where(out, 1.0 / np.maximum(rr, 1e-12), 1.0)
    x, y = x * inv, y * inv
    r2 = r * r
    for _ in range(MAX_FOLDS):
        rad = np.sqrt(x * x + y * y)
        a = np.mod(np.arctan2(y, x), 2.0 * alpha)
        a = np.where(a > alpha, 2.0 * alpha - a, a)
        x, y = rad * np.cos(a), rad * np.sin(a)
        dx, dy = x - x0, y
        d2 = dx * dx + dy * dy
        inside = d2 < r2
        if not inside.any():
            break
        k = np.where(inside, r2 / np.maximum(d2, 1e-12), 1.0)
        x, y = x0 + dx * k, dy * k
    rad = np.sqrt(x * x + y * y)                                       # a last fold into the wedge, so the cap
    a = np.mod(np.arctan2(y, x), 2.0 * alpha)                          # never leaves a point outside it
    a = np.where(a > alpha, 2.0 * alpha - a, a)
    x, y = rad * np.cos(a), rad * np.sin(a)
    k = bc(zoom, st) / rho_b
    return np.stack([x * k / (2.0 * asp[0]) + 0.5, y * k / 2.0 + 0.5], axis=-1).astype(np.float32)


_HELPERS = """
float _hyp_cosh(float x) { x = clamp(x, -20.0, 20.0); return 0.5 * (exp(x) + exp(-x)); }
float _hyp_sinh(float x) { x = clamp(x, -20.0, 20.0); return 0.5 * (exp(x) - exp(-x)); }
float _hyp_tanh(float x) { float e = exp(2.0 * clamp(x, -20.0, 20.0)); return (e - 1.0) / (e + 1.0); }
vec2 _hyp_to_disk(vec2 st, float model, vec2 asp) {
  float x = (st.x - 0.5) * asp.x * 2.0, y = (st.y - 0.5) * 2.0;
  if (model < 0.5) return vec2(x, y);
  float u = x * 1.5707963, v = y * 1.5707963;
  float d = _hyp_cosh(u) + cos(v);
  return vec2(_hyp_sinh(u), sin(v)) / d;
}
vec2 _hyp_from_disk(vec2 z, float model, vec2 asp) {
  if (model >= 0.5) {
    float den = (1.0 - z.x) * (1.0 - z.x) + z.y * z.y + 1e-12;
    float u = 0.5 * log(((1.0 + z.x) * (1.0 + z.x) + z.y * z.y) / den);
    float v = atan(2.0 * z.y, 1.0 - dot(z, z));
    z = vec2(u, v) / 1.5707963;
  }
  return vec2(z.x / (2.0 * asp.x) + 0.5, z.y * 0.5 + 0.5);
}
vec2 _hyp_mobius(vec2 z, float shift, float direction, float turn) {
  float t = _hyp_tanh(shift * 0.5);
  vec2 a = t * vec2(cos(direction), sin(direction));
  vec2 n = z - a;
  vec2 d = vec2(1.0 - dot(a, z), -(a.x * z.y - a.y * z.x));
  float dd = dot(d, d) + 1e-12;
  vec2 r = vec2(n.x * d.x + n.y * d.y, n.y * d.x - n.x * d.y) / dd;
  float c = cos(-turn), s = sin(-turn);
  return vec2(r.x * c - r.y * s, r.x * s + r.y * c);
}
"""

_TRIANGLE = """
  float alpha = 3.14159265 / max(p, 2.0001);
  float cosb = q >= 100.0 ? 1.0 : cos(3.14159265 / max(q, 1.0001));
  float coshoc = max(cosb / sin(alpha), 1.000001);
  float rhoc = sqrt((coshoc - 1.0) / (coshoc + 1.0));
  float x0 = (1.0 + rhoc * rhoc) / (2.0 * rhoc);
  float rad = x0 - rhoc;
  float coshob = q >= 100.0 ? 1e9 : max((1.0 / tan(alpha)) / tan(3.14159265 / q), 1.0);
  float rhob = coshob > 1e8 ? 1.0 : sqrt((coshob - 1.0) / (coshob + 1.0));"""

register(Op("hyperkaleid", "coord",
            [F("p", 4, scalar=True, doc="tiles meeting at the centre: angle pi/p"),
             F("q", 5, scalar=True, doc="tiles meeting at each corner: angle pi/q; 100 or more means infinity"),
             F("shift", 0, doc="hyperbolic distance the picture has moved along `direction`"),
             F("direction", 0, doc="radians"), F("turn", 0, doc="radians, about the centre"),
             F("model", 0, scalar=True, doc="0 Poincaré disk, 1 the band"),
             F("zoom", 1, doc="1 fits the fundamental triangle to the frame's half height")],
            np=_np_hyperkaleid, glsl_helpers=_HELPERS, validate=_validate_hyperkaleid,
            doc="Hyperbolic kaleidoscope: fold the frame into the fundamental triangle of the (2, p, q) triangle "
                "group, so what the chain draws there tiles the Poincaré disk (model 0) or the band (model 1) "
                "like a Circle Limit print; hyperbolic when (p-2)(q-2) > 4, and q >= 100 puts the corners on "
                "the boundary (p=3 is the modular group). `shift`/`direction`/`turn` move the tiling by an "
                "isometry, so it flows without a seam. Outside the disk the tiling is inverted; `.mult(hyperfade())` "
                "fades the picture to black at the boundary. The triangle read is the wedge from the centre to the top "
                "edge between angles 0 and 180/p degrees. Invalid signal-driven p or q falls back to the input picture "
                "in the same pass. `zoom` scales the source wedge after folding; it is not an output zoom. Frame the "
                "output after the fold. In the band model, hyperfade is radial in the mapped disk, so it also fades the "
                "band's ends.",
            glsl="""
  if (!(p > 2.0 && q > 2.0 && p <= 3.402823466e38 && q <= 3.402823466e38
        && (p - 2.0) * (q - 2.0) > 4.0)) return _st;
""" + ASP + _TRIANGLE + """
  vec2 z = _hyp_mobius(_hyp_to_disk(_st, model, asp), shift, direction, turn);
  float rr = dot(z, z);
  if (rr >= 1.0) z /= max(rr, 1e-12);
  float r2 = rad * rad;
  for (int i = 0; i < 24; i++) {
    float len = length(z);
    float a = mod(atan(z.y, z.x), 2.0 * alpha);
    a = a > alpha ? 2.0 * alpha - a : a;
    z = len * vec2(cos(a), sin(a));
    vec2 d = vec2(z.x - x0, z.y);
    float d2 = dot(d, d);
    if (d2 >= r2) break;
    z = vec2(x0, 0.0) + d * (r2 / max(d2, 1e-12));
  }
  float len = length(z);
  float a = mod(atan(z.y, z.x), 2.0 * alpha);
  a = a > alpha ? 2.0 * alpha - a : a;
  z = len * vec2(cos(a), sin(a));
  float k = zoom / rhob;
  return vec2(z.x * k / (2.0 * asp.x) + 0.5, z.y * k * 0.5 + 0.5);"""))


def _np_mobius(st, shift, direction, turn, model, env):
    asp = _asp(env)
    x, y = _to_disk(st, float(model), asp)
    x, y = _mobius(x, y, bc(shift, st), bc(direction, st), bc(turn, st))
    return _from_disk(x, y, float(model), asp)


register(Op("mobius", "coord",
            [F("shift", 0, doc="hyperbolic distance moved along `direction`"), F("direction", 0, doc="radians"),
             F("turn", 0, doc="radians, about the centre"), F("model", 0, scalar=True, doc="0 disk, 1 band")],
            np=_np_mobius, glsl_helpers=_HELPERS,
            doc="An isometry of the hyperbolic plane: translate the picture by `shift` along the geodesic through "
                "the centre at `direction`, and turn it about the centre. The way to scroll a hyperkaleid() "
                "picture, or anything drawn in the disk or the band, without breaking it.",
            glsl=ASP + """
  vec2 z = _hyp_mobius(_hyp_to_disk(_st, model, asp), shift, direction, turn);
  return _hyp_from_disk(z, model, asp);"""))


def _np_hyperfade(st, fade, model, env):
    asp = _asp(env)
    x, y = _to_disk(st, float(model), asp)
    s = np.clip(1.0 - (x * x + y * y), 0.0, 1.0) / np.maximum(bc(fade, st), 1e-6)
    s = np.clip(s, 0.0, 1.0)
    a = (s * s * (3.0 - 2.0 * s)).astype(np.float32)
    return np.stack([a, a, a, a], axis=-1)


register(Op("hyperfade", "src",
            [F("fade", 0.2, doc="width of the fade, in 1 - |z|^2"), F("model", 0, scalar=True, doc="0 disk, 1 band")],
            np=_np_hyperfade, glsl_helpers=_HELPERS,
            doc="Grey: both RGB and alpha are 1 in the middle of the disk (or band) falling to 0 at its boundary "
                "over `fade`, measured as 1 - |z|^2, which is how fast the tiles shrink there. `.mult()` a hyperbolic "
                "picture by it so the sub-pixel tiles at the edge fade to black instead of shimmering. For a fade over "
                "another picture, `ground.layer(pic.mask(hyperfade()))`; it is 0 outside the disk.",
            glsl=ASP + """
  vec2 z = _hyp_to_disk(_st, model, asp);
  float s = clamp((1.0 - dot(z, z)) / max(fade, 1e-6), 0.0, 1.0);
  float a = s * s * (3.0 - 2.0 * s);
  return vec4(a, a, a, a);"""))
