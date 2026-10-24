"""Strokes, brushes and print: the hand-made side of abstract animation.

Fischinger's Studie films are oscillators drawn in charcoal — comets, sine sweeps, arcs, pulsing
to the music — so here the oscillator is a *stroke*: ``harmonograph`` (Lissajous and damped
pendulum curves), ``wave`` (a sine drawn as a line), ``arc``; ``charcoal`` gives any of them paper
tooth and ragged edges, ``smear`` drags paint along a direction field (paint-on-glass, smudged
charcoal), ``dab`` is a bristle brush for accretive painting (``src(o0).layer(dab(...))`` adds one
touch a frame: Motion Painting No. 1).

For the flat-graphic end — the Zagreb school, Soyuzmultfilm's sixties, Polish poster colour —
``halftone`` is a three-angle screen print, ``boil`` the hand-drawn jitter of a line redrawn every
frame, ``polarRepeat`` rotational copies around the centre (ornament), and ``set_frame_rate(12)``
puts the whole patch on twos.  ``filmstock(4)`` is the muted Orwo / Sovcolor look.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Op, register
from .noise import h11, h21, h22, segd, snoise3
from .ops import F, I, TAU, bc, mix, rotate_xy, smoothstep

ASP = "vec2 asp = vec2(resolution.x / resolution.y, 1.0);"


def _asp(env):
    return np.array([env.res[0] / env.res[1], 1.0], np.float32)


def _stroke_grid(q):
    """Recognise rectilinear coordinates once, including scaled/pixelated grids."""
    xs, ys = q[0, :, 0], q[:, 0, 1]
    if (np.all(np.diff(xs) >= 0) and np.all(np.diff(ys) >= 0)
            and np.all(q[..., 0] == xs[None, :]) and np.all(q[..., 1] == ys[:, None])):
        return xs, ys
    return None


def _stroke_segment(best, q, a, b, width, amplitude, grid):
    # Beyond eight widths exp(-2*(d/width)^2) underflows to zero in
    # float32. This is a numerical bound, not a visible-tail quality setting.
    pad = abs(width) * 8
    if not np.isfinite(pad) or width == 0:
        d = segd(q, a, b)
        np.maximum(best, np.exp(-(d / width) ** 2 * 2) * amplitude, out=best)
        return
    lo, hi = np.minimum(a, b) - pad, np.maximum(a, b) + pad
    if grid is not None:
        xs, ys = grid
        x0, x1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0], side="right")
        y0, y1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1], side="right")
        if x0 == x1 or y0 == y1:
            return
        region = best[y0:y1, x0:x1]
        d = segd(q[y0:y1, x0:x1], a, b)
        np.maximum(region, np.exp(-(d / width) ** 2 * 2) * amplitude, out=region)
    else:
        inside = ((q[..., 0] >= lo[0]) & (q[..., 0] <= hi[0])
                  & (q[..., 1] >= lo[1]) & (q[..., 1] <= hi[1]))
        if np.any(inside):
            d = segd(q[inside], a, b)
            best[inside] = np.maximum(best[inside], np.exp(-(d / width) ** 2 * 2) * amplitude)


def _stroke_alpha(q, pts, width, taper, head_at_end=True):
    """Max over segments of a gaussian stroke profile; width/alpha taper towards the tail."""
    K = len(pts) - 1
    best = np.zeros(q.shape[:2], np.float32)
    grid = _stroke_grid(q)
    for i in range(K):
        u = (i + 0.5) / K
        tail = (1.0 - u) if head_at_end else u
        w = width * (1.0 - taper * tail * 0.85)
        _stroke_segment(best, q, pts[i], pts[i + 1], w, 1.0 - taper * tail * 0.7, grid)
    return best.astype(np.float32)


def _white(alpha):
    # Straight alpha: coverage belongs in alpha, not in the colour as well.
    one = np.ones_like(alpha)
    return np.stack([one, one, one, alpha], axis=-1).astype(np.float32)


# -----------------------------------------------------------------------------
# strokes
# -----------------------------------------------------------------------------

_STROKE_GLSL = """    float tail = 1.0 - u;
    float w = width * (1.0 - taper * tail * 0.85);
    float d = _segd(q, prev, p);
    float a = exp(-(d / w) * (d / w) * 2.0) * (1.0 - taper * tail * 0.7);
    best = max(best, a);
    prev = p;"""


def _harmonograph_segments(f1, f2, f3, f4, length) -> int:
    return int(min(256.0, max(16.0, math.ceil(length * max(f1, f2, f3, f4) * 24.0))))


def _np_harmonograph(st, f1, f2, f3, f4, phase, decay, length, size, width, taper, speed, env):
    K = _harmonograph_segments(f1, f2, f3, f4, length)
    q = (st - 0.5) * _asp(env)
    ph = phase + env.time * speed
    pts = []
    for i in range(K + 1):
        t = i / K
        s = t * length * TAU
        e = math.exp(-decay * t * length)
        x = math.sin(f1 * s + ph)
        if f3 > 0:
            x = 0.5 * (x + math.sin(f3 * s))
        y = math.sin(f2 * s)
        if f4 > 0:
            y = 0.5 * (y + math.sin(f4 * s))
        pts.append((size * e * x, size * e * y))
    return _white(_stroke_alpha(q, pts, width, taper))


register(Op("harmonograph", "src",
            [F("f1", 3, scalar=True), F("f2", 2, scalar=True), F("f3", 0, scalar=True), F("f4", 0, scalar=True),
             F("phase", 1.5708, scalar=True), F("decay", 0, scalar=True), F("length", 1, scalar=True),
             F("size", 0.4, scalar=True), F("width", 0.012, scalar=True), F("taper", 0, scalar=True), F("speed", 0.1, scalar=True)],
            np=_np_harmonograph,
            doc="A Lissajous / harmonograph curve drawn as a stroke (white, alpha = stroke): x = sin(f1 t + phase), y = sin(f2 t); f3/f4 > 0 add a second pendulum per axis, `decay` damps it over `length` turns, `speed` turns the phase. `taper` thins the tail; the head is the end of the curve. Keep length x frequency under ~10 for a smooth curve (256 segments).",
            glsl=ASP + """
  vec2 q = (_st - 0.5) * asp;
  float ph = phase + time * speed;
  float best = 0.0;
  float nx = f3 > 0.0 ? 0.5 : 1.0, ny = f4 > 0.0 ? 0.5 : 1.0;
  vec2 prev = size * vec2(nx * sin(ph), 0.0);
  float n = min(256.0, max(16.0, ceil(length * max(max(f1, f2), max(f3, f4)) * 24.0)));
  for (int i = 1; i <= 256; i++) {
    if (float(i) > n) break;
    float t = float(i) / n;
    float s = t * length * 6.283185307;
    float e = exp(-decay * t * length);
    float x = nx * (sin(f1 * s + ph) + (f3 > 0.0 ? sin(f3 * s) : 0.0));
    float y = ny * (sin(f2 * s) + (f4 > 0.0 ? sin(f4 * s) : 0.0));
    vec2 p = size * e * vec2(x, y);
    float u = (float(i) - 0.5) / n;
""" + _STROKE_GLSL + """
  }
  return vec4(vec3(1.0), best);"""))


def _wave_segments(frequency, length) -> int:
    return int(min(192.0, max(16.0, math.ceil(abs(frequency) * length * 24.0))))


def _np_wave(st, frequency, amplitude, width, speed, phase, length, taper, env):
    K = _wave_segments(frequency, length)
    asp = _asp(env)
    q = (st - 0.5) * asp
    L = length * asp[0]
    pts = []
    for i in range(K + 1):
        u = i / K
        x = (u - 0.5) * L
        y = amplitude * math.sin(TAU * (frequency * (x / asp[0] + 0.5) + phase + env.time * speed))
        pts.append((x, y))
    # both ends taper, so the window is symmetric: fold the segment index towards the middle
    best = np.zeros(st.shape[:2], np.float32)
    grid = _stroke_grid(q)
    for i in range(K):
        u = (i + 0.5) / K
        edge = abs(2.0 * u - 1.0)
        w = width * (1.0 - taper * edge * 0.85)
        _stroke_segment(best, q, pts[i], pts[i + 1], w, 1.0 - taper * edge * 0.7, grid)
    return _white(best)


register(Op("wave", "src",
            [F("frequency", 3, scalar=True), F("amplitude", 0.15, scalar=True), F("width", 0.012, scalar=True),
             F("speed", 1, scalar=True), F("phase", 0, scalar=True), F("length", 1, scalar=True), F("taper", 0, scalar=True)],
            np=_np_wave,
            doc="An oscillator drawn as a stroke: a sine across the frame, `frequency` cycles per width, travelling at `speed`. `length` is the stroke's extent (fraction of the width); `taper` thins both ends.",
            glsl=ASP + """
  vec2 q = (_st - 0.5) * asp;
  float L = length * asp.x;
  float best = 0.0;
  vec2 prev = vec2(-0.5 * L, amplitude * sin(6.283185307 * (frequency * (-0.5 * L / asp.x + 0.5) + phase + time * speed)));
  float n = min(192.0, max(16.0, ceil(abs(frequency) * length * 24.0)));
  for (int i = 1; i <= 192; i++) {
    if (float(i) > n) break;
    float u = float(i) / n;
    float x = (u - 0.5) * L;
    vec2 p = vec2(x, amplitude * sin(6.283185307 * (frequency * (x / asp.x + 0.5) + phase + time * speed)));
    float um = (float(i) - 0.5) / n;
    float edge = abs(2.0 * um - 1.0);
    float w = width * (1.0 - taper * edge * 0.85);
    float d = _segd(q, prev, p);
    float a = exp(-(d / w) * (d / w) * 2.0) * (1.0 - taper * edge * 0.7);
    best = max(best, a);
    prev = p;
  }
  return vec4(vec3(1.0), best);"""))


def _arc_segments(sweep) -> int:
    return int(min(96.0, max(6.0, math.ceil(abs(sweep) * 12.0))))


def _np_arc(st, radius, start, sweep, width, taper, speed, env):
    K = _arc_segments(sweep)
    q = (st - 0.5) * _asp(env)
    a0 = start + env.time * speed
    pts = [(radius * math.cos(a0 + sweep * i / K), radius * math.sin(a0 + sweep * i / K)) for i in range(K + 1)]
    return _white(_stroke_alpha(q, pts, width, taper))


register(Op("arc", "src",
            [F("radius", 0.3, scalar=True), F("start", 0, scalar=True), F("sweep", 3.1416, scalar=True),
             F("width", 0.012, scalar=True), F("taper", 0, scalar=True), F("speed", 0, scalar=True)],
            np=_np_arc,
            doc="An arc stroke about the centre from angle `start` through `sweep` radians (a full circle: 6.2832), turning at `speed`. `taper` thins the tail; the head is the end of the arc.",
            glsl=ASP + """
  vec2 q = (_st - 0.5) * asp;
  float a0 = start + time * speed;
  float best = 0.0;
  vec2 prev = radius * vec2(cos(a0), sin(a0));
  float n = min(96.0, max(6.0, ceil(abs(sweep) * 12.0)));
  for (int i = 1; i <= 96; i++) {
    if (float(i) > n) break;
    float ang = a0 + sweep * float(i) / n;
    vec2 p = radius * vec2(cos(ang), sin(ang));
    float u = (float(i) - 0.5) / n;
""" + _STROKE_GLSL + """
  }
  return vec4(vec3(1.0), best);"""))


# -----------------------------------------------------------------------------
# brush, paper, paint
# -----------------------------------------------------------------------------

def _np_dab(st, size, aspect, angle, bristles, softness, seed, env):
    q = (st - 0.5) * _asp(env)
    q = rotate_xy(q, bc(angle, st))                     # rotate_xy turns by -angle like the GLSL mat2 below
    e = np.stack([q[..., 0] / bc(size, st), q[..., 1] / (bc(size, st) * bc(aspect, st))], axis=-1)
    d = np.sqrt((e * e).sum(-1))
    base = 1.0 - smoothstep(1.0 - bc(softness, st), 1.0, d)
    row = np.floor((e[..., 1] + 1.0) * 12.0) + bc(seed, st) * 13.0
    b = h11(row * 1.7 + bc(seed, st) * 131.0)
    dry = h21(np.stack([np.floor(e[..., 0] * 40.0), row], axis=-1))
    a = base * mix(1.0, 0.3 + 0.7 * b, bc(bristles, st)) * mix(1.0, 0.6 + 0.4 * dry, bc(bristles, st) * 0.5)
    return _white(a)


register(Op("dab", "src", [F("size", 0.15), F("aspect", 0.5), F("angle", 0), F("bristles", 0.5), F("softness", 0.3), F("seed", 0)],
            np=_np_dab,
            doc="One touch of a bristle brush: an ellipse `size` tall (fraction of the height) and `aspect` as wide, turned by `angle`, streaked along its length by `bristles`. White, alpha = paint. Feed it back for accretive painting: src(o0).layer(dab(...).color(...)).",
            glsl=ASP + """
  vec2 q = (_st - 0.5) * asp;
  float c = cos(angle), s = sin(angle);
  q = mat2(c, -s, s, c) * q;
  vec2 e = q / vec2(size, size * aspect);
  float d = length(e);
  float base = 1.0 - smoothstep(1.0 - softness, 1.0, d);
  float row = floor((e.y + 1.0) * 12.0) + seed * 13.0;
  float b = _h11(row * 1.7 + seed * 131.0);
  float dry = _h21(vec2(floor(e.x * 40.0), row));
  float a = base * mix(1.0, 0.3 + 0.7 * b, bristles) * mix(1.0, 0.6 + 0.4 * dry, bristles * 0.5);
  return vec4(vec3(1.0), a);"""))


def _np_charcoal(tex, st, tooth, rough, scale, seed, env):
    block = np.floor(st * env.res / 2.0)
    j = (h22(block + seed * 131.0) - 0.5) * rough / env.res
    c = env.sample(tex, st + j)
    asp = _asp(env)
    n = 0.5 + 0.5 * snoise3(np.stack([st[..., 0] * scale * asp[0], st[..., 1] * scale, np.full(st.shape[:2], seed * 7.1 + 0.37, np.float32)], axis=-1))
    n2 = h21(np.floor(st * env.res) + seed)
    t = mix(1.0, n * (0.7 + 0.3 * n2), tooth * (1.0 - 0.6 * np.clip(c[..., 3], 0.0, 1.0)))
    c = c.copy()
    c[..., 3] *= t
    return c.astype(np.float32)


register(Op("charcoal", "texture", [F("tooth", 0.5, scalar=True), F("rough", 1.5, scalar=True), F("scale", 90, scalar=True), F("seed", 0, scalar=True)],
            np=_np_charcoal,
            doc="Paper tooth removes coverage from alpha while preserving straight RGB colour. Edges jitter by `rough` pixels; grain at `scale` stays fixed in image space. Composite over paper with `over` or `recipes.on_paper`.",
            glsl=ASP + """
  vec2 block = floor(_st * resolution / 2.0);
  vec2 j = (_h22(block + seed * 131.0) - 0.5) * rough / resolution;
  vec4 c = texture2D(_tex, fract(_st + j));
  float n = 0.5 + 0.5 * _noise(vec3(_st * scale * asp, seed * 7.1 + 0.37));
  float n2 = _h21(floor(_st * resolution) + seed);
  float t = mix(1.0, n * (0.7 + 0.3 * n2), tooth * (1.0 - 0.6 * clamp(c.a, 0.0, 1.0)));
  return vec4(c.rgb, c.a * t);"""))


def _np_smear(tex, st, angle, length, env):
    ang = bc(angle, st)
    asp = _asp(env)
    d = np.stack([np.cos(ang) / asp[0], np.sin(ang)], axis=-1) * bc(length, st)[..., None]
    acc = None
    wsum = 0.0
    for i in range(12):
        w = 1.0 - i / 12.0
        s = env.sample(tex, st - d * (i / 11.0))
        acc = s * w if acc is None else acc + s * w
        wsum += w
    return (acc / wsum).astype(np.float32)


register(Op("smear", "texture", [F("angle", 0), F("length", 0.02)], np=_np_smear,
            doc="Paint dragged: 12 taps behind each pixel along `angle` (radians; a chain makes it a flow field), `length` of the height. Paint on glass, smudged charcoal, motion blur.",
            glsl=ASP + """
  vec2 d = vec2(cos(angle) / asp.x, sin(angle)) * length;
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int i = 0; i < 12; i++) {
    float w = 1.0 - float(i) / 12.0;
    acc += w * texture2D(_tex, fract(_st - d * (float(i) / 11.0)));
    wsum += w;
  }
  return acc / wsum;"""))


# -----------------------------------------------------------------------------
# print and limited animation
# -----------------------------------------------------------------------------

def _np_halftone(c, size, angle, sharpness, env):
    out = []
    soft = mix(0.6, 0.05, bc(sharpness, c))
    for k, off in enumerate((0.0, 1.0472, 0.2618)):
        th = bc(angle, c) + off
        q = rotate_xy(env.px, th) / bc(size, c)[..., None]
        cell = (q - np.floor(q)) - 0.5
        d = np.sqrt((cell * cell).sum(-1)) * 2.0
        v = np.clip(c[..., k], 0.0, 1.0)
        ink = 1.0 - v
        r = 2.0 * np.sqrt(ink / math.pi) * mix(1.0, 1.254, ink)
        out.append(np.where(r <= 0.0, 1.0, smoothstep(r - soft, r + soft, d)))
    return np.concatenate([np.stack(out, axis=-1), c[..., 3:]], axis=-1).astype(np.float32)


register(Op("halftone", "color", [F("size", 6), F("angle", 0.7854), F("sharpness", 0.7)], np=_np_halftone,
            doc="Screen print: each channel becomes a grid of ink dots `size` pixels apart, sized by how dark the channel is, the three screens turned 0, 60 and 15 degrees from `angle` so they moiré like a cheap colour print.",
            glsl="""float soft = mix(0.6, 0.05, sharpness);
  vec3 o = vec3(0.0);
  for (int k = 0; k < 3; k++) {
    float th = angle + (k == 0 ? 0.0 : (k == 1 ? 1.0472 : 0.2618));
    float cs = cos(th), sn = sin(th);
    vec2 q = mat2(cs, -sn, sn, cs) * gl_FragCoord.xy / size;
    vec2 cell = fract(q) - 0.5;
    float d = length(cell) * 2.0;
    float v = clamp(k == 0 ? _c0.r : (k == 1 ? _c0.g : _c0.b), 0.0, 1.0);
    float ink = 1.0 - v;
    float r = 2.0 * sqrt(ink / 3.14159265) * mix(1.0, 1.254, ink);
    float dot_ = r <= 0.0 ? 1.0 : smoothstep(r - soft, r + soft, d);
    if (k == 0) o.r = dot_; else if (k == 1) o.g = dot_; else o.b = dot_;
  }
  return vec4(o, _c0.a);"""))


def _np_boil(st, amount, rate, scale, seed, env):
    f = np.floor(env.time * rate) * 7.3 + seed * 131.0
    z = bc(f + 0.5, st)
    nx = snoise3(np.stack([st[..., 0] * scale, st[..., 1] * scale, z], axis=-1))
    ny = snoise3(np.stack([st[..., 0] * scale + 31.7, st[..., 1] * scale + 31.7, z], axis=-1))
    return st + np.stack([nx, ny], axis=-1) * bc(amount, st)[..., None]


register(Op("boil", "coord", [F("amount", 0.004), F("rate", 12), F("scale", 6), F("seed", 0)], np=_np_boil,
            doc="Line boil: the drawing redrawn by hand `rate` times a second — coordinates wander by up to `amount` with smooth noise at `scale`, re-rolled each redraw. Pair it with set_frame_rate(12) for animation on twos.",
            glsl="""float f = floor(time * rate) * 7.3 + seed * 131.0;
  vec2 n = vec2(_noise(vec3(_st * scale, f + 0.5)), _noise(vec3(_st * scale + 31.7, f + 0.5)));
  return _st + n * amount;"""))


def _np_polar_repeat(st, n, offset, aspect, env):
    asp = np.stack([1.0 + (env.res[0] / env.res[1] - 1.0) * bc(aspect, st), np.ones(st.shape[:2])], axis=-1)
    q = (st - 0.5) * asp
    r = np.sqrt((q * q).sum(-1))
    a = np.arctan2(q[..., 1], q[..., 0])
    wedge = TAU / bc(n, st)
    a = np.mod(a + bc(offset, st), wedge)
    return 0.5 + r[..., None] * np.stack([np.cos(a), np.sin(a)], axis=-1) / asp


register(Op("polarRepeat", "coord", [F("n", 5), F("offset", 0), F("aspect", 1)], np=_np_polar_repeat,
            doc="`n` rotational copies in image space. `aspect=0` preserves the old normalized-coordinate distortion; default 1 keeps circular orbits on rectangular frames.",
            glsl="""vec2 asp = vec2(mix(1.0, resolution.x / resolution.y, aspect), 1.0);
  vec2 q = (_st - 0.5) * asp;
  float r = length(q);
  float a = atan(q.y, q.x);
  float wedge = 6.283185307 / n;
  a = mod(a + offset, wedge);
  return 0.5 + r * vec2(cos(a), sin(a)) / asp;"""))
