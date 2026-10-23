"""Video synthesis and video decay.

Analog video-synth vocabulary (the LZX / Rutt-Etra family): a scan processor, a contour keyer,
a sine staircase shaper, bitwise logic between two signals, shear, and a GPU pixel sorter.
Then the medium: tape (chroma bleed, tracking jitter, head switching, dropouts, ringing) and
small-gauge film (film stocks, the projector gate).  ``recipes.vhs()`` and ``recipes.super8()``
put the second group together.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .noise import h11, h21, h22, snoise3
from .ops import F, I, TAU, bc, lum, mix, ones_like_hw, smoothstep, step

ASPECT_GLSL = "vec2(resolution.x / resolution.y, 1.0)"


# =============================================================================
# video synthesis
# =============================================================================

def _np_psort(tex, st, threshold, vertical, env):
    parity = env.pass_index % 2
    axis = np.array([0.0, 1.0], np.float32) if vertical > 0.5 else np.array([1.0, 0.0], np.float32)
    idx = np.floor((st * env.res * axis).sum(-1))
    even = np.mod(idx + parity, 2.0) < 0.5
    partner = st + np.where(even[..., None], axis, -axis) / env.res
    me = env.sample(tex, st)
    other = env.sample(tex, partner)
    lm, lo = lum(me[..., :3]), lum(other[..., :3])
    inside = (partner[..., 0] >= 0) & (partner[..., 0] <= 1) & (partner[..., 1] >= 0) & (partner[..., 1] <= 1)
    active = inside & (lm >= threshold) & (lo >= threshold)
    take_other = np.where(even, lm > lo, lo > lm) & active
    return np.where(take_other[..., None], other, me).astype(np.float32)


register(Op("psort", "texture", [F("threshold", 0.3, scalar=True), F("vertical", 0, scalar=True), I("iterations", 8)],
            np=_np_psort, iterations=True,
            doc="Pixel sorting on the GPU: each iteration is one odd-even transposition step along rows (vertical=1: columns) within runs brighter than `threshold`. Feed it back (src(o0).psort()) and it keeps sorting frame after frame.",
            glsl="""vec2 px = 1.0 / resolution;
  float parity = mod(_pass, 2.0);
  vec2 axis = vertical > 0.5 ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
  float idx = floor(dot(_st * resolution, axis));
  bool even = mod(idx + parity, 2.0) < 0.5;
  vec2 partner = _st + (even ? axis : -axis) * px;
  vec4 me = texture2D(_tex, _st);
  if (partner.x < 0.0 || partner.x > 1.0 || partner.y < 0.0 || partner.y > 1.0) return me;
  vec4 other = texture2D(_tex, partner);
  float lm = _luminance(me.rgb), lo = _luminance(other.rgb);
  if (lm < threshold || lo < threshold) return me;
  if (even) return (lm <= lo) ? me : other;
  return (lo <= lm) ? me : other;"""))


def _np_ruttetra(tex, st, lines, amount, thickness, glow, env):
    # Lines are displaced towards +y for amount > 0, so a pixel can only be covered by the line
    # of its own cell, those below it (walked until they are out of reach) and the one just above.
    best = np.zeros(st.shape[:2], np.float32)
    col = np.zeros(st.shape[:2] + (3,), np.float32)
    k0 = np.floor(st[..., 1] * lines)
    direction = 1.0 if amount >= 0 else -1.0
    margin = 2.5 * thickness / env.res[1]
    for i in range(64):
        k = k0 + direction * (1.0 - i)
        yk = (k + 0.5) / lines
        if i > 0:
            reach = np.abs(st[..., 1] - yk) <= abs(amount) + margin
            if not reach.any():
                break
        else:
            reach = True
        valid = (k >= 0) & (k < lines) & reach
        s = env.sample(tex, np.stack([st[..., 0], yk], axis=-1))
        l = lum(s[..., :3])
        d = (st[..., 1] - (yk + amount * l)) * env.res[1]
        w = np.exp(-d * d / (thickness * thickness)) * valid
        better = w > best
        best = np.where(better, w, best)
        col = np.where(better[..., None], s[..., :3], col)
    out = mix(col, 1.0, glow) * best[..., None]
    return np.concatenate([out, best[..., None]], axis=-1).astype(np.float32)


register(Op("ruttetra", "texture", [F("lines", 90, scalar=True), F("amount", 0.25, scalar=True), F("thickness", 1.2, scalar=True), F("glow", 0.3, scalar=True)],
            np=_np_ruttetra,
            doc="Rutt-Etra scan processor: `lines` horizontal scan lines, each displaced upward by `amount` times the brightness under it, drawn `thickness` pixels wide (alpha = line). Rotate/scale the chain before it to tilt the raster.",
            glsl="""float best = 0.0;
  vec3 col = vec3(0.0);
  float k0 = floor(_st.y * lines);
  float dir = amount >= 0.0 ? 1.0 : -1.0;
  float margin = 2.5 * thickness / resolution.y;
  for (int i = 0; i < 64; i++) {
    float k = k0 + dir * (1.0 - float(i));
    float yk = (k + 0.5) / lines;
    if (i > 0 && abs(_st.y - yk) > abs(amount) + margin) break;
    if (k < 0.0 || k >= lines) continue;
    vec4 s = texture2D(_tex, vec2(_st.x, yk));
    float l = _luminance(s.rgb);
    float d = (_st.y - (yk + amount * l)) * resolution.y;
    float w = exp(-d * d / (thickness * thickness));
    if (w > best) { best = w; col = s.rgb; }
  }
  return vec4(mix(col, vec3(1.0), glow) * best, best);"""))


def _np_contours(c, levels, width, fill, env):
    l = lum(c[..., :3])
    f = np.abs(np.mod(l * levels + 0.5, 1.0) - 0.5) * 2.0
    line = (1.0 - smoothstep(0.0, width, f))[..., None]
    return np.concatenate([mix(c[..., :3] * fill, 1.0, line), c[..., 3:]], axis=-1)


register(Op("contours", "color", [F("levels", 6), F("width", 0.15), F("fill", 0)], np=_np_contours,
            doc="Topographic keyer: white isolines where luminance crosses `levels` thresholds; `fill` keeps that much of the image between them.",
            glsl="""float l = _luminance(_c0.rgb);
  float f = abs(fract(l * levels + 0.5) - 0.5) * 2.0;
  float line = 1.0 - smoothstep(0.0, width, f);
  return vec4(mix(_c0.rgb * fill, vec3(1.0), line), _c0.a);"""))


def _np_sinefold(c, stages, phase, env):
    st_ = bc(stages, c)[..., None]
    ph = bc(phase, c)[..., None]
    return np.concatenate([0.5 + 0.5 * np.sin((c[..., :3] * st_ + ph) * TAU), c[..., 3:]], axis=-1)


register(Op("sinefold", "color", [F("stages", 2), F("phase", 0)], np=_np_sinefold,
            doc="Sine staircase shaper: each channel through `stages` cycles of a sine; ramps become rainbows, gradients become bands.",
            glsl="return vec4(0.5 + 0.5 * sin((_c0.rgb * stages + phase) * 6.283185307), _c0.a);"))


def _np_bits(c0, c1, mode, depth, env):
    n = 2 ** int(depth) - 1
    a = np.floor(np.clip(c0[..., :3], 0, 1) * n + 0.5).astype(np.int64)
    b = np.floor(np.clip(c1[..., :3], 0, 1) * n + 0.5).astype(np.int64)
    m = bc(mode, c0)[..., None]
    r = np.where(m < 0.5, a ^ b, np.where(m < 1.5, a & b, a | b))
    return np.concatenate([(r / n).astype(np.float32), c0[..., 3:]], axis=-1)


register(Op("bits", "combine", [Input("tex", "chain"), F("mode", 0), I("depth", 4)], np=_np_bits,
            doc="Bitwise logic between two chains quantised to `depth` bits: mode 0 XOR, 1 AND, 2 OR. Two ramps XORed are the oldest video-synth texture there is.",
            glsl="""float n = pow(2.0, float(depth)) - 1.0;
  vec3 a = floor(clamp(_c0.rgb, 0.0, 1.0) * n + 0.5);
  vec3 b = floor(clamp(_c1.rgb, 0.0, 1.0) * n + 0.5);
  vec3 r = vec3(0.0);
  float w = 1.0;
  for (int i = 0; i < 8; i++) {
    if (i >= depth) break;
    vec3 ba = mod(floor(a / w), 2.0);
    vec3 bb = mod(floor(b / w), 2.0);
    vec3 bit = mode < 0.5 ? abs(ba - bb) : (mode < 1.5 ? ba * bb : max(ba, bb));
    r += bit * w;
    w *= 2.0;
  }
  return vec4(r / n, _c0.a);"""))


def _np_shear(st, x, y, env):
    ux = st[..., 0] + (st[..., 1] - 0.5) * x
    uy = st[..., 1] + (st[..., 0] - 0.5) * y
    return np.stack([ux, uy], axis=-1)


register(Op("shear", "coord", [F("x", 0.3), F("y", 0)], np=_np_shear, doc="Skew: x shifts rows by their height, y shifts columns by their width.",
            glsl="return vec2(_st.x + (_st.y - 0.5) * x, _st.y + (_st.x - 0.5) * y);"))


# =============================================================================
# tape
# =============================================================================

_TO_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
_TO_RGB = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]], np.float32)


def _np_yiq(tex, st, chroma, luma, shift, env):
    px = np.array([1.0 / env.res[0], 0.0], np.float32)
    c = env.sample(tex, st)[..., :3] @ _TO_YIQ.T
    Y = c[..., 0]
    if luma > 0:
        Y = Y * 3.0
        for k, w in ((-2, 1.0), (-1, 2.0), (1, 2.0), (2, 1.0)):
            Y = Y + w * (env.sample(tex, st + px * (luma * 0.5 * k))[..., :3] @ _TO_YIQ.T)[..., 0]
        Y = Y / 9.0
    iq = np.zeros(st.shape[:2] + (2,), np.float32)
    wsum = 0.0
    for i in range(-8, 9):
        w = 1.0 - abs(i) / 9.0
        s = env.sample(tex, st + np.array([i / 8.0 * chroma - shift, 0.0], np.float32))[..., :3] @ _TO_YIQ.T
        iq += s[..., 1:] * w
        wsum += w
    iq /= wsum
    rgb = np.concatenate([Y[..., None], iq], axis=-1) @ _TO_RGB.T
    return np.concatenate([rgb, ones_like_hw(st)[..., None]], axis=-1).astype(np.float32)


register(Op("yiq", "texture", [F("chroma", 0.03, scalar=True), F("luma", 1.0, scalar=True), F("shift", 0.004, scalar=True)],
            np=_np_yiq,
            doc="Composite-video bandwidth: chroma (I/Q) smeared horizontally over `chroma` of the width and delayed by `shift`, luma softened over `luma` pixels. The colour bleed of tape.",
            glsl="""const mat3 toYIQ = mat3(0.299, 0.596, 0.211,  0.587, -0.274, -0.523,  0.114, -0.322, 0.312);
  const mat3 toRGB = mat3(1.0, 1.0, 1.0,  0.956, -0.272, -1.106,  0.621, -0.647, 1.703);
  vec2 px = vec2(1.0 / resolution.x, 0.0);
  vec3 c = toYIQ * texture2D(_tex, fract(_st)).rgb;
  float Y = c.x;
  if (luma > 0.0) {
    Y = c.x * 3.0;
    for (int k = -2; k <= 2; k++) {
      if (k == 0) continue;
      Y += (3.0 - abs(float(k))) * (toYIQ * texture2D(_tex, fract(_st + px * (luma * 0.5 * float(k)))).rgb).x;
    }
    Y /= 9.0;
  }
  vec2 iq = vec2(0.0); float wsum = 0.0;
  for (int i = -8; i <= 8; i++) {
    float w = 1.0 - abs(float(i)) / 9.0;
    vec3 s = toYIQ * texture2D(_tex, fract(_st + vec2(float(i) / 8.0 * chroma - shift, 0.0))).rgb;
    iq += s.yz * w; wsum += w;
  }
  iq /= wsum;
  return vec4(toRGB * vec3(Y, iq), 1.0);"""))


def _np_linejitter(st, amount, rate, wobble, seed, env):
    line = np.floor(st[..., 1] * env.res[1])
    frame = np.floor(env.time * rate)
    j = (h21(np.stack([line, bc(frame + seed * 131.0, st)], axis=-1)) - 0.5) * 2.0
    tracking = 0.5 + 0.5 * snoise3(np.stack([line * 0.02, bc(env.time * 0.7, st), bc(seed, st)], axis=-1))
    w = np.sin(st[..., 1] * 25.0 + env.time * 3.1) * wobble
    return np.stack([st[..., 0] + j * amount * tracking * tracking + w, st[..., 1]], axis=-1)


register(Op("linejitter", "coord", [F("amount", 0.004), F("rate", 60), F("wobble", 0.002), F("seed", 0)], np=_np_linejitter,
            doc="Tracking error: every scan line shifts sideways by its own random amount, worse in drifting bands, plus a slow time-base wobble.",
            glsl="""float line = floor(_st.y * resolution.y);
  float frame = floor(time * rate);
  float j = (_h21(vec2(line, frame + seed * 131.0)) - 0.5) * 2.0;
  float tracking = 0.5 + 0.5 * _noise(vec3(line * 0.02, time * 0.7, seed));
  float w = sin(_st.y * 25.0 + time * 3.1) * wobble;
  return vec2(_st.x + j * amount * tracking * tracking + w, _st.y);"""))


def _np_headswitch(st, height, amount, rate, seed, env):
    frame = np.floor(env.time * rate)
    f = np.clip(1.0 - st[..., 1] / np.maximum(height, 1e-6), 0.0, 1.0)
    j = (h11(bc(frame + seed * 131.0, st)) - 0.5) * 2.0
    n = (h21(np.stack([np.floor(st[..., 1] * env.res[1]), bc(frame, st)], axis=-1)) - 0.5) * 0.5
    x = np.where(st[..., 1] > height, st[..., 0], st[..., 0] + (j + n) * amount * f * f)
    return np.stack([x, st[..., 1]], axis=-1)


register(Op("headswitch", "coord", [F("height", 0.06), F("amount", 0.08), F("rate", 60), F("seed", 0)], np=_np_headswitch,
            doc="Head-switching noise: the bottom `height` of the frame tears sideways, more towards the very bottom, re-rolled `rate` times a second.",
            glsl="""if (_st.y > height) return _st;
  float frame = floor(time * rate);
  float f = 1.0 - _st.y / height;
  float j = (_h11(frame + seed * 131.0) - 0.5) * 2.0;
  float n = (_h21(vec2(floor(_st.y * resolution.y), frame)) - 0.5) * 0.5;
  return vec2(_st.x + (j + n) * amount * f * f, _st.y);"""))


def _np_dropouts(st, density, extent, rate, seed, env):
    line = np.floor(st[..., 1] * env.res[1])
    frame = bc(np.floor(env.time * rate), st)
    h = h21(np.stack([line, frame + seed * 131.0], axis=-1))
    on = h <= density * 0.05
    r = h22(np.stack([line * 1.3 + 7.0, frame + seed * 131.0], axis=-1))
    x0 = r[..., 0]
    ln = extent * (0.2 + 0.8 * r[..., 1])
    inX = step(x0, st[..., 0]) * step(st[..., 0], x0 + ln)
    v = inX * (0.6 + 0.4 * h11(line + frame)) * on
    return np.concatenate([np.stack([v, v, v], axis=-1), v[..., None]], axis=-1).astype(np.float32)


register(Op("dropouts", "src", [F("density", 0.5), F("extent", 0.3), F("rate", 60), F("seed", 0)], np=_np_dropouts,
            doc="Tape dropouts: short bright streaks along random scan lines, new ones `rate` times a second (white on black, alpha = streak). lighten() them onto the picture.",
            glsl="""float line = floor(_st.y * resolution.y);
  float frame = floor(time * rate);
  float h = _h21(vec2(line, frame + seed * 131.0));
  if (h > density * 0.05) return vec4(0.0);
  vec2 r = _h22(vec2(line * 1.3 + 7.0, frame + seed * 131.0));
  float x0 = r.x;
  float len = extent * (0.2 + 0.8 * r.y);
  float inX = step(x0, _st.x) * step(_st.x, x0 + len);
  float v = inX * (0.6 + 0.4 * _h11(line + frame));
  return vec4(vec3(v), v);"""))


def _np_ringing(tex, st, amount, offset, env):
    px = np.array([offset / env.res[0], 0.0], np.float32)
    c = env.sample(tex, st)
    l1 = env.sample(tex, st - px)
    l2 = env.sample(tex, st - px * 2.0)
    rgb = c[..., :3] + amount * (c[..., :3] - l1[..., :3]) - amount * 0.5 * (l1[..., :3] - l2[..., :3])
    return np.concatenate([rgb, c[..., 3:]], axis=-1).astype(np.float32)


register(Op("ringing", "texture", [F("amount", 0.5, scalar=True), F("offset", 2, scalar=True)], np=_np_ringing,
            doc="Edge ringing: overshoot and a second ripple to the right of every vertical edge, `offset` pixels apart. The sharpening of a worn tape deck.",
            glsl="""vec2 px = vec2(offset / resolution.x, 0.0);
  vec4 c = texture2D(_tex, fract(_st));
  vec4 l = texture2D(_tex, fract(_st - px));
  vec4 l2 = texture2D(_tex, fract(_st - px * 2.0));
  return vec4(c.rgb + amount * (c.rgb - l.rgb) - amount * 0.5 * (l.rgb - l2.rgb), c.a);"""))


# =============================================================================
# film
# =============================================================================

_STOCKS = {
    0: (np.array([0.0, 0.0, 0.0]), np.array([1.08, 1.0, 0.95]), np.array([1.2, 1.1, 1.3]), 1.3),     # kodachrome
    1: (np.array([0.0, 0.01, 0.03]), np.array([0.98, 1.0, 1.08]), np.array([1.0, 1.0, 0.95]), 1.1),  # ektachrome
    2: (np.array([0.08, 0.07, 0.05]), np.array([0.9, 0.86, 0.78]), np.array([0.9, 0.95, 1.0]), 0.65),  # faded
    3: (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.0),       # tri-x
}


def _np_filmstock(c, stock, amount, env):
    lift, gain, gam, sat = _STOCKS.get(int(stock), _STOCKS[0])
    x = np.clip(c[..., :3], 0, 1)
    x = np.power(np.clip(x * gain.astype(np.float32) + lift.astype(np.float32), 0, 1), gam.astype(np.float32))
    if int(stock) == 3:
        l = smoothstep(0.06, 0.94, lum(x))
        x = np.repeat(l[..., None], 3, -1)
    L = lum(x)[..., None]
    x = mix(np.repeat(L, 3, -1), x, sat)
    am = bc(amount, c)[..., None]
    return np.concatenate([mix(c[..., :3], x, am), c[..., 3:]], axis=-1).astype(np.float32)


register(Op("filmstock", "color", [I("stock", 0), F("amount", 1)], np=_np_filmstock,
            doc="Film stock tone: 0 Kodachrome (warm, dense), 1 Ektachrome (cool shadows), 2 faded (lifted blacks, yellowed), 3 Tri-X (B&W S-curve).",
            glsl="""vec3 lift = vec3(0.0), gain = vec3(1.0), gam = vec3(1.0); float sat = 1.0;
  if (stock == 0) { gain = vec3(1.08, 1.0, 0.95); gam = vec3(1.2, 1.1, 1.3); sat = 1.3; }
  else if (stock == 1) { lift = vec3(0.0, 0.01, 0.03); gain = vec3(0.98, 1.0, 1.08); gam = vec3(1.0, 1.0, 0.95); sat = 1.1; }
  else if (stock == 2) { lift = vec3(0.08, 0.07, 0.05); gain = vec3(0.9, 0.86, 0.78); gam = vec3(0.9, 0.95, 1.0); sat = 0.65; }
  else { sat = 0.0; }
  vec3 c = clamp(_c0.rgb, 0.0, 1.0);
  c = pow(clamp(c * gain + lift, 0.0, 1.0), gam);
  if (stock == 3) { float l = smoothstep(0.06, 0.94, _luminance(c)); c = vec3(l); }
  float L = _luminance(c);
  c = mix(vec3(L), c, sat);
  return vec4(mix(_c0.rgb, c, amount), _c0.a);"""))


def _np_gate(st, width, height, radius, softness, env):
    asp = np.array([env.res[0] / env.res[1], 1.0], np.float32)
    p = np.abs(st - 0.5) * asp
    radius = bc(radius, st)
    q = np.stack([p[..., 0] - (bc(width, st) * asp[0] * 0.5 - radius), p[..., 1] - (bc(height, st) * 0.5 - radius)], axis=-1)
    d = np.sqrt((np.maximum(q, 0.0) ** 2).sum(-1)) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0) - radius
    softness = np.maximum(softness, 1e-4)
    m = 1.0 - smoothstep(-softness, softness, d)
    return np.concatenate([np.stack([m, m, m], axis=-1), m[..., None]], axis=-1).astype(np.float32)


register(Op("gate", "src", [F("width", 0.96), F("height", 0.94), F("radius", 0.06), F("softness", 0.01)], np=_np_gate,
            doc="The projector gate: a rounded-rectangle mask (white inside, alpha = inside). mult() the picture by it; pan it a little for a frame that doesn't sit still.",
            glsl="""vec2 asp = """ + ASPECT_GLSL + """;
  vec2 p = abs(_st - 0.5) * asp;
  vec2 half_ = vec2(width * asp.x, height) * 0.5 - radius;
  vec2 q = p - half_;
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
  float s = max(softness, 0.0001);
  float m = 1.0 - smoothstep(-s, s, d);
  return vec4(vec3(m), m);"""))
