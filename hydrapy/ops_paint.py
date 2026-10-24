"""Paint-on-film ops: dye compositing, drops, crackle, veins, scratches, dust, ink bleed, strip pan.

The model behind them: a projector shines white light through a strip of clear
leader that has been painted with transparent dyes.  Each dye layer multiplies
the light by its transmittance (Beer–Lambert), so colours mix the way inks do,
not the way RGB lights do, and thick or doubled paint goes black.  Drops dry
with dark rims; dried paint crackles; the print gets scratched and dusty.

A *layer* is a chain whose rgb is the colour the dye lets through and whose
alpha is its thickness (straight alpha, not premultiplied):

    leader = solid(1, 1, 1)
    leader.stain(drops(4, seed=SH(24, 0, 99))).stain(drops(7, hue=0.6, seed=2)).out(o0)

Aspect ratio is corrected in these sources (drops are round on a 16:9 frame),
unlike Hydra's own sources.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .noise import h11, h21, h22, hsv_to_rgb, snoise3
from .ops import F, I, TAU, bc, grey, lum, mix, ones_like_hw, smoothstep, step, vec4

ASPECT_GLSL = "vec2(resolution.x / resolution.y, 1.0)"


def _aspect(env, st):
    return np.array([env.res[0] / env.res[1], 1.0], dtype=np.float32)


# -----------------------------------------------------------------------------
# dye compositing
# -----------------------------------------------------------------------------

def _np_stain(c0, c1, density, env):
    dye = np.clip(c1[..., :3], 0.0, 1.0)
    t = np.maximum(c1[..., 3:4], 0.0) * (np.asarray(density, dtype=np.float32)[..., None] if np.ndim(density) else density)
    T = np.exp(-t * (1.0 - dye) * 3.0)
    return np.concatenate([c0[..., :3] * T, c0[..., 3:]], axis=-1)


register(Op("stain", "combine", [Input("tex", "chain"), F("density", 1)], np=_np_stain,
            doc="Shine the chain through a layer of transparent dye: rgb of the layer is what the dye lets through, alpha its thickness. Multiplicative, like ink; start from solid(1,1,1) as clear leader.",
            glsl="""vec3 dye = clamp(_c1.rgb, 0.0, 1.0);
  float t = max(_c1.a, 0.0) * density;
  vec3 T = exp(-t * (1.0 - dye) * 3.0);
  return vec4(_c0.rgb * T, _c0.a);"""))


def _np_alpha(c0, c1, env):
    return np.concatenate([c0[..., :3], lum(c1[..., :3])[..., None]], axis=-1)


register(Op("alpha", "combine", [Input("tex", "chain")], np=_np_alpha,
            doc="Set alpha from another chain's luminance without premultiplying (mask() premultiplies). Use it to build dye layers: solid(1,0.2,0.1).alpha(shape(4)).",
            glsl="return vec4(_c0.rgb, _luminance(_c1.rgb));"))


# -----------------------------------------------------------------------------
# sources
# -----------------------------------------------------------------------------

def _np_drops(st, scale, size, density, stretch, rim, softness, wobble, texture, seed, hue, hueSpread, sat, env):
    p = st * _aspect(env, st) * bc(scale, st)[..., None]
    i = np.floor(p)
    f = p - i
    seed = bc(seed, st)
    density = bc(density, st)
    cov = np.zeros(st.shape[:2], dtype=np.float32)
    rimv = np.zeros_like(cov)
    best = np.full_like(cov, 1e9)   # nearest drop (in radius units) owns the pixel's colour
    chosen = np.zeros(st.shape[:2], bool)
    chosen_hue = np.zeros(st.shape[:2], np.float32)
    wob = wobble * snoise3(np.stack([p[..., 0] * 2.0, p[..., 1] * 2.0, seed], axis=-1))
    tex = 1.0 - texture * (0.5 + 0.5 * snoise3(np.stack([p[..., 0] * 6.0, p[..., 1] * 6.0, seed + 5.0], axis=-1)))
    for y in (-2, -1, 0, 1, 2):
        for x in (-2, -1, 0, 1, 2):
            n = np.array([x, y], dtype=np.float32)
            cell = i + n
            h = h21(cell + seed[..., None] * 131.0)
            exists = h <= density
            c = n + h22(cell * 1.7 + seed[..., None] * 131.0)
            rr = h22(cell * 2.9 + seed[..., None] * 131.0 + 7.0)
            r = size * (0.35 + 0.65 * rr[..., 0])
            ang = TAU * rr[..., 1]
            d = f - c
            ca, sa = np.cos(ang), np.sin(ang)
            dx = (ca * d[..., 0] + sa * d[..., 1]) / stretch
            dy = -sa * d[..., 0] + ca * d[..., 1]
            dist = np.sqrt(dx * dx + dy * dy) / r + wob
            cv = (1.0 - smoothstep(1.0 - softness, 1.0 + softness, dist)) * exists
            rm = smoothstep(0.8, 1.0, dist) * cv
            better = exists & (cv > 0.0) & (dist < best)
            chosen_hue = np.where(better, h, chosen_hue)
            chosen |= better
            best = np.where(better, dist, best)
            cov = np.maximum(cov, cv)
            rimv = np.maximum(rimv, rm)
    hsv = np.stack([hue + hueSpread * (chosen_hue / np.maximum(density, 1e-3) - 0.5),
                    bc(sat, st), ones_like_hw(st)], axis=-1)
    col = np.where(chosen[..., None], hsv_to_rgb(hsv), 1.0)
    return np.concatenate([col, (cov * (1.0 + rim * rimv) * tex)[..., None]], axis=-1).astype(np.float32)


register(Op("drops", "src", [F("scale", 4), F("size", 0.5), F("density", 0.6), F("stretch", 1), F("rim", 0.6),
                             F("softness", 0.03), F("wobble", 0.1), F("texture", 0.4), F("seed", 0), F("hue", 0.02),
                             F("hueSpread", 0.15), F("sat", 0.9)],
            np=_np_drops,
            doc="Dye drops on a jittered grid: rgb = each drop's dye colour (hue ± hueSpread), alpha = thickness: thicker at the drying rim, uneven inside (texture). stretch > 1 smears them into strokes. Feed to stain().",
            glsl="""vec2 p = _st * """ + ASPECT_GLSL + """ * scale;
  vec2 i = floor(p);
  vec2 f = fract(p);
  float cov = 0.0, rimv = 0.0, best = 1e9;
  vec3 col = vec3(1.0);
  float wob = wobble * _noise(vec3(p * 2.0, seed));
  float tex = 1.0 - texture * (0.5 + 0.5 * _noise(vec3(p * 6.0, seed + 5.0)));
  for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
    vec2 n = vec2(float(x), float(y));
    vec2 cell = i + n;
    float h = _h21(cell + seed * 131.0);
    if (h > density) continue;
    vec2 c = n + _h22(cell * 1.7 + seed * 131.0);
    vec2 rr = _h22(cell * 2.9 + seed * 131.0 + 7.0);
    float r = size * (0.35 + 0.65 * rr.x);
    float ang = 6.283185307 * rr.y;
    vec2 d = f - c;
    d = mat2(cos(ang), -sin(ang), sin(ang), cos(ang)) * d;
    d.x /= stretch;
    float dist = length(d) / r + wob;
    float cv = 1.0 - smoothstep(1.0 - softness, 1.0 + softness, dist);
    float rm = smoothstep(0.8, 1.0, dist) * cv;
    if (cv > 0.0 && dist < best) { best = dist; col = _hsvToRgb(vec3(hue + hueSpread * (h / max(density, 1e-3) - 0.5), sat, 1.0)); }
    cov = max(cov, cv);
    rimv = max(rimv, rm);
  }
  return vec4(col, cov * (1.0 + rim * rimv) * tex);"""))


def _np_crackle(st, scale, width, warp, seed, env):
    p = st * _aspect(env, st) * bc(scale, st)[..., None]
    seedf = bc(seed, st)
    wx = snoise3(np.stack([p[..., 0] * 0.7, p[..., 1] * 0.7, seedf + 2.0], axis=-1))
    wy = snoise3(np.stack([p[..., 0] * 0.7, p[..., 1] * 0.7, seedf + 9.0], axis=-1))
    p = p + np.stack([wx, wy], axis=-1) * warp
    taper = 0.5 + 0.5 * snoise3(np.stack([p[..., 0] * 3.0, p[..., 1] * 3.0, seedf + 4.0], axis=-1))
    i = np.floor(p)
    f = p - i
    seed = bc(seed, st)[..., None]
    f1 = np.full(st.shape[:2], 8.0, dtype=np.float32)
    f2 = np.full_like(f1, 8.0)
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            n = np.array([x, y], dtype=np.float32)
            pt = h22(i + n + seed * 131.0)
            dd = n + pt - f
            d = np.sqrt((dd * dd).sum(-1))
            f2 = np.where(d < f1, f1, np.where(d < f2, d, f2))
            f1 = np.minimum(f1, d)
    line = (1.0 - smoothstep(0.0, width * (0.3 + 0.7 * taper), f2 - f1)) * (0.5 + 0.5 * taper)
    return np.concatenate([np.stack([line, line, line], axis=-1), line[..., None]], axis=-1).astype(np.float32)


register(Op("crackle", "src", [F("scale", 6), F("width", 0.04), F("warp", 0.4), F("seed", 0)], np=_np_crackle,
            doc="Cracks in dried paint: warped Voronoi cell borders as thin tapering lines (alpha = line).",
            glsl="""vec2 p = _st * """ + ASPECT_GLSL + """ * scale;
  p += vec2(_noise(vec3(p * 0.7, seed + 2.0)), _noise(vec3(p * 0.7, seed + 9.0))) * warp;
  float taper = 0.5 + 0.5 * _noise(vec3(p * 3.0, seed + 4.0));
  vec2 i = floor(p);
  vec2 f = fract(p);
  float f1 = 8.0, f2 = 8.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 n = vec2(float(x), float(y));
    vec2 pt = _h22(i + n + seed * 131.0);
    float d = length(n + pt - f);
    if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
  }
  float line = (1.0 - smoothstep(0.0, width * (0.3 + 0.7 * taper), f2 - f1)) * (0.5 + 0.5 * taper);
  return vec4(vec3(line), line);"""))


def _np_veins(st, scale, octaves, sharpness, speed, seed, env):
    p = st * _aspect(env, st) * bc(scale, st)[..., None]
    v = np.zeros(st.shape[:2], dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    z = bc(seed + env.time * speed, st)
    for k in range(int(octaves)):
        n = 1.0 - np.abs(snoise3(np.stack([p[..., 0] * freq, p[..., 1] * freq, z + k * 7.3], axis=-1)))
        v = v + amp * np.power(np.maximum(n, 0.0), sharpness)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return grey(v / max(norm, 1e-6), st)


register(Op("veins", "src", [F("scale", 3), I("octaves", 5), F("sharpness", 2), F("speed", 0), F("seed", 0)], np=_np_veins,
            doc="Ridged multifractal noise: leaf veins, wing membranes, cracked ice. Bright lines on dark.",
            glsl="""vec2 p = _st * """ + ASPECT_GLSL + """ * scale;
  float v = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
  for (int k = 0; k < 8; k++) {
    if (k >= octaves) break;
    float n = 1.0 - abs(_noise(vec3(p * freq, seed + time * speed + float(k) * 7.3)));
    v += amp * pow(max(n, 0.0), sharpness);
    norm += amp; amp *= 0.5; freq *= 2.0;
  }
  return vec4(vec3(v / norm), 1.0);"""))


def _np_scratches(st, density, extent, width, jitter, rate, seed, env):
    px = st[..., 0] * env.res[0]
    lane = np.floor(px / 3.0)
    seed = bc(seed, st)
    h = h21(np.stack([lane, np.full_like(lane, 17.0)], axis=-1) + seed[..., None] * 131.0)
    on = step(1.0 - density * 0.2, h)
    y0 = h11(lane * 3.1 + seed * 131.0)
    y1 = y0 + extent * (0.3 + 0.7 * h11(lane * 5.7 + seed * 131.0))
    inY = step(y0, st[..., 1]) * step(st[..., 1], y1)
    frame = np.floor(env.time * rate)
    jx = jitter * (h11(lane * 0.37 + frame * 1.7 + seed) - 0.5)
    cx = (h11(lane * 9.1 + seed * 131.0) - 0.5) * 2.0 + jx
    dx = np.abs((px / 3.0 - np.floor(px / 3.0)) * 3.0 - 1.5 - cx)
    depth = 0.4 + 0.6 * snoise3(np.stack([lane * 0.3, st[..., 1] * 15.0, seed], axis=-1))
    line = (1.0 - smoothstep(0.0, width, dx)) * on * inY * depth
    return np.concatenate([np.stack([line, line, line], axis=-1), line[..., None]], axis=-1).astype(np.float32)


register(Op("scratches", "src", [F("density", 0.5), F("extent", 0.6), F("width", 1.2), F("jitter", 1), F("rate", 24), F("seed", 0)],
            np=_np_scratches,
            doc="Vertical emulsion scratches (white on black, alpha = line), wiggling `jitter` pixels `rate` times a second. Drift the seed slowly; scratches run along the strip.",
            glsl="""float px = _st.x * resolution.x;
  float lane = floor(px / 3.0);
  float h = _h21(vec2(lane, 17.0) + seed * 131.0);
  float on = step(1.0 - density * 0.2, h);
  float y0 = _h11(lane * 3.1 + seed * 131.0);
  float y1 = y0 + extent * (0.3 + 0.7 * _h11(lane * 5.7 + seed * 131.0));
  float inY = step(y0, _st.y) * step(_st.y, y1);
  float frame = floor(time * rate);
  float jx = jitter * (_h11(lane * 0.37 + frame * 1.7 + seed) - 0.5);
  float cx = (_h11(lane * 9.1 + seed * 131.0) - 0.5) * 2.0 + jx;
  float dx = abs(fract(px / 3.0) * 3.0 - 1.5 - cx);
  float depth = 0.4 + 0.6 * _noise(vec3(lane * 0.3, _st.y * 15.0, seed));
  float line = (1.0 - smoothstep(0.0, width, dx)) * on * inY * depth;
  return vec4(vec3(line), line);"""))


def _np_dust(st, density, size, hairs, seed, env):
    pxp = st * env.res
    ci = np.floor(pxp / 24.0)
    seed = bc(seed, st)
    v = np.zeros(st.shape[:2], dtype=np.float32)
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            n = np.array([x, y], dtype=np.float32)
            cell = ci + n
            h = h21(cell + seed[..., None] * 131.0)
            exists = h <= density * 0.5
            centre = (cell + h22(cell * 2.3 + seed[..., None] * 131.0)) * 24.0
            ka = h22(cell * 3.7 + seed[..., None] * 131.0 + 3.0)
            lr = h22(cell * 5.3 + seed[..., None] * 131.0 + 9.0)
            kind = ka[..., 0]
            ang = TAU * ka[..., 1]
            ln = 8.0 + 16.0 * lr[..., 0]
            d_ = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
            rel = pxp - centre
            t = np.clip((rel * d_).sum(-1), -ln * 0.5, ln * 0.5)
            perp = rel - d_ * t[..., None]
            hair = (1.0 - smoothstep(0.0, 0.9, np.sqrt((perp * perp).sum(-1)))) * 0.7
            rad = size * (0.5 + lr[..., 1])
            speck = 1.0 - smoothstep(rad, rad + 1.0, np.sqrt((rel * rel).sum(-1)))
            val = np.where(kind < hairs, hair, speck) * exists
            v = np.maximum(v, val)
    return np.concatenate([np.stack([v, v, v], axis=-1), v[..., None]], axis=-1).astype(np.float32)


register(Op("dust", "src", [F("density", 0.3), F("size", 1.5), F("hairs", 0.3), F("seed", 0)], np=_np_dust,
            doc="Specks and hairs on the gate (white on black, alpha = dust). New seed every frame for the real thing: seed=SH(24, 0, 99).",
            glsl="""vec2 pxp = _st * resolution;
  vec2 ci = floor(pxp / 24.0);
  float v = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 cell = ci + vec2(float(x), float(y));
    float h = _h21(cell + seed * 131.0);
    if (h > density * 0.5) continue;
    vec2 centre = (cell + _h22(cell * 2.3 + seed * 131.0)) * 24.0;
    vec2 ka = _h22(cell * 3.7 + seed * 131.0 + 3.0);
    vec2 lr = _h22(cell * 5.3 + seed * 131.0 + 9.0);
    float kind = ka.x;
    if (kind < hairs) {
      float ang = 6.283185307 * ka.y;
      float len = 8.0 + 16.0 * lr.x;
      vec2 dir = vec2(cos(ang), sin(ang));
      vec2 rel = pxp - centre;
      float t = clamp(dot(rel, dir), -len * 0.5, len * 0.5);
      v = max(v, (1.0 - smoothstep(0.0, 0.9, length(rel - dir * t))) * 0.7);
    } else {
      float rad = size * (0.5 + lr.y);
      v = max(v, 1.0 - smoothstep(rad, rad + 1.0, length(pxp - centre)));
    }
  }
  return vec4(vec3(v), v);"""))


# -----------------------------------------------------------------------------
# coordinate op
# -----------------------------------------------------------------------------

def _np_pan(st, x, y, speedX, speedY, env):
    return st + np.stack([bc(x + env.time * speedX, st), bc(y + env.time * speedY, st)], axis=-1)


register(Op("pan", "coord", [F("x", 0), F("y", 0), F("speedX", 0), F("speedY", 0)], np=_np_pan,
            doc="Translate without wrapping (scroll wraps). pan(0, 0, 0, fps) moves one frame-height per frame: a painted strip running through the gate, so smears continue from frame to frame.",
            glsl="return _st + vec2(x + time * speedX, y + time * speedY);"))


# -----------------------------------------------------------------------------
# texture op: ink spreading
# -----------------------------------------------------------------------------

def _np_bleed(tex, st, grow, softness, roughness, env):
    acc = None
    wsum = 0.0
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            w = 2.0 if (x == 0 and y == 0) else 1.0
            s = env.sample_offset(tex, st, (x * 1.5, y * 1.5))
            pm = np.concatenate([s[..., :3] * s[..., 3:4], s[..., 3:4]], axis=-1) * w
            acc = pm if acc is None else acc + pm
            wsum += w
    acc = acc / wsum
    n = snoise3(np.stack([st[..., 0] * env.res[0] * 0.04, st[..., 1] * env.res[1] * 0.04, np.full(st.shape[:2], 11.0, np.float32)], axis=-1))
    a = smoothstep(0.5 - grow - softness, 0.5 - grow + softness, acc[..., 3] + (n - 0.5) * roughness) * np.maximum(acc[..., 3], 1.0)
    rgb = np.where(acc[..., 3:4] > 1e-4, acc[..., :3] / np.maximum(acc[..., 3:4], 1e-4), 1.0)
    return np.concatenate([rgb, a[..., None]], axis=-1).astype(np.float32)


register(Op("bleed", "texture", [F("grow", 0.08, scalar=True), F("softness", 0.15, scalar=True), F("roughness", 0.4, scalar=True), I("iterations", 3)],
            np=_np_bleed, iterations=True, compute="bleed",
            doc="Ink spreading: blurs a dye layer and re-thresholds its alpha with noise, so edges creep outward (grow) and go ragged (roughness). Iterate for more.",
            glsl="""vec2 px = 1.0 / resolution;
  vec4 acc = vec4(0.0); float wsum = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    float w = (x == 0 && y == 0) ? 2.0 : 1.0;
    vec4 s = texture2D(_tex, fract(_st + vec2(float(x), float(y)) * px * 1.5));
    acc += vec4(s.rgb * s.a, s.a) * w; wsum += w;
  }
  acc /= wsum;
  float n = _noise(vec3(_st * resolution * 0.04, 11.0));
  float a = smoothstep(0.5 - grow - softness, 0.5 - grow + softness, acc.a + (n - 0.5) * roughness) * max(acc.a, 1.0);
  vec3 rgb = acc.a > 1e-4 ? acc.rgb / acc.a : vec3(1.0);
  return vec4(rgb, a);"""))


# -----------------------------------------------------------------------------
# texture op: bake (render once, reuse)
# -----------------------------------------------------------------------------

def _np_bake(tex, st, env):
    return env.sample(tex, st)


register(Op("bake", "texture", [], np=_np_bake,
            doc="Render the chain so far into a texture and sample it from here on. Everything after sees one copy, so a chain used twice (bloom: x.add(x.blur())) is evaluated once. Coordinates after bake() wrap.",
            glsl="return texture2D(_tex, fract(_st));"))
