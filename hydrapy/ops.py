"""The op registry: every Hydra op plus the additions, each with a GLSL body and a NumPy twin.

GLSL bodies are GLSL ES 1.00 and may use ``time``, ``resolution``, ``gl_FragCoord`` and
the helpers in ``noise.GLSL_NOISE``.  NumPy twins receive the same arguments plus an
``env`` (see ``backend_np.Env``) and must broadcast: any float argument may arrive as a
scalar *or* an (H, W) array when a chain is plugged into that parameter.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .noise import bayer4, h11, h21, h22, h31, hash11, hash21, hash22, hash31, hsv_to_rgb, rgb_to_hsv, snoise3

TAU = 6.283185307179586
PI = 3.1416  # Hydra uses 3.1416 in shape()/kaleid(); keep it for pixel parity

W_LUMA = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)


def F(name, default=0.0, **kw):
    return Input(name, "float", float(default), **kw)


def I(name, default=1, **kw):
    return Input(name, "int", int(default), scalar=True, **kw)


def lum(rgb):
    return rgb[..., 0] * W_LUMA[0] + rgb[..., 1] * W_LUMA[1] + rgb[..., 2] * W_LUMA[2]


def bc(v, like):
    """Broadcast a scalar/array parameter to the (H, W) grid of ``like``."""
    return np.broadcast_to(np.asarray(v, dtype=np.float32), like.shape[:2])


def ones_like_hw(like):
    return np.ones(like.shape[:2], dtype=np.float32)


def vec4(r, g, b, a=None, like=None):
    like = like if like is not None else r
    if a is None:
        a = ones_like_hw(like)
    return np.stack([bc(r, like), bc(g, like), bc(b, like), bc(a, like)], axis=-1).astype(np.float32)


def grey(v, like=None):
    like = like if like is not None else v
    v = bc(v, like)
    return np.stack([v, v, v, ones_like_hw(like)], axis=-1).astype(np.float32)


def fract(x):
    return x - np.floor(x)


def step(edge, x):
    return (np.asarray(x) >= edge).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a * (1.0 - t) + b * t


def rotate_xy(xy, ang):
    ang = np.asarray(ang, dtype=np.float32)
    c, s = np.cos(ang), np.sin(ang)
    x, y = xy[..., 0], xy[..., 1]
    # mat2(cos, -sin, sin, cos) * xy  (GLSL column-major): x' = cos*x + sin*y ; y' = -sin*x + cos*y
    return np.stack([c * x + s * y, -s * x + c * y], axis=-1)


# =============================================================================
# SOURCES
# =============================================================================

def _np_noise(st, scale, offset, env):
    v = np.stack([st[..., 0] * scale, st[..., 1] * scale, bc(offset * env.time, st)], axis=-1)
    return grey(snoise3(v), st)


register(Op("noise", "src", [F("scale", 10), F("offset", 0.1)], hydra=True, np=_np_noise,
            doc="Simplex noise; offset scales its drift through time.",
            glsl="return vec4(vec3(_noise(vec3(_st*scale, offset*time))), 1.0);"))


def _np_voronoi(st, scale, speed, blending, env):
    p = st * bc(scale, st)[..., None]
    i_st = np.floor(p)
    f_st = p - i_st
    m_dist = np.full(st.shape[:2], 10.0, dtype=np.float32)
    m_point = np.zeros(st.shape, dtype=np.float32)
    tspeed = bc(env.time * speed, st)[..., None]
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            neighbor = np.array([i, j], dtype=np.float32)
            pt = hash22(i_st + neighbor)
            pt = 0.5 + 0.5 * np.sin(tspeed + TAU * pt)
            diff = neighbor + pt - f_st
            dist = np.sqrt((diff * diff).sum(-1))
            closer = dist < m_dist
            m_dist = np.where(closer, dist, m_dist)
            m_point = np.where(closer[..., None], pt, m_point)
    color = m_point[..., 0] * 0.3 + m_point[..., 1] * 0.6
    color = color * (1.0 - blending * m_dist)
    return grey(color, st)


register(Op("voronoi", "src", [F("scale", 5), F("speed", 0.3), F("blending", 0.3)], hydra=True, np=_np_voronoi,
            doc="Hydra's animated Voronoi cells.",
            glsl="""vec3 color = vec3(.0);
  vec2 st = _st * scale;
  vec2 i_st = floor(st);
  vec2 f_st = fract(st);
  float m_dist = 10.;
  vec2 m_point;
  for (int j=-1; j<=1; j++ ) {
    for (int i=-1; i<=1; i++ ) {
      vec2 neighbor = vec2(float(i),float(j));
      vec2 p = i_st + neighbor;
      vec2 point = fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
      point = 0.5 + 0.5*sin(time*speed + 6.2831*point);
      vec2 diff = neighbor + point - f_st;
      float dist = length(diff);
      if( dist < m_dist ) { m_dist = dist; m_point = point; }
    }
  }
  color += dot(m_point,vec2(.3,.6));
  color *= 1.0 - blending*m_dist;
  return vec4(color, 1.0);"""))


def _np_osc(st, frequency, sync, offset, env):
    x, t = st[..., 0], env.time
    r = np.sin((x - offset / frequency + t * sync) * frequency) * 0.5 + 0.5
    g = np.sin((x + t * sync) * frequency) * 0.5 + 0.5
    b = np.sin((x + offset / frequency + t * sync) * frequency) * 0.5 + 0.5
    return vec4(r, g, b, like=st)


register(Op("osc", "src", [F("frequency", 60), F("sync", 0.1), F("offset", 0)], hydra=True, np=_np_osc,
            doc="Sine oscillator across x; offset separates the colour channels.",
            glsl="""vec2 st = _st;
  float r = sin((st.x-offset/frequency+time*sync)*frequency)*0.5 + 0.5;
  float g = sin((st.x+time*sync)*frequency)*0.5 + 0.5;
  float b = sin((st.x+offset/frequency+time*sync)*frequency)*0.5 + 0.5;
  return vec4(r, g, b, 1.0);"""))


def _np_shape(st, sides, radius, smoothing, env):
    p = st * 2.0 - 1.0
    a = np.arctan2(p[..., 0], p[..., 1]) + PI
    r = (2.0 * PI) / sides
    d = np.cos(np.floor(0.5 + a / r) * r - a) * np.sqrt((p * p).sum(-1))
    v = 1.0 - smoothstep(radius, radius + smoothing + 0.0000001, d)
    return grey(v, st)


register(Op("shape", "src", [F("sides", 3), F("radius", 0.3), F("smoothing", 0.01)], hydra=True, np=_np_shape,
            doc="Regular polygon with `sides` sides.",
            glsl="""vec2 st = _st * 2. - 1.;
  float a = atan(st.x,st.y)+3.1416;
  float r = (2.*3.1416)/sides;
  float d = cos(floor(.5+a/r)*r-a)*length(st);
  return vec4(vec3(1.0-smoothstep(radius,radius + smoothing + 0.0000001,d)), 1.0);"""))


def _np_gradient(st, speed, env):
    return vec4(st[..., 0], st[..., 1], np.sin(env.time * speed) * ones_like_hw(st), like=st)


register(Op("gradient", "src", [F("speed", 0)], hydra=True, np=_np_gradient,
            doc="x in red, y in green, sin(time*speed) in blue.",
            glsl="return vec4(_st, sin(time*speed), 1.0);"))


def _np_src(st, tex, env):
    return env.sample(env.texture(tex), st)


register(Op("src", "src", [Input("tex", "sampler2D", None)], hydra=True, np=_np_src,
            doc="Read an output (o0..o3), Image, Video or Camera.",
            glsl="return texture2D(tex, fract(_st));"))


def _np_delay(st, tex, frames, env):
    return env.sample(env.history(tex, int(frames)), st)


register(Op("delay", "src", [Input("tex", "sampler2D", None), I("frames", 1)], np=_np_delay,
            doc="An output as it was `frames` frames earlier than src() gives (delay(o0, 0) == src(o0)). Each output keeps a ring of textures as deep as the largest delay used. Optical-printer moves: frame repeats, A/B rolls, superimposition at an offset.",
            glsl="return texture2D(tex, fract(_st));"))


def _np_prev(st, env):
    return env.sample(env.texture(env.current_output), st)


register(Op("prev", "src", [], hydra=True, np=_np_prev,
            doc="The previous frame of the output this chain renders into.",
            glsl="return texture2D(prevBuffer, fract(_st));"))


def _np_solid(st, r, g, b, a, env):
    return vec4(r, g, b, a, like=st)


register(Op("solid", "src", [F("r", 0), F("g", 0), F("b", 0), F("a", 1)], hydra=True, np=_np_solid,
            doc="A flat colour.", glsl="return vec4(r, g, b, a);"))


# --- new sources -------------------------------------------------------------

def _np_fbm(st, scale, octaves, speed, gain, lacunarity, env):
    total = np.zeros(st.shape[:2], dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(int(octaves)):
        v = np.stack([st[..., 0] * scale * freq, st[..., 1] * scale * freq,
                      bc(env.time * speed + i * 7.3, st)], axis=-1)
        total = total + amp * snoise3(v)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return grey(0.5 + 0.5 * total / max(norm, 1e-6), st)


register(Op("fbm", "src", [F("scale", 4), I("octaves", 4), F("speed", 0.1), F("gain", 0.5), F("lacunarity", 2.0)],
            np=_np_fbm, doc="Fractal Brownian motion: layered simplex noise, 0..1.",
            glsl="""float total = 0.0; float amp = 1.0; float freq = 1.0; float norm = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    total += amp * _noise(vec3(_st*scale*freq, time*speed + float(i)*7.3));
    norm += amp; amp *= gain; freq *= lacunarity;
  }
  return vec4(vec3(0.5 + 0.5*total/max(norm, 1e-6)), 1.0);"""))


def _np_rings(st, frequency, speed, offset, env):
    p = st - 0.5
    r = np.sqrt((p * p).sum(-1)) * 2.0
    return grey(0.5 + 0.5 * np.sin(r * frequency - env.time * speed + offset), st)


register(Op("rings", "src", [F("frequency", 10), F("speed", 0), F("offset", 0)], np=_np_rings,
            doc="Concentric sine rings.",
            glsl="float r = length(_st - 0.5)*2.0; return vec4(vec3(0.5 + 0.5*sin(r*frequency - time*speed + offset)), 1.0);"))


def _np_grid(st, n, thickness, env):
    f = np.abs(fract(st * n) - 0.5)
    line = step(0.5 - thickness, np.maximum(f[..., 0], f[..., 1]))
    return grey(line, st)


register(Op("grid", "src", [F("n", 8), F("thickness", 0.05)], np=_np_grid,
            doc="Line grid, n cells across.",
            glsl="vec2 f = abs(fract(_st*n) - 0.5); float line = step(0.5 - thickness, max(f.x, f.y)); return vec4(vec3(line), 1.0);"))


def _np_white(st, rate, seed, env):
    v = h31(np.stack([np.floor(st[..., 0] * env.w), np.floor(st[..., 1] * env.h),
                      bc(np.floor(env.time * rate) + seed, st)], axis=-1))
    return grey(v, st)


register(Op("white", "src", [F("rate", 60), F("seed", 0)], np=_np_white,
            doc="Per-pixel static, re-rolled `rate` times a second.",
            glsl="return vec4(vec3(_h31(vec3(floor(_st*resolution), floor(time*rate) + seed))), 1.0);"))


# =============================================================================
# COORDINATE OPS
# =============================================================================

def _np_rotate(st, angle, speed, env):
    xy = st - 0.5
    return rotate_xy(xy, angle + speed * env.time) + 0.5


register(Op("rotate", "coord", [F("angle", 10), F("speed", 0)], hydra=True, np=_np_rotate,
            doc="Rotate about the centre (radians).",
            glsl="""vec2 xy = _st - vec2(0.5);
  float ang = angle + speed *time;
  xy = mat2(cos(ang),-sin(ang), sin(ang),cos(ang))*xy;
  xy += 0.5;
  return xy;"""))


def _np_scale(st, amount, xMult, yMult, offsetX, offsetY, env):
    off = np.stack([bc(offsetX, st), bc(offsetY, st)], axis=-1)
    xy = st - off
    xy = xy / np.stack([bc(amount * xMult, st), bc(amount * yMult, st)], axis=-1)
    return xy + off


register(Op("scale", "coord", [F("amount", 1.5), F("xMult", 1), F("yMult", 1), F("offsetX", 0.5), F("offsetY", 0.5)],
            hydra=True, np=_np_scale, doc="Zoom about (offsetX, offsetY).",
            glsl="""vec2 xy = _st - vec2(offsetX, offsetY);
  xy*=(1.0/vec2(amount*xMult, amount*yMult));
  xy+=vec2(offsetX, offsetY);
  return xy;"""))


def _np_pixelate(st, pixelX, pixelY, env):
    xy = np.stack([bc(pixelX, st), bc(pixelY, st)], axis=-1)
    return (np.floor(st * xy) + 0.5) / xy


register(Op("pixelate", "coord", [F("pixelX", 20), F("pixelY", 20)], hydra=True, np=_np_pixelate,
            doc="Quantise coordinates into pixelX by pixelY cells.",
            glsl="vec2 xy = vec2(pixelX, pixelY); return (floor(_st * xy) + 0.5)/xy;"))


def _np_repeat(st, repeatX, repeatY, offsetX, offsetY, env):
    s = st * np.stack([bc(repeatX, st), bc(repeatY, st)], axis=-1)
    x = s[..., 0] + step(1.0, np.mod(s[..., 1], 2.0)) * offsetX
    y = s[..., 1] + step(1.0, np.mod(x, 2.0)) * offsetY
    return fract(np.stack([x, y], axis=-1))


register(Op("repeat", "coord", [F("repeatX", 3), F("repeatY", 3), F("offsetX", 0), F("offsetY", 0)],
            hydra=True, np=_np_repeat, doc="Tile, with alternating-row/column offsets.",
            glsl="""vec2 st = _st * vec2(repeatX, repeatY);
  st.x += step(1., mod(st.y,2.0)) * offsetX;
  st.y += step(1., mod(st.x,2.0)) * offsetY;
  return fract(st);"""))


def _np_repeatX(st, reps, offset, env):
    s = st * np.stack([bc(reps, st), ones_like_hw(st)], axis=-1)
    y = s[..., 1] + step(1.0, np.mod(s[..., 0], 2.0)) * offset
    return fract(np.stack([s[..., 0], y], axis=-1))


register(Op("repeatX", "coord", [F("reps", 3), F("offset", 0)], hydra=True, np=_np_repeatX,
            doc="Tile horizontally.",
            glsl="""vec2 st = _st * vec2(reps, 1.0);
  st.y += step(1., mod(st.x,2.0))* offset;
  return fract(st);"""))


def _np_repeatY(st, reps, offset, env):
    s = st * np.stack([ones_like_hw(st), bc(reps, st)], axis=-1)
    x = s[..., 0] + step(1.0, np.mod(s[..., 1], 2.0)) * offset
    return fract(np.stack([x, s[..., 1]], axis=-1))


register(Op("repeatY", "coord", [F("reps", 3), F("offset", 0)], hydra=True, np=_np_repeatY,
            doc="Tile vertically.",
            glsl="""vec2 st = _st * vec2(1.0, reps);
  st.x += step(1., mod(st.y,2.0))* offset;
  return fract(st);"""))


def _np_kaleid(st, nSides, env):
    p = st - 0.5
    r = np.sqrt((p * p).sum(-1))
    a = np.arctan2(p[..., 1], p[..., 0])
    pi = 2.0 * PI
    a = np.mod(a, pi / nSides)
    a = np.abs(a - pi / nSides / 2.0)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)


register(Op("kaleid", "coord", [F("nSides", 4)], hydra=True, np=_np_kaleid,
            doc="Kaleidoscope with nSides mirrors.",
            glsl="""vec2 st = _st;
  st -= 0.5;
  float r = length(st);
  float a = atan(st.y, st.x);
  float pi = 2.*3.1416;
  a = mod(a,pi/nSides);
  a = abs(a-pi/nSides/2.);
  return r*vec2(cos(a), sin(a));"""))


def _np_scroll(st, scrollX, scrollY, speedX, speedY, env):
    x = st[..., 0] + scrollX + env.time * speedX
    y = st[..., 1] + scrollY + env.time * speedY
    return fract(np.stack([x, y], axis=-1))


register(Op("scroll", "coord", [F("scrollX", 0.5), F("scrollY", 0.5), F("speedX", 0), F("speedY", 0)],
            hydra=True, np=_np_scroll, doc="Translate (wrapping).",
            glsl="""_st.x += scrollX + time*speedX;
  _st.y += scrollY + time*speedY;
  return fract(_st);"""))


def _np_scrollX(st, scrollX, speed, env):
    return fract(np.stack([st[..., 0] + scrollX + env.time * speed, st[..., 1]], axis=-1))


register(Op("scrollX", "coord", [F("scrollX", 0.5), F("speed", 0)], hydra=True, np=_np_scrollX,
            doc="Translate horizontally.", glsl="_st.x += scrollX + time*speed; return fract(_st);"))


def _np_scrollY(st, scrollY, speed, env):
    return fract(np.stack([st[..., 0], st[..., 1] + scrollY + env.time * speed], axis=-1))


register(Op("scrollY", "coord", [F("scrollY", 0.5), F("speed", 0)], hydra=True, np=_np_scrollY,
            doc="Translate vertically.", glsl="_st.y += scrollY + time*speed; return fract(_st);"))


# --- new coordinate ops ----------------------------------------------------------

def _np_polar(st, zoom, rotations, env):
    p = st - 0.5
    r = np.sqrt((p * p).sum(-1)) * 2.0 * zoom
    a = (np.arctan2(p[..., 1], p[..., 0]) / TAU + 0.5) * rotations
    return np.stack([fract(a), r], axis=-1)


register(Op("polar", "coord", [F("zoom", 1), F("rotations", 1)], np=_np_polar,
            doc="Cartesian -> polar: x becomes angle, y becomes radius. osc().polar() gives spokes.",
            glsl="""vec2 p = _st - 0.5;
  float r = length(p)*2.0*zoom;
  float a = (atan(p.y, p.x)/6.283185307 + 0.5)*rotations;
  return vec2(fract(a), r);"""))


def _np_unpolar(st, zoom, env):
    a = st[..., 0] * TAU
    r = st[..., 1] * 0.5 / zoom
    return np.stack([0.5 + r * np.cos(a), 0.5 + r * np.sin(a)], axis=-1)


register(Op("unpolar", "coord", [F("zoom", 1)], np=_np_unpolar,
            doc="Polar -> cartesian: wraps the image around the centre (tunnel).",
            glsl="float a = _st.x*6.283185307; float r = _st.y*0.5/zoom; return vec2(0.5 + r*cos(a), 0.5 + r*sin(a));"))


def _np_twist(st, amount, speed, env):
    p = st - 0.5
    r = np.sqrt((p * p).sum(-1))
    return rotate_xy(p, (amount + speed * env.time) * r) + 0.5


register(Op("twist", "coord", [F("amount", 3), F("speed", 0)], np=_np_twist,
            doc="Rotation that grows with radius (swirl).",
            glsl="""vec2 p = _st - 0.5;
  float r = length(p);
  float ang = (amount + speed*time)*r;
  p = mat2(cos(ang),-sin(ang), sin(ang),cos(ang))*p;
  return p + 0.5;"""))


def _np_fisheye(st, amount, env):
    p = (st - 0.5) * 2.0
    d = (p * p).sum(-1)
    return p * (1.0 + amount * d)[..., None] * 0.5 + 0.5


register(Op("fisheye", "coord", [F("amount", 1)], np=_np_fisheye,
            doc="Barrel (amount > 0) or pincushion (amount < 0) distortion.",
            glsl="vec2 p = (_st - 0.5)*2.0; p *= 1.0 + amount*dot(p, p); return p*0.5 + 0.5;"))


def _np_mirror(st, x, y, env):
    m = 0.5 - np.abs(st - 0.5)
    w = np.stack([bc(step(0.5, x), st), bc(step(0.5, y), st)], axis=-1)
    return mix(st, m, w)


register(Op("mirror", "coord", [F("x", 1), F("y", 0)], np=_np_mirror,
            doc="Reflect the left half into the right (x) and/or bottom into top (y).",
            glsl="vec2 m = vec2(0.5) - abs(_st - 0.5); return mix(_st, m, vec2(step(0.5, x), step(0.5, y)));"))


def _np_wobble(st, frequency, amount, speed, env):
    t = env.time * speed
    dx = np.sin(st[..., 1] * frequency * TAU + t) * amount
    dy = np.cos(st[..., 0] * frequency * TAU + t) * amount
    return st + np.stack([dx, dy], axis=-1)


register(Op("wobble", "coord", [F("frequency", 3), F("amount", 0.02), F("speed", 1)], np=_np_wobble,
            doc="Sinusoidal displacement.",
            glsl="""float t = time*speed;
  return _st + amount*vec2(sin(_st.y*frequency*6.283185307 + t), cos(_st.x*frequency*6.283185307 + t));"""))


def _np_glitch(st, blocks, amount, rate, seed, env):
    cell = np.floor(st * blocks)
    tick = bc(np.floor(env.time * rate) + seed, st)
    on = step(0.7, h31(np.stack([cell[..., 0], cell[..., 1], tick], axis=-1)))
    shift = (h22(cell + tick[..., None] * 0.37) - 0.5) * amount * on[..., None]
    return fract(st + shift)


register(Op("glitch", "coord", [F("blocks", 8), F("amount", 0.1), F("rate", 4), F("seed", 0)], np=_np_glitch,
            doc="Random block displacement, re-rolled `rate` times a second.",
            glsl="""vec2 cell = floor(_st*blocks);
  float tick = floor(time*rate) + seed;
  float on = step(0.7, _h31(vec3(cell, tick)));
  vec2 shift = (_h22(cell + tick*0.37) - 0.5)*amount*on;
  return fract(_st + shift);"""))


# =============================================================================
# COLOUR OPS
# =============================================================================

def _np_posterize(c, bins, gamma, env):
    g = np.asarray(gamma, dtype=np.float32)[..., None] if np.ndim(gamma) else gamma
    b = np.asarray(bins, dtype=np.float32)[..., None] if np.ndim(bins) else bins
    c2 = np.power(np.maximum(c, 0.0), g)
    c2 = np.floor(c2 * b) / b
    c2 = np.power(c2, 1.0 / g)
    return np.concatenate([c2[..., :3], c[..., 3:]], axis=-1)


register(Op("posterize", "color", [F("bins", 3), F("gamma", 0.6)], hydra=True, np=_np_posterize,
            doc="Quantise colour into `bins` levels.",
            glsl="""vec4 c2 = pow(_c0, vec4(gamma));
  c2 *= vec4(bins);
  c2 = floor(c2);
  c2/= vec4(bins);
  c2 = pow(c2, vec4(1.0/gamma));
  return vec4(c2.xyz, _c0.a);"""))


def _np_shift(c, r, g, b, a, env):
    return np.stack([fract(c[..., 0] + r), fract(c[..., 1] + g), fract(c[..., 2] + b), fract(c[..., 3] + a)], axis=-1)


register(Op("shift", "color", [F("r", 0.5), F("g", 0), F("b", 0), F("a", 0)], hydra=True, np=_np_shift,
            doc="Add to each channel and wrap.",
            glsl="""vec4 c2 = vec4(_c0);
  c2.r = fract(c2.r + r);
  c2.g = fract(c2.g + g);
  c2.b = fract(c2.b + b);
  c2.a = fract(c2.a + a);
  return vec4(c2.rgba);"""))


def _np_invert(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    rgb = (1.0 - c[..., :3]) * am + c[..., :3] * (1.0 - am)
    return np.concatenate([rgb, c[..., 3:]], axis=-1)


register(Op("invert", "color", [F("amount", 1)], hydra=True, np=_np_invert, doc="Invert colour.",
            glsl="return vec4((1.0-_c0.rgb)*amount + _c0.rgb*(1.0-amount), _c0.a);"))


def _np_contrast(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    rgb = (c[..., :3] - 0.5) * am + 0.5
    return np.concatenate([rgb, c[..., 3:]], axis=-1)


register(Op("contrast", "color", [F("amount", 1.6)], hydra=True, np=_np_contrast, doc="Contrast about 0.5.",
            glsl="vec4 c = (_c0-vec4(0.5))*vec4(amount) + vec4(0.5); return vec4(c.rgb, _c0.a);"))


def _np_brightness(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    return np.concatenate([c[..., :3] + am, c[..., 3:]], axis=-1)


register(Op("brightness", "color", [F("amount", 0.4)], hydra=True, np=_np_brightness, doc="Add to RGB.",
            glsl="return vec4(_c0.rgb + vec3(amount), _c0.a);"))


def _np_luma(c, threshold, tolerance, env):
    a = smoothstep(threshold - (tolerance + 0.0000001), threshold + (tolerance + 0.0000001), lum(c[..., :3]))
    return np.concatenate([c[..., :3] * a[..., None], a[..., None]], axis=-1)


register(Op("luma", "color", [F("threshold", 0.5), F("tolerance", 0.1)], hydra=True, np=_np_luma,
            doc="Keep bright parts; alpha follows luminance (for layer/mask).",
            glsl="""float a = smoothstep(threshold-(tolerance+0.0000001), threshold+(tolerance+0.0000001), _luminance(_c0.rgb));
  return vec4(_c0.rgb*a, a);"""))


def _np_thresh(c, threshold, tolerance, env):
    v = smoothstep(threshold - (tolerance + 0.0000001), threshold + (tolerance + 0.0000001), lum(c[..., :3]))
    return np.concatenate([np.stack([v, v, v], axis=-1), c[..., 3:]], axis=-1)


register(Op("thresh", "color", [F("threshold", 0.5), F("tolerance", 0.04)], hydra=True, np=_np_thresh,
            doc="Black/white by luminance.",
            glsl="return vec4(vec3(smoothstep(threshold-(tolerance+0.0000001), threshold+(tolerance+0.0000001), _luminance(_c0.rgb))), _c0.a);"))


def _np_color(c, r, g, b, a, env):
    col = vec4(r, g, b, a, like=c)
    pos = step(0.0, col)
    return mix((1.0 - c) * np.abs(col), col * c, pos)


register(Op("color", "color", [F("r", 1), F("g", 1), F("b", 1), F("a", 1)], hydra=True, np=_np_color,
            doc="Multiply channels; negative values invert that channel.",
            glsl="""vec4 c = vec4(r, g, b, a);
  vec4 pos = step(0.0, c);
  return vec4(mix((1.0-_c0)*abs(c), c*_c0, pos));"""))


def _np_saturate(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    inten = lum(c[..., :3])[..., None]
    return np.concatenate([mix(np.repeat(inten, 3, -1), c[..., :3], am), c[..., 3:]], axis=-1)


register(Op("saturate", "color", [F("amount", 2)], hydra=True, np=_np_saturate, doc="Saturation.",
            glsl="""const vec3 W = vec3(0.2125, 0.7154, 0.0721);
  vec3 intensity = vec3(dot(_c0.rgb, W));
  return vec4(mix(intensity, _c0.rgb, amount), _c0.a);"""))


def _np_hue(c, hue, env):
    h = rgb_to_hsv(c[..., :3])
    h = np.concatenate([h[..., :1] + bc(hue, c)[..., None], h[..., 1:]], axis=-1)
    return np.concatenate([hsv_to_rgb(h), c[..., 3:]], axis=-1)


register(Op("hue", "color", [F("hue", 0.4)], hydra=True, np=_np_hue, doc="Rotate hue.",
            glsl="""vec3 c = _rgbToHsv(_c0.rgb);
  c.r += hue;
  return vec4(_hsvToRgb(c), _c0.a);"""))


def _np_colorama(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    h = rgb_to_hsv(c[..., :3]) + am
    return np.concatenate([fract(hsv_to_rgb(h)), c[..., 3:]], axis=-1)


register(Op("colorama", "color", [F("amount", 0.005)], hydra=True, np=_np_colorama,
            doc="Shift all of HSV and wrap: the classic Hydra acid colours.",
            glsl="""vec3 c = _rgbToHsv(_c0.rgb);
  c += vec3(amount);
  c = _hsvToRgb(c);
  c = fract(c);
  return vec4(c, _c0.a);"""))


def _np_sum(c, scale, env):
    v = c * (np.asarray(scale, dtype=np.float32)[..., None] if np.ndim(scale) else scale)
    return grey(v.sum(-1), c)


register(Op("sum", "color", [F("scale", 1)], hydra=True, np=_np_sum, doc="Sum of channels as grey.",
            glsl="vec4 v = _c0 * scale; return vec4(vec3(v.r + v.g + v.b + v.a), 1.0);"))


def _mk_channel(ch, idx):
    def fn(c, scale, offset, env):
        return grey(c[..., idx] * scale + offset, c)
    register(Op(ch, "color", [F("scale", 1), F("offset", 0)], hydra=True, np=fn,
                doc=f"The {ch} channel as grey (scale, offset).",
                glsl=f"return vec4(_c0.{ch} * scale + offset);"))


for _ch, _i in (("r", 0), ("g", 1), ("b", 2), ("a", 3)):
    _mk_channel(_ch, _i)


# --- new colour ops ------------------------------------------------------------

def _np_fold(c, gain, bias, env):
    g = np.asarray(gain, dtype=np.float32)[..., None] if np.ndim(gain) else gain
    b = np.asarray(bias, dtype=np.float32)[..., None] if np.ndim(bias) else bias
    v = (c[..., :3] + b) * g
    return np.concatenate([np.abs(np.mod(v - 1.0, 2.0) - 1.0), c[..., 3:]], axis=-1)


register(Op("fold", "color", [F("gain", 2), F("bias", 0)], np=_np_fold,
            doc="Wavefolder: colour above 1 folds back down (gain drives it like a synth folder).",
            glsl="vec3 v = (_c0.rgb + bias)*gain; return vec4(abs(mod(v - 1.0, 2.0) - 1.0), _c0.a);"))


def _np_gamma(c, amount, env):
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    return np.concatenate([np.power(np.maximum(c[..., :3], 0.0), am), c[..., 3:]], axis=-1)


register(Op("gamma", "color", [F("amount", 2.2)], np=_np_gamma, doc="rgb ** amount.",
            glsl="return vec4(pow(max(_c0.rgb, 0.0), vec3(amount)), _c0.a);"))


def _np_levels(c, lo, hi, env):
    l_ = np.asarray(lo, dtype=np.float32)[..., None] if np.ndim(lo) else lo
    h_ = np.asarray(hi, dtype=np.float32)[..., None] if np.ndim(hi) else hi
    return np.concatenate([np.clip((c[..., :3] - l_) / (h_ - l_ + 1e-6), 0.0, 1.0), c[..., 3:]], axis=-1)


register(Op("levels", "color", [F("lo", 0), F("hi", 1)], np=_np_levels, doc="Stretch lo..hi to 0..1 and clip.",
            glsl="return vec4(clamp((_c0.rgb - lo)/(hi - lo + 1e-6), 0.0, 1.0), _c0.a);"))


def _np_dither(c, levels, size, env):
    th = bayer4(np.floor(env.px[..., 0] / size), np.floor(env.px[..., 1] / size))[..., None]
    n = np.maximum(levels - 1.0, 1.0)
    n = np.asarray(n, dtype=np.float32)[..., None] if np.ndim(n) else n
    return np.concatenate([np.floor(c[..., :3] * n + th) / n, c[..., 3:]], axis=-1)


register(Op("dither", "color", [F("levels", 2), F("size", 1)], np=_np_dither,
            doc="Ordered (Bayer 4x4) dithering to `levels` levels per channel; `size` = dot size in pixels.",
            glsl="""float th = _bayer4(floor(gl_FragCoord.xy / size));
  float n = max(levels - 1.0, 1.0);
  return vec4(floor(_c0.rgb*n + th)/n, _c0.a);"""))


def _np_palette(c, r, g, b, phase, frequency, env):
    t = lum(c[..., :3])
    out = []
    for off in (r, g, b):
        out.append(0.5 + 0.5 * np.cos(TAU * (frequency * t + phase + off)))
    return np.concatenate([np.stack(out, axis=-1), c[..., 3:]], axis=-1)


register(Op("palette", "color", [F("r", 0), F("g", 0.33), F("b", 0.67), F("phase", 0), F("frequency", 1)],
            np=_np_palette,
            doc="Map luminance through a cosine palette (Inigo Quilez); r/g/b are per-channel phase offsets.",
            glsl="""float t = _luminance(_c0.rgb);
  vec3 col = 0.5 + 0.5*cos(6.283185307*(frequency*t + phase + vec3(r, g, b)));
  return vec4(col, _c0.a);"""))


def _np_grain(c, amount, rate, size, env):
    n = h31(np.stack([np.floor(env.px[..., 0] / size), np.floor(env.px[..., 1] / size), bc(np.floor(env.time * rate), env.px)], axis=-1))
    am = np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount
    return np.concatenate([c[..., :3] + (n[..., None] - 0.5) * am, c[..., 3:]], axis=-1)


register(Op("grain", "color", [F("amount", 0.1), F("rate", 60), F("size", 1)], np=_np_grain, doc="Film grain; `size` in pixels (Super 8 is coarse: 2-3).",
            glsl="""float n = _h31(vec3(floor(gl_FragCoord.xy / size), floor(time*rate)));
  return vec4(_c0.rgb + (n - 0.5)*amount, _c0.a);"""))


def _np_vignette(c, amount, radius, env):
    p = env.base_st - 0.5
    d = np.sqrt((p * p).sum(-1)) * 2.0
    v = 1.0 - amount * smoothstep(radius, 1.42, d)
    return np.concatenate([c[..., :3] * v[..., None], c[..., 3:]], axis=-1)


register(Op("vignette", "color", [F("amount", 0.6), F("radius", 0.6)], np=_np_vignette, doc="Darken the corners.",
            glsl="""float d = length(gl_FragCoord.xy/resolution - 0.5)*2.0;
  float v = 1.0 - amount*smoothstep(radius, 1.42, d);
  return vec4(_c0.rgb*v, _c0.a);"""))


# =============================================================================
# COMBINE OPS
# =============================================================================

def _amt(amount):
    return np.asarray(amount, dtype=np.float32)[..., None] if np.ndim(amount) else amount


def _np_add(c0, c1, amount, env):
    am = _amt(amount)
    return (c0 + c1) * am + c0 * (1.0 - am)


register(Op("add", "combine", [Input("tex", "chain"), F("amount", 1)], hydra=True, np=_np_add, doc="Add.",
            glsl="return (_c0 + _c1)*amount + _c0*(1.0-amount);"))


def _np_sub(c0, c1, amount, env):
    am = _amt(amount)
    return (c0 - c1) * am + c0 * (1.0 - am)


register(Op("sub", "combine", [Input("tex", "chain"), F("amount", 1)], hydra=True, np=_np_sub, doc="Subtract.",
            glsl="return (_c0 - _c1)*amount + _c0*(1.0-amount);"))


def _np_layer(c0, c1, env):
    a = c1[..., 3:4]
    return np.concatenate([mix(c0[..., :3], c1[..., :3], a), np.clip(c0[..., 3:4] + a, 0.0, 1.0)], axis=-1)


register(Op("layer", "combine", [Input("tex", "chain")], hydra=True, np=_np_layer, doc="Alpha-composite on top.",
            glsl="return vec4(mix(_c0.rgb, _c1.rgb, _c1.a), clamp(_c0.a + _c1.a, 0.0, 1.0));"))


def _np_over(c0, c1, env):
    a0, a1 = np.clip(c0[..., 3:4], 0, 1), np.clip(c1[..., 3:4], 0, 1)
    a = a1 + a0 * (1 - a1)
    num = c1[..., :3] * a1 + c0[..., :3] * a0 * (1 - a1)
    rgb = np.where(a > 0, num / np.where(a > 0, a, 1.0), 0.0)
    return np.concatenate([rgb, a], axis=-1).astype(np.float32)


register(Op("over", "combine", [Input("tex", "chain")], np=_np_over,
            doc="Straight-alpha source-over, correct for transparent layers on transparent layers (pigment kept apart from its paper). Onto an opaque picture Hydra's `layer` gives the same colour.",
            glsl="""float a0 = clamp(_c0.a, 0.0, 1.0), a1 = clamp(_c1.a, 0.0, 1.0);
  float a = a1 + a0 * (1.0 - a1);
  vec3 num = _c1.rgb * a1 + _c0.rgb * a0 * (1.0 - a1);
  return vec4(a > 0.0 ? num / a : vec3(0.0), a);"""))


def _np_blend(c0, c1, amount, env):
    am = _amt(amount)
    return c0 * (1.0 - am) + c1 * am


register(Op("blend", "combine", [Input("tex", "chain"), F("amount", 0.5)], hydra=True, np=_np_blend, doc="Crossfade.",
            glsl="return _c0*(1.0-amount)+_c1*amount;"))


def _np_mult(c0, c1, amount, env):
    am = _amt(amount)
    return c0 * (1.0 - am) + (c0 * c1) * am


register(Op("mult", "combine", [Input("tex", "chain"), F("amount", 1)], hydra=True, np=_np_mult, doc="Multiply.",
            glsl="return _c0*(1.0-amount)+(_c0*_c1)*amount;"))


def _np_diff(c0, c1, env):
    return np.concatenate([np.abs(c0[..., :3] - c1[..., :3]), np.maximum(c0[..., 3:4], c1[..., 3:4])], axis=-1)


register(Op("diff", "combine", [Input("tex", "chain")], hydra=True, np=_np_diff, doc="Absolute difference.",
            glsl="return vec4(abs(_c0.rgb-_c1.rgb), max(_c0.a, _c1.a));"))


def _np_mask(c0, c1, env):
    a = lum(c1[..., :3])[..., None]
    return np.concatenate([c0[..., :3] * a, a], axis=-1)


register(Op("mask", "combine", [Input("tex", "chain")], hydra=True, np=_np_mask, doc="Mask by the other chain's luminance.",
            glsl="float a = _luminance(_c1.rgb); return vec4(_c0.rgb*a, a);"))


# --- new combine ops ----------------------------------------------------------

def _np_lighten(c0, c1, amount, env):
    return mix(c0, np.maximum(c0, c1), _amt(amount))


register(Op("lighten", "combine", [Input("tex", "chain"), F("amount", 1)], np=_np_lighten, doc="Per-channel max.",
            glsl="return mix(_c0, max(_c0, _c1), amount);"))


def _np_darken(c0, c1, amount, env):
    return mix(c0, np.minimum(c0, c1), _amt(amount))


register(Op("darken", "combine", [Input("tex", "chain"), F("amount", 1)], np=_np_darken, doc="Per-channel min.",
            glsl="return mix(_c0, min(_c0, _c1), amount);"))


def _np_screen(c0, c1, amount, env):
    return mix(c0, 1.0 - (1.0 - c0) * (1.0 - c1), _amt(amount))


register(Op("screen", "combine", [Input("tex", "chain"), F("amount", 1)], np=_np_screen, doc="Screen blend.",
            glsl="return mix(_c0, 1.0 - (1.0 - _c0)*(1.0 - _c1), amount);"))


def _np_overlay(c0, c1, amount, env):
    lo = 2.0 * c0 * c1
    hi = 1.0 - 2.0 * (1.0 - c0) * (1.0 - c1)
    return mix(c0, mix(lo, hi, step(0.5, c0)), _amt(amount))


register(Op("overlay", "combine", [Input("tex", "chain"), F("amount", 1)], np=_np_overlay, doc="Overlay blend.",
            glsl="""vec4 lo = 2.0*_c0*_c1;
  vec4 hi = 1.0 - 2.0*(1.0 - _c0)*(1.0 - _c1);
  return mix(_c0, mix(lo, hi, step(0.5, _c0)), amount);"""))


def _np_wipe(c0, c1, amount, softness, angle, env):
    p = env.base_st - 0.5
    x = p[..., 0] * np.cos(angle) + p[..., 1] * np.sin(angle) + 0.5
    m = smoothstep(amount - softness, amount + softness, x)[..., None]
    return mix(c0, c1, m)


register(Op("wipe", "combine", [Input("tex", "chain"), F("amount", 0.5), F("softness", 0.02), F("angle", 0)],
            np=_np_wipe, doc="Spatial crossfade along a direction (amount 0..1 sweeps it).",
            glsl="""vec2 p = gl_FragCoord.xy/resolution - 0.5;
  float x = dot(p, vec2(cos(angle), sin(angle))) + 0.5;
  float m = smoothstep(amount - softness, amount + softness, x);
  return mix(_c0, _c1, m);"""))


# =============================================================================
# MODULATE (combineCoord) OPS
# =============================================================================

def _np_modulateRepeat(st, c0, repeatX, repeatY, offsetX, offsetY, env):
    s = st * np.stack([bc(repeatX, st), bc(repeatY, st)], axis=-1)
    x = s[..., 0] + step(1.0, np.mod(s[..., 1], 2.0)) + c0[..., 0] * offsetX
    y = s[..., 1] + step(1.0, np.mod(x, 2.0)) + c0[..., 1] * offsetY
    return fract(np.stack([x, y], axis=-1))


register(Op("modulateRepeat", "combineCoord",
            [Input("tex", "chain"), F("repeatX", 3), F("repeatY", 3), F("offsetX", 0.5), F("offsetY", 0.5)],
            hydra=True, np=_np_modulateRepeat, doc="Tile with offsets driven by the modulator.",
            glsl="""vec2 st = _st * vec2(repeatX, repeatY);
  st.x += step(1., mod(st.y,2.0)) + _c0.r * offsetX;
  st.y += step(1., mod(st.x,2.0)) + _c0.g * offsetY;
  return fract(st);"""))


def _np_modulateRepeatX(st, c0, reps, offset, env):
    s = st * np.stack([bc(reps, st), ones_like_hw(st)], axis=-1)
    y = s[..., 1] + step(1.0, np.mod(s[..., 0], 2.0)) + c0[..., 0] * offset
    return fract(np.stack([s[..., 0], y], axis=-1))


register(Op("modulateRepeatX", "combineCoord", [Input("tex", "chain"), F("reps", 3), F("offset", 0.5)],
            hydra=True, np=_np_modulateRepeatX, doc="",
            glsl="""vec2 st = _st * vec2(reps, 1.0);
  st.y += step(1., mod(st.x,2.0)) + _c0.r * offset;
  return fract(st);"""))


def _np_modulateRepeatY(st, c0, reps, offset, env):
    s = st * np.stack([ones_like_hw(st), bc(reps, st)], axis=-1)
    x = s[..., 0] + step(1.0, np.mod(s[..., 1], 2.0)) + c0[..., 0] * offset
    return fract(np.stack([x, s[..., 1]], axis=-1))


register(Op("modulateRepeatY", "combineCoord", [Input("tex", "chain"), F("reps", 3), F("offset", 0.5)],
            hydra=True, np=_np_modulateRepeatY, doc="",
            glsl="""vec2 st = _st * vec2(1.0, reps);
  st.x += step(1., mod(st.y,2.0)) + _c0.r * offset;
  return fract(st);"""))


def _np_modulateKaleid(st, c0, nSides, env):
    p = st - 0.5
    r = np.sqrt((p * p).sum(-1))
    a = np.arctan2(p[..., 1], p[..., 0])
    pi = 2.0 * PI
    a = np.mod(a, pi / nSides)
    a = np.abs(a - pi / nSides / 2.0)
    rr = c0[..., 0] + r
    return np.stack([rr * np.cos(a), rr * np.sin(a)], axis=-1)


register(Op("modulateKaleid", "combineCoord", [Input("tex", "chain"), F("nSides", 4)], hydra=True,
            np=_np_modulateKaleid, doc="Kaleidoscope with radius pushed by the modulator.",
            glsl="""vec2 st = _st - 0.5;
  float r = length(st);
  float a = atan(st.y, st.x);
  float pi = 2.*3.1416;
  a = mod(a,pi/nSides);
  a = abs(a-pi/nSides/2.);
  return (_c0.r+r)*vec2(cos(a), sin(a));"""))


def _np_modulateScrollX(st, c0, scrollX, speed, env):
    return fract(np.stack([st[..., 0] + c0[..., 0] * scrollX + env.time * speed, st[..., 1]], axis=-1))


register(Op("modulateScrollX", "combineCoord", [Input("tex", "chain"), F("scrollX", 0.5), F("speed", 0)],
            hydra=True, np=_np_modulateScrollX, doc="",
            glsl="_st.x += _c0.r*scrollX + time*speed; return fract(_st);"))


def _np_modulateScrollY(st, c0, scrollY, speed, env):
    return fract(np.stack([st[..., 0], st[..., 1] + c0[..., 0] * scrollY + env.time * speed], axis=-1))


register(Op("modulateScrollY", "combineCoord", [Input("tex", "chain"), F("scrollY", 0.5), F("speed", 0)],
            hydra=True, np=_np_modulateScrollY, doc="",
            glsl="_st.y += _c0.r*scrollY + time*speed; return fract(_st);"))


def _np_modulate(st, c0, amount, env):
    return st + c0[..., :2] * _amt(amount)


register(Op("modulate", "combineCoord", [Input("tex", "chain"), F("amount", 0.1)], hydra=True, np=_np_modulate,
            doc="Displace coordinates by the modulator's red/green.",
            glsl="return _st + _c0.xy*amount;"))


def _np_modulateScale(st, c0, multiple, offset, env):
    xy = st - 0.5
    xy = xy / np.stack([offset + multiple * c0[..., 0], offset + multiple * c0[..., 1]], axis=-1)
    return xy + 0.5


register(Op("modulateScale", "combineCoord", [Input("tex", "chain"), F("multiple", 1), F("offset", 1)],
            hydra=True, np=_np_modulateScale, doc="",
            glsl="""vec2 xy = _st - vec2(0.5);
  xy*=(1.0/vec2(offset + multiple*_c0.r, offset + multiple*_c0.g));
  xy+=vec2(0.5);
  return xy;"""))


def _np_modulatePixelate(st, c0, multiple, offset, env):
    xy = np.stack([offset + c0[..., 0] * multiple, offset + c0[..., 1] * multiple], axis=-1)
    return (np.floor(st * xy) + 0.5) / xy


register(Op("modulatePixelate", "combineCoord", [Input("tex", "chain"), F("multiple", 10), F("offset", 3)],
            hydra=True, np=_np_modulatePixelate, doc="",
            glsl="""vec2 xy = vec2(offset + _c0.x*multiple, offset + _c0.y*multiple);
  return (floor(_st * xy) + 0.5)/xy;"""))


def _np_modulateRotate(st, c0, multiple, offset, env):
    xy = st - 0.5
    return rotate_xy(xy, offset + c0[..., 0] * multiple) + 0.5


register(Op("modulateRotate", "combineCoord", [Input("tex", "chain"), F("multiple", 1), F("offset", 0)],
            hydra=True, np=_np_modulateRotate, doc="",
            glsl="""vec2 xy = _st - vec2(0.5);
  float angle = offset + _c0.x * multiple;
  xy = mat2(cos(angle),-sin(angle), sin(angle),cos(angle))*xy;
  xy += 0.5;
  return xy;"""))


def _np_modulateHue(st, c0, amount, env):
    d = np.stack([c0[..., 1] - c0[..., 0], c0[..., 2] - c0[..., 1]], axis=-1)
    return st + d * _amt(amount) / env.res


register(Op("modulateHue", "combineCoord", [Input("tex", "chain"), F("amount", 1)], hydra=True, np=_np_modulateHue,
            doc="", glsl="return _st + (vec2(_c0.g - _c0.r, _c0.b - _c0.g) * amount * 1.0/resolution);"))


def _np_displace(st, c0, amount, env):
    return st + (c0[..., :2] - 0.5) * _amt(amount)


register(Op("displace", "combineCoord", [Input("tex", "chain"), F("amount", 0.1)], np=_np_displace,
            doc="Like modulate but centred: 0.5 in the modulator means no displacement.",
            glsl="return _st + (_c0.xy - 0.5)*amount;"))


# =============================================================================
# TEXTURE OPS (neighbourhood / multi-pass)
# =============================================================================
# The chain before the op is rendered to an intermediate texture `_tex`; the op then
# samples it around `_st`.  NumPy twins get the rendered prefix as `tex` (H, W, 4).

def _taps(env, tex, st, offsets):
    return [env.sample_offset(tex, st, o) for o in offsets]


def _np_blur(tex, st, radius, env):
    acc = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            s = env.sample_offset(tex, st, (i * radius, j * radius))
            acc = s if acc is None else acc + s
    return acc / 9.0


register(Op("blur", "texture", [F("radius", 1, scalar=True), I("iterations", 1)], np=_np_blur, iterations=True,
            doc="3x3 box blur with taps `radius` pixels apart; use iterations for a wider blur.",
            glsl="""vec2 px = radius / resolution;
  vec4 sum = vec4(0.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
    sum += texture2D(_tex, fract(_st + vec2(float(i), float(j))*px));
  return sum / 9.0;"""))


def _np_sharpen(tex, st, amount, env):
    c, l, r, u, d = _taps(env, tex, st, [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)])
    out = c * (1.0 + 4.0 * amount) - amount * (l + r + u + d)
    return np.concatenate([out[..., :3], c[..., 3:]], axis=-1)


register(Op("sharpen", "texture", [F("amount", 1, scalar=True)], np=_np_sharpen, doc="Unsharp mask.",
            glsl="""vec2 px = 1.0 / resolution;
  vec4 c = texture2D(_tex, fract(_st));
  vec4 n = texture2D(_tex, fract(_st + vec2(-px.x, 0.0))) + texture2D(_tex, fract(_st + vec2(px.x, 0.0)))
         + texture2D(_tex, fract(_st + vec2(0.0, px.y))) + texture2D(_tex, fract(_st + vec2(0.0, -px.y)));
  vec4 o = c*(1.0 + 4.0*amount) - amount*n;
  return vec4(o.rgb, c.a);"""))


def _np_edge(tex, st, amount, env):
    tl, t, tr, l, r, bl, b, br = [lum(x[..., :3]) for x in _taps(
        env, tex, st, [(-amount, amount), (0, amount), (amount, amount), (-amount, 0), (amount, 0),
                       (-amount, -amount), (0, -amount), (amount, -amount)])]
    gx = -tl - 2 * l - bl + tr + 2 * r + br
    gy = -bl - 2 * b - br + tl + 2 * t + tr
    return grey(np.sqrt(gx * gx + gy * gy), st)


register(Op("edge", "texture", [F("amount", 1, scalar=True)], np=_np_edge, doc="Sobel edge magnitude.",
            glsl="""vec2 px = amount / resolution;
  float tl = _luminance(texture2D(_tex, fract(_st + vec2(-px.x, px.y))).rgb);
  float t  = _luminance(texture2D(_tex, fract(_st + vec2(0.0, px.y))).rgb);
  float tr = _luminance(texture2D(_tex, fract(_st + vec2(px.x, px.y))).rgb);
  float l  = _luminance(texture2D(_tex, fract(_st + vec2(-px.x, 0.0))).rgb);
  float r  = _luminance(texture2D(_tex, fract(_st + vec2(px.x, 0.0))).rgb);
  float bl = _luminance(texture2D(_tex, fract(_st + vec2(-px.x, -px.y))).rgb);
  float b  = _luminance(texture2D(_tex, fract(_st + vec2(0.0, -px.y))).rgb);
  float br = _luminance(texture2D(_tex, fract(_st + vec2(px.x, -px.y))).rgb);
  float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
  float gy = -bl - 2.0*b - br + tl + 2.0*t + tr;
  return vec4(vec3(length(vec2(gx, gy))), 1.0);"""))


def _np_emboss(tex, st, amount, env):
    a, b = _taps(env, tex, st, [(1, 1), (-1, -1)])
    return grey(0.5 + lum((a - b)[..., :3]) * amount, st)


register(Op("emboss", "texture", [F("amount", 2, scalar=True)], np=_np_emboss, doc="Relief shading.",
            glsl="""vec2 px = 1.0 / resolution;
  vec4 d = texture2D(_tex, fract(_st + px)) - texture2D(_tex, fract(_st - px));
  return vec4(vec3(0.5 + _luminance(d.rgb)*amount), 1.0);"""))


def _np_dilate(tex, st, radius, env):
    out = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            s = env.sample_offset(tex, st, (i * radius, j * radius))
            out = s if out is None else np.maximum(out, s)
    return out


register(Op("dilate", "texture", [F("radius", 1, scalar=True), I("iterations", 1)], np=_np_dilate, iterations=True,
            doc="Grow bright regions (3x3 max).",
            glsl="""vec2 px = radius / resolution;
  vec4 m = vec4(0.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
    m = max(m, texture2D(_tex, fract(_st + vec2(float(i), float(j))*px)));
  return m;"""))


def _np_erode(tex, st, radius, env):
    out = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            s = env.sample_offset(tex, st, (i * radius, j * radius))
            out = s if out is None else np.minimum(out, s)
    return out


register(Op("erode", "texture", [F("radius", 1, scalar=True), I("iterations", 1)], np=_np_erode, iterations=True,
            doc="Shrink bright regions (3x3 min).",
            glsl="""vec2 px = radius / resolution;
  vec4 m = vec4(1.0);
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
    m = min(m, texture2D(_tex, fract(_st + vec2(float(i), float(j))*px)));
  return m;"""))


def _np_life(tex, st, threshold, env):
    taps = _taps(env, tex, st, [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
    n = sum(step(threshold, s[..., 0]) for s in taps)
    cur = env.sample(tex, st)
    alive = step(threshold, cur[..., 0])
    nxt = ((np.abs(n - 3.0) < 0.5) | ((alive > 0.5) & (np.abs(n - 2.0) < 0.5))).astype(np.float32)
    trail = np.maximum(cur[..., 1] * 0.9, nxt)
    return np.stack([nxt, trail, alive, ones_like_hw(st)], axis=-1)


register(Op("life", "texture", [F("threshold", 0.5, scalar=True), I("iterations", 1)], np=_np_life, iterations=True,
            doc="One step of Conway's Life on the red channel (red = alive). Green keeps a decaying trail, blue the previous state. Feed it back: src(o0).life().out(o0).",
            glsl="""vec2 px = 1.0 / resolution;
  float n = 0.0;
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
    if (i == 0 && j == 0) continue;
    n += step(threshold, texture2D(_tex, fract(_st + vec2(float(i), float(j))*px)).r);
  }
  vec4 cur = texture2D(_tex, fract(_st));
  float alive = step(threshold, cur.r);
  float next = (abs(n - 3.0) < 0.5 || (alive > 0.5 && abs(n - 2.0) < 0.5)) ? 1.0 : 0.0;
  float trail = max(cur.g*0.9, next);
  return vec4(next, trail, alive, 1.0);"""))


def _np_rd(tex, st, feed, kill, da, db, env):
    def ab(s):
        a, b = s[..., 0], s[..., 1]
        a = np.where(a + b < 0.001, 1.0, a)
        return a, b
    cur = env.sample(tex, st)
    A, B = ab(cur)
    lapA = -A
    lapB = -B
    for (i, j), w in (((-1, 0), 0.2), ((1, 0), 0.2), ((0, -1), 0.2), ((0, 1), 0.2),
                      ((-1, -1), 0.05), ((1, -1), 0.05), ((-1, 1), 0.05), ((1, 1), 0.05)):
        s = env.sample_offset(tex, st, (i, j))
        a, b = ab(s)
        lapA = lapA + w * a
        lapB = lapB + w * b
    abb = A * B * B
    A2 = A + (da * lapA - abb + feed * (1.0 - A))
    B2 = B + (db * lapB + abb - (kill + feed) * B)
    return np.stack([np.clip(A2, 0, 1), np.clip(B2, 0, 1), np.clip(B2, 0, 1), ones_like_hw(st)], axis=-1)


register(Op("rd", "texture", [F("feed", 0.055, scalar=True), F("kill", 0.062, scalar=True),
                              F("da", 1.0, scalar=True), F("db", 0.5, scalar=True), I("iterations", 4)],
            np=_np_rd, iterations=True, compute="rd",
            doc="Gray-Scott reaction-diffusion step: A in red, B in green/blue. Black counts as pure A, so seed B with green and feed it back: src(o0).rd().add(shape().color(0,1,0), 0.1).out(o0).",
            glsl="""vec2 px = 1.0 / resolution;
  vec4 cur = texture2D(_tex, fract(_st));
  float A = cur.r, B = cur.g;
  if (A + B < 0.001) A = 1.0;
  float lapA = -A, lapB = -B;
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
    if (i == 0 && j == 0) continue;
    float w = (i == 0 || j == 0) ? 0.2 : 0.05;
    vec4 s = texture2D(_tex, fract(_st + vec2(float(i), float(j))*px));
    float a = s.r, b = s.g;
    if (a + b < 0.001) a = 1.0;
    lapA += w*a; lapB += w*b;
  }
  float abb = A*B*B;
  float A2 = A + (da*lapA - abb + feed*(1.0 - A));
  float B2 = B + (db*lapB + abb - (kill + feed)*B);
  return vec4(clamp(A2, 0.0, 1.0), clamp(B2, 0.0, 1.0), clamp(B2, 0.0, 1.0), 1.0);"""))


def _np_chroma(tex, st, amount, angle, env):
    d = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * amount
    a = env.sample(tex, st + d)
    c = env.sample(tex, st)
    b = env.sample(tex, st - d)
    return np.stack([a[..., 0], c[..., 1], b[..., 2], c[..., 3]], axis=-1)


register(Op("chroma", "texture", [F("amount", 0.01, scalar=True), F("angle", 0, scalar=True)], np=_np_chroma,
            doc="Chromatic aberration: red and blue sampled on opposite sides.",
            glsl="""vec2 d = amount*vec2(cos(angle), sin(angle));
  vec4 c = texture2D(_tex, fract(_st));
  return vec4(texture2D(_tex, fract(_st + d)).r, c.g, texture2D(_tex, fract(_st - d)).b, c.a);"""))


def _np_pixelsort(tex, st, threshold, axis, env):
    img = tex.copy()
    if int(axis) == 1:
        img = img.transpose(1, 0, 2)
    l = lum(img[..., :3])
    mask = l > threshold
    for y in range(img.shape[0]):
        row = mask[y]
        if not row.any():
            continue
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if e - s > 1:
                order = np.argsort(l[y, s:e])
                img[y, s:e] = img[y, s:e][order]
    if int(axis) == 1:
        img = img.transpose(1, 0, 2)
    return env.sample(img, st)


register(Op("pixelsort", "texture", [F("threshold", 0.5, scalar=True), I("axis", 0)], np=_np_pixelsort, cpu_only=True,
            doc="Sort runs of bright pixels by luminance along rows (axis 0) or columns (axis 1). CPU only.",
            glsl=""))
