"""Vectorised noise and hash functions matching the GLSL used by the shader backend.

The 3D simplex noise is a direct NumPy transcription of the Ashima / Gustavson
``snoise(vec3)`` that Hydra uses, so ``noise()`` looks the same on both backends.
"""
from __future__ import annotations

import numpy as np


def _mod289(x):
    return x - np.floor(x * (1.0 / 289.0)) * 289.0


def _permute(x):
    return _mod289(((x * 34.0) + 1.0) * x)


def _taylor_inv_sqrt(r):
    return 1.79284291400159 - 0.85373472095314 * r


def snoise3(v: np.ndarray) -> np.ndarray:
    """3D simplex noise. ``v`` has shape (..., 3); returns (...), range about [-1, 1].

    Evaluated in float32 on purpose: the permutation hash is precision-sensitive and
    float32 is what the GPU does, so both backends land on the same noise.
    """
    v = np.asarray(v, dtype=np.float32)
    Cx, Cy = 1.0 / 6.0, 1.0 / 3.0
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    s = (vx + vy + vz) * Cy
    ix, iy, iz = np.floor(vx + s), np.floor(vy + s), np.floor(vz + s)
    t = (ix + iy + iz) * Cx
    x0x, x0y, x0z = vx - ix + t, vy - iy + t, vz - iz + t

    gx = (x0x >= x0y).astype(v.dtype)
    gy = (x0y >= x0z).astype(v.dtype)
    gz = (x0z >= x0x).astype(v.dtype)
    lx, ly, lz = 1.0 - gx, 1.0 - gy, 1.0 - gz
    i1x, i1y, i1z = np.minimum(gx, lz), np.minimum(gy, lx), np.minimum(gz, ly)
    i2x, i2y, i2z = np.maximum(gx, lz), np.maximum(gy, lx), np.maximum(gz, ly)

    x1x, x1y, x1z = x0x - i1x + Cx, x0y - i1y + Cx, x0z - i1z + Cx
    x2x, x2y, x2z = x0x - i2x + Cy, x0y - i2y + Cy, x0z - i2z + Cy
    x3x, x3y, x3z = x0x - 0.5, x0y - 0.5, x0z - 0.5

    ix, iy, iz = _mod289(ix), _mod289(iy), _mod289(iz)
    zero = np.zeros_like(ix)
    one = np.ones_like(ix)
    pz = np.stack([iz + zero, iz + i1z, iz + i2z, iz + one], axis=-1)
    py = np.stack([iy + zero, iy + i1y, iy + i2y, iy + one], axis=-1)
    px = np.stack([ix + zero, ix + i1x, ix + i2x, ix + one], axis=-1)
    p = _permute(_permute(_permute(pz) + py) + px)

    n_ = 1.0 / 7.0
    nsx, nsy, nsz = n_ * 2.0, n_ * 0.5 - 1.0, n_ * 1.0
    j = p - 49.0 * np.floor(p * nsz * nsz)
    x_ = np.floor(j * nsz)
    y_ = np.floor(j - 7.0 * x_)
    x = x_ * nsx + nsy
    y = y_ * nsx + nsy
    h = 1.0 - np.abs(x) - np.abs(y)

    # b0 = (x.x, x.y, y.x, y.y); b1 = (x.z, x.w, y.z, y.w)
    b0 = np.stack([x[..., 0], x[..., 1], y[..., 0], y[..., 1]], axis=-1)
    b1 = np.stack([x[..., 2], x[..., 3], y[..., 2], y[..., 3]], axis=-1)
    s0 = np.floor(b0) * 2.0 + 1.0
    s1 = np.floor(b1) * 2.0 + 1.0
    sh = -(h <= 0.0).astype(v.dtype)
    # a0 = b0.xzyw + s0.xzyw * sh.xxyy
    a0 = np.stack([
        b0[..., 0] + s0[..., 0] * sh[..., 0],
        b0[..., 2] + s0[..., 2] * sh[..., 0],
        b0[..., 1] + s0[..., 1] * sh[..., 1],
        b0[..., 3] + s0[..., 3] * sh[..., 1],
    ], axis=-1)
    a1 = np.stack([
        b1[..., 0] + s1[..., 0] * sh[..., 2],
        b1[..., 2] + s1[..., 2] * sh[..., 2],
        b1[..., 1] + s1[..., 1] * sh[..., 3],
        b1[..., 3] + s1[..., 3] * sh[..., 3],
    ], axis=-1)

    p0 = np.stack([a0[..., 0], a0[..., 1], h[..., 0]], axis=-1)
    p1 = np.stack([a0[..., 2], a0[..., 3], h[..., 1]], axis=-1)
    p2 = np.stack([a1[..., 0], a1[..., 1], h[..., 2]], axis=-1)
    p3 = np.stack([a1[..., 2], a1[..., 3], h[..., 3]], axis=-1)

    norm = _taylor_inv_sqrt(np.stack([
        (p0 * p0).sum(-1), (p1 * p1).sum(-1), (p2 * p2).sum(-1), (p3 * p3).sum(-1)], axis=-1))
    p0 = p0 * norm[..., 0:1]
    p1 = p1 * norm[..., 1:2]
    p2 = p2 * norm[..., 2:3]
    p3 = p3 * norm[..., 3:4]

    d0 = x0x * x0x + x0y * x0y + x0z * x0z
    d1 = x1x * x1x + x1y * x1y + x1z * x1z
    d2 = x2x * x2x + x2y * x2y + x2z * x2z
    d3 = x3x * x3x + x3y * x3y + x3z * x3z
    m = np.maximum(0.6 - np.stack([d0, d1, d2, d3], axis=-1), 0.0)
    m = m * m
    g0 = p0[..., 0] * x0x + p0[..., 1] * x0y + p0[..., 2] * x0z
    g1 = p1[..., 0] * x1x + p1[..., 1] * x1y + p1[..., 2] * x1z
    g2 = p2[..., 0] * x2x + p2[..., 1] * x2y + p2[..., 2] * x2z
    g3 = p3[..., 0] * x3x + p3[..., 1] * x3y + p3[..., 2] * x3z
    return 42.0 * ((m * m) * np.stack([g0, g1, g2, g3], axis=-1)).sum(-1)


def hash21(p: np.ndarray) -> np.ndarray:
    """GLSL-style ``fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453)``. ``p``: (..., 2). float32 like the GPU."""
    p = np.asarray(p, dtype=np.float32)
    d = p[..., 0] * 127.1 + p[..., 1] * 311.7
    return np.mod(np.sin(d) * 43758.5453, 1.0)


def hash31(p: np.ndarray) -> np.ndarray:
    """Hash of (..., 3) -> (...)."""
    p = np.asarray(p, dtype=np.float32)
    d = p[..., 0] * 127.1 + p[..., 1] * 311.7 + p[..., 2] * 74.7
    return np.mod(np.sin(d) * 43758.5453, 1.0)


def hash22(p: np.ndarray) -> np.ndarray:
    """Hash of (..., 2) -> (..., 2), same constants as Hydra's voronoi."""
    p = np.asarray(p, dtype=np.float32)
    d1 = p[..., 0] * 127.1 + p[..., 1] * 311.7
    d2 = p[..., 0] * 269.5 + p[..., 1] * 183.3
    return np.mod(np.stack([np.sin(d1), np.sin(d2)], axis=-1) * 43758.5453, 1.0)


def hash11(x: np.ndarray) -> np.ndarray:
    """Scalar hash in [0, 1)."""
    x = np.asarray(x, dtype=np.float32)
    return np.mod(np.sin(x * 12.9898 + 78.233) * 43758.5453, 1.0)


# --- sin-free hashes (Dave Hoskins) --------------------------------------------
# Only multiplies, adds and fract, so float32 NumPy and the GPU agree bit for bit
# (up to FMA contraction). Used by the new ops; Hydra's own ops keep Hydra's sin hash.

def h11(x) -> np.ndarray:
    p = np.mod(np.asarray(x, dtype=np.float32) * 0.1031, 1.0)
    p = p * (p + 33.33)
    p = p * (p + p)
    return np.mod(p, 1.0)


def h21(p) -> np.ndarray:
    """(..., 2) -> (...)"""
    p = np.asarray(p, dtype=np.float32)
    p3 = np.mod(np.stack([p[..., 0], p[..., 1], p[..., 0]], axis=-1) * 0.1031, 1.0)
    d = p3[..., 0] * (p3[..., 1] + 33.33) + p3[..., 1] * (p3[..., 2] + 33.33) + p3[..., 2] * (p3[..., 0] + 33.33)
    p3 = p3 + d[..., None]
    return np.mod((p3[..., 0] + p3[..., 1]) * p3[..., 2], 1.0)


def h22(p) -> np.ndarray:
    """(..., 2) -> (..., 2)"""
    p = np.asarray(p, dtype=np.float32)
    p3 = np.mod(np.stack([p[..., 0] * 0.1031, p[..., 1] * 0.1030, p[..., 0] * 0.0973], axis=-1), 1.0)
    d = p3[..., 0] * (p3[..., 1] + 33.33) + p3[..., 1] * (p3[..., 2] + 33.33) + p3[..., 2] * (p3[..., 0] + 33.33)
    p3 = p3 + d[..., None]
    return np.mod(np.stack([(p3[..., 0] + p3[..., 1]) * p3[..., 2], (p3[..., 0] + p3[..., 2]) * p3[..., 1]], axis=-1), 1.0)


def h31(p) -> np.ndarray:
    """(..., 3) -> (...)"""
    p3 = np.mod(np.asarray(p, dtype=np.float32) * 0.1031, 1.0)
    d = p3[..., 0] * (p3[..., 2] + 31.32) + p3[..., 1] * (p3[..., 1] + 31.32) + p3[..., 2] * (p3[..., 0] + 31.32)
    p3 = p3 + d[..., None]
    return np.mod((p3[..., 0] + p3[..., 1]) * p3[..., 2], 1.0)


# Same functions in GLSL, prepended to every generated shader.
GLSL_NOISE = r"""
vec3 _mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _permute(vec4 x) { return _mod289(((x*34.0)+1.0)*x); }
vec4 _taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float _noise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = _mod289(i);
  vec4 p = _permute(_permute(_permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = _taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
float _hash21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float _hash31(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
vec2 _hash22(vec2 p) { return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453); }
float _hash11(float x) { return fract(sin(x * 12.9898 + 78.233) * 43758.5453); }
float _h11(float p) { p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
float _h21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 _h22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
float _h31(vec3 p3) { p3 = fract(p3 * .1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float _luminance(vec3 rgb) { const vec3 W = vec3(0.2125, 0.7154, 0.0721); return dot(rgb, W); }
vec3 _rgbToHsv(vec3 c){
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 _hsvToRgb(vec3 c){
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
mat3 _rot3(float rx, float ry, float rz) {
  float cx = cos(rx), sx = sin(rx), cy = cos(ry), sy = sin(ry), cz = cos(rz), sz = sin(rz);
  mat3 Rx = mat3(1.0, 0.0, 0.0,  0.0, cx, sx,  0.0, -sx, cx);
  mat3 Ry = mat3(cy, 0.0, -sy,  0.0, 1.0, 0.0,  sy, 0.0, cy);
  mat3 Rz = mat3(cz, sz, 0.0,  -sz, cz, 0.0,  0.0, 0.0, 1.0);
  return Rz * Ry * Rx;
}
float _bayer4(vec2 px) {
  vec2 p = floor(mod(px, 4.0));
  float a = p.x + p.y * 4.0;
  // 4x4 Bayer matrix, row-major, values 0..15
  float m = 0.0;
  if (a == 0.0) m = 0.0;  else if (a == 1.0) m = 8.0;  else if (a == 2.0) m = 2.0;  else if (a == 3.0) m = 10.0;
  else if (a == 4.0) m = 12.0; else if (a == 5.0) m = 4.0; else if (a == 6.0) m = 14.0; else if (a == 7.0) m = 6.0;
  else if (a == 8.0) m = 3.0;  else if (a == 9.0) m = 11.0; else if (a == 10.0) m = 1.0; else if (a == 11.0) m = 9.0;
  else if (a == 12.0) m = 15.0; else if (a == 13.0) m = 7.0; else if (a == 14.0) m = 13.0; else m = 5.0;
  return (m + 0.5) / 16.0;
}
float _segd(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
  return length(pa - ba * h);
}
"""


def segd(p: np.ndarray, a, b) -> np.ndarray:
    """Distance from points p (..., 2) to the segment a-b (same maths as the GLSL `_segd`)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    pa = p - a
    ba = b - a
    h = np.clip((pa * ba).sum(-1) / max(float((ba * ba).sum()), 1e-9), 0.0, 1.0)
    d = pa - ba * h[..., None]
    return np.sqrt((d * d).sum(-1))

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32)


def bayer4(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Ordered-dither threshold in (0,1) for integer pixel coords (same matrix as GLSL)."""
    ix = np.mod(np.floor(px), 4).astype(int)
    iy = np.mod(np.floor(py), 4).astype(int)
    return (BAYER4[iy, ix] + 0.5) / 16.0


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) RGB -> HSV, matching the GLSL swizzle version above."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g))
    cond = (g >= b)
    px = np.where(cond, g, b)
    py = np.where(cond, b, g)
    pz = np.where(cond, 0.0, -1.0)
    pw = np.where(cond, -1.0 / 3.0, 2.0 / 3.0)
    # q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r))
    cond2 = (r >= px)
    qx = np.where(cond2, r, px)
    qy = np.where(cond2, py, py)
    qz = np.where(cond2, pz, pw)
    qw = np.where(cond2, px, r)
    d = qx - np.minimum(qw, qy)
    e = 1.0e-10
    h = np.abs(qz + (qw - qy) / (6.0 * d + e))
    s = d / (qx + e)
    return np.stack([h, s, qx], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    p = np.abs(np.mod(h[..., None] + np.array([1.0, 2.0 / 3.0, 1.0 / 3.0]), 1.0) * 6.0 - 3.0)
    return v[..., None] * (1.0 + (np.clip(p - 1.0, 0.0, 1.0) - 1.0) * s[..., None])
