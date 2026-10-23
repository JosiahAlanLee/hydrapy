"""The p5.js WEBGL move: the chain so far becomes a texture on a plane, sphere or cylinder
seen by a perspective camera.  Texture ops, so they work on both backends and compile to a
pass that renders the chain, then a pass that ray-casts it.

    osc(20).kaleid(4).plane3d(rotX=0.6, rotY=Time() * 0.3)       # the canvas tilted in space
    noise(4).sphere3d(rotY=Time() * 0.5, shade=0.6)             # a textured globe
    src(o0).cylinder3d(rotX=0.3, rotY=Time(), wrap=1)           # a drum

Units: the camera sits at the origin looking down -z with a ``fov`` degree vertical field
of view.  ``distance`` = 1 puts an unrotated plane exactly filling the frame; the sphere's
``radius`` and the cylinder's ``radius``/``height`` are in the same units (a radius of 0.5
is half the frame height at distance 1).  Outside the object the result is transparent
black, so ``.layer()`` it over something or ``.add()`` a background.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .ops import F, bc

# camera ray and object-space transform shared by the three ops (GLSL)
_RAY = """
  float aspect = resolution.x / resolution.y;
  float f = 1.0 / tan(radians(fov) * 0.5);
  vec3 rd = normalize(vec3((_st.x - 0.5) * 2.0 * aspect, (_st.y - 0.5) * 2.0, -f));
  vec3 c = vec3(0.0, 0.0, -distance * f * 0.5);
  mat3 R = _rot3(rotX, rotY, rotZ);
  mat3 Rt = mat3(R[0][0], R[1][0], R[2][0],  R[0][1], R[1][1], R[2][1],  R[0][2], R[1][2], R[2][2]);
  vec3 ro = Rt * (-c);          // camera in object space
  vec3 rdo = Rt * rd;           // ray direction in object space
  vec3 L = normalize(vec3(0.4, 0.6, 1.0));
"""


def _rot3(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def _ray(st, env, rotX, rotY, rotZ, distance, fov):
    """Camera ray in object space: returns (ro (3,), rdo (H,W,3), R (3,3), aspect, f)."""
    aspect = env.res[0] / env.res[1]
    f = 1.0 / np.tan(np.radians(fov) * 0.5)
    rd = np.stack([(st[..., 0] - 0.5) * 2.0 * aspect, (st[..., 1] - 0.5) * 2.0, np.full(st.shape[:2], -f, np.float32)], axis=-1)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    c = np.array([0.0, 0.0, -distance * f * 0.5], dtype=np.float32)
    R = _rot3(float(rotX), float(rotY), float(rotZ))
    ro = R.T @ (-c)
    rdo = rd @ R            # == R^T rd for each pixel
    return ro.astype(np.float32), rdo.astype(np.float32), R, aspect, f


_LIGHT = np.array([0.4, 0.6, 1.0], dtype=np.float32) / np.linalg.norm([0.4, 0.6, 1.0])


def _np_plane3d(tex, st, rotX, rotY, rotZ, distance, fov, wrap, shade, fade, env):
    ro, rdo, R, aspect, f = _ray(st, env, rotX, rotY, rotZ, distance, fov)
    denom = rdo[..., 2]
    t = np.where(np.abs(denom) > 1e-6, -ro[2] / np.where(np.abs(denom) > 1e-6, denom, 1.0), -1.0)
    p = ro[None, None, :] + rdo * t[..., None]
    uv = np.stack([p[..., 0] / aspect + 0.5, p[..., 1] + 0.5], axis=-1)
    inside = (t > 0)
    if wrap < 0.5:
        inside &= (uv[..., 0] >= 0) & (uv[..., 0] <= 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= 1)
    n_cam = R @ np.array([0, 0, 1], np.float32)
    lam = 1.0 - shade * (1.0 - abs(float(n_cam @ _LIGHT)))
    fog = np.exp(-np.maximum(t, 0.0) * fade)[..., None]
    out = env.sample(tex, uv) * lam * np.concatenate([fog, fog, fog, np.ones_like(fog)], -1)
    out = out * inside[..., None]
    return np.concatenate([out[..., :3], inside[..., None].astype(np.float32) * out[..., 3:4]], axis=-1)


register(Op("plane3d", "texture", [F("rotX", 0.5, scalar=True), F("rotY", 0, scalar=True), F("rotZ", 0, scalar=True),
                                   F("distance", 1, scalar=True), F("fov", 60, scalar=True), F("wrap", 0, scalar=True),
                                   F("shade", 0, scalar=True), F("fade", 0, scalar=True)],
            np=_np_plane3d,
            doc="The chain as a textured plane in perspective (rotations in radians). wrap=1 tiles it to the horizon (the Hydra floor); shade darkens it by its angle to the light; fade darkens it with distance (fog).",
            glsl=_RAY + """
  float denom = rdo.z;
  if (abs(denom) < 1e-6) return vec4(0.0);
  float t = -ro.z / denom;
  if (t <= 0.0) return vec4(0.0);
  vec3 p = ro + rdo * t;
  vec2 uv = vec2(p.x / aspect + 0.5, p.y + 0.5);
  float lam = (1.0 - shade * (1.0 - abs(dot(R * vec3(0.0, 0.0, 1.0), L)))) * exp(-t * fade);
  if (wrap > 0.5) return texture2D(_tex, fract(uv)) * vec4(vec3(lam), 1.0);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
  return texture2D(_tex, uv) * vec4(vec3(lam), 1.0);"""))


def _np_sphere3d(tex, st, rotX, rotY, rotZ, radius, distance, fov, shade, env):
    ro, rdo, R, aspect, f = _ray(st, env, rotX, rotY, rotZ, distance, fov)
    b = (rdo * ro[None, None, :]).sum(-1)
    cc = float(ro @ ro) - radius * radius
    disc = b * b - cc
    hit = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    hit &= t > 0
    p = ro[None, None, :] + rdo * t[..., None]
    n = p / max(radius, 1e-6)
    u = np.arctan2(n[..., 2], n[..., 0]) / (2 * np.pi) + 0.5
    v = np.arcsin(np.clip(n[..., 1], -1, 1)) / np.pi + 0.5
    n_cam = n @ R.T
    lam = 1.0 - shade * (1.0 - np.clip(n_cam @ _LIGHT, 0, 1))
    out = env.sample(tex, np.stack([u, v], axis=-1)) * np.concatenate([np.repeat(lam[..., None], 3, -1), np.ones_like(lam[..., None])], -1)
    out = out * hit[..., None]
    return out.astype(np.float32)


register(Op("sphere3d", "texture", [F("rotX", 0, scalar=True), F("rotY", 0, scalar=True), F("rotZ", 0, scalar=True),
                                    F("radius", 0.45, scalar=True), F("distance", 1, scalar=True), F("fov", 60, scalar=True),
                                    F("shade", 0.5, scalar=True)],
            np=_np_sphere3d,
            doc="The chain wrapped around a sphere (x = longitude, y = latitude), lit from the upper right by `shade`.",
            glsl=_RAY + """
  float b = dot(rdo, ro);
  float cc = dot(ro, ro) - radius * radius;
  float disc = b * b - cc;
  if (disc <= 0.0) return vec4(0.0);
  float t = -b - sqrt(disc);
  if (t <= 0.0) return vec4(0.0);
  vec3 p = ro + rdo * t;
  vec3 n = p / radius;
  float u = atan(n.z, n.x) / 6.283185307 + 0.5;
  float v = asin(clamp(n.y, -1.0, 1.0)) / 3.141592654 + 0.5;
  float lam = 1.0 - shade * (1.0 - clamp(dot(R * n, L), 0.0, 1.0));
  return texture2D(_tex, fract(vec2(u, v))) * vec4(vec3(lam), 1.0);"""))


def _np_cylinder3d(tex, st, rotX, rotY, rotZ, radius, height, distance, fov, wrap, shade, env):
    ro, rdo, R, aspect, f = _ray(st, env, rotX, rotY, rotZ, distance, fov)
    a = rdo[..., 0] ** 2 + rdo[..., 2] ** 2
    b = rdo[..., 0] * ro[0] + rdo[..., 2] * ro[2]
    cc = ro[0] ** 2 + ro[2] ** 2 - radius * radius
    disc = b * b - a * cc
    hit = (disc > 0) & (a > 1e-8)
    sq = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(a > 1e-8, a, 1.0)
    t0 = (-b - sq) / a_safe
    t1 = (-b + sq) / a_safe
    y0 = ro[1] + rdo[..., 1] * t0
    y1 = ro[1] + rdo[..., 1] * t1
    ok0 = (t0 > 0) & (np.abs(y0) <= height * 0.5)
    ok1 = (t1 > 0) & (np.abs(y1) <= height * 0.5)
    t = np.where(ok0, t0, t1)
    hit &= ok0 | ok1
    p = ro[None, None, :] + rdo * t[..., None]
    n = np.stack([p[..., 0], np.zeros_like(p[..., 0]), p[..., 2]], axis=-1) / max(radius, 1e-6)
    u = np.arctan2(p[..., 2], p[..., 0]) / (2 * np.pi) + 0.5
    v = p[..., 1] / max(height, 1e-6) + 0.5
    n_cam = n @ R.T
    facing = np.where(ok0, 1.0, -1.0)
    lam = 1.0 - shade * (1.0 - np.clip((n_cam @ _LIGHT) * facing, 0, 1))
    uv = np.stack([u, v], axis=-1)
    out = env.sample(tex, uv) * np.concatenate([np.repeat(lam[..., None], 3, -1), np.ones_like(lam[..., None])], -1)
    out = out * hit[..., None]
    return out.astype(np.float32)


register(Op("cylinder3d", "texture", [F("rotX", 0.3, scalar=True), F("rotY", 0, scalar=True), F("rotZ", 0, scalar=True),
                                      F("radius", 0.4, scalar=True), F("height", 0.9, scalar=True), F("distance", 1, scalar=True),
                                      F("fov", 60, scalar=True), F("wrap", 0, scalar=True), F("shade", 0.5, scalar=True)],
            np=_np_cylinder3d,
            doc="The chain wrapped around an open cylinder (x goes around, y along the axis); you see the inside through the open ends.",
            glsl=_RAY + """
  float a = rdo.x * rdo.x + rdo.z * rdo.z;
  float b = rdo.x * ro.x + rdo.z * ro.z;
  float cc = ro.x * ro.x + ro.z * ro.z - radius * radius;
  float disc = b * b - a * cc;
  if (disc <= 0.0 || a < 1e-8) return vec4(0.0);
  float sq = sqrt(disc);
  float t0 = (-b - sq) / a, t1 = (-b + sq) / a;
  float y0 = ro.y + rdo.y * t0, y1 = ro.y + rdo.y * t1;
  bool ok0 = t0 > 0.0 && abs(y0) <= height * 0.5;
  bool ok1 = t1 > 0.0 && abs(y1) <= height * 0.5;
  if (!ok0 && !ok1) return vec4(0.0);
  float t = ok0 ? t0 : t1;
  vec3 p = ro + rdo * t;
  vec3 n = vec3(p.x, 0.0, p.z) / radius;
  float u = atan(p.z, p.x) / 6.283185307 + 0.5;
  float v = p.y / height + 0.5;
  float facing = ok0 ? 1.0 : -1.0;
  float lam = 1.0 - shade * (1.0 - clamp(dot(R * n, L) * facing, 0.0, 1.0));
  return texture2D(_tex, fract(vec2(u, v))) * vec4(vec3(lam), 1.0);"""))
