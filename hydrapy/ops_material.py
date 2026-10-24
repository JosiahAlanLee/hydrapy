"""Height-field materials, using straight alpha and screen-height units.

Independent implementations of central-difference normals, Blinn lighting,
and environment lookup. See docs/DEMOSCENE.md for references and conventions.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .ops import F, bc


def _normalize(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(length > 1e-8, v / np.maximum(length, 1e-8), np.array([0, 0, 1], np.float32))


def _np_normal(tex, st, strength, radius, env):
    radius = max(abs(radius), .5)
    gx = (env.sample_offset(tex, st, (radius, 0))[..., 0] - env.sample_offset(tex, st, (-radius, 0))[..., 0]) * env.res[1] / (2 * radius)
    gy = (env.sample_offset(tex, st, (0, radius))[..., 0] - env.sample_offset(tex, st, (0, -radius))[..., 0]) * env.res[1] / (2 * radius)
    n = _normalize(np.stack([-gx * strength, -gy * strength, np.ones_like(gx)], axis=-1))
    return np.concatenate([n * .5 + .5, env.sample(tex, st)[..., 3:4]], axis=-1)


register(Op("normalMap", "texture", [F("strength", .03), F("radius", 1, scalar=True)],
    np=_np_normal,
    doc="Red channel as height; central-difference normals encoded in RGB as 0.5+0.5*n, +Y up. strength is height per screen height; radius is the sampling distance in pixels (minimum 0.5). Preserves alpha; sampling wraps.",
    glsl="""
  float r = max(abs(radius), .5);
  vec2 dx = vec2(r/resolution.x, 0.), dy = vec2(0., r/resolution.y);
  float gx = (texture2D(_tex, fract(_st+dx)).r-texture2D(_tex, fract(_st-dx)).r)*resolution.y/(2.*r);
  float gy = (texture2D(_tex, fract(_st+dy)).r-texture2D(_tex, fract(_st-dy)).r)*resolution.y/(2.*r);
  vec3 n = normalize(vec3(-gx*strength, -gy*strength, 1.));
  return vec4(.5+.5*n, texture2D(_tex, fract(_st)).a);
"""))


def _np_relight(c, azimuth, elevation, ambient, diffuse, specular, shininess, env):
    n = _normalize(c[..., :3] * 2 - 1)
    el = np.clip(elevation, -np.pi / 2, np.pi / 2)
    light = np.stack([bc(np.cos(el) * np.cos(azimuth), c),
                      bc(np.cos(el) * np.sin(azimuth), c), bc(np.sin(el), c)], axis=-1)
    half = _normalize(light + np.array([0, 0, 1], np.float32))
    ndotl = np.maximum(np.sum(n * light, axis=-1), 0)
    spec = np.maximum(np.sum(n * half, axis=-1), 0) ** np.maximum(shininess, 1)
    v = np.maximum(ambient, 0) + np.maximum(diffuse, 0) * ndotl + np.maximum(specular, 0) * spec * (ndotl > 0)
    return np.concatenate([np.repeat(v[..., None], 3, axis=-1), c[..., 3:4]], axis=-1)


_DECODE = "vec3 n=2.*_c0.rgb-1.; n=length(n)>1.e-8?normalize(n):vec3(0.,0.,1.);"

register(Op("relight", "color",
    [F("azimuth", .8), F("elevation", .7), F("ambient", .18), F("diffuse", .65), F("specular", .35), F("shininess", 32)],
    np=_np_relight,
    doc="Light an encoded normal map with Lambert diffuse + Blinn specular. Angles in radians; elevation=pi/2 faces the viewer. Returns HDR grayscale lighting with the input alpha; multiply by pigment separately.",
    glsl=_DECODE + """
  float el=clamp(elevation,-1.57079632679,1.57079632679);
  vec3 light=vec3(cos(el)*cos(azimuth),cos(el)*sin(azimuth),sin(el));
  vec3 h=light+vec3(0.,0.,1.); h=length(h)>1.e-8?normalize(h):vec3(0.,0.,1.);
  float lam=max(dot(n,light),0.);
  float spec=pow(max(dot(n,h),0.),max(shininess,1.))*(lam>0.?1.:0.);
  float v=max(ambient,0.)+max(diffuse,0.)*lam+max(specular,0.)*spec;
  return vec4(vec3(v),_c0.a);
"""))


def _np_env(st, c, rotation, env):
    n = _normalize(c[..., :3] * 2 - 1)
    r = 2 * n[..., 2:3] * n - np.array([0, 0, 1], np.float32)
    u = np.arctan2(r[..., 2], r[..., 0]) / (2 * np.pi) + .5 + rotation
    # Longitude is undefined at the poles. Choose one consistently on CPU/GPU.
    u = np.where(r[..., 0] ** 2 + r[..., 2] ** 2 < 1e-12, .5 + rotation, u)
    v = np.arcsin(np.clip(r[..., 1], -1, 1)) / np.pi + .5
    return np.stack([u, np.clip(v, .5 / env.res[1], 1 - .5 / env.res[1])], axis=-1)


register(Op("envMap", "combineCoord", [Input("normals", "chain"), F("rotation", 0)],
    np=_np_env,
    doc="Sample this chain as a latitude/longitude environment reflected by encoded normals. rotation is turns. A view-facing normal samples (0.75,0.5). Set coverage separately with alpha(normals.a()).",
    glsl=_DECODE + """
  vec3 r=2.*n.z*n-vec3(0.,0.,1.);
  float u=dot(r.xz,r.xz)<1.e-12?.5:atan(r.z,r.x)/6.28318530718+.5;
  float v=asin(clamp(r.y,-1.,1.))/3.14159265359+.5;
  return vec2(u+rotation,clamp(v,.5/resolution.y,1.-.5/resolution.y));
"""))


def _np_refract(st, c, amount, env):
    n = _normalize(c[..., :3] * 2 - 1)
    return st + n[..., :2] * bc(amount, st)[..., None] * np.array([env.res[1] / env.res[0], 1], np.float32)


register(Op("refractMap", "combineCoord", [Input("normals", "chain"), F("amount", .05)],
    np=_np_refract,
    doc="Screen-space glass distortion from encoded normals; amount in screen-height units. This is a normal-driven UV offset, not physical ray refraction. Zero amount is identity.",
    glsl=_DECODE + "return _st+n.xy*amount*vec2(resolution.y/resolution.x,1.);"))
