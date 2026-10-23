# Portions adapted from thi.ng/umbrella, Apache-2.0 (see NOTICE).
# New renderer and SDF code: hydrapy contributors, AGPL-3.0-or-later.
"""Animated point fields and composable signed distance geometry.

orbitVoronoi's alternating rings adapt the Apache-2.0 thi.ng rotating-voronoi
example (see NOTICE). Its analytic rasterizer is new; it does not reproduce
De/Frag, perform jump flooding, or construct/subdivide polygon meshes.
SDF operations store signed distance in RGB until sdfMask/sdfBands.
"""
from __future__ import annotations

import numpy as np

from .core import Input, Op, register
from .ops import F, TAU, bc, grey, smoothstep


def _np_orbit(st, points, rings, radius, speed, phase, width, mode, env):
    points = int(np.clip(np.floor(points), 3, 20))
    rings = int(np.clip(np.floor(rings), 1, 4))
    q = (st - .5) * np.array([env.res[0] / env.res[1], 1], np.float32)
    first = np.sum(q * q, axis=-1)
    second = np.full_like(first, 1e10)
    identity = np.zeros_like(first)
    radius = max(abs(radius), 1e-4)
    for ring in range(rings):
        angle = phase + env.time * speed * (1 if ring % 2 == 0 else -1) + ring * np.pi
        for i in range(points):
            a = np.float32(angle + TAU * i / points)
            p = np.array([np.cos(a), np.sin(a)], np.float32) * (radius * .5 ** ring)
            d = np.sum((q - p) ** 2, axis=-1)
            closer = d < first
            second = np.where(closer, first, np.minimum(second, d))
            first = np.minimum(first, d)
            identity = np.where(closer, 1 + ring * points + i, identity)
    gap = np.sqrt(second) - np.sqrt(first)
    aa = 1 / env.res[1]
    interior = smoothstep(max(width, 0), max(width, 0) + aa, gap)
    if mode < .5:
        value = (.2 + .8 * np.mod(identity * np.float32(.61803398875), 1)) * interior
    elif mode < 1.5:
        value = 1 - interior
    else:
        value = np.minimum(1, np.sqrt(first) / radius)
    return grey(value, st)


register(Op("orbitVoronoi", "src",
    [F("points", 9, scalar=True), F("rings", 4, scalar=True), F("radius", .42, scalar=True),
     F("speed", .18, scalar=True), F("phase", 0, scalar=True), F("width", .004, scalar=True), F("mode", 0, scalar=True)],
    np=_np_orbit,
    doc="Alternating rotating rings of Voronoi sites. points=3..20 per ring, rings=1..4; speed in radians/second. mode=0 shaded cells, 1 edges, 2 nearest distance. width uses the nearest/second-nearest distance gap (not a geometric polygon inset).",
    glsl="""
  vec2 q = (_st - .5) * vec2(resolution.x / resolution.y, 1.0);
  float first = dot(q,q), second = 1.e10, identity = 0.;
  float count = clamp(floor(points), 3., 20.);
  float nr = clamp(floor(rings), 1., 4.);
  float rad = max(abs(radius), .0001);
  for (int r=0; r<4; r++) {
    if (float(r) >= nr) break;
    float angle = phase + time * speed * (mod(float(r), 2.) < .5 ? 1. : -1.) + float(r)*3.14159265359;
    for (int i=0; i<20; i++) {
      if (float(i) >= count) break;
      float a = angle + 6.28318530718*float(i)/count;
      vec2 p = rad * pow(.5, float(r)) * vec2(cos(a), sin(a));
      float d = dot(q-p, q-p);
      if (d < first) {
        second = first; first = d; identity = 1. + float(r)*count + float(i);
      } else { second = min(second, d); }
    }
  }
  float gap = sqrt(second) - sqrt(first);
  float inside = smoothstep(max(width, 0.), max(width, 0.) + 1./resolution.y, gap);
  float value = mode < .5 ? (.2 + .8 * fract(identity * .61803398875)) * inside
    : (mode < 1.5 ? 1. - inside : min(1., sqrt(first)/rad));
  return vec4(vec3(value), 1.);
"""))


def _q(st, x, y, env):
    return (st - np.stack([bc(x, st), bc(y, st)], axis=-1)) * np.array([env.res[0] / env.res[1], 1], np.float32)


def _np_circle(st, radius, x, y, env):
    return grey(np.linalg.norm(_q(st, x, y, env), axis=-1) - np.abs(radius), st)


register(Op("sdfCircle", "src", [F("radius", .25), F("x", .5), F("y", .5)], np=_np_circle,
    doc="Signed distance to a circle; negative inside. radius is in screen-height units, x/y in normalized screen coordinates. Finish with sdfMask or sdfBands.",
    glsl="vec2 q = (_st - vec2(x,y)) * vec2(resolution.x/resolution.y, 1.); return vec4(vec3(length(q) - abs(radius)), 1.);"))


def _np_box(st, width, height, radius, x, y, env):
    radius = np.clip(radius, 0, np.minimum(np.abs(width), np.abs(height)) * .5)
    q = np.abs(_q(st, x, y, env)) - np.stack([bc(np.abs(width) * .5 - radius, st), bc(np.abs(height) * .5 - radius, st)], axis=-1)
    d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0) - radius
    return grey(d, st)


register(Op("sdfBox", "src", [F("width", .45), F("height", .35), F("radius", .04), F("x", .5), F("y", .5)], np=_np_box,
    doc="Signed distance to a rounded box. Full width/height and corner radius are screen-height units; center x/y is normalized. Finish with sdfMask or sdfBands.",
    glsl="""float r = clamp(radius, 0., .5 * min(abs(width), abs(height)));
  vec2 q = abs((_st-vec2(x,y))*vec2(resolution.x/resolution.y, 1.)) - .5*abs(vec2(width,height)) + r;
  float d = length(max(q,0.)) + min(max(q.x,q.y),0.) - r;
  return vec4(vec3(d),1.);"""))


def _np_union(a, b, smoothness, env):
    k = np.maximum(smoothness, 1e-8)
    h = np.clip(.5 + .5 * (b[..., 0] - a[..., 0]) / k, 0, 1)
    d = b[..., 0] * (1 - h) + a[..., 0] * h - k * h * (1 - h)
    return grey(np.where(np.asarray(smoothness) <= 0, np.minimum(a[..., 0], b[..., 0]), d), a)


register(Op("sdfUnion", "combine", [Input("tex", "chain"), F("smoothness", 0)], np=_np_union,
    doc="Union of signed distance fields; smoothness > 0 rounds the join with a polynomial smooth minimum.",
    glsl="""float a = _c0.r, b = _c1.r, k = max(smoothness, 1.e-8);
  float h = clamp(.5 + .5*(b-a)/k,0.,1.);
  return vec4(vec3(smoothness <= 0. ? min(a,b) : mix(b,a,h)-k*h*(1.-h)),1.);"""))

register(Op("sdfIntersect", "combine", [Input("tex", "chain")],
    np=lambda a, b, env: grey(np.maximum(a[..., 0], b[..., 0]), a),
    doc="Intersection of signed distance fields (maximum).",
    glsl="return vec4(vec3(max(_c0.r,_c1.r)),1.);"))

register(Op("sdfSubtract", "combine", [Input("tex", "chain")],
    np=lambda a, b, env: grey(np.maximum(a[..., 0], -b[..., 0]), a),
    doc="Cut the other signed distance field out of this field.",
    glsl="return vec4(vec3(max(_c0.r,-_c1.r)),1.);"))


def _np_mask(c, offset, softness, env):
    aa = np.maximum(np.abs(softness), .5 / env.res[1])
    return grey(1 - smoothstep(offset - aa, offset + aa, c[..., 0]), c)


register(Op("sdfMask", "color", [F("offset", 0), F("softness", 0)], np=_np_mask,
    doc="Finish an SDF as an opaque black/white mask. Positive offset expands; softness has a half-pixel minimum. Apply palettes/texture masks afterwards.",
    glsl="float aa=max(abs(softness),.5/resolution.y); return vec4(vec3(1.-smoothstep(offset-aa,offset+aa,_c0.r)),1.);"))


def _np_bands(c, frequency, width, phase, env):
    f = np.maximum(np.abs(frequency), 1e-5)
    d = np.abs(np.mod(c[..., 0] * f - phase + .5, 1) - .5)
    aa = np.minimum(.25, .5 * f / env.res[1])
    half = np.clip(width, 0, 1) * .5
    return grey(1 - smoothstep(half - aa, half + aa, d), c)


register(Op("sdfBands", "color", [F("frequency", 20), F("width", .3), F("phase", 0)], np=_np_bands,
    doc="Finish an SDF as repeated contour bands; frequency per height unit, width is the filled fraction, phase shifts the bands. Animate the field before this op for continuous morphs.",
    glsl="""float f=max(abs(frequency),1.e-5), d=abs(fract(_c0.r*f-phase+.5)-.5);
  float aa=min(.25,.5*f/resolution.y), halfWidth=clamp(width,0.,1.)*.5;
  return vec4(vec3(1.-smoothstep(halfWidth-aa,halfWidth+aa,d)),1.);"""))


def _np_metaballs(st, count, radius, spread, smoothness, speed, seed, env):
    q = (st - .5) * np.array([env.res[0] / env.res[1], 1], np.float32)
    d = np.full(st.shape[:2], 1e4, np.float32)
    k = max(smoothness, 1e-8)
    for i in range(int(np.clip(np.floor(count), 1, 16))):
        a = np.float32(i * 2.39996323 + seed)
        t = np.float32(env.time * speed)
        center = spread * np.array([np.sin(t * (1 + i * .07) + a),
                                     np.cos(t * (.83 + i * .05) + a * 1.37)], np.float32)
        b = np.linalg.norm(q - center, axis=-1) - abs(radius)
        h = np.clip(.5 + .5 * (b - d) / k, 0, 1)
        d = b * (1 - h) + d * h - k * h * (1 - h)
    return grey(d, st)


register(Op("sdfMetaballs", "src",
    [F("count", 5, scalar=True), F("radius", .12, scalar=True), F("spread", .28, scalar=True),
     F("smoothness", .12, scalar=True), F("speed", .5, scalar=True), F("seed", 0, scalar=True)],
    np=_np_metaballs,
    doc="Smooth union of 1..16 animated circles, a negative-inside distance bound. Deterministic motion in seconds; radius/spread/smoothness use screen-height units. Finish with sdfMask/sdfBands or derive a normal map.",
    glsl="""
  vec2 q=(_st-.5)*vec2(resolution.x/resolution.y,1.);
  float d=1.e4, k=max(smoothness,1.e-8);
  for(int i=0;i<16;i++) {
    if(float(i)>=clamp(floor(count),1.,16.)) break;
    float a=float(i)*2.39996323+seed, t=time*speed;
    vec2 p=spread*vec2(sin(t*(1.+float(i)*.07)+a),cos(t*(.83+float(i)*.05)+a*1.37));
    float b=length(q-p)-abs(radius), h=clamp(.5+.5*(b-d)/k,0.,1.);
    d=mix(b,d,h)-k*h*(1.-h);
  }
  return vec4(vec3(d),1.);
"""))
