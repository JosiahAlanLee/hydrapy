"""Composable distance bounds, compiled to a bounded fragment-shader marcher.

Independent implementations of standard SDF formulae; no third-party shader
source is embedded. See docs/DEMOSCENE.md for references and limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

import numpy as np

from .core import Chain, Op, Transform, normalize_args
from .ops import F
from .ops_3d import _rot3
from .signals import Signal


@dataclass(frozen=True)
class SDF3:
    """Immutable 3D field. Parameters accept numbers or scalar Signal objects.

    box() takes half-extents; translate/rotate/scale act on the object.
    Radius/extents are made positive during evaluation. All lengths use the
    same world units. A uniform scale preserves the distance bound.
    """
    kind: str
    params: tuple = ()
    children: tuple = ()
    iterations: int = 0

    def __post_init__(self):
        arity = {"sphere": (1, 0), "box": (3, 0), "torus": (2, 0),
                 "translate": (3, 1), "rotate": (3, 1), "scale": (1, 1),
                 "union": (1, 2), "intersect": (0, 2), "subtract": (0, 2), "fold": (5, 1)}
        if self.kind not in arity or (len(self.params), len(self.children)) != arity[self.kind]:
            raise ValueError("invalid SDF3 node")
        if not all(isinstance(c, SDF3) for c in self.children):
            raise TypeError("SDF3 children must be SDF3 fields")
        for value in self.params:
            if not isinstance(value, Signal) and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("SDF3 parameters must be finite numbers or scalar Signals")
        if self.kind == "fold" and (int(self.iterations) != self.iterations or not 1 <= self.iterations <= 10):
            raise ValueError("fold iterations must be an integer in 1..10")

    @staticmethod
    def sphere(radius=.6): return SDF3("sphere", (radius,))

    @staticmethod
    def box(x=.5, y=.5, z=.5): return SDF3("box", (x, y, z))

    @staticmethod
    def torus(major=.65, minor=.18): return SDF3("torus", (major, minor))

    def translate(self, x=0, y=0, z=0): return SDF3("translate", (x, y, z), (self,))
    def rotate(self, x=0, y=0, z=0): return SDF3("rotate", (x, y, z), (self,))
    def scale(self, amount=1): return SDF3("scale", (amount,), (self,))
    def union(self, other, smoothness=0): return SDF3("union", (smoothness,), (self, other))
    def intersect(self, other): return SDF3("intersect", (), (self, other))
    def subtract(self, other): return SDF3("subtract", (), (self, other))

    def fold(self, iterations=4, scale=2, x=.7, y=.7, z=.7, angle=.2):
        """Recursive abs/rotate/scale/translate folding; scale clamps above 1.

        A kaleidoscopic IFS distance bound with a user-chosen terminal shape.
        Iteration count is structural; other inputs can animate continuously.
        """
        return SDF3("fold", (scale, x, y, z, angle), (self,), iterations)

    def distance(self, points, ctx=None):
        """Evaluate the field on an array of (...,3) points (for geometry tools)."""
        from .core import Ctx
        p = np.asarray(points, np.float32)
        if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
            raise ValueError("points must be finite (...,3) coordinates")
        plan, values = _flatten(self)
        ctx = ctx or Ctx()
        args = [v(ctx) if isinstance(v, Signal) else v for v in values]
        return _distance(plan, p, args)


def _flatten(field):
    values = []
    def visit(node):
        indices = tuple(range(len(values), len(values) + len(node.params)))
        values.extend(node.params)
        return node.kind, indices, tuple(visit(c) for c in node.children), node.iterations
    plan = visit(field)
    return plan, values


def _distance(plan, p, args):
    kind, indices, children, iterations = plan
    a = [np.float32(args[i]) for i in indices]
    if kind == "sphere": return np.linalg.norm(p, axis=-1) - abs(a[0])
    if kind == "box":
        q = np.abs(p) - np.abs(a)
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
    if kind == "torus":
        q = np.stack([np.linalg.norm(p[..., [0, 2]], axis=-1) - abs(a[0]), p[..., 1]], axis=-1)
        return np.linalg.norm(q, axis=-1) - abs(a[1])
    if kind == "translate": return _distance(children[0], p - a, args)
    if kind == "rotate": return _distance(children[0], p @ _rot3(*a), args)
    if kind == "scale":
        s = max(abs(a[0]), 1e-4)
        return s * _distance(children[0], p / s, args)
    if kind == "fold":
        s = max(abs(a[0]), np.float32(1.001))
        q = p.copy()
        rot = _rot3(0, 0, a[4])
        factor = np.float32(1)
        for _ in range(iterations):
            q = (np.abs(q) @ rot.T) * s - a[1:4]
            factor *= s
        return _distance(children[0], q, args) / factor
    d, e = [_distance(c, p, args) for c in children]
    if kind == "intersect": return np.maximum(d, e)
    if kind == "subtract": return np.maximum(d, -e)
    k = max(a[0], 1e-8)
    h = np.clip(.5 + .5 * (e - d) / k, 0, 1)
    return e * (1-h) + d*h - k*h*(1-h)


def _bound(plan, args):
    """Radius R such that distance(p) >= length(p) - R.

    This bounds the epsilon shell as well as the zero surface, including
    folded fields and smooth unions. It is only used to reject certain misses;
    surviving rays keep their original start and marching budget.
    """
    kind, indices, children, iterations = plan
    a = [float(np.float32(args[i])) for i in indices]
    b = [_bound(c, args) for c in children]
    if kind == "sphere": return abs(a[0])
    if kind == "box": return math.sqrt(sum(v*v for v in a))
    if kind == "torus": return abs(a[0]) + abs(a[1])
    if kind == "translate": return b[0] + math.sqrt(sum(v*v for v in a))
    if kind == "rotate": return b[0]
    if kind == "scale": return b[0] * max(abs(a[0]), 1e-4)
    if kind == "fold":
        radius, shift = b[0], math.sqrt(sum(v*v for v in a[1:4]))
        for _ in range(iterations):
            radius = (radius + shift) / max(abs(a[0]), 1.001)
        return radius
    if kind == "intersect": return min(b)
    if kind == "subtract": return b[0]
    return max(b) + max(a[0], 1e-8) * .25


def _shader_bound(plan):
    kind, indices, children, iterations = plan
    a = [f"v{i}" for i in indices]
    b = [_shader_bound(c) for c in children]
    if kind == "sphere": return f"abs({a[0]})"
    if kind == "box": return f"length(vec3({','.join(a)}))"
    if kind == "torus": return f"(abs({a[0]})+abs({a[1]}))"
    if kind == "translate": return f"({b[0]}+length(vec3({','.join(a)})))"
    if kind == "rotate": return b[0]
    if kind == "scale": return f"({b[0]}*max(abs({a[0]}),.0001))"
    if kind == "fold":
        radius = b[0]
        for _ in range(iterations):
            radius = f"(({radius}+length(vec3({','.join(a[1:4])})))/max(abs({a[0]}),1.001))"
        return radius
    if kind == "intersect": return f"min({b[0]},{b[1]})"
    if kind == "subtract": return b[0]
    return f"(max({b[0]},{b[1]})+max({a[0]},1.e-8)*.25)"


_CAMERA = [("rotX", .25), ("rotY", .3), ("rotZ", 0), ("distance", 3), ("fov", 45),
           ("steps", 96), ("epsilon", .001), ("far", 10), ("mode", 0)]


def _np_march(plan, count, st, args, env):
    field_args = args[:count]
    rx, ry, rz, distance, fov, steps, epsilon, far, mode = args[count:]
    eps = max(abs(epsilon), 1e-5)
    far = max(abs(far), eps)
    f = np.tan(np.radians(np.clip(fov, 1, 178)) * .5)
    q = (st-.5) * 2 * np.array([env.res[0]/env.res[1], 1], np.float32)
    rd = np.concatenate([q*f, -np.ones(st.shape[:2]+(1,), np.float32)], axis=-1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    rotation = _rot3(rx, ry, rz)
    ro = np.array([0, 0, distance], np.float32) @ rotation
    rd = rd @ rotation
    shape = st.shape[:2]
    rd = rd.reshape(-1, 3)
    t = np.zeros(len(rd), np.float32)
    hit = np.zeros(len(rd), bool)
    radius = _bound(plan, field_args)
    radius += eps + 1e-4 * max(1, radius)  # account for float32 transforms
    closest = np.clip(-(rd @ ro), 0, far)
    active = np.flatnonzero(np.linalg.norm(ro + rd * closest[:, None], axis=-1) <= radius)
    for _ in range(int(np.clip(np.floor(steps), 1, 256))):
        if not len(active):
            break
        d = _distance(plan, ro + rd[active] * t[active, None], field_args)
        arrived = np.abs(d) <= eps
        hit[active[arrived]] = True
        active, d = active[~arrived], d[~arrived]
        t[active] += np.maximum(np.abs(d) * .85, eps * .25)
        active = active[t[active] <= far]
    rgb = np.zeros((len(rd), 3), np.float32)
    if mode >= 1.5:
        rgb[hit] = np.clip(t[hit] / far, 0, 1)[:, None]
    elif np.any(hit):
        pos = ro + rd[hit] * t[hit, None]
        grad = []
        for axis in np.eye(3, dtype=np.float32) * eps:
            grad.append(_distance(plan, pos+axis, field_args) - _distance(plan, pos-axis, field_args))
        n = np.stack(grad, axis=-1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        n = n @ rotation.T
        if mode >= .5:
            rgb[hit] = .5 + .5*n
        else:
            light = np.array([.4, .6, 1], np.float32); light /= np.linalg.norm(light)
            half = light + np.array([0, 0, 1], np.float32); half /= np.linalg.norm(half)
            lam = np.maximum(n @ light, 0)
            shade = .15 + .7*lam + .25*np.maximum(n @ half, 0)**32*(lam > 0)
            rgb[hit] = shade[..., None]
    return np.concatenate([rgb, hit[..., None]], axis=-1).reshape(*shape, 4).astype(np.float32)


def _shader_map(plan, name, count):
    lines = []
    serial = [0]
    def emit(node, point):
        kind, indices, children, iterations = node
        a = [f"v{i}" for i in indices]
        i = serial[0]; serial[0] += 1
        d, q = f"d{i}", f"q{i}"
        if kind == "sphere":
            lines.append(f"float {d}=length({point})-abs({a[0]});")
        elif kind == "box":
            lines.append(f"vec3 {q}=abs({point})-abs(vec3({','.join(a)}));")
            lines.append(f"float {d}=length(max({q},0.))+min(max({q}.x,max({q}.y,{q}.z)),0.);")
        elif kind == "torus":
            lines.append(f"float {d}=length(vec2(length({point}.xz)-abs({a[0]}),{point}.y))-abs({a[1]});")
        elif kind in ("translate", "rotate", "scale", "fold"):
            if kind == "translate": lines.append(f"vec3 {q}={point}-vec3({','.join(a)});")
            if kind == "rotate":
                lines.append(f"mat3 r{i}=_rot3({','.join(a)}); vec3 {q}=vec3(dot({point},r{i}[0]),dot({point},r{i}[1]),dot({point},r{i}[2]));")
            if kind == "scale": lines.append(f"float s{i}=max(abs({a[0]}),.0001); vec3 {q}={point}/s{i};")
            if kind == "fold":
                lines.append(f"float s{i}=max(abs({a[0]}),1.001), f{i}=1.; vec3 {q}={point};")
                lines.append(f"for(int j=0;j<{iterations};j++){{{q}=_rot3(0.,0.,{a[4]})*abs({q})*s{i}-vec3({','.join(a[1:4])}); f{i}*=s{i};}}")
            child = emit(children[0], q)
            suffix = f"*s{i}" if kind == "scale" else (f"/f{i}" if kind == "fold" else "")
            lines.append(f"float {d}={child}{suffix};")
        else:
            c, e = [emit(child, point) for child in children]
            if kind == "union":
                lines.append(f"float k{i}=max({a[0]},1.e-8), h{i}=clamp(.5+.5*({e}-{c})/k{i},0.,1.);")
                lines.append(f"float {d}=mix({e},{c},h{i})-k{i}*h{i}*(1.-h{i});")
            else:
                lines.append(f"float {d}=max({c},{'-' if kind == 'subtract' else ''}{e});")
        return d
    result = emit(plan, "p")
    params = "".join(f",float v{i}" for i in range(count))
    return f"float {name}(vec3 p{params}){{\n" + "\n".join(lines) + f"\nreturn {result};\n}}"


def _field_python(plan, args, formatter):
    kind, indices, children, iterations = plan
    a = [formatter(args[i]) for i in indices]
    if not children:
        return f"SDF3.{kind}({', '.join(a)})"
    c = [_field_python(child, args, formatter) for child in children]
    if kind == "fold": a = [str(iterations)] + a
    if len(c) == 2: a.insert(0, c[1])
    return f"{c[0]}.{kind}({', '.join(a)})"


@lru_cache(maxsize=128)
def _op(plan, count):
    digest = hashlib.sha256(repr(plan).encode()).hexdigest()[:16]
    map_name = "_sdfMap_" + digest
    call_args = "".join(f",v{i}" for i in range(count))
    call = lambda p: f"{map_name}({p}{call_args})"
    glsl = """
  float eps=max(abs(epsilon),.00001), limit=max(abs(far),eps);
  vec2 q=(_st-.5)*2.*vec2(resolution.x/resolution.y,1.);
  vec3 rd=normalize(vec3(q*tan(radians(clamp(fov,1.,178.))*.5),-1.));
  mat3 r=_rot3(rotX,rotY,rotZ);
  vec3 origin=vec3(0.,0.,distance);
  vec3 ro=vec3(dot(origin,r[0]),dot(origin,r[1]),dot(origin,r[2]));
  rd=vec3(dot(rd,r[0]),dot(rd,r[1]),dot(rd,r[2]));
  float radius=BOUND;
  radius+=eps+.0001*max(1.,radius);
  float closest=clamp(-dot(ro,rd),0.,limit);
  if(length(ro+rd*closest)>radius)return vec4(0.);
  float t=0.; bool hit=false;
  for(int i=0;i<256;i++) {
    if(float(i)>=clamp(floor(steps),1.,256.)||t>limit)break;
    float d=MAP(ro+rd*t);
    if(abs(d)<=eps){hit=true;break;}
    t+=max(abs(d)*.85,eps*.25);
  }
  if(!hit)return vec4(0.);
  if(mode>=1.5)return vec4(vec3(clamp(t/limit,0.,1.)),1.);
  vec3 p=ro+rd*t;
  vec3 n=vec3(MAP(p+vec3(eps,0.,0.))-MAP(p-vec3(eps,0.,0.)),
              MAP(p+vec3(0.,eps,0.))-MAP(p-vec3(0.,eps,0.)),
              MAP(p+vec3(0.,0.,eps))-MAP(p-vec3(0.,0.,eps)));
  n=r*(n/max(length(n),1.e-8));
  if(mode>=.5)return vec4(.5+.5*n,1.);
  vec3 light=normalize(vec3(.4,.6,1.)), h=normalize(light+vec3(0.,0.,1.));
  float lam=max(dot(n,light),0.);
  float shade=.15+.7*lam+.25*pow(max(dot(n,h),0.),32.)*(lam>0.?1.:0.);
  return vec4(vec3(shade),1.);
"""
    glsl = glsl.replace("BOUND", _shader_bound(plan))
    # MAP is only a Python template marker, never a GLSL macro. Balance nested
    # parentheses to replace every call without assuming a particular argument.
    while "MAP(" in glsl:
        start = glsl.index("MAP("); end = start + 4; depth = 1
        while depth:
            if glsl[end] == "(": depth += 1
            elif glsl[end] == ")": depth -= 1
            end += 1
        glsl = glsl[:start] + call(glsl[start+4:end-1]) + glsl[end:]
    def np_march(st, *args):
        return _np_march(plan, count, st, args[:-1], args[-1])
    def python_source(args, formatter):
        scene = _field_python(plan, args, formatter)
        camera = ", ".join(f"{name}={formatter(value)}" for (name, _), value in zip(_CAMERA, args[count:]))
        return f"raymarch({scene}, {camera})"
    return Op("sdfRay_" + digest, "src",
              [F(f"v{i}", 0, scalar=True) for i in range(count)] + [F(n, v, scalar=True) for n, v in _CAMERA],
              np=np_march, glsl=glsl, glsl_helpers=_shader_map(plan, map_name, count), python_source=python_source)


def raymarch(field, **camera):
    """Render an SDF3 field as a regular Hydra chain.

    mode=0: lit grayscale; 1: encoded view-space normals; 2: ray distance/far.
    Misses are transparent. steps is limited to 1..256; epsilon defaults .001.
    There are no shadows, environment bounces, arbitrary texture displacement,
    or implicit depth testing between separate Hydra layers.
    """
    if not isinstance(field, SDF3):
        raise TypeError("raymarch expects an SDF3 field")
    plan, values = _flatten(field)
    if len(values) > 192:
        raise ValueError("SDF3 scene exceeds the 192-parameter limit; split or simplify it")
    op = _op(plan, len(values))
    return Chain([Transform(op, normalize_args(op, tuple(values), camera))])
