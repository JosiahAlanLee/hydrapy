"""Euclidean dihedral reflection folding, the first Coxeter-family extension.

The finite I2(n) reflection group has two mirror generators meeting at pi/n.
Unlike hyperkaleid this is Euclidean and tiles angular sectors, not the
hyperbolic disk. Coordinate folding is bounded and independent of time.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_coxeter2(st, order, centerX, centerY, angle, env):
    aspect = env.res[0] / env.res[1]
    x = (st[..., 0] - bc(centerX, st)) * aspect
    y = st[..., 1] - bc(centerY, st)
    a = np.pi / max(float(order), 2.0)
    theta = np.mod(np.arctan2(y, x) - bc(angle, st), 2*a)
    theta = np.where(theta > a, 2*a-theta, theta) + bc(angle, st)
    radius = np.hypot(x, y)
    return np.stack([bc(centerX, st) + radius*np.cos(theta)/aspect,
                     bc(centerY, st) + radius*np.sin(theta)], axis=-1).astype(np.float32)

register(Op("coxeter2", "coord",
    [F("order", 6, scalar=True, doc="Integer dihedral order, at least 2"),
     F("centerX", 0.5), F("centerY", 0.5),
     F("angle", 0, doc="Rotation of fundamental wedge, radians")],
    np=_np_coxeter2,
    doc="Euclidean dihedral reflection group I2(n): fold into a wedge of angle pi/n. order should be an integer >=2; unlike hyperkaleid this is not hyperbolic.",
    glsl="""float aspect = resolution.x / resolution.y;
  vec2 xy = (_st - vec2(centerX, centerY))*vec2(aspect, 1.0);
  float a = 3.141592653589793 / max(order, 2.0);
  float theta = mod(atan(xy.y, xy.x) - angle, 2.0*a);
  theta = (theta > a ? 2.0*a - theta : theta) + angle;
  float radius = length(xy);
  return vec2(centerX, centerY) + radius*vec2(cos(theta)/aspect, sin(theta));"""))
