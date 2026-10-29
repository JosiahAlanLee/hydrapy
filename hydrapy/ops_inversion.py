"""Circle inversion coordinate transform, with CPU and GLSL parity."""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_circleinvert(st, radius, centerX, centerY, mix, env):
    aspect = env.res[0] / env.res[1]
    x = (st[..., 0] - bc(centerX, st)) * aspect
    y = st[..., 1] - bc(centerY, st)
    r2 = x*x + y*y
    scale = bc(radius, st)**2 / np.maximum(r2, 1e-8)
    weight = np.clip(bc(mix, st), 0, 1)
    factor = 1 + weight * (scale - 1)
    return np.stack([bc(centerX, st) + x*factor/aspect,
                     bc(centerY, st) + y*factor], axis=-1).astype(np.float32)

register(Op("circleinvert", "coord",
    [F("radius", 0.3, doc="Inversion circle radius in frame-height units"),
     F("centerX", 0.5, doc="Circle center x in normalized coordinates"),
     F("centerY", 0.5, doc="Circle center y in normalized coordinates"),
     F("mix", 1.0, doc="0 identity, 1 full circle inversion")],
    np=_np_circleinvert,
    doc="Invert sample coordinates through a circle, with aspect correction. The center is singular and clamped; extreme magnification can alias.",
    glsl="""float aspect = resolution.x / resolution.y;
  vec2 delta = (_st - vec2(centerX, centerY)) * vec2(aspect, 1.0);
  float scale = radius*radius / max(dot(delta, delta), 1e-8);
  float factor = 1.0 + clamp(mix, 0.0, 1.0) * (scale - 1.0);
  return vec2(centerX, centerY) + delta * factor / vec2(aspect, 1.0);"""))
