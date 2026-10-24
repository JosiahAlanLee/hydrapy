"""Independent pixel clocks, inspired by Memo Akten's PixlFlippr description.

Original spatial frequency/phase fields; no edition source or assets copied.
Every sample depends only on its coordinates, time and the supplied controls.
There are no neighbor reads, feedback, random draws or simulation steps.
"""
from __future__ import annotations

import numpy as np

from .core import Op, register
from .ops import F, I, TAU, grey


def _np_pixel_osc(st, rate, spread, density, pixel, geometry, phase, threshold, env):
    pitch = max(1., float(np.floor(pixel)))
    # Sample cell centers in units of half the image height. A whole cell
    # shares one oscillator even when it occupies several output pixels.
    p = ((np.floor(st * env.res / pitch) + .5) * pitch - env.res * .5) * (2. / env.res[1])
    x, y = p[..., 0], p[..., 1]
    xx, yy, xy = x*x, y*y, x*y
    mode = int(np.clip(geometry, 0, 3))
    if mode == 0:
        carrier = xy + .19*(xx-yy)
        detune = .18*(xx+yy) + .045*np.abs(xy)
    elif mode == 1:
        carrier = np.sqrt(xx+yy+.01) + .12*(xx-yy)
        detune = .24*(xx+yy) + .05*xy
    elif mode == 2:
        ax, ay = np.abs(x), np.abs(y)
        carrier = np.maximum(ax,ay) + .17*np.minimum(ax,ay)
        detune = .32*ax*ay + .045*(xx-yy)
    else:
        carrier = xx-yy + .075*np.sin(TAU*x)*np.sin(TAU*y)
        detune = .20*(xx+.7*yy) + .06*xy
    cycles = density*carrier + phase + np.float32(env.time)*(rate+spread*detune)
    # Reduce the argument before sin to limit drift for dense phase fields.
    wave = np.sin(np.float32(TAU)*(cycles-np.floor(cycles)))
    return grey((wave >= threshold).astype(np.float32), st)


register(Op("pixelOsc", "src",
    [F("rate", .035), F("spread", 1.6), F("density", 90),
     F("pixel", 4, scalar=True), I("geometry", 0), F("phase", .137), F("threshold", 0)],
    np=_np_pixel_osc,
    doc="Independent binary pixel clocks. With constant controls each cell has a fixed frequency and phase: rate + spread*field (Hz), density*geometry + phase (cycles). pixel is an integer screen-pixel pitch >=1. geometry: 0 saddle, 1 radial, 2 folded, 3 lattice. threshold compares with sine, normally -1..1. No neighbor reads or history; deliberate spatial aliasing.",
    glsl="""
  float pitch=max(1.,floor(pixel));
  vec2 p=((floor(_st*resolution/pitch)+.5)*pitch-resolution*.5)*(2./resolution.y);
  float x=p.x, y=p.y, xx=x*x, yy=y*y, xy=x*y;
  float carrier, detune;
  if(geometry<=0) {
    carrier=xy+.19*(xx-yy);
    detune=.18*(xx+yy)+.045*abs(xy);
  } else if(geometry==1) {
    carrier=sqrt(xx+yy+.01)+.12*(xx-yy);
    detune=.24*(xx+yy)+.05*xy;
  } else if(geometry==2) {
    vec2 a=abs(p);
    carrier=max(a.x,a.y)+.17*min(a.x,a.y);
    detune=.32*a.x*a.y+.045*(xx-yy);
  } else {
    carrier=xx-yy+.075*sin(6.28318530718*x)*sin(6.28318530718*y);
    detune=.20*(xx+.7*yy)+.06*xy;
  }
  float cycles=density*carrier+phase+time*(rate+spread*detune);
  float value=step(threshold,sin(6.28318530718*fract(cycles)));
  return vec4(vec3(value),1.);
"""))
