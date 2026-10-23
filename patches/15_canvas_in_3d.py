"""The p5.js WEBGL move: the chain becomes a texture on a plane, a sphere or a cylinder.

Rotations in radians; distance=1 puts an unrotated plane exactly in the frame; wrap=1 tiles the
plane to the horizon. Outside the object the result is transparent, so layer it over a ground.
"""
from hydrapy import *

ground = solid(0.03, 0.03, 0.05)
tex = osc(24, 0.08, 1.2).kaleid(4).modulate(noise(2, 0.15), 0.2).palette(0.05, 0.3, 0.6, Lfo(0.04))

ground.layer(tex.plane3d(rotX=1.25, rotY=0, rotZ=Lfo(0.02, lo=-0.3, hi=0.3), distance=0.55, fov=70, wrap=1, shade=0.4, fade=0.35)) \
      .layer(tex.sphere3d(rotX=0.2, rotY=Time() * 0.4, radius=0.32, distance=1.1, shade=0.6)) \
      .add(src(o0).scale(1.01).rotate(0.002), 0.35) \
      .out(o0)

ground.layer(src(Text("CANVAS", font="DejaVuSans-Bold", size=0.3, color=(1, 1, 1))).cylinder3d(
    rotX=0.25, rotY=Time() * 0.6, radius=0.4, height=0.8, shade=0.6)) \
      .layer(tex.plane3d(rotX=0.4, rotY=Lfo(0.11, lo=-0.8, hi=0.8), distance=2.2, shade=0.3).scroll(0, 0.28)) \
      .out(o1)
render(o0)
