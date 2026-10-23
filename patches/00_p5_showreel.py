"""The p5.js side of a Hydra practice, four outputs (1-4 in the HTML export).

o0  text and line drawings as dye on paper     o1  the canvas in 3D: a floor and a sphere
o2  delay(): echoes of a moving shape          o3  text on a cylinder over the same floor
"""
from hydrapy import *
from hydrapy.recipes import film

set_bpm(120)
title = Text("hydrapy", font="DejaVuSerif-Bold", size=0.26, color=(0.85, 0.1, 0.05), y=0.62)
sub = Text("a video synth, on paper", font="DejaVuSerif", size=0.07, color=(0.1, 0.1, 0.1), y=0.36)
plants = Bank([Lines(meadow(seed=s, n=9), stroke=1.6, color=(0.05, 0.08, 0.05)) for s in range(6)], rate=12, hold=2)
paper = solid(0.98, 0.96, 0.9).stain(fbm(6, 4, 0, 0.5).alpha(solid(0.15, 0.15, 0.15)), 0.3)
film(paper.stain(src(plants), 1.0).stain(src(title).bleed(0.12, 0.15, 0.3, iterations=1), 1.0).stain(src(sub), 0.9),
     grain=0.05, scratch=0.15, specks=0.1, bloom=0.15, fps=24).out(o0)

ground = solid(0.03, 0.03, 0.05)
tex = osc(24, 0.08, 1.2).kaleid(4).modulate(noise(2, 0.15), 0.2).palette(0.05, 0.3, 0.6, Lfo(0.04))
ground.layer(tex.plane3d(rotX=1.25, rotY=0, rotZ=Lfo(0.02, lo=-0.3, hi=0.3), distance=0.55, fov=70, wrap=1, shade=0.4, fade=0.35)) \
      .layer(tex.sphere3d(rotX=0.2, rotY=Time() * 0.4, radius=0.32, distance=1.1, shade=0.6)) \
      .add(src(o1).scale(1.01).rotate(0.002), 0.35) \
      .out(o1)

mover = shape(5, 0.18, 0.02).scroll(Lfo(0.3, "tri", -0.3, 0.3), Lfo(0.21, "sine", -0.2, 0.2)).rotate(Time() * 2).color(1, 0.6, 0.2) \
    .add(shape(3, 0.1, 0.01).scroll(Lfo(0.43, lo=-0.35, hi=0.35), 0).color(0.2, 0.6, 1))
mover.add(delay(o2, 4).color(0.8, 0.3, 0.9), 0.7).add(delay(o2, 8).color(0.3, 0.9, 0.8), 0.5).add(delay(o2, 12).color(0.9, 0.9, 0.3), 0.35).out(o2)

ground.layer(tex.plane3d(rotX=1.25, distance=0.55, fov=70, wrap=1, shade=0.4, fade=0.35)) \
      .layer(src(Text("CANVAS", font="DejaVuSans-Bold", size=0.3, color=(1, 1, 1))).cylinder3d(rotX=0.25, rotY=Time() * 0.6, radius=0.4, height=0.8, shade=0.6)) \
      .out(o3)
render(o0)
