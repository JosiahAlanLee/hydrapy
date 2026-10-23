"""Super 8.

The whole patch steps at 18 frames a second (set_frame_rate / recipes.super8(hold=True)), so
everything on every output judders like the projector, including the signals.

A Driftless roadside: procedural plants (drawing.meadow) as a line drawing against a warm sky,
swaying a little.

o0  Kodachrome: dense, warm, the halation reddish around the highlights
o1  the same on faded stock: lifted blacks, yellowed, half the saturation
o2  Tri-X reversal, black and white, more grain and scratches
o3  the stages by hand: filmstock + grain + a weaving gate on the bare picture
    (change set_frame_rate below: 16 for Regular 8, 24 for sound speed, None to release)

Pair it with the painted-film recipes:  super8(handpainted(seed=SH(6, 0, 99)), stock=0)
"""
from hydrapy import *
from hydrapy.recipes import super8

sky = osc(1.5, 0.0, 0.6).rotate(1.5708).color(0.95, 0.72, 0.45).brightness(-0.05) \
      .add(noise(3, 0.02).color(0.2, 0.15, 0.1), 0.5)
plants = src(Lines(meadow(seed=7, n=16), stroke=3.5, color=(0.08, 0.06, 0.04), fit=False)) \
         .scrollX(Lfo(0.4, "sine", lo=-0.003, hi=0.003)).scrollY(0.03)
pic = sky.layer(plants)

super8(pic, stock=0).out(o0)                      # sets the patch to 18 fps
super8(pic, stock=2, flicker=0.12, specks=0.25).out(o1)
super8(pic, stock=3, grain=0.25, scratch=0.6).out(o2)

pic.filmstock(0).grain(0.12, 18, 2).mult(gate(0.95, 0.93, 0.06, 0.008).pan(SH(18, -0.003, 0.003), 0)).out(o3)

set_frame_rate(18)
render(o0)
