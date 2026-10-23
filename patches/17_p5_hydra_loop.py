"""The p5js-hydra-fxhash template, in one place.

The original: Hydra renders o0; p5 (WEBGL) draws the Hydra canvas as a texture on a plane() that
is a little smaller than the window; Hydra reads that p5 canvas back as s0 and layers it over o0:

    src(o0).layer(src(s0)).blend(osc()).pixelate(150, 150).out(o0)      // hydra
    texture(pg); plane(pg.width - 100, pg.height - 100)                  // p5 draw()

Here the p5 half is src(o0).plane3d(): the output on a plane in perspective, fed back. Nudge the
rotations and the plane becomes a tumbling window into its own past. Fx("oo...", legacy=True) is
the template's fxrand() (plain Fx("oo...") is today's $fx.rand()): the same hash gives the same
parameters every time, as it did on fxhash. For a page that draws them per mint, see 34_fxhash.py.
"""
from hydrapy import *

fx = Fx("ooHydrapyTemplate1")                       # swap the hash for a new edition

tilt = fx.rnd_btw(-0.35, 0.35)
spin = fx.rnd_btwexp(0.02, 0.2) * fx.pick([-1, 1])
freq = fx.rnd_btw(20, 80)
px = fx.rnd_int(60, 200)

# o0: the template as written, plus the plane in 3D (the original plane was flat and a touch smaller)
src(o0).layer(src(o0).plane3d(rotX=tilt, rotY=Time() * spin, distance=1.08, shade=0.2)) \
       .blend(osc(freq, 0.1, fx.rnd_btw(0, 2))) \
       .pixelate(px, px) \
       .out(o0)

# o1: the same loop with the window tumbling harder, a frame behind itself, and no pixelation
src(o1).layer(delay(o1, 2).plane3d(rotX=Lfo(0.07, lo=-0.5, hi=0.5), rotY=Time() * spin * 3, distance=1.15, shade=0.3)) \
       .blend(osc(freq / 2, 0.1, fx.rnd_btw(0, 2)).kaleid(fx.rnd_int(3, 7)), 0.4) \
       .out(o1)
render(o0)
