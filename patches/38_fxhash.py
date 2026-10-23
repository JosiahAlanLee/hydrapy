"""An fxhash edition: every number here is drawn per mint, in the page, from the page's hash.

`Mint()` records the draws; `to_fxhash` (or `hydrapy fxhash 38_fxhash.py --sdk fxhash.min.js`)
writes a project whose page makes the same draws again from `$fx.hash`, declares the fx(params)
and sets the features, then calls `$fx.preview()` after two seconds.  In Python the draws take the
preview hash's values: `hydrapy render 38_fxhash.py --fxhash oo… -o mint.png` shows another mint,
and the exported page's n key jumps to a fresh one.

Choices between chains (`switch`) compile every branch, crossfaded with 0/1 weights.
"""
from hydrapy import *

fx = Mint()

source = fx.pick(["waves", "noise", "cells"])
pattern = switch(source,
                 osc(fx.rnd_btw(8, 40), 0.05, fx.rnd_btw(0.6, 1.8)),
                 noise(fx.rnd_btw(2, 7), 0.08),
                 voronoi(fx.rnd_btw(4, 12), 0.2, 0.4))
sides = fx.rnd_int(3, 9)
spin = fx.rnd_btw(-0.08, 0.08)
palette = fx.pick(["ember", "tide", "moss"])
glitch = EntropyLock(fx, rate=12, lock=fx.rnd_btw(0.005, 0.03))     # glitches that keep coming back

density = fx.number("density", default=0.6, min=0.1, max=1.0, update="sync")   # fxlens slider, live
ink = fx.color("ink", default="#f2b04a")

ground = switch(palette, solid(0.09, 0.04, 0.03), solid(0.02, 0.05, 0.09), solid(0.04, 0.06, 0.03))
figure = pattern.kaleid(sides).rotate(0, spin).modulate(noise(2.5, 0.05), density * 0.15).color(*ink)

ground.layer(figure.luma(0.35, 0.25)) \
      .tear(glitch * 7 + 3, 3, 0.08, fx.rnd_btw(0.05, 0.3)) \
      .grain(0.06) \
      .vignette(0.5, 0.7) \
      .out(o0)

fx.features({"Source": source, "Palette": palette, "Sides": sides})
render(o0)
