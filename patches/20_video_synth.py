"""Analog video synthesis, the LZX / Rutt-Etra end of it.

o0  Rutt-Etra scan processor: a raster of scan lines displaced by the brightness under them,
    on a slowly tumbling noise field. Rotate the picture before it and the raster tilts with it.
o1  Two ramps through bitwise XOR at 5 bits, then a sine staircase shaper: the oldest
    video-synth texture there is (Sandin / Jones colorizer territory).
o2  Contour keyer over a feedback loop that pixel-sorts itself a little more every frame.
o3  Sinefold on a shear: gradients become rainbows, then the picture is cut into isolines.
"""
from hydrapy import *

field = noise(2.5, 0.15).brightness(0.1).contrast(1.6)

# o0: scan processor. amount is the "Z" of the Rutt-Etra: how far brightness lifts the line.
field.rotate(Lfo(0.02, lo=-0.15, hi=0.15)).ruttetra(lines=72, amount=Lfo(0.07, lo=0.1, hi=0.4), thickness=1.4, glow=0.35) \
     .color(0.7, 1.0, 0.8).out(o0)

# o1: ramp XOR ramp, through the staircase shaper. Changing `depth` changes the texture's scale.
rampX = gradient(0).scroll(Lfo(0.05))
rampY = gradient(0).rotate(1.5708).scroll(0, Lfo(0.03))
rampX.bits(rampY, 0, 6).sinefold(Lfo(0.01, lo=1, hi=2), Lfo(0.1)).out(o1)

# o2: the GPU pixel sorter is progressive: every frame sorts the bright runs 8 steps further.
# A fresh picture is loaded every 6 seconds (blend 1 for a few frames), then left to melt.
src(o2).psort(0.3, 0, iterations=8).blend(field.kaleid(3), Lfo(1 / 6, "pulse", width=0.03)).out(o2)

# o3: sinefold on a shear, then isolines
src(o2).shear(Lfo(0.04, lo=-0.3, hi=0.3)).sinefold(2, Lfo(0.07)).contours(8, 0.2, 0.5).out(o3)

render(o0)
