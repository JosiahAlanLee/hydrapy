"""polar / unpolar coordinate ops: an oscillator in polar space becomes spokes, unpolar wraps a tiling into a tunnel."""
from hydrapy import *

osc(12, 0.2, 1).repeat(2, 3).polar(1, 2).twist(Lfo(0.07, lo=-3, hi=3)).out(o0)
src(o0).unpolar(Lfo(0.1, lo=0.5, hi=1.5)).scrollY(0, 0.1).mirror(1, 1).fisheye(Lfo(0.13, "tri", -0.6, 0.8)).out(o1)
render(o1)
