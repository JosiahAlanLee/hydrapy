"""Morph geometry before drawing contours; all operations run in GLSL or NumPy."""
from hydrapy import *

a = sdfCircle(.22, Lfo(.06, lo=.36, hi=.54), .5)
b = sdfBox(.38, .42, .07, .57).rotate(Lfo(.04, lo=-.5, hi=.5))
field = a.sdfUnion(b, Lfo(.05, lo=.02, hi=.18))
hole = sdfCircle(Lfo(.07, lo=.05, hi=.16), .53, .51)
field = field.sdfSubtract(hole)
field.sdfBands(18, .24, Time() * .17).palette(.06, .23, .46, Lfo(.03), .8).out(o0)
field.sdfMask().color(.4, .85, 1).out(o1)
render(o0)
