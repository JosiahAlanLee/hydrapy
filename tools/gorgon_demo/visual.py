"""Gorgon demo: tracker controls geometry; Rack's LFO controls the palette."""
from hydrapy import *

z = Param('Dolly Z', .35, min=0, max=1)
roll = Param('Roll', 0, min=-45, max=45, step=1, unit='degrees')
feedback = Param('Feedback', .25, min=0, max=.9)
hue = Param('Hue', .2, min=0, max=1)
set_bpm(138)
shape(8,.35,.012).repeat(3,3).scale(1+z*2).rotate(roll*.01745329252).modulate(
    noise(3,.08),.06).color(.7,.9,.65).hue(hue).out(o1)
src(o3).scale(1.008).rotate(.003).blend(src(o1),1-feedback).out(o3)
src(o1).blend(src(o3),.6).saturate(1.3).out(o0)
src(o3).edge().color(.7,.9,1).out(o2)
render(o0)
