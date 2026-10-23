"""Tape.

o0  the picture: hard-edged, saturated, the kind of thing tape does the most to
o1  the picture played back from a first-generation VHS copy, with the deck's PLAY caption
o2  a fourth-generation dub (each copy softer, noisier and shakier than the last)
o3  the stages by hand, for taking apart: colour bleed, ringing, jitter + head switching,
    dropouts, hiss — all CV-able, so bad tracking can come from an LFO or a knob.

The ops are in hydrapy/ops_video.py; the recipe that strings them together is recipes.vhs().
"""
from hydrapy import *
from hydrapy.recipes import vhs

pic = osc(4, 0.03, 1.4).posterize(4, 1.0).kaleid(5).rotate(0.1).modulate(noise(2, 0.05), 0.04)
osd = src(Text("PLAY", size=0.07, x=0.14, y=0.86, color=(1, 1, 1)))      # the deck draws this after the tape, so it stays crisp

pic.out(o0)
vhs(pic).layer(osd).out(o1)
vhs(pic, generation=4).layer(osd).out(o2)

tracking = Lfo(0.05, "tri", lo=0.0, hi=1.0)                 # the tracking knob drifting back and forth
pic.yiq(0.03, 2.0, 0.005).ringing(0.6, 2) \
   .linejitter(tracking * 0.012, 30, 0.003) \
   .headswitch(0.05 + tracking * 0.08, tracking * 0.25, 30) \
   .lighten(dropouts(tracking * 2.0, 0.3, 30), 0.9) \
   .grain(0.05 + tracking * 0.1, 30, 1) \
   .out(o3)

render(o1)
