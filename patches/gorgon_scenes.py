"""Gorgon scenes: three named looks for the tracker's SCENE lane, played by its lanes.

Write `drift`, `fracture` or `bloom` in the SCENE column; a fade crossfades over
that many rows. Lane 1's hits kick the image, lane 2's notes set the pattern's
density, and the automation lanes still drive Dolly Z, Roll and Feedback.
"""
from hydrapy import *

STUDIO = {'title': 'Gorgon / Scenes', 'tags': ['Tracker', 'Scenes', 'Feedback']}
z = Param('Dolly Z', .35, min=0, max=1)
roll = Param('Roll', 0, min=-45, max=45, step=1, unit='degrees')
feedback = Param('Feedback', .25, min=0, max=.9)
set_bpm(138)

kick = tracker.hit(1, decay=.18)              # 1 on each lane-1 hit, gone in 180 ms
density = (tracker.note(2) - 48) / 12         # lane 2's note, in octaves above C3

# Shared feedback memory: every scene feeds it, so changing scene leaves a trail.
src(o3).scale(1.006).rotate(.002).blend(src(o0), 1 - feedback).out(o3)


@scene('drift')
def _():
    return osc(6 + density * 8, .05, 1.1).rotate(roll * .01745).scale(1 + z + kick * .3).blend(src(o3), .4)


@scene('fracture')
def _():
    return voronoi(4 + density * 6, .3, .2).kaleid(6).color(1, .55, .25).modulate(src(o3), .05 + kick * .2)


@scene('bloom')
def _():
    return shape(6, .3 + kick * .15, .05).repeat(3, 3).rotate(roll * .01745).blend(src(o3), .6).saturate(1.4)


render(o0)
