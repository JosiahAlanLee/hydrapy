"""Seeded smooth overdraw of a botanical drawing; fixed canvas avoids fit jitter.

Replace plant(...) with flora_lines('plant.json') for Generative_Art geometry,
then normalize its coordinates to a fixed view before calling fuzzy_paths.
"""
from hydrapy import *

# These existing plant() paths are already normalized. No new species model.
paths = [subdivide(p, 1) for p in plant("fern", seed=14)]

def draw(t):
    return Lines(fuzzy_paths(paths, passes=4, jitter=.004, seed=21, time=t, rate=.12),
                 canvas=(960, 540), fit=False, stroke=1.1, color=(.7, .9, .55)).render()

solid(.018, .027, .025).layer(src(FrameSource(draw))).out(o0)
