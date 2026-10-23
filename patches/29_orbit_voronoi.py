"""Opposing rings of Voronoi sites: analytic GPU adaptation of thi.ng's example.

Mode 0 is cell tone, 1 draws edges, 2 exposes distance for downstream effects.
This is not the De/Frag artwork renderer. This patch also exports to live HTML.
"""
from hydrapy import *

cells = orbitVoronoi(11, 4, .46, .22, width=.003)
cells.palette(.02, .19, .43, Lfo(.025, lo=0, hi=.25), 1.3).out(o0)
orbitVoronoi(11, 4, .46, .22, width=.004, mode=1).color(.6, .9, 1).out(o1)
render(o0)
