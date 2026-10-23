"""Public thi.ng cellular engine: interacting rules, scrolling age and glyphs.

hydrapy render patches/28_cellular.py --gpu --seconds 12 --out cells.mp4
No C-SCAPE/ASCII-SCAPE edition code or curated artwork parameters are used.
"""
from hydrapy import *

set_bpm(120)
# 24 generations per beat; simulation speed is independent of export fps.
cells = Cellular(width=384, height=216, rate=24 * 120 / 60, seed=17, stripes=48)
src(cells).palette(.04, .23, .53, Lfo(.025, lo=0, hi=.18), 1.4).out(o0)

glyphs = Cellular(width=96, height=54, rate=24, seed=17, stripes=12,
                  chars=" .:-=+*#%@", cell=(10, 14))
solid(.012, .018, .025).add(src(glyphs).color(.45, .9, .8)).out(o1)
render(o0)
