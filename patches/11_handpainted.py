"""Hand-painted film, three ways. Switch with render(o0 / o1 / o2); 1-4 in the HTML export.

o0  a new painting every 1/8 s (step-printed: each held for three frames at 24 fps)
o1  one continuous painted strip running through the gate at 24 fps
o2  black leader: scratched so light gets through, then dyed
"""
from hydrapy import *
from hydrapy.recipes import film, handpainted

film(handpainted(seed=SH(8, 0, 99), bleed=2), fps=24).out(o0)
film(handpainted(seed=3, strip=24, hues=(0.95, 0.08, 0.55), stretch=(2.0, 3.0, 1.0)), fps=24).out(o1)
film(handpainted(seed=SH(12, 0, 99), leader="black", hues=(0.05, 0.12, 0.6), density=(0.7, 0.5, 0.3)),
     scratch=0.2, bloom=0.5, fps=24).out(o2)
render(o0)
