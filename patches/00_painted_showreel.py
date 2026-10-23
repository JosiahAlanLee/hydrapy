"""Painted film, four outputs (1-4 in the HTML export).

o0  a new painting every 1/8 s, step-printed       o1  one painted strip running at 24 fps
o2  black leader, scratched then dyed              o3  Mothlight, procedural wings and veins
"""
from hydrapy import *
from hydrapy.recipes import film, handpainted, mothlight

film(handpainted(seed=SH(8, 0, 99), bleed=2), fps=24).out(o0)
film(handpainted(seed=3, strip=24, hues=(0.95, 0.08, 0.55), stretch=(2.0, 3.0, 1.0)), fps=24).out(o1)
film(handpainted(seed=SH(12, 0, 99), leader="black", hues=(0.05, 0.12, 0.6), density=(0.7, 0.5, 0.3)),
     scratch=0.2, bloom=0.5, fps=24).out(o2)
film(mothlight(seed=SH(8, 0, 99)), scratch=0.3, bloom=0.3, fps=24).out(o3)
render(o0)
