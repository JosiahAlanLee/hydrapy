"""Mothlight: organic matter on a light table.

Put your own photos (leaves, wings, grasses, paint on acetate, shot against white) in a folder
next to this file called `matter/` and they become the dye, one per frame. Without the folder,
the procedural version runs: stretched drops as wings, ridged noise as membrane, dark veins.
"""
import os
from hydrapy import *
from hydrapy.recipes import film, mothlight

folder = os.path.join(os.path.dirname(__file__), "matter")
bank = Bank(folder, rate=24, hold=2, mode="random") if os.path.isdir(folder) else None

film(mothlight(seed=SH(8, 0, 99), bank=bank), scratch=0.3, bloom=0.3, fps=24).out(o0)
film(mothlight(seed=5, strip=24, hue=0.08), fps=24).out(o1)
render(o0)
