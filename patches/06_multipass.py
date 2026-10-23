"""Texture ops split the chain into GPU passes: blur -> edge -> chroma on top of a kaleidoscope."""
from hydrapy import *

base = voronoi(6, 0.4, 0.5).kaleid(4).rotate(Lfo(0.05, lo=0, hi=6.28))
base.blur(1.5, iterations=2).edge(1.5).palette(0.3, 0.6, 0.9, Lfo(0.04), 2) \
    .chroma(0.006, Lfo(0.3, lo=0, hi=6.28)) \
    .glitch(12, 0.08, 6) \
    .grain(0.08) \
    .out(o0)
