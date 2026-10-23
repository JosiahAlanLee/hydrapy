"""Marbling: suminagashi and ebru, as chains of closed-form ink moves (hydrapy/ops_marble.py).

Every drop, stylus and comb is an exactly invertible map, so a chain is the bath's history in
order, rendered fresh each frame with every parameter live: the combs can be drawn through and
back again, and the pattern un-marbles.

o0  suminagashi: two pipettes, sumi and indigo with clear water between, stirred by a slow swirl
o1  gel-git: bands of colour combed one way, then the other — the comb strokes swing on LFOs
o2  battal (stone) ebru, then a single stylus drawn through, back and forth
o3  stones combed through and back again on a slow triangle: the moves are exactly invertible, so at
    the bottom of the triangle the stones are home
"""
from hydrapy import *
from hydrapy.recipes import bands, ebru_stones, suminagashi

set_bpm(96)

suminagashi(rings=14, radius=0.065).swirl(0.5, 0.52, Lfo(0.03, lo=-0.35, hi=0.35), 0.24, 0.1) \
    .ripple(0.3, Lfo(0.05, lo=0.0, hi=0.012), 0.2, Time() * 0.2).out(o0)

bands(repeats=4).tine(1.5708, 0, 0.07 + Lfo(0.04, lo=-0.03, hi=0.03), 0.02, 0.1) \
                .tine(-1.5708, 0.05, 0.07 + Lfo(0.033, lo=-0.03, hi=0.03), 0.02, 0.1).out(o1)

ebru_stones(n=30, seed=4).tine(0.4, 0.0, Lfo(0.06, "tri", lo=-0.35, hi=0.35), 0.05).out(o2)

# the same stones combed through and back: the comb is exactly invertible, so they come home
ebru_stones(n=30, seed=9).tine(1.5708, 0, Lfo(0.05, "tri", lo=0.0, hi=0.3), 0.025, 0.12) \
                         .tine(0.0, 0, Lfo(0.05, "tri", lo=0.0, hi=0.15, phase=0.25), 0.03, 0.2).out(o3)

render(o0)
