"""Unfolding: a symmetric automaton that keeps doubling itself out from the centre.

Long-form and slow. Every few seconds the field is redrawn at twice the size from its middle (each
cell becomes a 2x2 block), and between those folds the rules work the enlarged blocks into finer
structure again. Fredkin's parity rule copies whatever it is given eight times over, so a glyph at
the centre opens out like a folded sheet; Anneal and Diamoeba melt the copies into soft shapes;
Brian's Brain and Star Wars send sparks along their edges.

* Symmetric in both axes, exactly: everything that varies across the field is read through
  `mirrorgrid(N, M)` (whole cells, folded in integers, so mirrored cells see identical numbers),
  the rules look at symmetric neighbourhoods, and `ca(..., sym=1)` gives mirrored cells the same
  random draws. Any floating-point asymmetry, however small, would be amplified into the whole field.
* A disc of one rule sits inside a field of another. The inner rules cycle through seven every 41 s,
  the outer through eleven every 67 s, and the disc breathes on its own slow period, so the
  combination doesn't come round again for at least 59 hours.
* Folds come every 32, 64 or 128 generations; every sixth fold clears the middle and drops a new glyph.

UNFOLDING_SIZE=720x1280 makes a portrait version. Generations run at 30 a second (set_frame_rate),
so the timing is the same on any display.
"""
import os

from hydrapy import *

W, H = (int(v) for v in os.environ.get("UNFOLDING_SIZE", "1280x720").split("x"))
set_resolution(W, H)
set_frame_rate(30)
N, M = W // 4, H // 4                                   # cells: 320x180, each 4 px
R = CA_MODES

gen = (Time() * 30).floor()                             # the generation number


def every(k, phase=0):
    """1 on one generation in k, else 0."""
    return 1 - ((gen + phase) % k).clip(0, 1)


def cycle(rules, seconds):
    return Seq([R[r] for r in rules]).fast(2 / seconds)


inner = cycle(["parity", "parity", "parity-4", "parity", "parity-4", "parity", "parity-4"], 41)
outer = cycle(["anneal", "brians-brain", "diamoeba", "frogs", "coral", "star-wars", "anneal",
               "sticks", "day-and-night", "brians-brain", "walled-cities"], 67)
radius = Lfo(1 / 173, lo=0.14, hi=0.3)
disc = shape(64, radius, 0.002).mirrorgrid(N, M)        # 1 inside the inner rule's disc; exactly symmetric
outside = disc.invert()
rule = solid(outer, outer, outer).add(disc.mult(solid(inner - outer, inner - outer, inner - outer)))
pace = solid(1, 1, 1).sub(outside.mult(solid(0.25, 0.25, 0.25)))     # inside: in step; outside: asynchronous
flips = outside.mult(solid(0.0004, 0.0004, 0.0004))                   # parity turns any flip into noise: none inside

# Folds come every 2^k generations: parity's state at 2^k steps is the pattern plus eight clean copies.
period = Seq([32, 32, 64, 32, 64, 64, 32, 128]).fast(2 / 53)
fold = every(period)
drop = every(period * 6)                                # every sixth fold, a new glyph at the centre
glyph_seed = (gen / 97).floor() * 0.731
glyph = (noise(30, 0).scroll(glyph_seed, glyph_seed * 0.53).thresh(0.1, 0.001)
         .mult(shape(4, 0.022, 0.001)).mirrorgrid(N, M))
centre = shape(4, 0.05, 0.001).mirrorgrid(N, M)

unfolded = src(o1).pixelate(N, M).scale(2)              # each cell becomes a 2x2 block, from the centre out
(src(o1).ca(rule, pace, flips, 0.95, 0, 1)
        .blend(unfolded, fold)
        .mult(solid(1, 1, 1).sub(centre.mult(solid(drop, drop, drop))))   # the drop clears the middle ...
        .add(glyph, drop)                                                 # ... and leaves a glyph
        .out(o1, scale=1 / 4))

# --- the display: cells as tiles, a palette that drifts over minutes --------------------------
cell = src(o1)
tiles = shape(4, 0.86, 0.06).repeat(N, M)               # a small gap round every cell
ink = solid(0.035, 0.04, 0.06)
paper = solid(0.95, 0.91, 0.82)
glow = solid(0.25, 0.55, 0.9).hue(Lfo(1 / 311, lo=-0.12, hi=0.12)).add(disc.mult(solid(0.45, -0.1, -0.35)), 0.6)
(ink.add(cell.g().mult(glow), 0.42)
    .add(cell.r().mult(paper.blend(glow, cell.b().mult(solid(0.7, 0.7, 0.7)))), 1.0)
    .pixelate(N, M)                                     # colour once per cell ...
    .mult(tiles.add(solid(0.18, 0.18, 0.18)))           # ... then the gap, at full resolution
    .out(o0))
render(o0)
