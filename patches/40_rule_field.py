"""A field of rules: eighteen Life-like automata side by side, each region running its own.

`ca(mode, p, flip)` is one generation of a Life-like automaton with a rule per pixel. Here the rule
map is itself alive at a coarser scale:

* o2 is a coarse automaton, 80x45 cells at 1280x720 (scale 1/16), Brian's Brain run asynchronously: each cell
  updates with probability 0.5 a frame, so it boils rather than ticks.
* The fine field, o1 (320x180, scale 1/4), takes its rule from a tile map that drifts — each of
  the tiles (8x6 at 16:9) changes rule at its own moment — shifted by four wherever the coarse automaton is
  alive. Fronts form where rules meet: Day & Night eats Life, Coral grows into Seeds' static.
* p (the chance a cell updates) varies across the frame too: crisp synchronous gliders in some
  places, a slow asynchronous boil in others. flip keeps a little fluctuation everywhere, so no
  region dies out for good.

Colour: each region is tinted by its rule; newborn cells are near white and take the tint as they
age, the trail glows in it, and the coarse automaton shows through dimly underneath.
"""
import os

from hydrapy import *

# RULE_FIELD_SIZE=720x1280 for a portrait (phone) version; cell counts follow the frame
W, H = (int(v) for v in os.environ.get("RULE_FIELD_SIZE", "1280x720").split("x"))
set_resolution(W, H)
CELLS, COARSE = (W // 4, H // 4), (W // 16, H // 16)
TILES = (max(1, round(W / 160)), max(1, round(H / 120)))

once = ((0.05 - Time()) * 1000).clip(0, 1)              # 1 on the first frames: the initial seeding


def seeded(field, density, scale):
    return field.blend(noise(scale, 0).thresh(density, 0.001), once)


# --- o2: the coarse automaton (Brian's Brain, asynchronous) -------------------------------------
seeded(src(o2).ca(CA_MODES["brians-brain"], 0.5, 0.004, 0.93, 7), 0.55, 30).out(o2, scale=1 / 16)

# --- the rule map: tiles that change rule asynchronously, offset where o2 is alive ---------------
tiles = noise(2.2, 0.04).pixelate(*TILES)                  # each tile drifts; floor() makes it jump at its own moment
coarse = src(o2).r().pixelate(*COARSE)
rule = tiles.mult(solid(10, 10, 10)).add(solid(8, 8, 8)).add(coarse.mult(solid(4, 4, 4)))
pace = noise(1.1, 0.03).mult(solid(0.45, 0.45, 0.45)).add(solid(0.62, 0.62, 0.62))   # p: 0.17 .. 1

# --- o1: the fine field --------------------------------------------------------------------------
storm = coarse.mult(solid(0.03, 0.03, 0.03)).add(solid(0.0003, 0.0003, 0.0003))   # fine cells flicker under live coarse ones
seeded(src(o1).ca(rule, pace, storm, 0.9, 3), 0.4, 90).out(o1, scale=1 / 4)

# --- o0: colour ----------------------------------------------------------------------------------
fine = src(o1)
tint = solid(0.95, 0.42, 0.22).hue(rule.mult(solid(1 / 16, 1 / 16, 1 / 16)))
cells = solid(1.0, 0.96, 0.88).blend(tint, fine.b().add(solid(0.35, 0.35, 0.35)))
(solid(0.025, 0.025, 0.035)
    .add(coarse.mult(tint).mult(solid(0.5, 0.5, 0.6)), 0.16)
    .add(fine.g().mult(tint), 0.38)
    .add(fine.r().mult(cells), 1.0)
    .pixelate(*CELLS)
    .out(o0))
render(o0)
