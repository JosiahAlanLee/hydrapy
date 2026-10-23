"""Conway's Life as a video feedback texture op. Red = alive, green = decaying trail, blue = last state.
The board is reseeded by a 120 ms pulse every 2 s (a pulse LFO, not an envelope, so no frame rate can miss it)."""
from hydrapy import *

src(o0).life().add(noise(40, 0.5).thresh(0.72).color(1, 0, 0), Lfo(0.5, "pulse", width=0.06)) \
    .out(o0)
src(o0).color(1, 0.6, 0.2).blur(1).add(src(o0).r().color(1, 1, 1), 0.5).out(o1)
render(o1)
