"""Four outputs, one patch: press 1-4 in the HTML export (or `render(oN)` here) to switch.

o3  Gray-Scott reaction-diffusion state, seeded by a wandering hexagon (the workhorse; raw A/B)
o0  that state, coloured through a cosine palette
o1  per-pixel CV: an oscillator FM'd by noise, wavefolded by a beat envelope, then multipass chroma + glitch
o2  Conway's Life as video feedback, reseeded by a 120 ms pulse every 2 s
"""
from hydrapy import *

set_bpm(120)

seed = shape(6, 0.06, 0.01).scroll(Lfo(0.11, lo=-0.3, hi=0.3), Lfo(0.17, "tri", lo=-0.3, hi=0.3)).color(0, 1, 0)
src(o3).rd(feed=0.037, kill=0.06, iterations=6).add(seed, 0.1).out(o3)

src(o3).g().palette(0.1, 0.4, 0.7, Lfo(0.02), 1.5).vignette(0.4).out(o0)

env = Env(Clock(1), attack=0.02, decay=0.4)
osc(noise(2, 0.1).r() * 20 + 6, 0.05, 1.5 + SH(8, lo=-0.4, hi=0.4)) \
    .rotate(Drift(0.3, 0, 6.28)) \
    .kaleid(noise(1.5, 0.05).posterize(3, 1).r() * 4 + 3) \
    .fold(1 + env * 1.5) \
    .chroma(0.004, Lfo(0.3, lo=0, hi=6.28)) \
    .glitch(12, 0.06, 6) \
    .out(o1)

src(o2).life().add(noise(40, 0.5).thresh(0.62).color(1, 0, 0), Lfo(0.5, "pulse", width=0.06)) \
    .color(1, 0.6, 0.2) \
    .out(o2)

render(o0)
