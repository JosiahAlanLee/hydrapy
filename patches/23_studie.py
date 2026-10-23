"""Oskar Fischinger.

o0  Studie: charcoal oscillators on black paper to a beat — a ring of comets breathing, a long
    comet sweeping round, two sine strokes crossing, a pulse at the centre, smoke behind everything
    (recipes.studie). Swap the Clock for a record: a = Audio.from_file("brahms.mp3"); studie(o0, beat=a.flux)
o1  the same, printed (recipes.film): weave, flicker, bloom, grain
o2  the harmonograph: a damped double pendulum drawn in charcoal, turning, with trails
o3  Motion Painting No. 1: one touch of the brush per frame, never erased — the painting is the
    whole history of where the brush has been
"""
from hydrapy import *
from hydrapy.recipes import film, studie

set_bpm(126)

studie(o0, comets=6, waves=2, trails=0.8, paper=0.4).out(o0)
film(src(o0), grain=0.1, flicker=0.05, weave=0.002, bloom=0.35, scratch=0.15, specks=0.08, vignette=0.5).out(o1)

h = harmonograph(3, 4, 3.01, 4.02, 0, 0.6, 2.5, 0.42, 0.01, 0.4, 0.2).charcoal(0.4, 1.0, 100)
src(o2).mult(solid(0.9, 0.9, 0.9)).layer(h).out(o2)       # layer: a straight-alpha stroke onto an opaque picture

# the brush circles, its radius wandering (spirals, rings), turned along its own motion, a new colour every few seconds
r = Drift(0.5, 0.05, 0.42, 9)
brush = dab(0.07, 0.4, Time() * 0.4 * 6.2832 + 1.5708, 0.8, 0.2, SH(24, 0, 99)) \
    .scroll(Lfo(0.4, "sine", lo=-1, hi=1) * r * -1, Lfo(0.4, "sine", lo=-1, hi=1, phase=0.25) * r * -1) \
    .color(SH(0.4, 0.2, 1.0, 5), SH(0.4, 0.1, 0.9, 6), SH(0.4, 0.1, 0.9, 7))
src(o3).layer(brush).out(o3)

render(o1)
