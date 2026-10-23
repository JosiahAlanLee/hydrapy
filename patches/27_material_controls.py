"""Separate pigment, paper, motion and feedback lifetime.

    hydrapy render patches/27_material_controls.py --gpu --seconds 12 --out materials.mp4
    hydrapy html patches/27_material_controls.py

o0 is transparent pigment; o1 prints it onto stationary paper; o2 is the charcoal
Studie with independent motion/surface controls; o3 preserves the old radial distortion.
"""
from hydrapy import *
from hydrapy.recipes import on_paper, studie

set_bpm(140)
set_frame_rate(18)
set_composite_alpha(True)     # o0 and o3 are transparent pigment: show RGB x alpha over black

decay = Decay(0.7)  # halve accumulated pigment coverage every 0.7 seconds
old = src(o0).alpha(src(o0).a().mult(solid(decay, decay, decay)))
pigment = arc(0.3, 0.02, 0.66, 0.018, 0.7).polarRepeat(7, Time() * 0.18) \
    .scale(Lfo(0.08, lo=0.75, hi=1.1)).boil(0.0015, 12, 8, 3).color(0.58, 0.13, 0.11)
old.over(pigment).out(o0)
on_paper(src(o0), tooth=0.28, texture=0.025, seed=11).out(o1)

studie(o2, decay_seconds=0.45, motion=0.45, surface=0.0015, paper=0.35).out(o2)
arc(0.3, 0.02, 0.66, 0.018, 0.7).polarRepeat(7, Time() * 0.18, aspect=0).out(o3)
render(o1)
