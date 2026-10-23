"""Listening and watching: a record's beats, and the motion in footage.

o0  Fischinger to the record: studie() driven by the beats the tracker found in test.wav (swap in
    any record: Beats.from_file("brahms.wav"), or give the tempo if you know it: bpm=96)
o1  a pulse on every beat and a phase across each bar of four
o2  the onset-strength envelope itself, as a scale
o3  footage, if you have some: set FOOTAGE to a video and the painted texture is carried along its
    motion (optical flow) — Brakhage over the Driftless. Without footage, a stand-in.
"""
import os

from hydrapy import *
from hydrapy.recipes import studie

HERE = os.path.dirname(os.path.abspath(__file__))
track = os.path.join(HERE, "test.wav")
beats = Beats.from_file(track)
a = Audio.from_file(track)                                   # plays the record in the HTML export, in step

studie(o0, beat=beats.phase, comets=5, waves=2, paper=0.4).out(o0)

shape(4, 0.15 + beats.pulse(0.02, 0.5) * 0.25, 0.01).rotate(beats.bar(4) * 6.2832) \
    .color(1.0, 0.6, 0.3).out(o1)

osc(12, 0.05, 1.2).kaleid(4).scale(1 + beats.onset * 0.6).out(o2)

FOOTAGE = os.environ.get("FOOTAGE")
if FOOTAGE:
    video = Video(FOOTAGE)
    flow = Flow(video, gain=1.0)
    src(o3).displace(src(flow), -1).blend(src(video), 0.03) \
           .layer(drops(4, 0.4, 0.3, seed=SH(2, 0, 99)).alpha(drops(4, 0.4, 0.3, seed=SH(2, 0, 99)).a().mult(solid(0.05, 0.05, 0.05)))) \
           .out(o3)
else:
    noise(3, 0.2).modulate(osc(4, 0.1), 0.2).color(0.6, 0.8, 0.5).out(o3)

render(o0)
