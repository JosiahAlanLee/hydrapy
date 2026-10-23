"""A painted oscillator, patched into the modular.

Two drawn voices. On o0 a sine stroke (``wave``) stands upright down the middle of the frame; the
Soundtrack reads that middle column, so every place the stroke crosses the column is a pulse, and
the crossings per frame set the pitch. On o2 a damped harmonograph does the same with a far less
regular pulse train. Both leave the ES-3 at audio rate (``CvOut.send(k, track)``) — a VCO made of
charcoal, to be put through a filter — while a Probe sends the picture's brightness out as a CV,
and three ES-6 inputs come back in: pitch at 1 V/oct, the wave's wobble (its amplitude: from a
straight line, which is DC, to a pulse train), and grit (paper tooth, which is noise).

Without an interface the CV inputs are stand-ins and the outputs go to painted_oscillator.wav
(3 channels, 48 kHz) to play from a DAW into the ES-3.

    hydrapy live patches/26_painted_oscillator.py --gpu     # the live loop
    hydrapy render patches/26_painted_oscillator.py --gpu --seconds 10 --out po.mp4   # o1's picture + the first voice as its audio
"""
import atexit

from hydrapy import *

try:
    cv = CvIn(channels=(0, 1, 2))                       # ES-6 inputs 1-3
    pitch, wobble, grit = cv.cv                         # volts
except Exception:                                       # no interface: stand-ins
    pitch = Seq([0, 0.58, 1, 1.58, 2, 1.58, 1]).fast(0.5)      # 1 V/oct steps
    wobble, grit = Lfo(0.11, lo=0.5, hi=4), Lfo(0.07, lo=0, hi=5)

# voice A: a sine stroke, upright. `frequency` is cycles across the frame, so 1 V/oct from 2.
voiceA = wave(2 ** pitch * 2, wobble * 0.04, 0.02, 0, 0, 1.0, 0).rotate(1.5708).charcoal(grit * 0.12, 1.0, 100)
voiceA.out(o0)
trackA = Soundtrack(o0, x=0.5, width=0.004, fps=24, gain=0.5)      # ±5 V

# voice B: a damped harmonograph turning slowly; the column through it is an irregular pulse train
voiceB = harmonograph(3, 2 ** pitch, 3.01, 0, 0, 0.5, 2, 0.45, 0.015, 0.3, 0.15).charcoal(grit * 0.1, 1.0, 90)
voiceB.out(o2)
trackB = Soundtrack(o2, x=0.5, width=0.004, fps=24, gain=0.5)

# the picture: both voices with trails, the read column marked, through the print
src(o1).mult(solid(0.82, 0.82, 0.82)) \
    .layer(src(o0).color(0.9, 0.95, 1.0)).layer(src(o2).color(1.0, 0.75, 0.4)) \
    .layer(gate(0.004, 1.0, 0.0, 0.002).color(1, 0.2, 0.1, 0.35)) \
    .out(o1)

try:
    out = CvOut(channels=(0, 1, 2))                     # ES-3 outputs 1-3
except Exception:
    out = CvOut(file="painted_oscillator.wav", channels=(0, 1, 2), fps=24, samplerate=48000)
    atexit.register(out.close)                          # written when the render ends
out.send(0, trackA).send(1, trackB).send(2, Probe(o0, 0.5, 0.5, 1.0, lo=0, hi=5))   # audio, audio, brightness as 0-5 V

render(o1)
