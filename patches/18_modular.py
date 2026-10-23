"""The modular as the controller, the image as a voltage source.

Live (hydrapy live patches/18_modular.py): an ES-6 arrives as DC-coupled audio inputs on the ADAT
channels of your interface, the ES-3 takes DC-coupled outputs the same way. Pick the device and
zero-based channel numbers from `hydrapy devices`. Offline (hydrapy render): record the same CV as
a multichannel WAV next to this file as cv.wav and the render follows it frame-accurately; with no
file, synthetic signals stand in so the patch always runs.
"""
import os
from hydrapy import *

HERE = os.path.dirname(__file__)
LIVE = os.environ.get("HYDRAPY_LIVE") == "1"          # set when you run `HYDRAPY_LIVE=1 hydrapy live ...`
CV_FILE = os.path.join(HERE, "cv.wav")

if LIVE:
    cv = CvIn(device="ES-8", channels=[8, 9, 10, 11], volts_per_fs=10)      # ES-6 -> ADAT in 1-4, say
    out = CvOut(device="ES-8", channels=[8, 9], volts_per_fs=10)            # ADAT out 1-2 -> ES-3
    default_patch().transport = Link(bpm=120)                              # beat grid from Ableton Link
    lfo_a, lfo_b, env_cv, trig = cv.uni(0), cv.bi(1), cv.uni(2), cv.trig(3, threshold=1.0, length=0.6)
elif os.path.exists(CV_FILE):
    cv = CvIn.from_file(CV_FILE, channels=[0, 1, 2, 3], volts_per_fs=10)
    out = CvOut(file=os.path.join(HERE, "cv_out.wav"), channels=[0, 1], fps=30)
    lfo_a, lfo_b, env_cv, trig = cv.uni(0), cv.bi(1), cv.uni(2), cv.trig(3, threshold=1.0, length=0.6)
else:
    out = None
    lfo_a, lfo_b, env_cv, trig = Lfo(0.11, "tri"), Lfo(0.07, "sine", -1, 1), Lfo(0.05, "saw"), Clock(0.5)

set_bpm(120)
env = Env(trig, attack=0.02, decay=1.0, curve=2)       # retriggered by the gate on channel 3

# o0: the modular drives the synth
osc(lfo_a.map(6, 50), 0.1, 1 + lfo_b).rotate(lfo_b * 2).kaleid(env_cv.map(2, 8).quantize(1) + 1) \
    .scale(1 + env * 0.6).fold(1 + env_cv * 2) \
    .modulate(src(o0).scale(1.02), 0.1) \
    .out(o0)

# o1: a quieter view, and two probes that go back out as voltages
src(o0).blur(2).palette(0, 0.33, 0.67, Lfo(0.03)).out(o1)
centre = Probe(o0, 0.5, 0.5, 0.08, "luma", lo=0, hi=8)     # 0..8 V from the centre's brightness
edge = Probe(o0, 0.1, 0.5, 0.08, "r", lo=-5, hi=5)         # bipolar from the left edge's red
if out is not None:
    out.send(0, centre).send(1, edge)
render(o0)
