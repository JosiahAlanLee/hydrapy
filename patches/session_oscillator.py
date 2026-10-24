"""OSC-controlled drawn voice; no CvIn/CvOut, audio device, or neural model.

Open in Studio with the CPU backend, then arm live I/O and run the example
session. Soundtrack still generates the optical audio for recording/export;
native live speaker playback is deliberately not configured by this patch.
"""
import os
from hydrapy import *

control = OscIn(host="127.0.0.1", port=int(os.environ.get("HYDRAPY_SESSION_PORT", "57130")))
pitch = control.value("/voice/pitch", default=0, lo=-2, hi=3)
grit = control.value("/voice/grit", default=.05, lo=0, hi=1)
level = control.value("/voice/level", default=.4, lo=0, hi=1)

voice = wave(2 ** pitch * 2, .08, .02).rotate(1.5708).charcoal(grit, 1, 100)
solid(0, 0, 0).layer(voice).mult(solid(level, level, level)).out(o0)
Soundtrack(o0, x=.5, width=.004, fps=24, gain=.25, keep=4)
render(o0)
