"""The fold op is a wavefolder: push a smooth gradient past 1 and it folds back, like a Serge/Buchla folder."""
from hydrapy import *

rings(4, 1).add(gradient().r().mult(osc(3, 0.2)), 0.8) \
    .fold(Lfo(0.05, "tri", lo=1, hi=9)) \
    .palette(0, 0.33, 0.67, Lfo(0.03), 1.0) \
    .dither(6, 2) \
    .out(o0)
