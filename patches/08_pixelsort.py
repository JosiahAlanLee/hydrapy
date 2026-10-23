"""CPU-only op: pixel sorting runs of bright pixels, feedback-smeared."""
from hydrapy import *

noise(4, 0.3).thresh(0.4, 0.2).mult(osc(20, 0.1, 0.5)).pixelsort(0.3, 0) \
    .blend(src(o0).scrollX(0.002), 0.6).out(o0)
