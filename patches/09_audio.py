"""Audio-reactive. Offline it reads the file and renders in sync; the HTML export embeds the track."""
import os
from hydrapy import *

a = Audio.from_file(os.path.join(os.path.dirname(__file__), "test.wav")).set_scale(6).set_cutoff(1)

osc(20, 0.1, a.fft[0] * 4).rotate(a.fft[1] * 2, 0.1) \
    .modulateScale(shape(4, 0.4 + a.vol).color(1, 0.4, 0.2), a.fft[0] * 2, 1) \
    .kaleid(3 + a.flux.quantize(1) * 4) \
    .scale(1 + a.flux * 3) \
    .out(o0)
src(o0).blend(src(o1).scale(1.02).rotate(0.01), 0.7).out(o1)
render(o1)
