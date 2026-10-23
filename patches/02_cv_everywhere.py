"""Every knob takes a signal, and any knob takes a *chain* (per-pixel CV) — Hydra can't do osc(noise())."""
from hydrapy import *

set_bpm(120)
freq = Lfo(0.07, lo=6, hi=40)                    # slow LFO on oscillator frequency
jitter = SH(8, lo=-0.4, hi=0.4)                  # sample & hold on the colour offset
env = Env(Clock(1), attack=0.02, decay=0.4)      # AD envelope once per beat

osc(freq, 0.05, 1.5 + jitter) \
    .rotate(fbm(2, 3, 0.2) * 3, 0) \
    .rotate(Drift(0.3, 0, 6.28)) \
    .scale(1 + env * 0.5) \
    .kaleid(noise(1.5, 0.05).posterize(3, 1).r() * 4 + 3) \
    .fold(1 + env * 2) \
    .out(o0)
