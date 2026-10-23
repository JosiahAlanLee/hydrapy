"""delay(oN, k): an output as it was k frames ago. Each output keeps a ring of textures as deep as
the largest delay used. Optical-printer moves without spare outputs: echoes, A/B rolls, frame holds.
"""
from hydrapy import *

set_bpm(120)
# o0: something that moves
shape(5, 0.18, 0.02).scroll(Lfo(0.3, "tri", -0.3, 0.3), Lfo(0.21, "sine", -0.2, 0.2)).rotate(Time() * 2) \
    .color(1, 0.6, 0.2).add(shape(3, 0.1, 0.01).scroll(Lfo(0.43, lo=-0.35, hi=0.35), 0).color(0.2, 0.6, 1)).out(o0)

# o1: echoes — the same output 0, 4, 8, 12 frames back, each tinted
src(o0).add(delay(o0, 4).color(0.8, 0.3, 0.9), 0.7).add(delay(o0, 8).color(0.3, 0.9, 0.8), 0.5) \
       .add(delay(o0, 12).color(0.9, 0.9, 0.3), 0.35).out(o1)

# o2: A/B roll — alternate between now and a 6-frame-old frame every beat, as a projector would
a_or_b = Clock(1).smoothstep().quantize(1)
src(o0).blend(delay(o0, 6), a_or_b).out(o2)

# o3: the difference between now and 2 frames ago: motion only
src(o0).diff(delay(o0, 2)).thresh(0.1, 0.05).color(1, 0.2, 0.2).out(o3)
render(o1)
