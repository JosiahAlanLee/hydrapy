"""A Hydra sketch with the same semantics. `hydrapy hydra patches/01_hydra_classic.py` gives back JS to paste into hydra.ojack.xyz.
Time() * 0.1 exports as ({time}) => time*0.1; a Python lambda would be baked into a lookup table instead."""
from hydrapy import *

osc(10, 0.1, 1.2).rotate(Time() * 0.1).kaleid(Seq([3, 4, 5]).fast(0.5).smooth()) \
    .modulate(noise(3, 0.1), 0.2).blend(src(o0).scale(1.01), 0.5).color(1, 0.7, 0.4).out(o0)
