"""Gray-Scott reaction-diffusion on o0, seeded by a wandering hexagon, then coloured and fed back into itself."""
from hydrapy import *

seed = shape(6, 0.06, 0.01).scroll(Lfo(0.11, lo=-0.3, hi=0.3), Lfo(0.17, "tri", lo=-0.3, hi=0.3)).color(0, 1, 0)

src(o0).rd(feed=0.037, kill=0.06, iterations=8).add(seed, 0.1).out(o0)
src(o0).g().palette(0.1, 0.4, 0.7, Lfo(0.02), 1.5).vignette(0.4).out(o1)
render(o1)
