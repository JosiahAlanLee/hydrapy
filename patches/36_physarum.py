"""Physarum: slime-mould transport networks (Jeff Jones, 2010).

o0  32,768 agents laying and following trails on a 320x180 grid: a network that keeps re-routing
o1  the same mould fed by a picture — it follows the bright parts of o2 as well as its own trails
o2  the food: a slow voronoi, so the network grows along the cell walls and migrates as they drift
o3  the network retuned live: the sensor angle swings between long straight veins and fine mesh
    (CV it: sensor_angle=cv.cv[0] * 0.1 + 0.6)
"""
from hydrapy import *

src(Physarum(seed=1)).out(o0)

voronoi(5, 0.15, 0.1).out(o2)
src(Physarum(food=o2, food_weight=0.6, colour=(0.6, 0.9, 1.0), seed=2)).out(o1)

swing = Lfo(0.03, "tri", lo=0.25, hi=1.1)
src(Physarum(sensor_angle=swing, turn=swing * 0.7, colour=(1.0, 0.55, 0.35), seed=3)) \
    .palette(0.0, 0.1, 0.2, Lfo(0.01), 0.8).out(o3)

render(o0)
