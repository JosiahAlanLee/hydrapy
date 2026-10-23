"""A signed field supplies both moving contour bands and metallic blobs."""
from hydrapy import *

field = sdfMetaballs(count=6, radius=.13, spread=.26, smoothness=.16, speed=.6, seed=2)
normals = field.invert().normalMap(.2, radius=3)
light = normals.relight(azimuth=Time()*.3, elevation=.85, ambient=.2, specular=.6)
metal = osc(3, .1, .5).rotate(.8).envMap(normals).mult(light).alpha(field.sdfMask())
solid(.025, .035, .055).layer(metal).out(o0)
field.sdfBands(30, .18, Time()*.15).color(.7, .85, 1).out(o1)
render(o0)
