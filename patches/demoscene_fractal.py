"""A bounded kaleidoscopic IFS: recursive folds ending in a small box."""
from hydrapy import *

field = SDF3.box(.45, .45, .45).fold(
    iterations=3, scale=2.1, x=.7, y=.6, z=.65,
    angle=Lfo(.06, lo=.05, hi=.32)).scale(.85)
normals = raymarch(field, rotX=.35, rotY=Time()*.16, distance=3, steps=160, epsilon=.001, mode=1)
lit = normals.relight(azimuth=Time()*.2, elevation=.9, specular=.3).color(.95, .7, .4)
solid(.015, .02, .035).layer(lit).out(o0)
