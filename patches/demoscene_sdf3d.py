"""A composed 3D distance field, with sparse timeline control of its camera."""
from hydrapy import *

cues = Timeline({
    "rotation": Track([(0, -.5, "smooth"), (4, .8, "smooth"), (8, 2.4)]),
    "distance": Track([(0, 3.8, "smooth"), (3, 2.9, "smooth"), (8, 3.6)]),
})
scene = SDF3.torus(.65, .22).rotate(.4, 0, .2).union(
    SDF3.sphere(.42).translate(Lfo(.12, lo=-.5, hi=.5), .15, 0), .22)
scene = scene.subtract(SDF3.box(.13, .9, .13).rotate(0, 0, .3))
normals = raymarch(scene, rotY=cues["rotation"], distance=cues["distance"], mode=1)
lit = normals.relight(azimuth=.8, elevation=.8, ambient=.15, diffuse=.7, specular=.45).color(.55, .76, 1)
solid(.015, .02, .04).layer(lit).out(o0)
raymarch(scene, rotY=cues["rotation"], distance=cues["distance"], mode=2).out(o1)
render(o0)
