"""Normal maps, moving relief lighting, environment reflections and glass.

All four outputs run natively in the standalone HTML exporter.
"""
from hydrapy import *

height = fbm(4).add(veins(3, speed=.12), .3).bake()
n = height.normalMap(.035)
light = n.relight(azimuth=Time()*.35, elevation=.65, specular=.5, shininess=48)
environment = osc(3, .08, .7).rotate(.7).add(gradient(), .25).bake()
n.out(o0)
light.color(.48, .72, .78).out(o1)
environment.envMap(n, Time()*.025).mult(light).out(o2)
grid(12).refractMap(n, .08).color(.3, .8, 1).out(o3)
render(o2)
