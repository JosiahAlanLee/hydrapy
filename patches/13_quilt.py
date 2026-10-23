"""Texture synthesis instead of a GAN: image quilting from one photo, a fresh synthesis per frame.

Point `source` at a photo of an organic texture (lichen, bark, a moth wing, dried paint). If the
file is missing, a procedural stand-in is quilted so the patch still runs. The 16 frames are
synthesised once (about a second each on a CPU) and then shown in random order, 12 a second.
"""
import os
from hydrapy import *
from hydrapy.recipes import film

source = os.path.join(os.path.dirname(__file__), "texture.jpg")
if not os.path.exists(source):
    source = veins(3, 6, 2.5, 0, 7).palette(0.05, 0.25, 0.45, 0.1, 0.8).mult(fbm(3, 4, 0, 0.5, 2).add(solid(0.5, 0.5, 0.5)), 0.6)

q = Quilt(source, size=(320, 180), count=16, block=32, overlap=6, seed=1, rate=12, mode="random")
photo = src(q)
film(solid(1, 1, 1).stain(photo.alpha(photo.invert().contrast(1.3)), 1.1).stain(drops(5, 0.4, 0.3, 1.5, seed=SH(12, 0, 99), hue=0.95), 0.6),
     fps=24).out(o0)
render(o0)
