"""Neural cellular automata: textures that grow themselves, and heal.

Two rules, each trained from one image (hydrapy nca train <image>; the trainer is
hydrapy/nca_train.py): `gravel` (scikit-image's CC0 gravel photograph) and `painted` (a frame of
hydrapy's own handpainted() film: red and blue stains on white). Neither has seen anything else.

o0  the gravel rule growing from nothing, then holding itself together
o1  the painted rule, wiped wherever o2 is bright: watch it heal behind the moving bar
o2  the damage: a bar sweeping back and forth
o3  the gravel rule mutating: noise pushed into its hidden channels on a slow LFO, then left to
    settle, through a palette (CV this from the modular: perturb=cv.cv[0] * 0.05)
"""
import os

from hydrapy import *

HERE = os.path.dirname(os.path.abspath(__file__))
gravel = os.path.join(HERE, "models", "gravel.npz")
painted = os.path.join(HERE, "models", "painted.npz")

src(NCA(gravel, size=(160, 90), speed=40)).out(o0)

shape(4, 0.5, 0.01).scale(1, 0.12, 4).scroll(Lfo(0.07, "tri", lo=-0.6, hi=0.6), 0).out(o2)
src(NCA(painted, size=(160, 90), speed=40, damage=o2)).out(o1)

mutating = NCA(gravel, size=(128, 72), speed=40, perturb=Lfo(0.05, "pulse", lo=0, hi=0.4, width=0.15), seed=3)
src(mutating).palette(0.0, 0.33, 0.67, Lfo(0.02), 1.2).out(o3)

render(o1)
