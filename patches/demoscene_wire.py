"""Retained 3D spline paths, projected for animation or a chosen SVG frame.

    python patches/demoscene_wire.py --svg sweep.svg --time 2

Wireframes are transparent. The FrameSource animation renders in Python/GL;
bake into a Bank for HTML. SVG contains the original projected line paths.
"""
import numpy as np
from hydrapy import *


def wire_at(t):
    controls = [[-.7, -.65, .1], [.4, -.45, .3*np.sin(t*.4)],
                [-.4, .05, .45], [.4, .4, -.4], [-.1, .75, .15]]
    curve = catmull_rom(controls, samples=16)
    radii = .07 + .035*np.sin(np.linspace(0, 4*np.pi, len(curve)) + t*.7)
    return sweep_wire(curve, radii, sides=10, twist=t*.4, rings=3)


def drawing_at(t, aspect=16/9):
    return project_wire(wire_at(t), rotX=.2, rotY=t*.25, distance=2.5, fov=48, aspect=aspect)


def draw(t):
    return Lines(drawing_at(t), canvas=(640, 360), fit=False, stroke=1.3, color=(.45, .9, .8)).render()


solid(.015, .025, .035).layer(src(FrameSource(draw))).out(o0)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--time", type=float, default=2)
    args = parser.parse_args()
    # to_svg fits geometry uniformly; use square coordinate units for that path.
    Path(args.svg).write_text(to_svg(drawing_at(args.time, aspect=1), size=(2970, 2100), stroke=2), encoding="utf-8")
