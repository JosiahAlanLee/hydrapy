"""Render a captioned demoscene tour; seconds are per chapter.

    python docs/demoscene_demo.py --out demoscene.mp4 --contact demoscene.png

Reuses the streaming tour renderer. Hardware encoding is probed by default.
"""
import argparse
import math
import generative_demo as tour
from hydrapy import save_mp4

tour.CHAPTERS = [
    ("demoscene_materials.py", 1, "01 / RELIEF", "Procedural height + moving light + specular highlights"),
    ("demoscene_materials.py", 2, "02 / REFLECTION", "Another Hydra chain becomes the reflected environment"),
    ("demoscene_metaballs.py", 0, "03 / IMPLICIT SURFACES", "Smooth distance unions + normal-driven material"),
    ("demoscene_wire.py", 0, "04 / SPLINE SWEEP", "Retained 3D paths + near-plane clipping + SVG export"),
    ("demoscene_sdf3d.py", 0, "05 / DISTANCE GEOMETRY", "Composed solids + normal pass + authored camera tracks"),
    ("demoscene_fractal.py", 0, "06 / RECURSIVE FOLDS", "Kaleidoscopic IFS with a bounded sphere tracer"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demoscene.mp4")
    parser.add_argument("--contact")
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--encoder", default="auto")
    args = parser.parse_args()
    size = tuple(map(int, args.size.lower().split("x")))
    if len(size) != 2 or min(size) < 64 or not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.fps)):
        parser.error("positive duration/fps and a size of at least 64x64 required")
    save_mp4(tour.frames(size, args.fps, args.seconds, args.contact), args.out,
             fps=args.fps, encoder=args.encoder,
             on_encoder=lambda e: print("Encoder:", e.name, flush=True))
