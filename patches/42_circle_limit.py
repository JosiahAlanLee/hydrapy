"""Circle Limit: a plotter motif drawn once, tiled over the hyperbolic plane, and scrolled along a geodesic.

`hyperkaleid(p, q)` folds the frame into the fundamental triangle of the (2, p, q) triangle group, so whatever
the chain draws in the wedge from the centre to the top edge (between 0 and 180/p degrees) tiles the Poincaré
disk, or the band (model 1), which fills a 16:9 frame. `shift` is a hyperbolic distance along a geodesic, so
the tiling flows without a seam; `hyperfade()` fades the sub-pixel tiles at the boundary, which would otherwise
shimmer. Try p, q = 3, 7 or 7, 3; q=100 is the modular group's cusped tiling.
"""
from hydrapy import *
from hydrapy.drawing import Lines

# the motif: a fern frond and a ring, as polylines in the wedge (0..1 frame coordinates, y up)
frond = [[(.5 + k * .012, .5 + k * .011 + .02 * __import__("math").sin(k * .9)) for k in range(30)]]
for k in range(4, 28, 4):
    x, y = .5 + k * .012, .5 + k * .011
    frond.append([(x, y), (x - .02 + .004 * k / 28, y + .035)])
    frond.append([(x, y), (x + .03, y + .01)])
ring = [[(.78 + .035 * __import__("math").cos(a / 24 * 6.2832), .62 + .035 * __import__("math").sin(a / 24 * 6.2832)) for a in range(25)]]
ink = Lines(frond + ring, stroke=3, color=(.12, .13, .3), fit=False)

paper = solid(.95, .93, .86).mult(fbm(3, 2, 0).color(.06, .06, .06).add(solid(.94, .94, .94)))
motif = paper.over(src(ink))
tiled = motif.hyperkaleid(4, 5, shift=Time() * .12, direction=Lfo(.02, lo=0, hi=6.2832), turn=Time() * .03, model=1)
tiled.mult(hyperfade(.18, 1)).out(o0)
render(o0)
