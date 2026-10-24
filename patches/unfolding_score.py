"""Opus's patch 41 with externally supplied fold/drop triggers.

Only the event clock changes: parity, symmetric draws, rule regions and
recursive enlargement follow 41_unfolding.py. Rendering uses a fixed cell grid.
"""
import hydrapy as H

INNER = (16, 16, 17, 16, 17, 16, 17)
OUTER = (8, 4, 9, 13, 7, 5, 8, 15, 2, 4, 14)


def make_field(controls, size=(320,180)):
    p=H.Patch(size=size)
    sig=lambda name:H.Fn(lambda _,name=name:controls[name])
    outer,inner,radius,fold,drop,seed=(sig(k) for k in ("outer","inner","radius","fold","drop","seed"))
    n,m=size
    disc=H.shape(64,radius,.002).mirrorgrid(n,m)
    outside=disc.invert()
    rule=H.solid(outer,outer,outer).add(disc.mult(H.solid(inner-outer,inner-outer,inner-outer)))
    pace=H.solid(1,1,1).sub(outside.mult(H.solid(.25,.25,.25)))
    flips=outside.mult(H.solid(.0004,.0004,.0004))
    glyph=(H.noise(30,0).scroll(seed,seed*.53).thresh(.1,.001)
           .mult(H.shape(4,.022,.001)).mirrorgrid(n,m))
    centre=H.shape(4,.05,.001).mirrorgrid(n,m)
    enlarged=H.src(p.o1).pixelate(n,m).scale(2)
    (H.src(p.o1).ca(rule,pace,flips,.95,0,sym=1)
     .blend(enlarged,fold)
     .mult(H.solid(1,1,1).sub(centre.mult(H.solid(drop,drop,drop))))
     .add(glyph,drop).out(p.o1,patch=p))
    return p.render(p.o1)
