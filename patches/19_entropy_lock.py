"""Entropy locking, after aebrer (CC0): a seeded random stream that keeps falling back into earlier
sequences, driving pixel feedback — samples of the image shredded onto itself, bands torn sideways —
so the same glitches recur and the recurrence grows a fractal texture.

Fx("oo...") is fxhash's own generator bit for bit; EntropyLock draws from a locked copy of it at a
steady rate. Change the hash for a new edition; raise `lock` for shorter loops.
"""
from hydrapy import *

HASH = "ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567"
fx = Fx(HASH)                                           # the piece's fixed parameters, as on fxhash
lock = EntropyLock(HASH, rate=24, lock=fx.rnd_btw(0.003, 0.02))   # the locked stream, 24 draws a second
hue = fx.rnd_btw(0, 1)
sides = fx.rnd_int(3, 7)
feed = osc(fx.rnd_btw(10, 40), 0.1, fx.rnd_btw(0.5, 2)).kaleid(sides).hue(hue)

# o0: the texture. Everything on screen is sampled by the next frame's shred/tear, which the lock steers.
src(o0).shred(lock, 4, fx.rnd_btw(0.15, 0.35), fx.rnd_btw(1.0, 2.0)) \
       .tear(lock * 7 + 3, 3, 0.1, fx.rnd_btw(0.1, 0.5)) \
       .tear(lock * 13 + 5, 2, 0.08, 0.2, vertical=1) \
       .blend(feed, 0.04) \
       .out(o0)

# o1: the same loop seen through posterize + edge, closer to the print look
src(o0).posterize(5, 0.8).edge(1).invert().contrast(1.4).out(o1)
render(o0)
