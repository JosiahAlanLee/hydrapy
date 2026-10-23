"""Recipes: Python functions that compose chains. Nothing here is a new op; it is how the
paint-on-film ops are meant to be put together.  Copy, edit, ignore.

    from hydrapy import *
    from hydrapy.recipes import handpainted, mothlight, film

    film(handpainted(seed=SH(8, 0, 99), strip=0)).out(o0)     # a new painting every 1/8 s
    film(handpainted(seed=3, strip=24)).out(o1)                # one long painted strip running at 24 fps
    film(mothlight(bank=Bank("wings/"))).out(o2)               # your light-table photos, step-printed

Three ways to get Brakhage's time:
  * a fresh painting per frame:   seed=SH(24, 0, 99)      (SH(8, …) holds each one for three frames: step printing)
  * paint that runs along the strip: strip=24            (the painting is one tall image; the gate moves one frame height per frame)
  * both, with different seeds per layer, is also allowed.
"""
from __future__ import annotations

from typing import Optional, Sequence

from . import (Chain, Drift, Lfo, SH, Seq, Signal, Time, crackle, drops, dust, fbm, scratches, solid, src, veins)


def _offset(seed, k: float):
    """seed + k for numbers and signals alike."""
    return seed + k


def handpainted(seed=0, hues: Sequence[float] = (0.02, 0.62, 0.12), density: Sequence[float] = (0.5, 0.45, 0.35),
                scale: Sequence[float] = (3, 5, 8), stretch: Sequence[float] = (1.2, 2.5, 1.0), size: float = 0.55,
                rim: float = 0.6, texture: float = 0.4, bleed: int = 0, cracks: float = 0.3, leader: str = "clear",
                strip: float = 0.0, sat: float = 0.9) -> Chain:
    """Dye layers on film leader.

    ``hues``/``density``/``scale``/``stretch`` describe one layer each (so three layers by default).
    ``bleed`` > 0 runs that many ink-spreading iterations per layer.  ``cracks`` stains dried-paint
    crackle on top.  ``leader`` is 'clear' (white light through dye) or 'black' (unexposed stock,
    scratched so light gets through, then dyed).  ``strip`` = fps makes the painting a strip that
    moves one frame height per frame.
    """
    if leader == "black":
        chain = solid(0, 0, 0).lighten(scratches(3.0, 0.9, 1.5, 1.0, 24, _offset(seed, 41)), 1.0) \
                              .lighten(dust(0.4, 2.0, 0.5, _offset(seed, 43)), 0.8)
    else:
        chain = solid(1, 1, 1)
    pan = (lambda c: c.pan(0, 0, 0, strip)) if strip else (lambda c: c)   # the strip moves under the gate
    for i, (h, d, sc, st) in enumerate(zip(hues, density, scale, stretch)):
        layer = pan(drops(sc, size, d, st, rim, 0.02, 0.15, texture, _offset(seed, 7 * i), hue=h, hueSpread=0.12, sat=sat))
        if bleed:
            layer = layer.bleed(0.08, 0.15, 0.5, iterations=int(bleed))
        chain = chain.stain(layer, 1.0 - 0.15 * i)
    if cracks:
        chain = chain.stain(solid(0, 0, 0).alpha(pan(crackle(7, 0.03, 0.4, _offset(seed, 29)))), cracks)
    return chain


def mothlight(seed=0, bank=None, scale: float = 2.5, density: float = 0.5, hue: float = 0.1, sat: float = 0.55,
              membrane: float = 0.6, veining: float = 0.6, strip: float = 0.0) -> Chain:
    """Translucent organic matter on a light table.

    With ``bank`` (a ``Bank`` or ``Quilt`` of your own photos: leaves, wings, grasses, shot against
    white) the photos become the dye: dark matter blocks light, pale matter tints it.  Without one,
    procedural wings: stretched drops whose thickness follows a ridged-noise membrane, with dark
    veins inside.
    """
    leader = solid(1, 1, 1)
    pan = (lambda c: c.pan(0, 0, 0, strip)) if strip else (lambda c: c)
    if bank is not None:
        photo = pan(src(bank))
        layer = photo.alpha(photo.invert().contrast(1.4))
        chain = leader.stain(layer, 1.2)
    else:
        # bake(): the wings are used three times below but rendered once on the GPU
        wings = pan(drops(scale, 0.7, density, 2.4, 0.3, 0.01, 0.3, 0.3, seed, hue=hue, hueSpread=0.06, sat=sat)).bake()
        mem = pan(veins(6, 5, 2.0, 0, _offset(seed, 3))).mult(solid(membrane, membrane, membrane)).add(solid(1 - membrane, 1 - membrane, 1 - membrane))
        layer = wings.alpha(wings.a().mult(mem))
        dark = solid(0, 0, 0).alpha(pan(veins(4, 6, 3.0, 0, _offset(seed, 4))).thresh(0.6, 0.1).mult(wings.a()))
        chain = leader.stain(layer, 0.9).stain(dark, veining)
    return chain


def film(chain: Chain, grain: float = 0.08, flicker: float = 0.06, weave: float = 0.003, bloom: float = 0.25,
         scratch: float = 0.4, specks: float = 0.15, vignette: float = 0.4, focus: float = 0.0, fps: float = 24,
         seed: float = 0.0) -> Chain:
    """The print and the projector: gate weave, flicker, scratches, dust, grain, bloom, vignette.

    Scratches run along the strip (their seed drifts slowly; they jitter per frame); dust and grain
    are new every frame.  ``focus`` > 0 blurs by that many pixels (a soft projector).
    """
    out = chain
    if weave:
        out = out.pan(SH(fps, -weave, weave, seed + 1), SH(fps, -weave, weave, seed + 2))
    out = out.bake()                      # the painting is rendered once; bloom and the rest sample it
    if focus:
        out = out.blur(focus)
    if bloom:
        out = out.add(out.blur(2, iterations=2).gamma(2.0), bloom)
    if scratch:
        out = out.lighten(scratches(scratch, 0.7, 1.2, 1.0, fps, Drift(0.15, 0, 50, seed + 3)), 0.8)
    if specks:
        out = out.mult(dust(specks, 1.5, 0.25, SH(fps, 0, 99, seed + 4)).invert(), 0.9)
    if flicker:
        g = 1.0 + SH(fps, -flicker, flicker, seed + 5)
        out = out.color(g, g, g)
    if grain:
        out = out.grain(grain, fps)
    if vignette:
        out = out.vignette(vignette, 0.55)
    return out
