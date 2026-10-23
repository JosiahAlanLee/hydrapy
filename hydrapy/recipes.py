"""Recipes: Python functions that compose chains. Nothing here is a new op; it is how the
paint-on-film and video-decay ops are meant to be put together.  Copy, edit, ignore.

    from hydrapy import *
    from hydrapy.recipes import handpainted, mothlight, film, vhs, super8

    film(handpainted(seed=SH(8, 0, 99), strip=0)).out(o0)     # a new painting every 1/8 s
    film(handpainted(seed=3, strip=24)).out(o1)                # one long painted strip running at 24 fps
    film(mothlight(bank=Bank("wings/"))).out(o2)               # your light-table photos, step-printed
    vhs(osc(8).kaleid(3), generation=3).out(o3)                # a third-generation dub
    super8(src(Image("holiday.jpg")), stock=2).out(o0)         # faded Super 8, 18 fps

Three ways to get Brakhage's time:
  * a fresh painting per frame:   seed=SH(24, 0, 99)      (SH(8, …) holds each one for three frames: step printing)
  * paint that runs along the strip: strip=24            (the painting is one tall image; the gate moves one frame height per frame)
  * both, with different seeds per layer, is also allowed.
"""
from __future__ import annotations

from typing import Optional, Sequence

from . import (Chain, Clock, Decay, Drift, Env, Lfo, Output, SH, Seq, Signal, Time, arc, crackle, drops, dust, fbm, harmonograph, scratches,
               set_frame_rate, solid, src, veins, wave)
from . import dropouts as _dropouts, gate as _gate


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


def vhs(chain: Chain, tracking: float = 0.5, bleed: float = 0.5, noise: float = 0.3, dropouts: float = 0.5,
        ringing: float = 0.5, generation: int = 1, fps: float = 30, seed: float = 0.0) -> Chain:
    """The picture recorded to tape and played back: composite colour bleed, edge ringing, line
    jitter and a drifting tracking band, head-switching tear at the bottom, dropouts, hiss.

    ``generation`` dubs the tape onto itself that many times; every copy is a little softer,
    noisier and shakier than the last (three or four is where the picture starts to swim).
    Every strength is 0..1-ish; 0 switches that stage off.
    """
    out = chain
    gens = max(int(generation), 1)
    for g in range(gens):
        k = 1.0 + 0.6 * g                                   # each dub is worse than the one before
        out = out.yiq(0.008 + 0.03 * bleed * k, 1.0 + 2.0 * bleed * k, 0.002 + 0.005 * bleed * k)
        if ringing:
            out = out.ringing(0.6 * ringing, 2 + g)
        if tracking:
            out = out.linejitter(0.006 * tracking * k, fps, 0.002 * tracking, seed + g) \
                     .headswitch(0.04 + 0.03 * tracking, 0.12 * tracking * k, fps, seed + 10 + g)
        if dropouts:
            out = out.lighten(_dropouts(0.6 * dropouts * k, 0.25 + 0.1 * g, fps, seed + 20 + g), 0.9)
        if noise:
            out = out.grain(0.07 * noise * k, fps, 1)
    # tape never quite reaches black or white
    return out.contrast(1.0 - 0.06 * gens).brightness(0.015 * gens)


def super8(chain: Chain, stock: int = 0, fps: float = 18, grain: float = 0.15, weave: float = 0.004,
           flicker: float = 0.08, focus: float = 0.6, halation: float = 0.25, specks: float = 0.15,
           scratch: float = 0.3, gate: float = 0.95, vignette: float = 0.3, hold: bool = True,
           seed: float = 0.0) -> Chain:
    """Small-gauge film: a soft lens, a stock (0 Kodachrome, 1 Ektachrome, 2 faded, 3 Tri-X),
    reddish halation around highlights, 18 fps judder, gate weave, flicker, coarse grain, dust,
    scratches, the rounded projector gate and a vignette.

    ``hold=True`` sets the default patch's frame rate to ``fps`` (the whole patch steps at that
    rate: ``set_frame_rate(None)`` releases it, or pass hold=False and set it yourself).
    """
    if hold and fps:
        set_frame_rate(fps)
    out = chain
    if weave:
        out = out.pan(SH(fps, -weave, weave, seed + 1), SH(fps, -weave, weave, seed + 2))
    out = out.bake()
    if focus:
        out = out.blur(focus)
    if halation:
        out = out.add(out.blur(2.5, iterations=2).gamma(2.5).color(1.0, 0.55, 0.35), halation)
    out = out.filmstock(stock)
    if scratch:
        out = out.lighten(scratches(scratch, 0.6, 1.0, 1.0, fps, Drift(0.12, 0, 50, seed + 3)), 0.7)
    if specks:
        out = out.mult(dust(specks, 1.8, 0.3, SH(fps, 0, 99, seed + 4)).invert(), 0.9)
    if flicker:
        g = 1.0 + SH(fps, -flicker, flicker, seed + 5)
        out = out.color(g, g, g)
    if grain:
        out = out.grain(grain, fps, 2)
    if vignette:
        out = out.vignette(vignette, 0.6)
    if gate:
        out = out.mult(_gate(gate, gate - 0.02, 0.06, 0.008))
    return out


def studie(out: Output, beat: Optional[Signal] = None, comets: int = 6, waves: int = 2, trails: float = 0.8,
           paper: float = 0.45, seed: float = 0.0, *, decay_seconds: Optional[float] = None,
           motion=1.0, surface=0.0, surface_rate=12.0) -> Chain:
    """Fischinger's *Studie* films: charcoal oscillators on black paper, dancing to a beat.

    A ring of comets breathing with the beat, one long comet sweeping round, `waves` sine strokes
    crossing, a pulse at the centre; every stroke on rough paper (`paper`), the previous frame kept
    at `trails` so the forms leave smoke behind them.  ``out`` is the output the chain will be
    assigned to — it reads it back for the trails: ``studie(o1).out(o1)``.  ``beat`` is a phase
    0..1 per beat (default ``Clock(1)`` at the patch bpm; a record is ``Audio.from_file(...)`` with
    its flux as the beat).  Put ``film()`` after it for the print.
    ``decay_seconds`` is the feedback half-life and overrides the old per-frame ``trails``.
    ``motion`` scales the composition's movement/pulses; ``surface`` is independent line-boil
    displacement, applied before the fixed paper tooth, at ``surface_rate`` redraws/second.
    """
    beat = beat if beat is not None else Clock(1)
    hit = Env(beat, 0.06, 0.5, 2.0, 0.0, 1.0) * motion
    def material(c, k, scale=80):
        if surface:
            c = c.boil(surface, surface_rate, 8, seed + k)
        return c.charcoal(paper, 1.0, scale, seed + k)
    sweep_per = 6.2832 / max(int(comets), 1) * 0.8
    ring = material(arc(0.3, 0.0, sweep_per, 0.05, 1.0, 0.0).polarRepeat(comets, Time() * 0.2 * motion + seed)
                    .scale(0.75 + 0.45 * hit), 1)
    sweep = material(arc(0.44, Lfo(0.11, lo=0, hi=6.2832, phase=seed) * motion,
                         2.6, 0.035 + 0.04 * hit, 1.0, 0.0), 2, 60)
    strokes = ring.over(sweep)
    for i in range(int(waves)):
        speed = 0.6 * (1 + i) * (1 if i % 2 == 0 else -1) * motion
        strokes = strokes.over(material(wave(2 + i, 0.14 + 0.14 * hit, 0.02, speed, i * 0.3, 0.92, 1.0)
                                       .rotate(0.25 * i - 0.12), 3 + i, 90))
    strokes = strokes.over(material(arc(0.025 + 0.06 * hit, 0, 6.2832, 0.03, 0, 0), 9, 120))
    strokes = strokes.contrast(1.25)
    if decay_seconds is not None:
        trails = Decay(decay_seconds)
    if trails:
        return solid(0, 0, 0).over(src(out).mult(solid(trails, trails, trails)).over(strokes))
    return solid(0, 0, 0).over(strokes)


def on_paper(pigment: Chain, paper_color=(0.94, 0.90, 0.81), tooth=0.35, texture=0.025, seed=0) -> Chain:
    """Move/warp pigment upstream; the paper grain and tooth stay fixed in image space.

    The pigment should carry its coverage in alpha (strokes do this by default).
    This returns an opaque print; keep a separate pigment output for feedback.
    """
    paper = solid(*paper_color).grain(texture, 0, 2)
    return paper.over(pigment.charcoal(tooth, 0, 80, seed))


def cutout(chain: Chain, levels: float = 3, paper: float = 0.35, boil: float = 0.004, misregister: float = 0.003,
           screen: float = 0, stock: int = 4, fps: float = 12, hold: bool = True, seed: float = 0.0) -> Chain:
    """The flat-graphic animation of the sixties East — Zagreb, Soyuzmultfilm, Pannónia, the Polish
    poster school: flat colour in a few `levels`, redrawn by hand every frame (`boil`), printed
    slightly off register (`misregister`), on paper (`paper`), optionally through a halftone
    `screen` (dot pitch in pixels), graded like Orwo stock, on twos (12 fps).

    ``hold=True`` sets the default patch's frame rate to ``fps``.
    """
    if hold and fps:
        set_frame_rate(fps)
    out = chain.posterize(levels, 1.0)
    if boil:
        out = out.boil(boil, fps, 5, seed)
    out = out.bake()
    if misregister:
        out = out.chroma(misregister, 0.5)
    if screen:
        out = out.halftone(screen, 0.7854, 0.7)
    if paper:
        out = solid(0, 0, 0).over(out.charcoal(paper, 1.0, 60, seed + 1))   # the tooth lives in alpha: flatten it
    out = out.filmstock(stock, 0.6)
    return out.grain(0.035, fps, 2).vignette(0.25, 0.7)
