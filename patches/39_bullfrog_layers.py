"""Bullfrog, in layers: a music video for Anchor Rite's "Bullfrog" (Pinned) about how Hydra layers.

The picture is built like the record's cover photograph, in depth planes: an out-of-focus ground
(bokeh), the burrs, and silk threads catching the light. Each plane is its own chain. The piece
adds one way of combining them at a time, captioned with the code that does it, and twice takes
the stack apart on screen:

  0:00 ground        noise and shape() discs, blended and screened
  0:09 + silk        add: thin shape(2) lines, glinting on the highs
  0:47 + burrs       layer + luma: osc() rays in polar(), keyed over the ground (o2)
  0:59 feedback      src(o1) zoomed and blended back into itself
  1:20 exploded      the four planes side by side, each with its code
  1:48 modulate      the burrs refract the ground behind them
  2:15 mask          the cover photograph, seen only through the burrs
  2:45 blend modes   diff / mult / overlay / screen against the photograph, one per phrase
  3:27 marbling      the feedback loop combed and swirled on the beat
  3:44 physarum      a slime mould that feeds on the burrs, screened in
  4:20 exploded      the new planes, and the display inside itself
  4:40 everything    the whole stack
  5:13 peel          the planes come off in reverse, down to the photograph

Section times come from a novelty analysis of the track (band energies and stereo width), snapped
to its beats (115 bpm). HYDRAPY_TRACK=/path/to/Bullfrog.flac drives it from the record (beats and
Hydra-style fft bands); without it, slow LFOs stand in so it can be auditioned in Studio.
HYDRAPY_COVER is the cover image (docs/bullfrog_layers.py extracts the track's embedded art).
Render: python docs/bullfrog_layers.py /path/to/Bullfrog.flac -o bullfrog_layers.mp4
"""
import math
import os

import numpy as np

from hydrapy import *

TRACK = os.environ.get("HYDRAPY_TRACK")
COVER = os.environ.get("HYDRAPY_COVER")
BPM = 115.04
set_bpm(BPM)

# --- the record -------------------------------------------------------------------------------
if TRACK:
    beats = Beats.from_file(TRACK)
    a = Audio.from_file(TRACK, bins=4)
    a.set_cutoff(0).set_scale(1).set_smooth(0.45)
    # raw bins span roughly 2-4, 0.3-3, 0.4-2.2, 0.5-1.8 over this track: map each to 0..1
    bass = ((a.fft[0] - 2.2) * (1 / 1.8)).clip(0, 1)
    body = ((a.fft[1] - 0.4) * (1 / 2.4)).clip(0, 1)
    mids = ((a.fft[2] - 0.4) * (1 / 1.6)).clip(0, 1)
    air = ((a.fft[3] - 0.5) * (1 / 1.3)).clip(0, 1)
    kick = beats.pulse(0.02, 0.45)
    bar = beats.bar(4)
else:
    bass, body, mids, air = Lfo(0.21, lo=0.3, hi=0.8), Lfo(0.07, lo=0.2, hi=0.8), Lfo(0.11, lo=0.2, hi=0.7), Lfo(0.5, lo=0.1, hi=0.6)
    kick = Env(Clock(1))
    bar = Clock(0.25)


def window(t0, t1, fade_in=2.0, fade_out=2.0):
    """1 between t0 and t1 (seconds into the track), with smooth fades: the score."""
    def w(ctx):
        t = ctx.time
        u = min(max((t - t0) / max(fade_in, 1e-6), 0.0), 1.0)
        v = min(max((t1 - t) / max(fade_out, 1e-6), 0.0), 1.0)
        return u * u * (3 - 2 * u) * v * v * (3 - 2 * v)
    return Fn(w, bake_seconds=340, bake_rate=24)


END = 335.9
P = {                                          # when each plane is on screen
    "ground": window(0.0, END - 0.5, 6, 3),
    "silk":   window(8.9, 322.0, 4, 5),
    "burrs":  window(46.8, 318.0, 10, 5),
    "phys":   window(223.9, 314.0, 6, 5),
}
T = {                                          # what is being shown off
    "feedback": window(58.8, 316.0, 6, 6),
    "explode1": window(80.0, 105.5, 2.5, 2.5),
    "modulate": window(107.7, 313.4, 4, 6),
    "mask":     window(135.3, 329.0, 5, 6),
    "modes":    window(164.9, 207.2, 3, 3),
    "marble":   window(207.2, 223.9, 3, 3),
    "explode2": window(260.5, 278.5, 2.5, 2.5),
    "all":      window(279.8, 313.4, 3, 4),
    "photo":    window(318.0, END, 9, 2),
}
PHRASE = 16 * 60 / BPM                         # sixteen beats


def phrase_is(k, n=4):
    return Fn(lambda c: 1.0 if int(math.floor(c.time / PHRASE)) % n == k else 0.0, bake_seconds=340, bake_rate=24)


UMBER = (0.16, 0.10, 0.06)
TAN = (0.76, 0.58, 0.38)
STRAW = (0.93, 0.84, 0.64)
SAGE = (0.36, 0.43, 0.28)
MOSS = (0.16, 0.19, 0.11)
POLLEN = (0.86, 0.72, 0.36)
WIDE = 16 / 9


def grey(v):
    return solid(v, v, v)


def discs(radius, soft, nx, ny, ox, oy):
    """Soft round discs on a staggered grid: shape() in each repeat() cell, made round for its cell."""
    cell = (WIDE * ny) / nx                    # the cell's aspect on a 16:9 frame
    return shape(64, radius, soft).scale(1, 0.72 / cell).repeat(nx, ny, ox, oy)


# --- plane 1: the ground, out of focus ---------------------------------------------------------
def ground():
    tone = solid(*UMBER).blend(solid(*MOSS), noise(0.8, 0.015).add(grey(0.3)))
    tone = tone.blend(solid(*SAGE), noise(1.3, 0.02).thresh(0.35, 0.35).color(0.7, 0.7, 0.7))
    near = discs(0.15, 0.13, 3, 2, 0.43, 0.27).modulate(noise(0.4, 0.01), 0.12).scrollX(0, -0.004)
    far = discs(0.09, 0.07, 5, 3, 0.21, 0.6).modulate(noise(0.6, 0.012), 0.1).scrollX(0, -0.009)
    glow = 0.32 + bass * 0.45 + kick * 0.18
    return tone.screen(near.color(*POLLEN), glow).screen(far.color(*STRAW), glow * 0.55)


# --- plane 2: the burrs (o2) ---------------------------------------------------------------------
def burr(x, y, size, spikes, seed, spin):
    sway = Lfo(0.031 + seed * 0.004, lo=-0.018, hi=0.018, phase=seed * 0.37)      # the stems move in the wind
    x, y = sway + x, sway * 0.6 + y
    # polar(): x becomes the angle, y the radius. Stripes become rays; a threshold that rises with
    # the radius tapers each ray to a point, and noise along the angle gives them ragged lengths.
    r = gradient().g()
    taper = r.mult(grey(1.1)).add(noise(11, 0.04).scrollY(seed), 0.2)
    bracts = osc(spikes * 2 * math.pi, 0, seed).thresh(taper, 0.05).polar()
    spines = osc(spikes * 3.1 * 2 * math.pi, 0, seed * 2).thresh(r.mult(grey(1.5)).add(grey(0.15)), 0.02).polar()
    tips = r.polar()                                                 # 0 at the centre, 1 at the rim
    head = shape(64, 0.3, 0.14)
    form = (bracts.add(spines, 0.6).color(*TAN).mult(tips.mult(grey(0.9)).add(grey(0.35)))   # pale towards the tips
                  .layer(head.color(*UMBER).luma(0.1, 0.05)))                               # the dark seed head
    grow = size * (P["burrs"] * 0.995 + 0.005) * (1 + 0.6 * T["mask"]) * (0.94 + (0.06 + 0.08 * T["all"]) * kick)   # scale(0) divides by zero
    return form.rotate(seed, spin * (1 + 3 * T["all"])).scale(grow, 1 / WIDE).scroll(0.5 - x, 0.5 - y)


(burr(0.66, 0.68, 0.5, 40, 0.3, 0.004)
    .add(burr(0.39, 0.6, 0.42, 36, 1.7, -0.006))
    .add(burr(0.30, 0.3, 0.34, 34, 2.9, 0.005))
    .add(burr(0.56, 0.2, 0.28, 30, 4.1, -0.003))
    .out(o2))


# --- plane 3: silk ---------------------------------------------------------------------------------
def silk():
    s = shape(2, 0.0016, 0.0022)
    threads = (s.rotate(0.55).add(s.rotate(-0.25).scrollY(0.12), 0.8)
                .add(s.rotate(1.1).scrollX(0.21), 0.6).add(s.rotate(-0.9).scrollX(-0.3).scrollY(-0.2), 0.5))
    glint = noise(3, 0.5).thresh(0.62 - air * 0.3, 0.2)            # light runs along them on the highs
    return threads.mult(glint.add(grey(0.2))).color(*STRAW).scrollY(0, 0.002)


# --- the cover, and the slime mould ----------------------------------------------------------------
if COVER and os.path.exists(COVER):
    photo = src(Image(COVER)).scale(1, 1, WIDE)                    # fill the width, crop top and bottom
else:
    photo = ground().contrast(1.4).hue(0.05)
class Mould(Physarum):
    """A Physarum that only starts growing at `start` seconds: a network left running for minutes
    coarsens into a few thick strands, so it is seeded just before its section."""

    def __init__(self, start, **kw):
        super().__init__(**kw)
        self.start = float(start)

    def steps_due(self, t):
        return super().steps_due(t - self.start) if t >= self.start else 0


mould = Mould(220.0, size=(320, 180), agents=(256, 128), food=o2, food_weight=0.4, deposit=0.05, speed=30,
              sensor_angle=0.7 + mids * 0.3, colour=STRAW, exposure=0.55)

# --- o3: the stack; o1: blend modes and feedback ----------------------------------------------------
keyed = src(o2).luma(0.06, 0.05)
m = T["mask"]
(ground().modulate(src(o2), T["modulate"] * (0.05 + mids * 0.12))              # refraction
         .layer(keyed.color(P["burrs"], P["burrs"], P["burrs"]))
         .add(silk(), P["silk"] * (0.5 + air + T["all"] * 0.4))
         .layer(photo.mask(src(o2).color(m, m, m)))                            # the photo through the burrs
         .screen(src(mould), P["phys"] * (0.3 + body * 0.35 + T["all"] * 0.2))
         .out(o3))

# blend modes, one per phrase, against the photograph drifting at another scale
other = photo.scale(1.35).scroll(Lfo(0.013, lo=-0.06, hi=0.06), Lfo(0.009, lo=-0.04, hi=0.04))
d, x, o, s = (T["modes"] * phrase_is(k) * 0.85 for k in range(4))
stack = (src(o3).blend(src(o3).diff(other), d)
                .blend(src(o3).mult(other.brightness(0.25)), x)
                .blend(src(o3).overlay(other), o)
                .blend(src(o3).screen(other.color(*TAN)), s))

feedback = (T["feedback"] * (0.5 + body * 0.3) + T["all"] * 0.15).clip(0, 0.88)
marble = T["marble"]
(src(o1).modulate(noise(2.5, 0.05), 0.003 * T["feedback"])
        .scale(1.0 + 0.005 * T["feedback"] + 0.01 * T["all"] * kick)
        .swirl(0.5, 0.5, marble * (bar - 0.5) * 1.2, 0.28, 0.08)
        .tine(1.5708, 0, marble * (kick - 0.25) * 0.25, 0.04, 0.16)
        .blend(stack, 1 - feedback)
        .out(o1))


# --- o0: the display ----------------------------------------------------------------------------
def rect(x0, x1, y0, y1):
    gx, gy = gradient().r(), gradient().g()
    return gx.thresh(x0, 0.001).mult(gx.thresh(x1, 0.001).invert()).mult(gy.thresh(y0, 0.001)).mult(gy.thresh(y1, 0.001).invert())


QUADRANTS = ((0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25))


def exploded(planes, w):
    """Each plane shrinks into its quadrant as w goes 0 -> 1."""
    shown = solid(0.03, 0.025, 0.02)
    k = 1 - 0.53 * w
    for content, (qx, qy) in zip(planes, QUADRANTS):
        dx, dy = (qx - 0.5) * w, (qy - 0.5) * w
        box = rect(0.5 + dx - 0.5 * k, 0.5 + dx + 0.5 * k, 0.5 + dy - 0.5 * k, 0.5 + dy + 0.5 * k)
        shown = shown.layer(content.scale(k).scroll(-dx, -dy).mask(box))
    return shown


def captions(items, canvas=(1280, 720)):
    """Text that fades in and out on the score, drawn in Python into one texture (one sampler).

    Each caption is rendered once and cropped to its letters; a frame only touches those boxes."""
    rendered = []
    for weight, text, x, y, size in items:
        img = Text(text, font="DejaVuSansMono", size=size, color=STRAW, x=x, y=y, align="left", canvas=canvas).render()
        ys, xs = np.nonzero(img[..., 3])
        if len(ys):
            box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
            rendered.append((weight, box, img[box[0]:box[1], box[2]:box[3]].astype(np.float32) / 255))
    frame = np.zeros((canvas[1], canvas[0], 4), np.uint8)
    dirty = []

    shown = [None]

    def draw(t):
        ctx = Ctx(time=t)
        weights = tuple(round(float(weight(ctx)), 3) for weight, _, _ in rendered)
        if weights == shown[0]:
            return None                       # nothing moved: FrameSource keeps (and GL doesn't re-upload) the frame
        shown[0] = weights
        for y0, y1, x0, x1 in dirty:
            frame[y0:y1, x0:x1] = 0
        dirty.clear()
        for weight, (y0, y1, x0, x1), img in rendered:
            v = float(weight(ctx))
            if v <= 1e-3:
                continue
            under = frame[y0:y1, x0:x1].astype(np.float32) / 255
            a = img[..., 3:4] * v
            under[..., :3] = under[..., :3] * (1 - a) + img[..., :3] * a
            under[..., 3:4] = a + under[..., 3:4] * (1 - a)
            frame[y0:y1, x0:x1] = (under * 255 + 0.5).astype(np.uint8)
            dirty.append((y0, y1, x0, x1))
        return frame
    return FrameSource(draw)


first = [ground(), src(o2), silk(), src(o1)]
first_code = ["ground  noise().blend().screen(shape().repeat())", "o2  osc().thresh(gradient()).polar()",
              "silk  shape(2).rotate().add().mult(noise())", "o1  src(o1).blend(ground.layer(o2).add(silk))"]
second = [photo.mask(src(o2)), src(mould), src(o1), src(o0)]
second_code = ["photo.mask(src(o2))", "Physarum(food=o2)", "o1  .swirl().tine() inside the feedback", "o0  the display, inside itself"]

STEPS = [                                      # what was just added, in the code that adds it
    (1.0, 8.0, "o1 = ground    noise().blend(solid()).screen(shape(64).repeat())"),
    (10.0, 17.0, "+ silk        .add(shape(2).rotate().mult(noise()))"),
    (48.0, 55.0, "+ burrs       .layer(src(o2).luma())    o2 = osc().thresh().polar()"),
    (60.0, 67.0, "feedback      src(o1).scale(1.005).blend(stack)"),
    (109.0, 116.0, "modulate      ground.modulate(src(o2))"),
    (136.5, 143.5, "mask          photo.mask(src(o2))"),
    (166.0, 172.0, "diff          stack.diff(photo.scale(1.35))"),
    (174.3, 180.3, "mult          stack.mult(photo)"),
    (182.6, 188.6, "overlay       stack.overlay(photo)"),
    (191.0, 197.0, "screen        stack.screen(photo)"),
    (208.5, 215.5, "marbling      src(o1).swirl().tine()   inside the feedback"),
    (225.0, 232.0, "+ physarum    .screen(src(Physarum(food=o2)))"),
    (281.0, 288.0, "everything"),
    (314.0, 320.0, "- physarum  - burrs  - silk"),
]
items = [(window(t0, t1, 0.8, 1.2), text, 0.04, 0.06, 0.03) for t0, t1, text in STEPS]
for weight, codes in ((T["explode1"], first_code), (T["explode2"], second_code)):
    for code, (qx, qy) in zip(codes, QUADRANTS):
        items.append((weight, code, qx - 0.225, qy - 0.212, 0.021))
overlay_text = captions(items)

display = (src(o1).blend(photo, T["photo"])
                  .blend(exploded(first, T["explode1"]), T["explode1"])
                  .blend(exploded(second, T["explode2"]), T["explode2"])
                  .layer(src(overlay_text)))
fade = P["ground"] * 0.98 + 0.02
(display.contrast(1.12).brightness(0.03).saturate(1.08)
        .grain(0.035, 24, 1.5)
        .vignette(0.4, 0.82)
        .color(fade, fade, fade)
        .out(o0))
render(o0)
