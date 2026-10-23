# Film, print and drawn sound

[Documentation](README.md) · [Project home](../README.md)

## Painted film (the Brakhage layer)

Hand-painted film is physics plus artifacts, and both are cheap to model. `hydrapy.recipes`
composes the pieces; the ops underneath are ordinary ops.

**The dye model.** A projector shines white light through clear leader painted with transparent
dyes. Each layer multiplies the light by its transmittance (Beer–Lambert), so red over blue goes
black the way inks do, doubled paint goes dark, and thin paint tints. A *layer* is any chain whose
rgb is the colour the dye lets through and whose alpha is its thickness (straight, not
premultiplied; `alpha(chain)` sets it from another chain's luminance without premultiplying).

```python
leader = solid(1, 1, 1)                                     # clear stock; solid(0,0,0) is black leader
painting = leader.stain(drops(3, 0.6, 0.5, 1.2, seed=1)) \
                 .stain(drops(5, 0.5, 0.5, 2.5, seed=2, hue=0.62), 0.8) \
                 .stain(solid(0, 0, 0).alpha(crackle(7, 0.03)), 0.3)
```

**Sources.** `drops` are dye drops on a jittered grid, thicker at the drying rim and mottled inside,
each its own colour, stretched into strokes with `stretch`; `crackle` is dried-paint cracking
(warped Voronoi borders); `veins` is ridged noise for membranes and leaf veins; `scratches` run
along the strip and jitter per frame; `dust` is specks and hairs in the gate. `bleed` spreads a
layer's edges like ink. All of them are seeded, and the seed can be a signal.

**Time.** Three ways to get his time, which you can combine:

* a fresh painting per frame: `seed=SH(24, 0, 99)`; `SH(8, …)` holds each one for three frames
  (step printing);
* paint that runs along the strip: `.pan(0, 0, 0, 24)` moves one frame height per frame, so the
  painting is one tall image and smears continue from frame to frame — `handpainted(strip=24)`;
* the projector: `film(chain)` adds gate weave, flicker, scratches, dust, grain, bloom and a
  vignette, each frame-rate aware.

**Real matter.** Brakhage pressed moth wings and grasses between strips of tape. Photograph your
own on a light table and point a `Bank` at the folder: `src(Bank("matter/", rate=24, hold=2,
mode="random"))` shows one photo per frame. `mothlight(bank=…)` turns the photos into dye (dark
matter blocks light, pale matter tints it). Banks work on all three backends; the HTML export embeds
the images.

**Instead of a GAN.** `Quilt("texture.jpg", size, count=24)` synthesises `count` new images from
one photo by image quilting (Efros–Freeman: random patches stitched along minimum-error seams).
Every frame is a different image with the same statistics as the original, which is the per-frame
discontinuity a GAN's smooth latent space would work against. About a second a frame on a CPU, no
training; `src(q)` uses the bank. (`quilt()` is also available on its own.) If you do want a learned
model later, the frames hydrapy exports are a reasonable init for an img2img diffusion pass with
independent seeds per frame; that is a separate pipeline.

Patches `11_handpainted.py`, `12_mothlight.py`, `13_quilt.py` show all of it.

## Video synthesis, tape and small-gauge film

The analog video-synth vocabulary (the LZX / Rutt-Etra end of it), then the two media that most
of that work was ever seen on. `patches/20_video_synth.py`, `21_vhs.py`, `22_super8.py`.

```python
noise(2.5).ruttetra(lines=72, amount=Lfo(0.07, lo=0.1, hi=0.4), glow=0.3)   # scan processor: lines lifted by brightness
gradient().bits(gradient().rotate(1.5708), 0, 6).sinefold(2)                 # ramp XOR ramp at 6 bits, through a sine staircase
src(o0).contours(8, 0.2, fill=0.5)                                           # isolines of luminance (a contour keyer)
src(o2).psort(0.3, 0, iterations=8).blend(pic, Lfo(1/6, "pulse", width=0.03)).out(o2)   # a picture loaded every 6 s, then left to melt
```

* **`ruttetra(lines, amount, thickness, glow)`** — a raster of scan lines, each displaced upward by
  `amount` × the brightness beneath it. Rotate or scale the chain before it to tilt the raster.
* **`contours`**, **`sinefold`**, **`bits`**, **`shear`** — a contour keyer; the sine staircase
  shaper (ramps become rainbows); bitwise XOR/AND/OR between two chains quantised to `depth` bits;
  skew.
* **`psort`** — pixel sorting on the GPU. Each iteration is one odd-even transposition step along
  rows (or columns) inside runs brighter than `threshold`, so `iterations=8` sorts eight pixels'
  worth per frame. Feed it back and the picture melts into sorted streaks over seconds; the
  CPU-only `pixelsort` still sorts a frame completely in one go.

Tape: `yiq(chroma, luma, shift)` is composite-video bandwidth (chroma smeared over a fraction of
the width and delayed, luma softened), `ringing` the overshoot of a deck's sharpening circuit,
`linejitter` the time-base error (every scan line shifts by its own amount, worse in drifting
tracking bands), `headswitch` the tear in the bottom band, `dropouts` the white streaks. Film:
`filmstock` (Kodachrome, Ektachrome, faded, Tri-X), `gate` (the rounded projector aperture),
`grain(amount, rate, size)` with `size` in pixels, and **`set_frame_rate(18)`** — the whole patch
holds `time` to 18 steps a second, so everything judders like the projector, signals included
(16 for Regular 8, 24 for sound speed, `None` to release). All of it is CV-able, so bad tracking can
come from an LFO or from your modular.

```python
from hydrapy.recipes import vhs, super8

vhs(pic).out(o0)                                    # first-generation copy
vhs(pic, generation=4).out(o1)                      # dubbed four times: softer, noisier, shakier each time
super8(pic, stock=0).out(o2)                        # Kodachrome, 18 fps, reddish halation, weave, grain, gate
super8(handpainted(seed=SH(6, 0, 99)), stock=2)     # the painted-film recipes, projected
```

## Fischinger, cut paper, and the drawn soundtrack

Three more things from the history of abstract animation. `patches/23_studie.py`, `24_cutout.py`,
`25_ornament_sound.py`.

**Painted oscillators.** Fischinger's *Studie* films (1929–34) are oscillators drawn in charcoal —
comets, sine sweeps, arcs, pulsing to a record — so here the oscillator is a *stroke*:

```python
harmonograph(3, 2, 0, 0, 1.5708, 0, 1, 0.4, 0.012)        # a Lissajous drawn as a line (f3/f4 > 0: damped double pendulum)
wave(3, 0.2, 0.015, 1, 0, 0.9, taper=1)                    # a sine across the frame, thinning at the ends
arc(0.3, 0, 2.5, 0.04, taper=1, speed=0.5)                 # a comet circling: thick head, thin tail
arc(0.3, 0, 1.0, 0.04, 1).polarRepeat(6, Time() * 0.2)     # six of them around the centre
   .charcoal(0.5, 1.5, 90)                                 # paper tooth and ragged edges
src(o0).layer(dab(0.07, 0.4, angle, 0.8).scroll(x, y).color(r, g, b)).out(o0)   # Motion Painting No. 1: one touch a frame, never erased
src(o0).smear(noise(2, 0.1) * 6.28, 0.02)                  # paint dragged along a flow field (paint on glass)
```

`recipes.studie(o0, beat=Clock(1)).out(o0)` is the whole thing — comets breathing with the beat,
smoke behind them; give it `beat=a.flux` from an `Audio` and it dances to the record. Every stroke
is white with alpha = stroke, so put it on something (`solid().layer(arc(...))`); `layer` /
`over` / `stain` / `alpha` all apply.

**Cut paper.** The flat-graphic animation of the sixties East — Zagreb, Soyuzmultfilm's Khitruk
years, Pannónia, the Polish poster school: `halftone` (a three-angle screen print), `boil` (the line
redrawn by hand every frame), `polarRepeat` (ornament), `filmstock(4)` (Orwo / Sovcolor), and
`set_frame_rate(12)` for animation on twos. `recipes.cutout(chain, levels=3, screen=5)` does the
posterising, boil, misregistration, paper and grading in one go; `Text("КИНО")` works, DejaVu has
Cyrillic.

**The drawn soundtrack.** In 1932 Fischinger photographed ornaments onto the optical track of the
film; the same year in Leningrad Avraamov, Sholpo and Yankovsky were drawing sound by hand and by
machine. `Soundtrack(o0, x=0.5)` reads a vertical strip of an output every frame, top to bottom,
and makes it the next 1/24 s of audio: a pattern repeating *n* times per frame height sounds at
*n × 24* Hz, and the shape of the pattern is the timbre.

```python
notes = Seq([10, 12, 16, 20]).fast(1)                      # stripes per frame -> 240, 288, 384, 480 Hz
shape(Seq([3, 4, 64]).fast(0.25), 0.2).repeat(7, notes).out(o0)   # triangle, square, round: the waveform
track = Soundtrack(o0, x=0.5)                              # what you see down the middle is what you hear
save_mp4(render_frames(seconds=12, fps=24), "ornament.mp4", fps=24, audio=track)   # or `hydrapy render ... --out x.mp4`
```

[Ornament sound](ornament_sound.mp4) is twelve seconds of it. The track keeps its own clock (a frame is read
only when 1/fps has passed, so a 60 fps window and a 24 fps render sound the same) and follows a
frame hold (`set_frame_rate(18)` reads at 18, as Super 8 sound did). The HTML export plays it live
(`readPixels` of the strip into the audio output; click to start), so a browser patch can score
itself; `CvOut().send(k, track)` plays it out of a DC-coupled interface into the modular (see the
[live I/O guide](live-io.md#the-modular-osc-midi-and-link)).
