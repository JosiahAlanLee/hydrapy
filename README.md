# hydrapy

[Hydra](https://hydra.ojack.xyz) — Olivia Jack's browser video synth — ported to Python, then pushed in a
modular-synth direction: every parameter is a CV input, there's a wavefolder, Conway's Life and
Gray–Scott reaction–diffusion run as feedback ops, and patches export back to Hydra, to standalone
WebGL pages, or to video files.

```python
from hydrapy import *

osc(10, 0.1, 0.8).rotate(Lfo(0.1, lo=0, hi=6.28)).kaleid(5).out(o0)
src(o0).blur(2).edge().palette(0, 0.33, 0.67, Lfo(0.05)).out(o1)
render(o1)

save_gif(render_frames(seconds=4), "out.gif")        # CPU, anywhere
to_html(path="out.html")                              # live WebGL page, no server needed
print(patch_to_hydra(strict=False))                   # paste into hydra.ojack.xyz
```

Two backends evaluate the same chain:

* **NumPy** — runs anywhere, deterministic, slow (≈1–5 fps at 320×180 for feedback patches). Home of
  the CPU-only ops (`pixelsort`).
* **GLSL** — the chain compiles to fragment shaders exactly the way Hydra compiles it (same nesting,
  same function bodies). Used by the HTML export (WebGL2, with a WebGL1 fallback) and by the
  [moderngl](https://github.com/moderngl/moderngl) backend for headless GPU rendering or a live window.

Every op is defined once, with a GLSL body *and* a NumPy twin; the test suite renders every op on
both and asserts they agree to within 8-bit tolerance (they do, except for per-pixel hash noise,
where float32 `sin` of huge arguments isn't portable even between GPUs).

## Install

```
pip install numpy pillow imageio            # core: CPU renderer, GIF, PNG, HTML export
pip install moderngl                        # GPU rendering (headless via EGL on Linux)
pip install glfw                            # `hydrapy live`: a window
pip install imageio-ffmpeg                  # MP4 export, decoding mp3/flac for Audio
pip install opencv-python                   # webcam in the GL backend (the HTML export uses getUserMedia)
```

Or `pip install -e .[all]` from this directory. Python 3.9+.

## The sixty-second tour

```python
from hydrapy import *

# 1. Hydra, as you know it. camelCase and snake_case both work.
osc(20, 0.1, 1).modulate(noise(3), 0.3).kaleid(4).out(o0)

# 2. Arrays are Seq, functions are Fn (or just a lambda), bpm is set_bpm.
set_bpm(120)
osc(Seq([10, 20, 40]).fast(2).smooth(), 0.1, Fn(lambda ctx: ctx.time % 1)).out(o1)

# 3. Signals: LFOs, clocks, envelopes, sample & hold, random drift — with arithmetic.
env = Env(Clock(1), attack=0.02, decay=0.5)          # retriggered every beat
osc(Lfo(0.1, lo=5, hi=40), 0.1, SH(4) * 2).scale(1 + env * 0.5).out(o2)

# 4. Per-pixel CV: any float parameter accepts a chain. Hydra can't do this.
osc(noise(2).r() * 30 + 5).rotate(fbm(3) * 6.28).out(o3)

# 5. Feedback ops that need neighbours: Life and reaction–diffusion.
src(o0).rd(feed=0.037, kill=0.06, iterations=8).add(shape(6, 0.05).color(0, 1, 0), 0.1).out(o0)

# 6. Chains combine with operators.
(osc(30) * noise(4) + shape(3).luma() | rings(8) ^ gradient()).out(o1)

render(o0)
```

Render it:

```
hydrapy render patch.py -o out.gif --seconds 6 --fps 30 --size 320x180 --warmup 60
hydrapy render patch.py -o out.mp4 --seconds 60 --size 1280x720 --gpu
hydrapy html   patch.py -o patch.html          # standalone WebGL page (keys: s = save png, f = fullscreen, space = pause)
hydrapy live   patch.py                        # window, needs moderngl + glfw
hydrapy ansi   patch.py                        # in the terminal, half-block ANSI
hydrapy hydra  patch.py --loose                # Hydra JS (drops the ops Hydra lacks)
hydrapy glsl   patch.py                        # the fragment shaders, pass by pass
hydrapy import sketch.js -o patch.py           # Hydra JS -> Python
hydrapy ops                                    # list every op
```

`hydrapy render sketch.js` works too — Hydra sketches are imported on the fly.

## What's different from Hydra

| Hydra | hydrapy |
|---|---|
| `osc(10).out(o0)` | same |
| `[1, 2, 3].fast(2).smooth()` | `Seq([1, 2, 3]).fast(2).smooth()` — or just pass a list; identical timing (bpm/60 steps per second, default bpm 30) |
| `() => time * 0.1`, `({time}) => ...` | `Fn(lambda ctx: ctx.time * 0.1)` or a bare lambda; `ctx` has `time, frame, bpm, width, height, mouse` |
| `bpm = 60`, `speed = 2` | `set_bpm(60)`, `speed(2)` |
| `s0.initCam()`, `s0.initImage(url)` | `src(Camera())`, `src(Image("file.png"))`, `src(Video("clip.mp4"))` |
| `a.fft[0]`, `a.setSmooth()` | `a = Audio.from_file("track.wav")` (offline, frame-accurate) or `Audio()` (microphone in the HTML export); `a.fft[i]`, `a.vol`, `a.flux`, `a.set_smooth()`… |
| `render()` shows all four outputs | `render(oN)` shows one; default `o0` |
| `.r()`, `.g()`, `.b()`, `.a()` | same (they return the channel as grey) |
| Hydra's 12 extra-ish GLSL helpers (`_luminance` etc.) | same names, same numbers |

Semantics that are reproduced on purpose: coordinates run 0–1 with y up; coordinate ops apply to the
*coordinate* so later ones act first (`osc().rotate(a).scale(b)` → `osc(rotate(scale(st, b), a))`);
`src(oN)` inside `oN`'s own chain reads the previous frame, inside a later output it reads the current
one; `shape` and `kaleid` keep Hydra's `3.1416`; `color()` with negative values inverts; the
`luminance` weights are Hydra's (0.2125, 0.7154, 0.0721).

What Hydra has that this doesn't (yet): `initScreen`, `setFunction` (write ops in Python instead,
see below), the multi-output `render()` grid, `.setResolution` mid-run, and the `p5`/`hydra-synth`
extensions. Hydra's `sum()` returned a float where a vec4 was expected; here it returns grey.

## The additions

### Signals: CV for every knob

All of these are pure functions of time, so they evaluate identically in Python, in the exported
HTML, and (as arrow functions) in Hydra.

| signal | what |
|---|---|
| `Lfo(freq, shape, lo, hi, phase, width)` | `sine`, `tri`, `saw`, `square`, `pulse` |
| `Seq([...]).fast().smooth().offset().ease().fit()` | Hydra arrays, same clock |
| `Clock(div, bpm)` | phase 0–1, `div` cycles per beat |
| `Env(clock, attack, decay, curve, lo, hi)` | attack/decay retriggered by a clock (fractions of its cycle) |
| `SH(rate, lo, hi, seed)` | sample & hold noise |
| `Drift(rate, lo, hi, seed)` | smooth random wander (1-D value noise) |
| `Time()`, `Mouse('x' | 'y')` | the obvious |
| `Fn(callable)` | anything; baked into a lookup table for HTML/Hydra export |
| `Audio.from_file(path).fft[i]`, `.vol`, `.flux` | Bark-band loudness like Hydra's Meyda pipeline; `flux` is onset strength (a free kick envelope) |

Arithmetic: `Lfo(1) * 2 + 0.5`, `(a.fft[0] ** 2).clip(0, 1)`, `.map(lo, hi)`, `.quantize(n)`,
`.sin()`, `.abs()`, `.smoothstep()`.

### Per-pixel parameters

Any float input accepts a chain. The chain is evaluated at the same coordinate the op receives and
its red channel is used (`.r()`, `.g()`, `.luma()` to choose). `osc(noise(2) * 40)` is an FM
oscillator; `rotate(fbm(2) * 3)` is a domain warp; `kaleid(noise().posterize(3, 1).r() * 4 + 3)` is a
kaleidoscope whose mirror count varies across the frame. Hydra's export can't express this and says
so.

### Texture ops and multi-pass compilation

`blur sharpen edge emboss dilate erode life rd chroma pixelsort` need the neighbourhood of a pixel,
which a pointwise shader can't see. The compiler splits the chain before such an op into its own
pass that renders to an intermediate target; the op then samples that texture. Nested and repeated
splits work (`osc().blur(2, iterations=3).edge().add(noise().blur(), 0.5)` is five passes). On the
CPU the prefix is just an array.

* `life()` — Conway's Life on the red channel; green keeps a decaying trail, blue the previous state.
  `src(o0).life().add(noise(40).thresh(0.7).color(1, 0, 0), 0.1).out(o0)`
* `rd(feed, kill, da, db, iterations)` — Gray–Scott. Black counts as pure A, so seed B with green:
  `src(o0).rd(0.055, 0.062).add(shape(4, 0.05).color(0, 1, 0), 0.1).out(o0)`; `.g()` is the pattern.
  Classic parameter pairs: 0.037/0.06 worms, 0.03/0.062 coral, 0.055/0.062 spots, 0.025/0.06 mitosis.
* `iterations=n` runs the op n times per frame (ping-ponging two targets) — RD wants 4–16.

### The rest

* `fold(gain, bias)` — wavefolder on colour, `abs(mod(x - 1, 2) - 1)`. Push `gain` with an envelope.
* `polar()` / `unpolar()` — in and out of polar coordinates; `twist`, `fisheye`, `mirror`, `wobble`,
  `glitch` (block displacement), `displace` (centred `modulate`).
* `fbm`, `rings`, `grid`, `white` sources; `gamma`, `levels`, `dither` (ordered Bayer), `palette`
  (Quilez cosine palette on luminance), `grain`, `vignette` colour ops; `lighten`, `darken`,
  `screen`, `overlay`, `wipe` blends.
* `pixelsort` — CPU only; the compiler refuses it for GLSL with a clear error.
* Operators: `a + b` add, `a - b` sub, `a * b` mult, `a | b` layer, `a ^ b` diff. Numbers become
  `solid()`. Mind Python precedence (`^` binds tighter than `|`).
* `Patch()` objects for more than one synth; `o0..o3` are matched by index so they work with any patch.


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
## Text, line drawings, the canvas in 3D, and the delay line

These are the things a Hydra + p5.js practice did outside Hydra.

**Text and drawings** are static textures like `Image`, rendered once with PIL (straight alpha), so
they work on all three backends and inside `Bank`:

```python
title = Text("ANCHORITE", font="DejaVuSerif-Bold", size=0.2, color=(0.85, 0.1, 0.05))   # any font file or name
plot = Lines("drawing.svg", stroke=1.6, color=(0.05, 0.08, 0.05))                         # paths, polylines, transforms, arcs
plants = Lines(meadow(seed=3), stroke=1.5)                                                 # or polylines in 0..1 coordinates
paper.stain(src(plot)).stain(src(title))                     # as dye
field.mult(src(plot).a().invert())                           # as a mask
Bank([Lines(meadow(seed=s)) for s in range(8)], rate=12)     # one drawing per frame
```

`hydrapy.drawing.plant()` / `meadow()` generate roadside plants as polylines (a stand-in until
your own drawings are in the folder); `to_svg()` writes polylines back out for the plotter.

**The canvas in 3D.** `plane3d`, `sphere3d`, `cylinder3d` are texture ops: the chain so far is
rendered, then ray-cast onto the shape by a perspective camera, with optional Lambert shading and
fog. Outside the shape the result is transparent, so layer it over a ground:

```python
ground.layer(tex.plane3d(rotX=1.25, distance=0.55, fov=70, wrap=1, fade=0.35))   # the floor
      .layer(tex.sphere3d(rotY=Time() * 0.4, radius=0.32, shade=0.6))
      .layer(src(Text("CANVAS")).cylinder3d(rotX=0.25, rotY=Time() * 0.6))
```

**The delay line.** `delay(oN, k)` is output N as it was k frames before what `src(oN)` gives
(`delay(o0, 0) == src(o0)`). Each output keeps a ring of textures as deep as the largest delay used,
on every backend. Echoes, A/B rolls, frame holds and motion extraction without spending outputs:

```python
src(o0).add(delay(o0, 4).color(0.8, 0.3, 0.9), 0.7).add(delay(o0, 8).color(0.3, 0.9, 0.8), 0.5)   # echoes
src(o0).blend(delay(o0, 6), Clock(1).smoothstep().quantize(1))                                     # A/B roll
src(o0).diff(delay(o0, 2)).thresh(0.1)                                                              # motion only
```

Patches `14_text_and_lines.py`, `15_canvas_in_3d.py`, `16_delay_line.py`; `docs/tour.mp4` walks
through every primitive and the combination patches (`python docs/tour.py` re-renders it).

## Op reference

### Sources

| op | args | | notes |
|---|---|---|---|
| `noise` | `scale=10, offset=0.1` | Hydra | Simplex noise; offset scales its drift through time. |
| `voronoi` | `scale=5, speed=0.3, blending=0.3` | Hydra | Hydra's animated Voronoi cells. |
| `osc` | `frequency=60, sync=0.1, offset=0` | Hydra | Sine oscillator across x; offset separates the colour channels. |
| `shape` | `sides=3, radius=0.3, smoothing=0.01` | Hydra | Regular polygon with `sides` sides. |
| `gradient` | `speed=0` | Hydra | x in red, y in green, sin(time*speed) in blue. |
| `src` | `tex` | Hydra | Read an output (o0..o3), Image, Video or Camera. |
| `delay` | `tex, frames=1` | **new** | An output as it was `frames` frames earlier than src() gives (delay(o0, 0) == src(o0)). Each output keeps a ring of textures as deep as the largest delay used. Optical-printer moves: frame repeats, A/B rolls, superimposition at an offset. |
| `prev` | `` | Hydra | The previous frame of the output this chain renders into. |
| `solid` | `r=0, g=0, b=0, a=1` | Hydra | A flat colour. |
| `fbm` | `scale=4, octaves=4, speed=0.1, gain=0.5, lacunarity=2` | **new** | Fractal Brownian motion: layered simplex noise, 0..1. |
| `rings` | `frequency=10, speed=0, offset=0` | **new** | Concentric sine rings. |
| `grid` | `n=8, thickness=0.05` | **new** | Line grid, n cells across. |
| `white` | `rate=60, seed=0` | **new** | Per-pixel static, re-rolled `rate` times a second. |
| `drops` | `scale=4, size=0.5, density=0.6, stretch=1, rim=0.6, softness=0.03, wobble=0.1, texture=0.4, seed=0, hue=0.02, hueSpread=0.15, sat=0.9` | **new** | Dye drops on a jittered grid: rgb = each drop's dye colour (hue ± hueSpread), alpha = thickness: thicker at the drying rim, uneven inside (texture). stretch > 1 smears them into strokes. Feed to stain(). |
| `crackle` | `scale=6, width=0.04, warp=0.4, seed=0` | **new** | Cracks in dried paint: warped Voronoi cell borders as thin tapering lines (alpha = line). |
| `veins` | `scale=3, octaves=5, sharpness=2, speed=0, seed=0` | **new** | Ridged multifractal noise: leaf veins, wing membranes, cracked ice. Bright lines on dark. |
| `scratches` | `density=0.5, extent=0.6, width=1.2, jitter=1, rate=24, seed=0` | **new** | Vertical emulsion scratches (white on black, alpha = line), wiggling `jitter` pixels `rate` times a second. Drift the seed slowly; scratches run along the strip. |
| `dust` | `density=0.3, size=1.5, hairs=0.3, seed=0` | **new** | Specks and hairs on the gate (white on black, alpha = dust). New seed every frame for the real thing: seed=SH(24, 0, 99). |

### Coordinate ops

| op | args | | notes |
|---|---|---|---|
| `rotate` | `angle=10, speed=0` | Hydra | Rotate about the centre (radians). |
| `scale` | `amount=1.5, xMult=1, yMult=1, offsetX=0.5, offsetY=0.5` | Hydra | Zoom about (offsetX, offsetY). |
| `pixelate` | `pixelX=20, pixelY=20` | Hydra | Quantise coordinates into pixelX by pixelY cells. |
| `repeat` | `repeatX=3, repeatY=3, offsetX=0, offsetY=0` | Hydra | Tile, with alternating-row/column offsets. |
| `repeatX` | `reps=3, offset=0` | Hydra | Tile horizontally. |
| `repeatY` | `reps=3, offset=0` | Hydra | Tile vertically. |
| `kaleid` | `nSides=4` | Hydra | Kaleidoscope with nSides mirrors. |
| `scroll` | `scrollX=0.5, scrollY=0.5, speedX=0, speedY=0` | Hydra | Translate (wrapping). |
| `scrollX` | `scrollX=0.5, speed=0` | Hydra | Translate horizontally. |
| `scrollY` | `scrollY=0.5, speed=0` | Hydra | Translate vertically. |
| `polar` | `zoom=1, rotations=1` | **new** | Cartesian -> polar: x becomes angle, y becomes radius. osc().polar() gives spokes. |
| `unpolar` | `zoom=1` | **new** | Polar -> cartesian: wraps the image around the centre (tunnel). |
| `twist` | `amount=3, speed=0` | **new** | Rotation that grows with radius (swirl). |
| `fisheye` | `amount=1` | **new** | Barrel (amount > 0) or pincushion (amount < 0) distortion. |
| `mirror` | `x=1, y=0` | **new** | Reflect the left half into the right (x) and/or bottom into top (y). |
| `wobble` | `frequency=3, amount=0.02, speed=1` | **new** | Sinusoidal displacement. |
| `glitch` | `blocks=8, amount=0.1, rate=4, seed=0` | **new** | Random block displacement, re-rolled `rate` times a second. |
| `pan` | `x=0, y=0, speedX=0, speedY=0` | **new** | Translate without wrapping (scroll wraps). pan(0, 0, 0, fps) moves one frame-height per frame: a painted strip running through the gate, so smears continue from frame to frame. |

### Colour ops

| op | args | | notes |
|---|---|---|---|
| `posterize` | `bins=3, gamma=0.6` | Hydra | Quantise colour into `bins` levels. |
| `shift` | `r=0.5, g=0, b=0, a=0` | Hydra | Add to each channel and wrap. |
| `invert` | `amount=1` | Hydra | Invert colour. |
| `contrast` | `amount=1.6` | Hydra | Contrast about 0.5. |
| `brightness` | `amount=0.4` | Hydra | Add to RGB. |
| `luma` | `threshold=0.5, tolerance=0.1` | Hydra | Keep bright parts; alpha follows luminance (for layer/mask). |
| `thresh` | `threshold=0.5, tolerance=0.04` | Hydra | Black/white by luminance. |
| `color` | `r=1, g=1, b=1, a=1` | Hydra | Multiply channels; negative values invert that channel. |
| `saturate` | `amount=2` | Hydra | Saturation. |
| `hue` | `hue=0.4` | Hydra | Rotate hue. |
| `colorama` | `amount=0.005` | Hydra | Shift all of HSV and wrap: the classic Hydra acid colours. |
| `sum` | `scale=1` | Hydra | Sum of channels as grey. |
| `r` | `scale=1, offset=0` | Hydra | The r channel as grey (scale, offset). |
| `g` | `scale=1, offset=0` | Hydra | The g channel as grey (scale, offset). |
| `b` | `scale=1, offset=0` | Hydra | The b channel as grey (scale, offset). |
| `a` | `scale=1, offset=0` | Hydra | The a channel as grey (scale, offset). |
| `fold` | `gain=2, bias=0` | **new** | Wavefolder: colour above 1 folds back down (gain drives it like a synth folder). |
| `gamma` | `amount=2.2` | **new** | rgb ** amount. |
| `levels` | `lo=0, hi=1` | **new** | Stretch lo..hi to 0..1 and clip. |
| `dither` | `levels=2, size=1` | **new** | Ordered (Bayer 4x4) dithering to `levels` levels per channel; `size` = dot size in pixels. |
| `palette` | `r=0, g=0.33, b=0.67, phase=0, frequency=1` | **new** | Map luminance through a cosine palette (Inigo Quilez); r/g/b are per-channel phase offsets. |
| `grain` | `amount=0.1, rate=60` | **new** | Film grain. |
| `vignette` | `amount=0.6, radius=0.6` | **new** | Darken the corners. |

### Combine ops (take another chain)

| op | args | | notes |
|---|---|---|---|
| `add` | `amount=1` | Hydra | Add. |
| `sub` | `amount=1` | Hydra | Subtract. |
| `layer` | `` | Hydra | Alpha-composite on top. |
| `blend` | `amount=0.5` | Hydra | Crossfade. |
| `mult` | `amount=1` | Hydra | Multiply. |
| `diff` | `` | Hydra | Absolute difference. |
| `mask` | `` | Hydra | Mask by the other chain's luminance. |
| `lighten` | `amount=1` | **new** | Per-channel max. |
| `darken` | `amount=1` | **new** | Per-channel min. |
| `screen` | `amount=1` | **new** | Screen blend. |
| `overlay` | `amount=1` | **new** | Overlay blend. |
| `wipe` | `amount=0.5, softness=0.02, angle=0` | **new** | Spatial crossfade along a direction (amount 0..1 sweeps it). |
| `stain` | `density=1` | **new** | Shine the chain through a layer of transparent dye: rgb of the layer is what the dye lets through, alpha its thickness. Multiplicative, like ink; start from solid(1,1,1) as clear leader. |
| `alpha` | `` | **new** | Set alpha from another chain's luminance without premultiplying (mask() premultiplies). Use it to build dye layers: solid(1,0.2,0.1).alpha(shape(4)). |

### Modulate ops (coordinates driven by another chain)

| op | args | | notes |
|---|---|---|---|
| `modulateRepeat` | `repeatX=3, repeatY=3, offsetX=0.5, offsetY=0.5` | Hydra | Tile with offsets driven by the modulator. |
| `modulateRepeatX` | `reps=3, offset=0.5` | Hydra |  |
| `modulateRepeatY` | `reps=3, offset=0.5` | Hydra |  |
| `modulateKaleid` | `nSides=4` | Hydra | Kaleidoscope with radius pushed by the modulator. |
| `modulateScrollX` | `scrollX=0.5, speed=0` | Hydra |  |
| `modulateScrollY` | `scrollY=0.5, speed=0` | Hydra |  |
| `modulate` | `amount=0.1` | Hydra | Displace coordinates by the modulator's red/green. |
| `modulateScale` | `multiple=1, offset=1` | Hydra |  |
| `modulatePixelate` | `multiple=10, offset=3` | Hydra |  |
| `modulateRotate` | `multiple=1, offset=0` | Hydra |  |
| `modulateHue` | `amount=1` | Hydra |  |
| `displace` | `amount=0.1` | **new** | Like modulate but centred: 0.5 in the modulator means no displacement. |

### Texture ops (neighbourhood; multi-pass on the GPU)

| op | args | | notes |
|---|---|---|---|
| `blur` | `radius=1, iterations=1` | **new** | 3x3 box blur with taps `radius` pixels apart; use iterations for a wider blur. |
| `sharpen` | `amount=1` | **new** | Unsharp mask. |
| `edge` | `amount=1` | **new** | Sobel edge magnitude. |
| `emboss` | `amount=2` | **new** | Relief shading. |
| `dilate` | `radius=1, iterations=1` | **new** | Grow bright regions (3x3 max). |
| `erode` | `radius=1, iterations=1` | **new** | Shrink bright regions (3x3 min). |
| `life` | `threshold=0.5, iterations=1` | **new** | One step of Conway's Life on the red channel (red = alive). Green keeps a decaying trail, blue the previous state. Feed it back: src(o0).life().out(o0). |
| `rd` | `feed=0.055, kill=0.062, da=1, db=0.5, iterations=4` | **new** | Gray-Scott reaction-diffusion step: A in red, B in green/blue. Black counts as pure A, so seed B with green and feed it back: src(o0).rd().add(shape().color(0,1,0), 0.1).out(o0). |
| `chroma` | `amount=0.01, angle=0` | **new** | Chromatic aberration: red and blue sampled on opposite sides. |
| `pixelsort` | `threshold=0.5, axis=0` | **new**, CPU only | Sort runs of bright pixels by luminance along rows (axis 0) or columns (axis 1). CPU only. |
| `bleed` | `grow=0.08, softness=0.15, roughness=0.4, iterations=3` | **new** | Ink spreading: blurs a dye layer and re-thresholds its alpha with noise, so edges creep outward (grow) and go ragged (roughness). Iterate for more. |
| `bake` | `` | **new** | Render the chain so far into a texture and sample it from here on. Everything after sees one copy, so a chain used twice (bloom: x.add(x.blur())) is evaluated once. Coordinates after bake() wrap. |
| `plane3d` | `rotX=0.5, rotY=0, rotZ=0, distance=1, fov=60, wrap=0, shade=0, fade=0` | **new** | The chain as a textured plane in perspective (rotations in radians). wrap=1 tiles it to the horizon (the Hydra floor); shade darkens it by its angle to the light; fade darkens it with distance (fog). |
| `sphere3d` | `rotX=0, rotY=0, rotZ=0, radius=0.45, distance=1, fov=60, shade=0.5` | **new** | The chain wrapped around a sphere (x = longitude, y = latitude), lit from the upper right by `shade`. |
| `cylinder3d` | `rotX=0.3, rotY=0, rotZ=0, radius=0.4, height=0.9, distance=1, fov=60, wrap=0, shade=0.5` | **new** | The chain wrapped around an open cylinder (x goes around, y along the axis); you see the inside through the open ends. |

## Backend notes

* **Texture sampling**: outputs are sampled bilinear with wrap-around (`fract(st)` as in Hydra) on
  both backends; `Renderer(filter="nearest")` for the crunchy version.
* **Precision**: NumPy evaluates in float32 to match the GPU, including the hash and noise
  functions, so noise and voronoi cells land in the same places. The new ops use sin-free hashes
  (Hoskins), which agree between CPU and GPU bit for bit; Hydra's own `sin`-based hash is kept for
  Hydra's ops, so `voronoi` after a `rotate` can differ in a cell or two.
* **Feedback buffers**: RGBA16F on the GPU when `EXT_color_buffer_float` is available (WebGL2),
  else 8-bit — reaction–diffusion looks best with float buffers. Hydra itself uses 8-bit.
* **Resolution**: the CPU default is 320×180 (`set_resolution`, `Patch(size=...)`, `--size`). Signals
  that use `ctx.width` adapt.
* **Speed**: on this machine's software GL, `rd(iterations=8)` at 320×180 renders ~25 fps on the
  GPU path and ~1 fps on NumPy; a real GPU is far faster. Use `--gpu` for anything long.
* **Audio in the HTML export**: the file is embedded as base64 and analysed with an `AnalyserNode`
  using the same Bark/loudness pipeline, so what you tuned offline is what plays. The page waits for a
  click (autoplay policy). `Audio()` with no file uses the microphone.

## Writing an op

Ops are plain declarations; both backends pick them up at once:

```python
import numpy as np
from hydrapy import *

def np_solarize(c, threshold, env):             # NumPy twin: c is (H, W, 4); params may be (H, W) arrays
    rgb = np.where(c[..., :3] > threshold, 1.0 - c[..., :3], c[..., :3])
    return np.concatenate([rgb, c[..., 3:]], axis=-1)

add_op(Op("solarize", "color", [Input("threshold", "float", 0.5)], np=np_solarize,
          glsl="vec3 rgb = mix(_c0.rgb, 1.0 - _c0.rgb, step(threshold, _c0.rgb)); return vec4(rgb, _c0.a);"))

osc(10).solarize(0.6).out()
```

Kinds are `src`, `coord`, `color`, `combine`, `combineCoord`, `texture`; the fixed first arguments
are Hydra's (`_st`, `_c0`, `_c1`, plus `_tex` for texture ops). GLSL bodies may use `time`,
`resolution`, `gl_FragCoord` and the helpers (`_noise`, `_luminance`, `_rgbToHsv`, `_hsvToRgb`,
`_hash11/21/22/31`, `_bayer4`).

## Layout

```
hydrapy/core.py        Chain, Op registry, Output/Patch, Image/Video/Camera
hydrapy/ops.py         every op: GLSL body + NumPy twin
hydrapy/noise.py       simplex noise, hashes, HSV — NumPy and GLSL
hydrapy/signals.py     Lfo/Seq/Clock/Env/SH/Drift/Fn/Audio, with JS emission
hydrapy/backend_np.py  CPU renderer
hydrapy/glsl.py        chain -> multi-pass GLSL compiler
hydrapy/backend_gl.py  moderngl renderer, headless and windowed
hydrapy/ops_paint.py   painted-film ops: stain alpha drops crackle veins scratches dust bleed bake pan
hydrapy/ops_3d.py      plane3d sphere3d cylinder3d
hydrapy/drawing.py     Text, Lines (SVG flattener), plant()/meadow(), to_svg()
hydrapy/recipes.py     handpainted(), mothlight(), film()
hydrapy/synth.py       image quilting, Quilt frame bank
hydrapy/export.py      GIF/MP4/PNG/ANSI, Hydra JS, Python, standalone HTML runtime
hydrapy/hydra_import.py Hydra JS -> patch
hydrapy/cli.py
patches/               examples (00–16)
docs/tour.py           renders docs/tour.mp4, the feature tour
tests/                 ~300 tests incl. CPU/GPU parity for every op
```

## License

AGPL-3.0-or-later, because Hydra is. See LICENSE.
