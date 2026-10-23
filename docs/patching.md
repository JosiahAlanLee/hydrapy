# Signals, feedback and geometry

[Documentation](README.md) · [Project home](../README.md)

## Signals and feedback

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
| `Time()`, `Mouse('x')`, `Mouse('y')` | elapsed time and mouse coordinates |
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

## Animated generative sources

`Cellular` adapts the public thi.ng engine that powers C-SCAPE, including
interacting rules, short-term memory and optional glyph rendering.
`orbitVoronoi` adapts thi.ng's open rotating-rings example into a native shader.
These use verified open components; they do not bundle the FXhash editions.

```python
cells = Cellular(width=384, height=216, rate=48, seed=17)
src(cells).palette(.04, .23, .53).out(o0)
orbitVoronoi(11, 4, speed=.22).palette().out(o1)

field = sdfCircle(.22, .42).sdfUnion(sdfBox(.35, .4, .06, .58), .12)
field.sdfSubtract(sdfCircle(.1)).sdfBands(18, .25, Time() * .2).out(o2)
```

Also included: `FrameSource(draw)` for Python animation textures, `subdivide`
and `fuzzy_paths` for moving linework, and `flora_lines` for your
Generative_Art plant JSON. Existing `dither` and `halftone` work on these sources.
Python textures run in NumPy/desktop GL and require baking into a `Bank` for
HTML. The Voronoi and SDF ops export directly to HTML.

See [sources, provenance and examples](TOXI_ANIMATION_SOURCES.md), patches
28–31, and `python docs/generative_demo.py --out generative_demo.mp4`.

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

**The p5 loop.** A Hydra + p5 practice often ran Hydra's canvas through p5's `plane()` and back
in as `s0`. That round trip is `src(o0).plane3d(...)` fed back into `o0` here; `patches/17_p5_hydra_loop.py`
is the `p5js-hydra-fxhash` template rebuilt that way, and `Fx("oo…")` is `fxrand()` with the
`rnd_btw / rnd_btwexp / rnd_int` helpers, so a hash reproduces its edition. `hydrapy import file.js`
is lenient: it pulls the Hydra statements out of a p5 sketch, and a source it can't read offline
(`s0.init({src: canvas})`) becomes an `External` placeholder to swap for an Image, Bank or the loop.

Patches `14_text_and_lines.py`, `15_canvas_in_3d.py`, `16_delay_line.py`, `17_p5_hydra_loop.py`;
`docs/tour.mp4` walks through every primitive and the combination patches (`python docs/tour.py`
re-renders it).

## Entropy locking (after aebrer)

[aebrer](https://www.raster.art/artwork/entropy-locked-recursive-glitch-textures-by-aebrer/about)'s
entropy locking is a seeded random stream that, with a small probability on each draw, gets pulled
back to its seed state — `if (random_int(1,1000) > 997) fxrand = sfc32(...hashes)` — so the
sequence keeps falling into earlier sequences, loops of unpredictable length, "random bottlenecks"
that make a system with too much entropy occupy a [lesser infinity](https://www.raster.art/artwork/lesser-infinities-by-aebrer/about).
When that stream steers pixel feedback the same glitches recur, and recurrence on top of feedback
grows the fractal texture. CC0; ported here with thanks.

```python
fx = Fx("ooYourFxhashHash")                      # fxhash's own sfc32, bit for bit: a hash gives the same numbers as the browser did
lock = EntropyLock("ooYourFxhashHash", rate=24, lock=0.01)   # 24 draws a second from a locked copy of the stream
src(o0).shred(lock, 4, 0.25, 1.5)               # samples of the image copied onto itself (magnified)
       .tear(lock * 7, 3, 0.1, 0.3)              # bands torn sideways
       .blend(osc(20).kaleid(fx.rnd_int(3, 7)), 0.05)
       .out(o0)
```

`Fx(hash, lock=p)` locks any draw sequence (`rand`, `rnd_btw`, `rnd_btwexp`, `rnd_int`, `pick`,
`table`; `reseed()`, `save()`/`restore()` for bottlenecks by hand). `EntropyLock` precomputes its
steps so the HTML export carries the same loops. `tear` and `shred` are coordinate ops, so they are
free and work on any source; `patches/19_entropy_lock.py` is the full loop.
