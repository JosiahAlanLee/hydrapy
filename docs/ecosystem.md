# Grown textures, vectors and listening

[Documentation](README.md) · [Example patches](../patches/)

Things Python makes easy that a browser sketch can't: texture rules learned from a photograph,
agent simulations, mathematical marbling, the picture traced as a beam for a scope, laser or
plotter, and a recording's beats or footage's motion as signals.

## Neural cellular automata: textures that grow and heal

The honest version of "GAN textures". A neural cellular automaton (after Mordvintsev, Niklasson,
Randazzo and Levin, *Self-Organising Textures*, Distill 2021) is a grid of cells, each holding 12
numbers (the first three its colour). Every step a cell looks at its 3×3 neighbourhood through
four fixed filters, runs that through a two-layer network of about six thousand weights, and adds
the result to itself — at a random half of the cells. Trained on one photograph, the rule grows
that texture from nothing, keeps it alive, and repairs it when it's damaged.

```python
moss = NCA("moss.npz", size=(160, 90), speed=40)       # 40 steps a second on a 160x90 grid
src(moss).out(o0)

shape(4, 0.5).scale(1, 0.1, 4).scroll(Lfo(0.07, "tri", -0.6, 0.6), 0).out(o1)
src(NCA("moss.npz", damage=o1)).out(o2)                # wiped wherever o1 is bright; it regrows

NCA("moss.npz", perturb=cv.cv[0] * 0.05)               # the modular pushing noise into its hidden channels
NCA("moss.npz", paint=o3, paint_amount=0.5)            # colour written into the cells, then integrated
```

`rate` (fraction of cells updating per step), `perturb` and `paint_amount` take signals; `damage`
and `paint` take outputs. The rule runs on the NumPy and GL backends and in the HTML export (its
weights are compiled into the shaders; WebGL2 float targets).

**Training your own** (`pip install torch`; CPU is fine — tens of minutes on a laptop, a few on a GPU or Apple MPS):

```
hydrapy nca train lichen.jpg --scale 128 --iters 2000      # -> lichen.npz
```

The loss compares texture statistics without a pretrained network: Gram matrices of fixed *random*
filters at four scales (random multi-scale filters capture texture surprisingly well) plus a
sliced-Wasserstein match of the colour distribution. Nothing is downloaded and no one else's
images are involved; the rule has only ever seen your photograph. `--scale` is the photograph's
shorter side in cells: features 4–16 cells across work best. The rule trains on a pool of states it
keeps running, so it learns to sustain the texture, not just to make it once — over the rollout
lengths it trains on. A rule that holds for a few hundred steps and then drifts (colours shift,
blobs coarsen, the state pins at its clamp) needs longer rollouts: fine-tune it with
`--init lichen.npz --steps 64 128 --lr 3e-4 --iters 500`. Check one by stepping it for a thousand
steps before you rely on it (`nca.step_np` in a loop).

[`patches/32_nca.py`](../patches/32_nca.py) uses the three rules shipped in `patches/models/`: `cells` (scikit-image's
immunohistochemistry slide), `gravel` (scikit-image's CC0 gravel photograph) and `painted` (a frame
of `recipes.handpainted()`). `cells` is the drift case above: trained with 24–40-step rollouts it
held for ~100 steps and then wandered off-palette and pinned at the clamp; 500 iterations of
fine-tuning at 64–128 steps fixed it (steady colour statistics over 2,500 steps).

## Vectors out: the oscilloscope, the laser, the plotter

One set of paths, three destinations. A path is an `(N, 2)` array in -1..1; a frame is a list of
them. They come from parametric curves (`Harmonograph`, `Lissajous`, `Rose`, all CV-able), from
drawings (`Paths.svg("drawing.svg")`, `Paths.lines(meadow())`), or from the picture itself:
`Contours(o0, levels=4)` traces isolines of an output every frame, the way a scan converter or a
Vectrex would show it.

**XY.** `XY(source, rate=80, gain=0.5)` turns a frame of paths into an audio-rate beam: it traces
the frame `rate` times a second, travels between paths blanked (`xy.z`), and keeps its phase across
frames, so a changing shape doesn't click. Out of the ES-3 into a scope's X/Y inputs (or a laser
DAC) it draws; into a mixer it's oscilloscope music — the beam's path *is* the waveform, `x` left,
`y` right.

```python
beam = XY(Harmonograph(3, 3 * ratio, 3.01, 0, 1.5708, 0.4, 4), rate=55, gain=0.5)
traced = XY(Contours(o0, levels=3), rate=60)
CvOut(channels=(0, 1, 2, 3, 4)).send(0, traced.x).send(1, traced.y).send(2, traced.z) \
                               .send(3, beam.x).send(4, beam.y)
src(beam).out(o1)                                      # a phosphor preview of what the scope shows
```

`xy.stereo()` / `xy.wav(path)` give the audio; `save_mp4(frames, path, audio=(xy.stereo(), xy.sr))`
muxes it (or just `audio=xy`). The phosphor preview works on the NumPy and GL backends (it's computed in Python, so not
in the HTML export). [`patches/33_vector_out.py`](../patches/33_vector_out.py) is the full rig.

**Plotter.** `to_plotter_svg(paths, "plot.svg", size_mm=(297, 210))` writes millimetres, fits the
paths inside the margins, orders them to cut pen-up travel, and puts each `layers=[...]` list on its
own Inkscape layer (one pen each) — vpype and the AxiDraw software read it. `chain_contours(chain)`
gives isolines of any field and `streamlines(chain)` flow-field lines through it (red as an angle,
or red/green as a direction). `python patches/34_plotter.py` writes a topography, a flow field and
a harmonograph.

## Marbling

Suminagashi and ebru are ink floating on water, pushed by drops and drawn through by styluses and
combs. After Lu, Jaffer, Jin, Zhao and Mao (*Mathematical Marbling*, IEEE CG&A 2012), each of those
moves is a closed-form, area-preserving, exactly invertible map — so each is a coordinate op, and a
chain is the bath's history in the order it happened:

```python
water = solid(0.93, 0.9, 0.82)
water.layer(disc(0.4, 0.5, 0.15).color(0.1, 0.1, 0.3)) \                    # a drop of indigo
     .inkdrop(0.6, 0.5, 0.12).layer(disc(0.6, 0.5, 0.12).color(0.6, 0.1, 0.1)) \   # a second drop pushes it aside
     .tine(1.5708, 0, Lfo(0.05, lo=-0.3, hi=0.3), 0.03, 0.12)               # a comb drawn through both, back and forth
```

`inkdrop(x, y, radius)` (drops compose: two of radius *a*, *b* are one of √(a²+b²)), `tine(angle,
offset, amount, sharpness, period)` (a stylus, or a comb with `period` > 0), `swirl(x, y, amount,
radius, sharpness)` (a circular stroke), `ripple(angle, amplitude, wavelength, phase)` (the wavy
comb), and `disc` (a round drop of ink to `layer` into the hole). No feedback, every parameter a
signal: the comb can be drawn through and back again, and the pattern un-marbles. Recipes:
`suminagashi(rings, radius, inks)`, `ebru_stones(n, seed, palette)`, `bands(colours)`,
`floating_drops(base, [(x, y, r, colour), ...])` (colour `None` is clear water). [`patches/35_marbling.py`](../patches/35_marbling.py).

## Physarum

Slime-mould agents (Jeff Jones, 2010): tens of thousands of them wander a grid, smell the trail at
three points ahead (left, centre, right), turn towards the strongest, step, and deposit; the trail
spreads and fades. Transport networks form, thicken where traffic is heavy, and re-route.

```python
src(Physarum(size=(320, 180), agents=(256, 128))).out(o0)          # 32,768 agents
voronoi(5, 0.15).out(o1)
src(Physarum(food=o1, food_weight=0.6)).out(o2)                   # they follow the picture too
Physarum(sensor_angle=cv.cv[0] * 0.1 + 0.6)                         # small angles: long veins; large: fine mesh
```

`sensor_angle`, `sensor_distance` (cells), `turn`, `step`, `deposit`, `decay`, `diffuse`,
`food_weight` and `exposure` take signals. On the GPU the agents live in a float texture and
deposit as points with additive blending; the HTML export runs it too (WebGL2). [`patches/36_physarum.py`](../patches/36_physarum.py).

## Listening and watching: beats, onsets, optical flow

`Beats.from_file("record.wav")` finds the beats of a recording — Ellis's dynamic-programming beat
tracker, in NumPy (or librosa's, if installed) — and hands them back as signals:

```python
beats = Beats.from_file("brahms.wav")            # or bpm=96 if you know the tempo
studie(o0, beat=beats.phase).out(o0)             # Fischinger on the record's actual downbeats
shape(4, beats.pulse(0.02, 0.4) * 0.3)           # an envelope on every beat
osc(10, 0.1, beats.bar(4))                       # a phase across each four-beat bar
noise(beats.onset * 20 + 3)                      # the onset-strength envelope
```

Tempo estimates can land an octave off (accents every other beat, busy hi-hats): give `bpm=` or
use `.half()` / `.double()`. With the tempo right, beats land within ~10 ms. Everything exports to
the HTML page (the beat times and the envelope are embedded); play the record there with
`Audio.from_file` and the page stays in step.

`Flow(Video("footage.mp4"))` is the footage's optical flow (OpenCV Farneback) as a texture:
red/green = 0.5 + each pixel's motion per frame, blue = speed. `src(o0).displace(src(flow), -1)`
carries whatever is in `o0` along the real motion — painted texture over landscape, datamosh.
Flow is computed in Python: NumPy and GL backends, not the HTML export. [`patches/37_beats_and_flow.py`](../patches/37_beats_and_flow.py).
