# Rendering and export

[Documentation](README.md) · [Project home](../README.md)

## Choosing a backend

Two backends evaluate the same chain:

* **NumPy** — runs anywhere; speed depends on the operators and resolution. Home of
  the CPU-only ops (`pixelsort`).
* **GLSL** — the chain compiles to fragment shaders exactly the way Hydra compiles it (same nesting,
  same function bodies). Used by the HTML export (WebGL2, with a WebGL1 fallback) and by the
  [moderngl](https://github.com/moderngl/moderngl) backend for headless GPU rendering or a live window.

Every op is defined once, with a GLSL body *and* a NumPy twin; the test suite renders every op on
both and asserts they agree to within 8-bit tolerance (they do, except for per-pixel hash noise,
where float32 `sin` of huge arguments isn't portable even between GPUs).

## Hardware video encoding

MP4/MOV/MKV export now defaults to `--encoder auto`: test a real frame at the output size
and frame rate, choose a working hardware H.264 encoder, or fall back to software `libx264`.
The CLI prints which encoder and FFmpeg executable it selected. Rendering (`--gpu`) and
encoding (`--encoder`) are independent; hardware encoding also works with the NumPy renderer.

```sh
hydrapy encoders --size 1920x1080 --fps 30
hydrapy render patch.py -o out.mp4 --seconds 60 --size 1920x1080 --gpu
hydrapy render patch.py -o out.mp4 --gpu --encoder nvenc --bitrate 16M
hydrapy render patch.py -o out.mp4 --gpu --encoder videotoolbox --bitrate 16M
hydrapy render patch.py -o out.mp4 --encoder software --quality 8
```

| Choice | FFmpeg encoder | Hardware |
|---|---|---|
| `nvenc` | `h264_nvenc` | NVIDIA, including RTX 4090 |
| `videotoolbox` | `h264_videotoolbox` | macOS VideoToolbox |
| `qsv` | `h264_qsv` | Intel Quick Sync |
| `amf` | `h264_amf` | AMD AMF |
| `vaapi` | `h264_vaapi` | Linux VAAPI (`--device /dev/dri/renderD128` to choose a node) |
| `software` | `libx264` | CPU |

An explicit encoder fails with the FFmpeg error if unavailable; only `auto` falls back.
Auto tries NVIDIA, VideoToolbox, Quick Sync, AMF, then VAAPI (VideoToolbox first on macOS),
across system and imageio-provided FFmpeg builds, before trying software. Drivers, device
access, and an FFmpeg build containing the encoder are required; appearing in FFmpeg's
encoder list alone is insufficient. Use `--ffmpeg /path/to/ffmpeg` (Python: `ffmpeg=...`)
or `IMAGEIO_FFMPEG_EXE` to restrict probing and export to one executable.

```python
from hydrapy import iter_frames, save_mp4, video_encoders

for candidate in video_encoders(size=(1920, 1080), fps=30):
    print(candidate.name, candidate.available, candidate.ffmpeg, candidate.reason)

save_mp4(iter_frames(seconds=60, fps=30), "minute.mp4", fps=30,
         encoder="auto", bitrate="16M")
```

`quality`/`--quality` is 1–10 (default 8). Software retains the previous x264 CRF mapping;
hardware quality settings are codec-specific. VideoToolbox and AMF use a bitrate estimate
based on size, frame rate, and quality. `bitrate`/`--bitrate` overrides quality control for
any encoder and requests a target bitrate, not a fixed output size or identical visual quality.
Output remains H.264 with 4:2:0 chroma; WebM is not supported by this export path.

Frames stream once; the first is buffered while encoders are tested. If encoding fails
mid-stream, export stops without retrying the stateful renderer, removes temporary files,
and preserves any existing destination. Soundtrack audio is muxed as AAC without re-encoding
the video. Its duration is the rendered frame count divided by fps: short audio
is padded with silence and long audio is cut at the video end. This also applies
to CLI and Studio offline exports. Hardware encoding offloads compression; frames still pass through CPU memory,
so this does not yet eliminate GPU readback or RGB conversion costs.

## Timing, transparent materials, and streaming exports

`set_frame_rate(18)` holds the actual image and feedback history between updates. Display and
CV-output hooks still run at the presentation rate; the optical soundtrack keeps playing held
images. Frame-rate-independent feedback uses a half-life in seconds:

```python
decay = Decay(0.7)
src(o0).mult(solid(decay, decay, decay)).layer(wave(3).color(0.8, 0.2, 0.1)).out(o0)
```

For transparent pigment, fade its **alpha** instead of its RGB. `patches/27_material_controls.py`
demonstrates this, stationary paper under moving pigment, and both radial-repeat modes.

* Strokes carry straight RGB colour (white) with coverage in alpha; `charcoal()` removes coverage
  once. Put them on a picture to see them: `solid().layer(wave(3))` — Hydra's `layer` is exact for
  a straight-alpha layer on an opaque picture. `over()` is source-over for transparent layers on
  transparent layers (pigment kept apart from its paper).
* The display is Hydra's: RGB, with alpha as a fourth channel that `.r()/.g()/.b()`, `mult`,
  `luma` and `mask` use as a number. `set_composite_alpha(True)` shows RGB × alpha over black
  instead — in the window, the HTML page, exports and brightness/sound probes — for patches built
  in straight alpha throughout (patch 27). Raw `Renderer.step()` and `GLRenderer.read_rgba()`
  always return uncomposited RGBA. (`tests/test_patch_brightness.py` holds every example patch to
  its baseline brightness, so a change to this can't darken them silently.)
* `polarRepeat(n)` uses image-space angles, preserving circles on wide and tall frames.
  `polarRepeat(n, aspect=0)` preserves the former normalized-coordinate distortion.
* `recipes.on_paper(pigment, paper_color=(0.94, 0.90, 0.81), tooth=0.35, texture=0.025)` applies
  fixed paper after the pigment's motion. Keep the transparent pigment on its own feedback
  output so the paper does not accumulate or move with it.
* `recipes.studie(..., decay_seconds=0.5, motion=0.4, surface=0.002, surface_rate=12)` separates
  trail lifetime, large-scale movement, and small line jitter. The older `trails` setting remains
  available when `decay_seconds` is omitted.

The CLI streams video frames directly to the encoder. `render_frames()` and `render_frames_gl()`
still return lists for callers that need random access; use their iterator counterparts for
long videos:

```python
from hydrapy.backend_gl import iter_frames_gl
save_mp4(iter_frames_gl(seconds=60, fps=30), "minute.mp4", fps=30)
# CPU: save_mp4(iter_frames(seconds=60, fps=30), "minute.mp4", fps=30)
```

Soundtracks carry fractional samples between optical frames (44.1 kHz / 24 fps alternates
1,837 and 1,838 samples), fill missing audio intervals with the held picture, and resample CV
audio routes using cumulative sample counts. CV files use presentation timestamps rather than
the synthesis hold rate; render warm-up is excluded. Set `record=True` and a sufficiently long
`keep` when exporting audio through the Python API. The CLI does this for its selected soundtrack.
GPU brightness probes reduce their regions before readback; audio strips are read only when due.

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
* **Speed**: use `--gpu` for long or high-resolution renders. See the
  [performance guide](PERFORMANCE.md) for reproducible CPU/GL benchmarks and
  measured 1080p/4K improvements. Check the renderer name: software OpenGL still uses the CPU.
* **Faded layers cost nothing**: a GLSL shader computes every branch of a chain, so a layer blended
  in at 0 still costs its full price. The GL renderer compiles a variant of each output's program
  without the transforms that are the identity this frame — `blend`/`add`/`sub`/`mult`/`screen`/
  `overlay`/`lighten`/`darken` at amount 0, `modulate`/`displace`/`swirl`/`tine` at amount 0,
  `ripple` at amplitude 0 — and caches the variants (`GLRenderer.specialise = False` turns it off).
  A long piece whose sections fade layers in and out on signals only pays for what's on screen.
  Custom ops can declare their own identity with `Op(..., identity=("amount", 0.0))`. Each signal is
  also evaluated once per frame and shared by every pass that uses it. Offline renders compile the
  variants they will need before the first frame (`GLRenderer.prewarm(times)`); live windows wait
  `GLRenderer.settle` frames (12) before compiling a new one.
* **Cheaper by choice**: `chain.out(o1, scale=0.25)` renders an output smaller, `mipblur` blurs wide
  at a fixed cost, `set_fast_noise()` swaps simplex noise for a texture lookup, and
  `hydrapy html --phone` makes a page for phones. See the [performance guide](PERFORMANCE.md#spending-pixels-where-they-show).
* **Audio in the HTML export**: the file is embedded as base64 and analysed with an `AnalyserNode`
  using the same Bark/loudness pipeline, so what you tuned offline is what plays. The page waits for a
  click (autoplay policy). `Audio()` with no file uses the microphone.

See also the [CLI commands](getting-started.md#the-sixty-second-tour) and
[audio timing regression notes](audio-timing.md).

## fxhash: one patch, every mint

`Fx` draws plain numbers, fixed when the patch is built: an exported page is one edition. A `Mint`
records its draws instead, as signals; the exported page makes the same draws again, in the same
order, from its own hash — `$fx.hash` on fxhash, `?fxhash=` in the URL, or the preview hash — with
`Fx`'s generator ported to JavaScript, so a hash gives the same piece in Python, in the page and on
fxhash.

```python
fx = Mint()                                      # preview hash: FXHASH / --fxhash, else a fixed development hash
source = fx.pick(["waves", "noise", "cells"])    # labels: the draw is the index, the label a feature
pattern = switch(source, osc(fx.rnd_btw(8, 40)), noise(fx.rnd_btw(2, 7)), voronoi(8))   # chosen per mint
sides = fx.rnd_int(3, 9)
glitch = EntropyLock(fx, rate=12, lock=fx.rnd_btw(0.005, 0.03))   # a locked table, drawn per mint too
density = fx.number("density", default=0.6, min=0.1, max=1, update="sync")   # fx(params): fxlens sliders, live
ink = fx.color("ink", default="#f2b04a")
pattern.kaleid(sides).modulate(noise(2), density * 0.2).color(*ink).tear(glitch * 7 + 3).out()
fx.features({"Source": source, "Sides": sides})
```

```
hydrapy fxhash patch.py --sdk fxhash.min.js     # patch_fxhash/: index.html, fxhash.min.js, upload.zip
hydrapy render patch.py --fxhash oo… -o mint.png --fxparam density=0.3   # look at any mint from Python
```

* A draw goes wherever a signal goes: op arguments, signal arithmetic, NCA and physarum
  parameters, `EntropyLock(mint, …)`. Python control flow on one (`if`, `range`, `float()`) would
  freeze it, so it raises; `.preview` is this hash's value when freezing is what you want.
* `pick` over chains, or `switch(index, *chains)`, compiles every branch and crossfades them with
  weights that are exactly 0 or 1 — the cost is all branches together. With a moving index
  (`switch(Lfo(0.05, lo=0, hi=2), a, b, c)`) it morphs.
* fx(params): `number`, `boolean`, `select` (the option's index), `color` (`.r .g .b .a`); `update="sync"`
  follows fxlens without reloading. Python previews use the default (or `params=` / `--fxparam`).
* The page calls `$fx.preview()` once it has run `preview` seconds (default 2). fxhash needs the
  SDK file in the project: `npx fxhash create` writes one, or `npm pack @fxhash/project-sdk` and take
  `package/dist/fxhash.min.js`. Without it the page uses a stand-in `$fx` that reads `?fxhash=`
  — enough to look at, not for fxlens or uploading. Everything else is inlined.
* `to_html` pages replay a Mint too: `?fxhash=` picks the mint, and the n key jumps to a fresh one.
* A patch that waits for a click (audio, the drawn soundtrack, a camera) can't be captured by
  fxhash; the export warns. [`patches/38_fxhash.py`](../patches/38_fxhash.py) is a complete piece.
