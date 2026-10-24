# Rendering performance

[Rendering and export](rendering-and-export.md) · [Bullfrog demo](bullfrog-demo.md)

Use the GL backend for 1080p/4K work. Rendering and video compression are separate:
`--gpu` selects OpenGL; `--encoder auto` tries working hardware encoders before
falling back to software. Check the reported renderer as well as the encoder.
`llvmpipe` is CPU software OpenGL, even when the GL backend is selected.

## Measure your machine

Run from the repository root with the GPU extra installed:

```sh
python docs/benchmark.py --backend gl --size 1920x1080 --readback --json timing-1080.json
python docs/benchmark.py --backend gl --size 3840x2160 --readback \
  --case bullfrog_surface --case raymarch --case nca --json timing-4k.json
OPENBLAS_NUM_THREADS=1 python docs/benchmark.py --backend cpu --size 320x180
```

The benchmark reports setup, first frame, warmed median and p95 frame times.
Each GL frame is synchronized with `finish()` so timings include execution.
`--readback` also includes the RGB download needed by video export. Encoding,
audio analysis, titles and crossfades are outside these operator timings.
Use `--frames 60 --warmup 10` for longer samples; repeat `--case` to select operators.
The NCA case keeps its simulation grid at 160×90 (12 channels, 96 hidden units,
one update per 30 fps frame) at every output resolution.

## Measured changes

Compared with commit `5362dbb`, using Python 3.12.14, NumPy 2.3.5, one BLAS thread,
AMD EPYC 9V74 and Mesa llvmpipe LLVM 20.1.2. Each entry is the median of nine
frames after three warm-up frames. Baseline and optimized runs used separate
processes, sequentially, with identical inputs, resolution and iteration counts.
Short samples can vary with scheduling; these are measurements of this machine,
not predictions for a hardware GPU.

| NumPy, 320×180 | Before (ms/frame) | After | Speedup |
| --- | ---: | ---: | ---: |
| Blur | 82.6 | 2.5 | 33.0× |
| Reaction–diffusion | 320.6 | 9.8 | 32.6× |
| Bleed | 311.6 | 105.7 | 2.9× |
| Harmonograph | 205.7 | 11.5 | 17.8× |
| Wave stroke | 223.6 | 9.2 | 24.3× |
| Arc stroke | 108.5 | 4.8 | 22.5× |
| YIQ | 188.9 | 14.3 | 13.2× |
| Normal map | 48.9 | 3.8 | 12.9× |
| Folded raymarch scene | 1270.1 | 51.6 | 24.6× |
| NCA | 24.6 | 14.5 | 1.7× |

| Software OpenGL, including readback | Before (ms/frame) | After | Speedup |
| --- | ---: | ---: | ---: |
| Bullfrog Surface, 1920×1080 | 87.2 | 47.4 | 1.84× |
| Folded raymarch, 1920×1080 | 21.5 | 16.4 | 1.31× |
| NCA, 1920×1080 | 15.8 | 11.5 | 1.38× |
| Bullfrog Surface, 3840×2160 | 329.9 | 182.2 | 1.81× |
| Folded raymarch, 3840×2160 | 82.9 | 54.0 | 1.54× |
| NCA, 3840×2160 | 43.0 | 36.1 | 1.19× |

These changes do not establish real-time 4K performance. A 30 fps budget is
33.3 ms/frame; 60 fps is 16.7 ms. Whole-demo speed also includes transitions,
simulation warm-up and encoding. Hardware GPU and hardware-encoder validation
must be done on the target machine.

## Where the work was removed

* NumPy neighbourhood effects use shifted arrays on matching pixel grids rather
  than rebuilding coordinates and bilinear-index arrays for every tap. Warped
  coordinates and different texture sizes retain general sampling. Identity
  samples remain independent copies so custom operators cannot alter a source.
* CPU strokes evaluate distances only inside conservative segment bounds. No
  visible Gaussian tails, segment counts or thickness settings are reduced.
* CPU raymarching removes finished rays from subsequent distance evaluations and
  computes normals only for hits. Both backends reject rays guaranteed to miss
  a conservative bound, including folds and smooth unions. Surviving rays keep
  the original start, step limit and epsilon.
* NCA perception uses separable CPU stencils and evaluates the neural network
  only for firing cells. Desktop GL writes all state textures in one draw when
  draw-buffer limits permit, sharing perception and hidden activations. The
  browser and limited contexts retain single-target shaders.
* The GLSL compiler reuses materialized prefixes shared by branches, including
  an immediately preceding `bake()`. Iterative effects that overwrite their
  input get private targets. Bullfrog Surface falls from six passes to three.
* Bullfrog dissolves and fades run on the GPU, downloading one RGB frame per
  video frame. Only stateful scenes receive simulation warm-up; PIL is used
  when titles or contact snapshots are needed. Display-byte blending is retained
  within one 8-bit level of CPU rounding.

Float32 operation ordering can change tiny errors, which chaotic feedback can
amplify over long runs. Regression tests cover wrapped/fractional/warped sampling,
stroke coverage, distance bounds, neural state, multi-target fallback, texture
branch aliasing and display blending, alongside the existing image and timing
suite. This is numerical equivalence, not a promise of bit-identical feedback
trajectories across backends or drivers.

High-iteration GPU simulations, dense stroke shaders, procedural noise and
multiple full-resolution outputs still cost work. 4K has four times as many
pixels as 1080p. A `pixelate()` effect does not reduce that pixel count; reduce
the actual render or simulation size when intentionally trading detail for speed.

## Spending pixels where they show

Most of a frame's cost is per pixel, so the biggest wins come from computing fewer of them, or
computing each one more cheaply. These are opt-in, because each one trades something.

**Per-output resolution.** `chain.out(o1, scale=0.25)` renders that output at a quarter of the
frame's width and height (1/16 of the pixels); whatever samples it upsamples it bilinearly. Use it
for soft material — out-of-focus grounds, blurs, feedback trails, simulations — and keep sharp
lines at full size. Pixel-sized ops (`blur`, `rd`) work in that output's own pixels. Supported by
the GL, NumPy and HTML renderers.

```python
ground = noise(3, 0.2).add(voronoi(6), 0.4).modulate(noise(2), 0.3).mipblur(4)
ground.out(o1, scale=0.25)                     # soft: a quarter size is invisible
src(o1).add(shape(4, 0.2, 0.002).repeat(4, 3), 0.8).out(o0)   # sharp: full size
```

**`mipblur(radius)`.** A wide blur at a fixed price: the input's mip levels are built after its
pass, and each pixel takes a B-spline sample of the level for `radius`, blended between levels.
`blur(r, iterations=n)` costs `n` full-size passes and grows with the radius; `mipblur` doesn't, and
`radius` animates smoothly. WebGL1 pages have no mip levels and fall back to sparse taps.

**`set_fast_noise()`** (`--fast-noise`). `noise()` and every op built on it read a 256×256 lattice
texture in two fetches instead of computing 3D simplex noise per pixel. It is value noise matched
to simplex's range and feature size: similar, not identical — thresholded noise shows axis-aligned
creases. Desktop GL and HTML pages build the same texture; NumPy renders and compute kernels keep
the exact noise, so CPU and GPU renders differ when it is on.

**Phones** (`hydrapy html patch.py --phone`, `to_html(phone=True)`):

* `precision="mixed"`: colour math in mediump (half precision, twice the rate on most phone GPUs);
  time, uniforms, coordinates, scalars, and anything that hashes or computes noise stay highp. A
  desktop or the software renderers here run mediump at full precision, so the visual effect of
  this mode has to be checked on the phone itself.
* `adaptive`: when frames arrive slower than the display refreshes, outputs render smaller (steps of
  0.85, down to 0.35 of the size) and grow back when there is headroom, but not straight back to a
  size that was too slow. Feedback is resampled into the new targets rather than cleared. The canvas
  keeps its size. Pixel-sized ops look softer while it is scaled down.
* Add `--8bit` for 8-bit output buffers (as Hydra; half the memory traffic of half floats, with
  more banding in slow feedback) and `--fast-noise`. The page's default `--pixel-ratio 1` already
  renders CSS pixels, not the 3× device pixels of a typical phone.

## Work skipped automatically

* **Faded layers cost nothing** (GL): each output's program is compiled in variants without the
  transforms that are the identity this frame — a `blend`/`add`/`modulate` at 0, a `mipblur` at
  radius 0 — see [rendering notes](rendering-and-export.md#backend-notes).
  `iter_frames_gl` (and so `hydrapy render --gpu`) compiles every variant the render will need
  before the first frame, for gates driven by predictable signals; live windows and Studio wait 12
  frames before compiling a new combination, so a value touching zero never stalls the show.
* **Shared sub-chains**: a chain (or an equal one built separately) used more than once at the same
  coordinates in a pass is computed once into a local; equal signals share one uniform.
* **Pipelined readback**: video export reads frame *n−1* through a pixel buffer while frame *n*
  renders, instead of stalling the GPU each frame.
* **Unchanged `FrameSource` frames**: a draw function may return `None` for "same as before"; the
  frame isn't copied, compared or re-uploaded.

Measured on Mesa llvmpipe (software OpenGL on two cores; relative numbers only), median ms/frame at
1280×720, against `c195d65` (which already had the faded-layer variants):

| Case | Before | After |
| --- | ---: | ---: |
| Patch 39 (Bullfrog layers) at 0:30 | 132.7 | 109.4 |
| Patch 39 at 1:35 | 185.4 | 177.1 |
| Patch 39 at 2:50 | 198.6 | 173.5 |
| Patch 39 at 4:50 | 190.3 | 154.6 |
| Offline `osc().kaleid()`, render + readback | 19.2 | 16.3 |

| Option, same machine | Off | On |
| --- | ---: | ---: |
| Soft ground on o1 (noise, voronoi, blur), `scale=0.5` / `0.25` | 63.8 | 21.9 / 11.8 |
| `blur(4, iterations=6)` → `mipblur(40)` (wider, smoother) | 157.3 | 49.7 |
| Three noise layers and a modulation, `set_fast_noise()` | 18.2 | 7.4 |
| Patch 39 at 2:50, `set_fast_noise()` | 157.9 | 126.8 |
| Heavy feedback page, 960×540, SwiftShader, `--phone` | 4.5 fps | 20 fps (at 0.35) |

Tried and dropped: uniform buffer objects for signal uniforms (no measurable change; per-frame
Python overhead is already ~0.3 ms for 80 uniforms). The pipelined readback showed no change
under software GL, where "GPU" and CPU are the same cores; its gain is overlap on a real GPU.

## Experimental compute simulations

`rd()` and `bleed()` have optional tiled compute kernels for desktop OpenGL 4.3+.
They are **off by default**: the available software renderer measured slower
than fragment rendering, and no hardware GPU speedup has been established.
Enable them for a comparison on the target GPU:

```sh
hydrapy render patch.py --compute --size 1920x1080 -o compute.mp4
hydrapy render patch.py --gpu --size 1920x1080 -o fragment.mp4
```

`--compute` implies `--gpu`. Python callers can pass `compute=True` to
`GLRenderer`, `iter_frames_gl`, or `render_frames_gl`. The renderer requests a
4.3 context and falls back to 3.3 if needed. An existing `ctx` is kept as supplied.
An older context or rejected compute shader produces a warning and uses the
fragment implementation. `renderer.compute_passes` and
`renderer.compute_fallbacks` report what was selected. HTML remains WebGL.

Only whole, unwarped `rd`/`bleed` passes qualify. For example, `rd(iterations=4)`
uses four compute dispatches; `rd(iterations=4).rotate(.2)` uses three and retains
the final fragment pass, where rotation affects sampling. Colors and composites
also retain their fused final fragment pass. No extra materialization pass is
inserted to qualify an expression for compute.

Each 16×16 workgroup loads a shared tile with a one-pixel halo for RD or a
two-pixel halo for bleed. Bleed still interpolates straight RGBA at its 1.5-pixel
offsets before premultiplying. Iterations use separate dispatches, the existing
ping-pong textures and RGBA16F rounding, with memory barriers between passes.
There is no readback between iterations. Partial workgroups, wrapped edges and
one-pixel dimensions are covered by tests, along with feedback, signals, delayed
outputs, resizing, shared branches, transformed final passes and fallback.

Compare identical inputs and iteration counts in separate, sequential runs:

```sh
python docs/benchmark.py --backend gl --size 1920x1080 --case rd --case bleed_alpha \
  --frames 60 --warmup 10 --json fragment.json
python docs/benchmark.py --backend gl --size 1920x1080 --case rd --case bleed_alpha \
  --frames 60 --warmup 10 --compute --json compute.json
```

Repeat at `3840x2160`, then add `--readback` to both commands for video export
cost. `--iterations 16` overrides the simulation iteration count in both cases.
`bleed_alpha` uses varying colors and alpha; the original `bleed` benchmark
retains its opaque input. JSON includes actual compute/fragment pass counts,
fallback reasons, GL version and renderer so fallback cannot masquerade as a
compute measurement. These timings exclude video encoding.

On Mesa llvmpipe LLVM 20.1.2 / OpenGL 4.5, Python 3.12.14, NumPy 2.3.5 and
AMD EPYC 9V74, seven measured frames after three warm-up frames gave these
medians. Both paths ran sequentially with matching inputs; execution was
synchronized, without RGB download:

| Case | Size | Fragment (ms/frame) | Compute (ms/frame) |
| --- | --- | ---: | ---: |
| RD, 4 iterations | 1920×1080 | 78.1 | 86.4 |
| Bleed with alpha, 3 iterations | 1920×1080 | 51.5 | 96.4 |
| RD, 4 iterations | 3840×2160 | 296.5 | 349.5 |
| Bleed with alpha, 3 iterations | 3840×2160 | 220.4 | 392.5 |

Including RGB download at 1080p gave 79.4 vs 90.2 ms for RD and 56.0 vs
109.7 ms for bleed. These are software-OpenGL results, not RTX measurements.
Keep fragment rendering unless repeated measurements on the actual hardware
show a benefit for the intended patch.

The paths are not bit-identical. In seeded 1080p/4K comparisons, RD's largest
channel difference was 0.000977; bleed's largest alpha difference was 0.002442
and premultiplied RGB difference was 0.001249. A few fully transparent paint
pixels had larger straight-RGB differences (up to 0.656) where tiny rounding
changes crossed the existing `acc.a > 1e-4` color cutoff. Revealing or using
those hidden colors, or running long chaotic feedback, can amplify differences.
Use the default fragment path when preserving that behavior matters.
