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
