# Development

[Documentation](README.md) · [Project home](../README.md)

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
`_hash11/21/22/31` (Hydra's sin hashes), `_h11/21/22/31` (sin-free, bit-exact with NumPy), `_bayer4`, `_segd`, `_rot3`).

## Layout

```
hydrapy/core.py        Chain, Op registry, Output/Patch, Image/Video/Camera
hydrapy/ops.py         core ops: GLSL body + NumPy twin
hydrapy/noise.py       simplex noise, hashes, HSV — NumPy and GLSL
hydrapy/signals.py     Lfo/Seq/Clock/Env/SH/Drift/Fn/Audio, with JS emission
hydrapy/backend_np.py  CPU renderer
hydrapy/glsl.py        chain -> multi-pass GLSL compiler
hydrapy/backend_gl.py  moderngl renderer, headless and windowed
hydrapy/ops_paint.py   painted-film ops: stain alpha drops crackle veins scratches dust bleed bake pan
hydrapy/ops_3d.py      plane3d sphere3d cylinder3d
hydrapy/ops_glitch.py  tear, shred (entropy-locked glitch feedback)
hydrapy/ops_video.py   psort ruttetra contours sinefold bits shear · yiq linejitter headswitch dropouts ringing · filmstock gate
hydrapy/ops_stroke.py  harmonograph wave arc dab charcoal smear · halftone boil polarRepeat
hydrapy/drawing.py     Text, Lines (SVG flattener), plant()/meadow(), to_svg()
hydrapy/cellular.py    MultiCA1D engine, Cellular animation/glyph source
hydrapy/ops_fields.py  orbitVoronoi, SDF primitives/combinations/masks/contours
hydrapy/ops_material.py height-field normals, relighting, environment and refraction maps
hydrapy/sdf3d.py       composable 3D fields and raymarching
hydrapy/wire.py        spline, sweep and lathe wire geometry
hydrapy/timeline.py    parameter tracks and cue timelines
hydrapy/geometry.py    subdivision, coherent fuzzy paths, flora JSON projection
hydrapy/recipes.py     handpainted(), mothlight(), film(), vhs(), super8(), studie(), cutout()
hydrapy/live.py        CvIn/CvOut, OscIn/OscOut, Midi, Link, Probe, Soundtrack (drawn sound), per-frame hooks
hydrapy/synth.py       image quilting, Quilt frame bank
hydrapy/export.py      GIF/MP4/PNG/ANSI, Hydra JS, Python, standalone HTML runtime
hydrapy/encoding.py    encoder discovery, selection and streaming video output
hydrapy/hydra_import.py Hydra JS -> patch
hydrapy/cli.py
patches/               numbered examples and demoscene patches
docs/tour.py           renders docs/tour.mp4, the feature tour
docs/optable.py        regenerates docs/operators.md from the registry
tests/                 tests including CPU/GPU parity, exports and audio timing
```

## Check a change

```sh
python -m pip install -e ".[test,gpu,video]"
python -m pytest -q
```

GPU tests need a working GL context; encoded-video inspection also uses `ffprobe`.
Optional integrations are skipped when their dependencies are unavailable.

## Maintain the operator reference

Edit operator declarations in `hydrapy/ops*.py`, then regenerate the
[operator reference](operators.md) from the repository root:

```sh
python docs/optable.py
python docs/optable.py --print
```

The generator writes only `docs/operators.md`; the project README stays a short
entry point. Add new tutorials to the [documentation index](README.md).
