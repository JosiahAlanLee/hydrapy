# Animated thi.ng sources and hydrapy adaptations

Checked 2026-10-23. Upstream is moving to Codeberg; links below use the
readable GitHub mirror pinned at
[`5c5e334`](https://github.com/thi-ng/umbrella/tree/5c5e334241a88fbd3ea31b906a091746539e682b).

## What was actually found

| Work / repository | Verified source | What hydrapy uses |
| --- | --- | --- |
| [C-SCAPE](https://www.fxhash.xyz/generative/13992) | [`@thi.ng/cellular`](https://github.com/thi-ng/umbrella/tree/5c5e334241a88fbd3ea31b906a091746539e682b/packages/cellular); its README explicitly identifies it as C-SCAPE's engine | Python adaptation of shared-rule 1D automata with age, arbitrary historical kernels and per-cell masks. Only the two public README demonstration rules ship as defaults. |
| [ASCII-SCAPE](https://www.fxhash.xyz/generative/16205) | Artist describes a text rendering successor to C-SCAPE. No complete licensed edition repository located. | Original glyph renderer over our cellular adapter, using installed fonts. No curated catalogs, themes or Iosevka font bundled. |
| [Rotating Voronoi example](https://github.com/thi-ng/umbrella/tree/5c5e334241a88fbd3ea31b906a091746539e682b/examples/rotating-voronoi) | Apache-2.0 example by Alberto Massa; credits Mike Bostock and Mario Klingemann | Alternating concentric seed rings, implemented as an analytic NumPy/GLSL field. |
| [De/Frag V3](https://www.fxhash.xyz/generative/4304) | Artist describes animated temporal Voronoi / jump flooding. No complete licensed project repository located. | No claim to port its renderer. `orbitVoronoi` is the separate public example above; it has neither jump flooding nor the painterly accumulation engine. |
| [Quasiflock](https://www.fxhash.xyz/generative/6671) | [Artist's Zig/WASM project notes](https://github.com/thi-ng/blog/blob/main/2022/20220903-zig-projects-wip.md) describe strand/flocking work derived from a 2006 project; some listed projects were not open source | No exact engine port. [`@thi.ng/boids`](https://github.com/thi-ng/umbrella/tree/5c5e334241a88fbd3ea31b906a091746539e682b/packages/boids) is a separate open starting point for a future original strand simulation. Quasiflock is not the quasi-lattice example. |
| [`tpl-umbrella-fxhash`](https://github.com/thi-ng/tpl-umbrella-fxhash), [`genart-api`](https://github.com/thi-ng/genart-api) | MIT application/template infrastructure | Useful for platform integration; these are not repositories of the finished animated artworks. |

The reusable engine's Apache license is distinct from the artwork editions'
licenses. C-SCAPE and ASCII-SCAPE descriptions specify CC BY-NC-ND 4.0;
we do not copy their artwork assets or edition code. Attribution and changes
are recorded in [NOTICE](../NOTICE), with the full upstream license alongside it.
"Not located" means this search did not find a complete licensed source, not
that no source exists anywhere.

## Cellular: animation as a composable texture

```python
from hydrapy import *

bpm = 120
cells = Cellular(width=384, height=216, rate=24 * bpm / 60, seed=17)
src(cells).palette(.04, .23, .53).kaleid(5).out(o0)
```

`rate` is a fixed number of generations per second. This example advances
24 generations per beat. Parameters on the resulting shader chain accept
the usual clocks, envelopes, LFOs and audio signals. A later change to
`set_bpm()` does not automatically change an already constructed Cellular's
rate. Set its rate when constructing the source for the intended tempo.

`CARule(rule, kernel=WOLFRAM3, states=2, reset=False, positional=True)` uses
the first kernel entry as the **least significant** neighborhood bit, matching
thi.ng. Kernel pairs are `(horizontal_offset, history_lag)`. Nonzero cells
count as occupied regardless of age. Negative lags are ignored. A rule's
selected bit increments cell age; an absent bit clears it. `reset=True`
wraps age to zero at `states`, while `False` saturates at `states - 1`.
Our default `False` makes binary rules useful directly; upstream defaults
to resetting. Kernel size is capped at 16 and historical lag at 64.

`MultiCA1D` exposes `.current`, `.history` (row 0 = current), `.mask`, and
`.update()` for custom simulations. Rules share history and can interact at
mask boundaries. For example, create a history-dependent rule with
`CARule(0x73ed2ac2, ((-2,1),(-1,0),(0,0),(1,0),(2,1)), 64, True)`.
This is an example configuration, not a recovered edition parameter.

`Cellular` prefills its displayed history at construction, advances in fixed
steps, and keeps only the current history window. Backwards seeks replay from
the seed, so a long backwards seek costs simulation time. Its perturbation
is our per-cell injection rule, not a port of upstream's probabilistic update
and mask-perturbation APIs. Seeded results are independent of output fps and
of intermediate render calls; they do not reproduce thi.ng's PRNG sequence.

Use `chars=" .:-=+*#%@"` for a glyph grid. `width`/`height` become cell counts;
`cell=(10,14)` controls pixel size. The first character represents an empty
cell. Font rendering can vary with the installed fallback font.

## Voronoi and signed distance geometry

```python
orbitVoronoi(points=11, rings=4, speed=.22, mode=0).palette().out(o0)

field = sdfCircle(.22, .42).sdfUnion(sdfBox(.35, .4, .06, .58), .12)
field.sdfSubtract(sdfCircle(.1)).sdfBands(18, .25, Time() * .2).out(o1)
```

`orbitVoronoi` accepts 3–20 sites per ring and 1–4 rings. These counts are
floored and clamped. Mode 0 draws cell tones, 1 edges, and 2 nearest distance.
Its width threshold is a nearest/second-nearest distance gap, not an exact
polygon inset. The public example uses mesh polygons and subdivision; this
adaptation uses a bounded per-pixel search of at most 81 sites, keeping the
whole operation on the GPU in GL/HTML. It is heavier than a local 3×3 Voronoi
search; reduce sites/rings for faster rendering.

The SDF operations store negative-inside signed distances in RGB. Combine
with `sdfUnion`, `sdfIntersect` or `sdfSubtract`; `blend(other, amount)`
interpolates the field to morph shapes. Finish with `sdfMask` or `sdfBands`
before applying a palette or using it as a texture mask. A blended field
preserves the interpolated zero contour but is not necessarily an exact
Euclidean distance function. Circle radii and full box dimensions are in
screen-height units, centers are normalized screen coordinates.

Keep distance math before the final display conversion: exporting a raw
distance field to PNG/MP4 clips negative values. Float GL buffers can retain
them between passes; HTML's 8-bit fallback buffers cannot.

## Curves, print effects and shared vegetation

`subdivide(points, iterations=2, closed=False)` applies Chaikin corner
cutting in 2D or 3D. `fuzzy_paths(paths, passes=4, jitter=.003, seed=0,
time=t)` adds stable overdraw with smoothly varying offsets. Both return
plain arrays usable by `Lines`, `to_svg`, or external geometry code.
These are new small implementations, not the full Generative_Art fuzz library.
No Shapely or Blender dependency is introduced.

```python
paths = flora_lines("plant.json", azimuth=25, elevation=12, detail=1)
solid(.02, .03, .02).layer(src(Lines(paths, color=(.6,.9,.4)))).out(o0)
```

`flora_lines` reads the `wisconsin-flora-1` JSON schema from
[`Generative_Art/lib/flora`](https://github.com/JosiahAlanLee/Generative_Art/tree/181c161072f7737e1a61e6797a7e89d13cc6c386/lib/flora)
or accepts its Plant object directly. It projects existing strokes; surfaces
do not occlude them. For hidden-line removal, import the SVG produced by
Generative_Art's renderer with `Lines("plant.svg")` instead.
Use the existing models' wind deformation before projection for consistent
plant motion. For animated geometry, keep a fixed view and use
`Lines(..., fit=False)` after normalizing to that view: fitting each frame
independently causes apparent zooming. This work does not introduce new
species or interpolate between independently generated growth stages.

`dither` (Bayer) and `halftone` already operate on animated sources in hydrapy.
Those remain the realtime print paths. Error diffusion and Poisson stippling
in Generative_Art are appropriate for baked frames or offline geometry;
regenerating random stipples every frame would shimmer. They were not copied
as redundant realtime modules here.

## Python source bridge and export

`FrameSource(draw)` accepts a callable returning a **uint8 RGB/RGBA** image
with row 0 at the top, for any requested time in seconds. It caches one frame;
GL reuses one uploaded texture, reallocating only if dimensions change.
The callable is responsible for reproducible seeks. NumPy generation still
runs on the CPU, followed by a texture upload for GPU composition.

Python sources (`Cellular`, `FrameSource`) run in NumPy and desktop GL.
For standalone HTML, bake a finite cycle with
`Bank([source.frame_at(i / fps) for i in range(count)], rate=fps)`.
HTML export otherwise raises an explicit error. `orbitVoronoi` and all SDF
ops are native and export directly to HTML. Cellular supports Python source
roundtrip; arbitrary draw callbacks must remain in their original patch.

See patches 28–31 and `docs/generative_demo.py`. The demo tries hardware
H.264 encoding through `encoder="auto"`, falling back to software when no
working hardware encoder is available.
