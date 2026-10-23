# Operator reference

[Documentation](README.md) · [Project home](../README.md)

Generated from the operator registry with `python docs/optable.py`.
Edit operator declarations, then regenerate this page.

[Sources](#sources) · [Coordinates](#coordinate-ops) · [Colour](#colour-ops) ·
[Combining](#combine-ops-take-another-chain) · [Modulation](#modulate-ops-coordinates-driven-by-another-chain) ·
[Textures](#texture-ops-neighbourhood-multi-pass-on-the-gpu)

## Sources

| op | args | origin | notes |
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
| `dropouts` | `density=0.5, extent=0.3, rate=60, seed=0` | **new** | Tape dropouts: short bright streaks along random scan lines, new ones `rate` times a second (white on black, alpha = streak). lighten() them onto the picture. |
| `gate` | `width=0.96, height=0.94, radius=0.06, softness=0.01` | **new** | The projector gate: a rounded-rectangle mask (white inside, alpha = inside). mult() the picture by it; pan it a little for a frame that doesn't sit still. |
| `harmonograph` | `f1=3, f2=2, f3=0, f4=0, phase=1.5708, decay=0, length=1, size=0.4, width=0.012, taper=0, speed=0.1` | **new** | A Lissajous / harmonograph curve drawn as a stroke (white, alpha = stroke): x = sin(f1 t + phase), y = sin(f2 t); f3/f4 > 0 add a second pendulum per axis, `decay` damps it over `length` turns, `speed` turns the phase. `taper` thins the tail; the head is the end of the curve. Keep length x frequency under ~10 for a smooth curve (256 segments). |
| `wave` | `frequency=3, amplitude=0.15, width=0.012, speed=1, phase=0, length=1, taper=0` | **new** | An oscillator drawn as a stroke: a sine across the frame, `frequency` cycles per width, travelling at `speed`. `length` is the stroke's extent (fraction of the width); `taper` thins both ends. |
| `arc` | `radius=0.3, start=0, sweep=3.1416, width=0.012, taper=0, speed=0` | **new** | An arc stroke about the centre from angle `start` through `sweep` radians (a full circle: 6.2832), turning at `speed`. `taper` thins the tail; the head is the end of the arc. |
| `dab` | `size=0.15, aspect=0.5, angle=0, bristles=0.5, softness=0.3, seed=0` | **new** | One touch of a bristle brush: an ellipse `size` tall (fraction of the height) and `aspect` as wide, turned by `angle`, streaked along its length by `bristles`. White, alpha = paint. Feed it back for accretive painting: src(o0).layer(dab(...).color(...)). |
| `orbitVoronoi` | `points=9, rings=4, radius=0.42, speed=0.18, phase=0, width=0.004, mode=0` | **new** | Alternating rotating rings of Voronoi sites. points=3..20 per ring, rings=1..4; speed in radians/second. mode=0 shaded cells, 1 edges, 2 nearest distance. width uses the nearest/second-nearest distance gap (not a geometric polygon inset). |
| `sdfCircle` | `radius=0.25, x=0.5, y=0.5` | **new** | Signed distance to a circle; negative inside. radius is in screen-height units, x/y in normalized screen coordinates. Finish with sdfMask or sdfBands. |
| `sdfBox` | `width=0.45, height=0.35, radius=0.04, x=0.5, y=0.5` | **new** | Signed distance to a rounded box. Full width/height and corner radius are screen-height units; center x/y is normalized. Finish with sdfMask or sdfBands. |
| `sdfMetaballs` | `count=5, radius=0.12, spread=0.28, smoothness=0.12, speed=0.5, seed=0` | **new** | Smooth union of 1..16 animated circles, a negative-inside distance bound. Deterministic motion in seconds; radius/spread/smoothness use screen-height units. Finish with sdfMask/sdfBands or derive a normal map. |

## Coordinate ops

| op | args | origin | notes |
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
| `tear` | `seed=0, count=3, size=0.12, shift=0.3, vertical=0` | **new** | `count` horizontal bands (vertical=1: columns) torn sideways by up to `shift`; which bands and how far come from `seed`. |
| `shred` | `seed=0, count=4, size=0.2, scale=1` | **new** | `count` rectangular samples of the image copied onto other places (magnified by `scale`); which, where and how big come from `seed`. Later copies copy from earlier ones. |
| `shear` | `x=0.3, y=0` | **new** | Skew: x shifts rows by their height, y shifts columns by their width. |
| `linejitter` | `amount=0.004, rate=60, wobble=0.002, seed=0` | **new** | Tracking error: every scan line shifts sideways by its own random amount, worse in drifting bands, plus a slow time-base wobble. |
| `headswitch` | `height=0.06, amount=0.08, rate=60, seed=0` | **new** | Head-switching noise: the bottom `height` of the frame tears sideways, more towards the very bottom, re-rolled `rate` times a second. |
| `boil` | `amount=0.004, rate=12, scale=6, seed=0` | **new** | Line boil: the drawing redrawn by hand `rate` times a second — coordinates wander by up to `amount` with smooth noise at `scale`, re-rolled each redraw. Pair it with set_frame_rate(12) for animation on twos. |
| `polarRepeat` | `n=5, offset=0, aspect=1` | **new** | `n` rotational copies in image space. `aspect=0` preserves the old normalized-coordinate distortion; default 1 keeps circular orbits on rectangular frames. |

## Colour ops

| op | args | origin | notes |
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
| `grain` | `amount=0.1, rate=60, size=1` | **new** | Film grain; `size` in pixels (Super 8 is coarse: 2-3). |
| `vignette` | `amount=0.6, radius=0.6` | **new** | Darken the corners. |
| `contours` | `levels=6, width=0.15, fill=0` | **new** | Topographic keyer: white isolines where luminance crosses `levels` thresholds; `fill` keeps that much of the image between them. |
| `sinefold` | `stages=2, phase=0` | **new** | Sine staircase shaper: each channel through `stages` cycles of a sine; ramps become rainbows, gradients become bands. |
| `filmstock` | `stock=0, amount=1` | **new** | Film stock tone: 0 Kodachrome (warm, dense), 1 Ektachrome (cool shadows), 2 faded (lifted blacks, yellowed), 3 Tri-X (B&W S-curve), 4 Orwo / Sovcolor (muted, green mids, magenta shadows). |
| `halftone` | `size=6, angle=0.7854, sharpness=0.7` | **new** | Screen print: each channel becomes a grid of ink dots `size` pixels apart, sized by how dark the channel is, the three screens turned 0, 60 and 15 degrees from `angle` so they moiré like a cheap colour print. |
| `sdfMask` | `offset=0, softness=0` | **new** | Finish an SDF as an opaque black/white mask. Positive offset expands; softness has a half-pixel minimum. Apply palettes/texture masks afterwards. |
| `sdfBands` | `frequency=20, width=0.3, phase=0` | **new** | Finish an SDF as repeated contour bands; frequency per height unit, width is the filled fraction, phase shifts the bands. Animate the field before this op for continuous morphs. |
| `relight` | `azimuth=0.8, elevation=0.7, ambient=0.18, diffuse=0.65, specular=0.35, shininess=32` | **new** | Light an encoded normal map with Lambert diffuse + Blinn specular. Angles in radians; elevation=pi/2 faces the viewer. Returns HDR grayscale lighting with the input alpha; multiply by pigment separately. |

## Combine ops (take another chain)

| op | args | origin | notes |
|---|---|---|---|
| `add` | `amount=1` | Hydra | Add. |
| `sub` | `amount=1` | Hydra | Subtract. |
| `layer` | `` | Hydra | Alpha-composite on top. |
| `over` | `` | **new** | Straight-alpha source-over, correct for transparent layers on transparent layers (pigment kept apart from its paper). Onto an opaque picture Hydra's `layer` gives the same colour. |
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
| `bits` | `mode=0, depth=4` | **new** | Bitwise logic between two chains quantised to `depth` bits: mode 0 XOR, 1 AND, 2 OR. Two ramps XORed are the oldest video-synth texture there is. |
| `sdfUnion` | `smoothness=0` | **new** | Union of signed distance fields; smoothness > 0 rounds the join with a polynomial smooth minimum. |
| `sdfIntersect` | `` | **new** | Intersection of signed distance fields (maximum). |
| `sdfSubtract` | `` | **new** | Cut the other signed distance field out of this field. |

## Modulate ops (coordinates driven by another chain)

| op | args | origin | notes |
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
| `envMap` | `rotation=0` | **new** | Sample this chain as a latitude/longitude environment reflected by encoded normals. rotation is turns. A view-facing normal samples (0.75,0.5). Set coverage separately with alpha(normals.a()). |
| `refractMap` | `amount=0.05` | **new** | Screen-space glass distortion from encoded normals; amount in screen-height units. This is a normal-driven UV offset, not physical ray refraction. Zero amount is identity. |

## Texture ops (neighbourhood; multi-pass on the GPU)

| op | args | origin | notes |
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
| `psort` | `threshold=0.3, vertical=0, iterations=8` | **new** | Pixel sorting on the GPU: each iteration is one odd-even transposition step along rows (vertical=1: columns) within runs brighter than `threshold`. Feed it back (src(o0).psort()) and it keeps sorting frame after frame. |
| `ruttetra` | `lines=90, amount=0.25, thickness=1.2, glow=0.3` | **new** | Rutt-Etra scan processor: `lines` horizontal scan lines, each displaced upward by `amount` times the brightness under it, drawn `thickness` pixels wide (alpha = line). Rotate/scale the chain before it to tilt the raster. |
| `yiq` | `chroma=0.03, luma=1, shift=0.004` | **new** | Composite-video bandwidth: chroma (I/Q) smeared horizontally over `chroma` of the width and delayed by `shift`, luma softened over `luma` pixels. The colour bleed of tape. |
| `ringing` | `amount=0.5, offset=2` | **new** | Edge ringing: overshoot and a second ripple to the right of every vertical edge, `offset` pixels apart. The sharpening of a worn tape deck. |
| `charcoal` | `tooth=0.5, rough=1.5, scale=90, seed=0` | **new** | Paper tooth removes coverage from alpha while preserving straight RGB colour. Edges jitter by `rough` pixels; grain at `scale` stays fixed in image space. Composite over paper with `over` or `recipes.on_paper`. |
| `smear` | `angle=0, length=0.02` | **new** | Paint dragged: 12 taps behind each pixel along `angle` (radians; a chain makes it a flow field), `length` of the height. Paint on glass, smudged charcoal, motion blur. |
| `normalMap` | `strength=0.03, radius=1` | **new** | Red channel as height; central-difference normals encoded in RGB as 0.5+0.5*n, +Y up. strength is height per screen height; radius is the sampling distance in pixels (minimum 0.5). Preserves alpha; sampling wraps. |

