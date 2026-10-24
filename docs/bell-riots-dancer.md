# Bell Riots — half-time dancer / edit 5

This pass keeps the fourth edit's 6:01.093 audio, shot timings and upper-register
responses. In the paint/moiré passages, a slow Body Roll silhouette passes
through HydraPy's real `delay()` output history and becomes a matte for the
paint. The moiré remains behind it. Other scenes are unchanged.

## Timing

The requested 69 BPM feel is locked to the track's **137 BPM** grid: **68.5 BPM
half time**, selected to avoid drift without stretching the audio. One dance
cycle spans 16 half-time beats / 32 track beats / eight bars: **14.01459854 s**.
This is 0.460829× the source clip's native playback speed.

Detected kick positions anchor the phase. Each onset is assigned its score
beat number; gaps interpolate without renumbering the choreography. Phase
continues across cuts. The original 155-frame action is evaluated at 480
subframe positions, with 30 samples per half-time beat. The printed footage
uses the exact rational frame rate **137/4 fps**; the final video is 30 fps.
This avoids accumulating a rounded loop-duration error.

The delay patch runs at 15 phase ticks per track beat. It uses
`delay(o0, 30)`, `delay(o0, 60)` and `delay(o0, 90)` at gains 0.60, 0.34 and
0.18, with small horizontal offsets. These are one, two and three half-time
beats behind. The delays are timed in musical phase, including local kick
timing variation. A bounded 90-tick pre-roll gives excerpts and scene returns
the same history as continuous playback.

The 1080p source print is reduced to a 960×540 grayscale matte for the live
delay ring. Only that matte needs temporal storage. The two rendered materials
are still moiré and paint: the current/delayed figure shapes and lights the
paint rather than adding another independent image texture.

## Reproduce

Use the self-contained `Dancer_Overlay.blend` from the CC0 dancer kit. Its
mannequin, rig and three original actions are local to the file. Blender 5.1+
is needed for this source scene.

```bash
blender -b Dancer_Overlay.blend -t 6 -P docs/render_body_roll.py -- \
  --output /tmp/body-roll
```

Add `--warp` to land the movement's holds on the half-time beats. The script
measures the source action's mean bone speed and takes its minima below 0.7×
the mean. Each hold must be a minimum over ±12 output frames, the window used by
`bell_riots_checks.py`. A monotonic piecewise-linear time map then moves each
hold to its nearest free multiple of 30 frames, strongest first, keeping 480
frames per cycle. Holds that would collide, or would make a segment more than 2×
faster or slower than the linear stretch, are left where they fall. For Body
Roll, holds at source frames 37.6, 50.1, 61.9, 71.6, 96.1, 124.6 and 137.6 land
on half-beats 4, 5, 6, 7, 10, 13 and 14. Segment speeds stay within 0.75–1.19× of
the linear stretch. The loop-point hold (155.1) stays 2.7 frames before the bar
line. `timing.json` records the holds and the breakpoints.

The print directory contains a retimed `.blend`, `timing.json` and 480 RGBA
PNGs. Convert their alpha planes to a `uint8` NumPy array of shape
`[480, 540, 960]`, in top-to-bottom image order:

```bash
python docs/prepare_dancer_matte.py /tmp/body-roll/frames -o body-roll-matte.npy
```

All 480 frames are kept and the duplicate endpoint is omitted. The renderer
accepts this `.npy` file with `--matte`.

```bash
python docs/bell_riots_dancer.py Bell_Riots_Counterpoint_Mix.flac \
  --matte body-roll-matte.npy --controls upper-controls.npz \
  --kicks bell-kicks.npz --art-root /path/to/Generative_Art \
  --assets-cache /tmp/bell-organic-assets \
  -o Bell_Riots_Edit_5_1080p.mp4 --checkpoints render-checkpoints
```

Use `--offset` and `--seconds` for an excerpt. `--title-scale 2.5` enlarges the
title block for phone viewing and keeps it centred in the same place. The
default is edit 5's size. The audio file and controls are
the outputs of [the counterpoint mix](bell-riots-counterpoint.md). The input
recording and Generative_Art drawings remain external assets.

## Validation and sources

```bash
python -m pytest -q tests/test_bell_riots_dancer.py \
  tests/test_bell_riots_counterpoint.py tests/test_audio_timing.py
```

Checks cover missing/duplicate kicks, long-duration phase, exact GPU delay
taps, excerpt pre-roll, unchanged non-paint scenes and the two-material limit.

The mannequin and Body Roll animation come from
[Mesh2Motion's CC0 art assets](https://github.com/Mesh2Motion/mesh2motion-assets)
at revision `c5b0b6c821dbf2fa1dfe49590a6f431fd9f707ce`:

- `rigs/rig-human.blend`
- `rigs/human/animation-human-dance-body-roll.blend`

The [CC0 license](https://github.com/Mesh2Motion/mesh2motion-assets/blob/c5b0b6c821dbf2fa1dfe49590a6f431fd9f707ce/LICENSE)
permits reuse of the assets. Credit: Mesh2Motion contributors. The source kit
retains the upstream license. This repository contains the print/render
adapters, not the binary character assets. Earlier visual-source credits are
in [edit 4](bell-riots-counterpoint.md#validation-and-provenance).
