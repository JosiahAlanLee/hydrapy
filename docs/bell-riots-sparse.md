# Bell Riots Pt 2 — Unfolding edit

A third pass on the same closing edit: **7:14.462–13:15.555** of the recording,
lasting **6:01.093**. Opus's [Unfolding](../patches/41_unfolding.py) stays on
screen while individual materials from the earlier videos enter through sparse
masks. Large outer cell regions are drawn as contours, leaving their interiors
dark; the central rosette keeps its finer, filled structure.

## Two-layer composition

The base is always Unfolding. At most one accent is visible: mirrored thi.ng
glyphs, pixel oscillators, bit registers, Physarum, one Driftless drawing, or
one baked painting. There are 24 accent passages, lasting 6.3–11.4 seconds, with
4.3–7.9 seconds of the base alone between them. Each accent fades completely
out before another enters. Titles occupy the second layer at the opening and
closing; accents are disabled then.

The accent mask selects about 18% of a mirrored 16×10 block field before
insetting the blocks and suppressing them over bright base pixels. This is a
selection probability, not a guaranteed screen-coverage percentage. The warm
accent hue stays opposite the cool base hue on a 311-second palette cycle.
Mask seeds use the existing `Fx` entropy lock at `.72`. Accent timings, rule
changes, palette drift and material motion are independent of the kick clock.

## Kick clock and unfolding

`bell_riots_kicks.py` measures 30–155 Hz onset energy from the actual trimmed
audio. It finds 825 candidates in this recording. Existing beat estimates only
constrain the early edge of a nearby measured attack; they do not fill in
missing kicks. This is low-frequency onset detection, not drum-stem separation,
so strong bass transients can still be mistaken for kicks in other recordings.

Each attack starts a small zoom/exposure pulse with a 105 ms decay. Eight CA
generations are distributed between successive attacks. Recursive 2× folds
occur every four or eight kicks: their intervals therefore stay at exactly
32 or 64 generations, preserving Unfolding's power-of-two construction. The
cadence changes only at a fold, and every sixth fold plants a new central
glyph. At 30 fps a trigger is shown on the first frame at or after its detected
time, adding less than one frame of display quantization.

Inner/outer rules retain their independent 41/67-second schedules, with the
173-second rule-region cycle from patch 41. The simulation uses a fixed
320×180 cell field. The altered clock is factored into
[`patches/unfolding_score.py`](../patches/unfolding_score.py); the original
standalone patch is unchanged.

## Render

```bash
python docs/bell_riots_sparse.py /path/to/track.flac \
  --art-root /path/to/Generative_Art \
  -o Bell_Riots_Unfolding_Edit.mp4 --size 1920x1080 --fps 30 \
  --cache bell-analysis.npz --kick-cache bell-kicks.npz \
  --assets-cache /tmp/bell-organic-assets \
  --checkpoints render-checkpoints
```

Use a GL-capable HydraPy environment, FFmpeg, SciPy and either Inkscape or
CairoSVG. The same analysis and raster caches as the organic edit can be reused.
The recording and source vectors stay outside this repository; source paths
and attribution are recorded in [the drawing manifest](assets/bell-organic/README.md).

`--offset 36 --seconds 24` renders an excerpt within the closing edit. The
offset is rounded to an output frame for both audio and video. Hidden
simulations catch up without rendering full-resolution preroll frames; no
composited-image feedback needs to be reconstructed. Fixed simulation clocks
make sparse and sequential presentation calls agree on the same GL driver.
Floating-point hash differences can change CA evolution between GPU drivers.

The renderer shares one GL context, updates Physarum at 15 Hz on a 384×216
trail field, bakes six paintings once, and samples only the selected accent.
Default delivery is 1080p30 H.264 at 8 Mbit/s with 320 kbit/s AAC. The audio
trim is not stretched. A `.cues.json` sidecar records the input hash and trim,
kick/fold times, accent cues, drawing hashes, dark-area samples and render speed.
4K output is accepted; this pass is validated at 1080p.

## Checks

```bash
python -m pytest tests/test_bell_riots_sparse.py -q
```

Tests cover kick/hi-hat discrimination and missing beats, 32/64-generation
fold intervals, trigger quantization, the layer budget including titles and
transitions, deterministic simulation catch-up, and all six accent render paths.
The render test uses synthetic vector fixtures. Full-video QA checks decoding,
frame count, duration, audio alignment and the layer budget over every frame.

The older material's upstream attribution remains in
[the original video documentation](bell-riots-video.md#provenance) and `NOTICE`.
