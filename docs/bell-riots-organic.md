# Bell Riots Pt 2 — organic edit

The later [Unfolding edit](bell-riots-sparse.md) revisits these materials one at
a time, with more negative space and a strict two-layer limit.

A second composition using the final approximately six minutes of the original
recording. The new edit begins at **434.461896 seconds (7:14.462)** and preserves
the tail at 795.554694 seconds: **361.092798 seconds (6:01.093)**. The start is the
nearest 32-beat boundary on the estimated 136.88 BPM grid, not a verified DAW
downbeat. `--edit-start` accepts a replacement edit point.

The 123 shots keep the original pixel clocks, per-pixel rule fields, mirrored
thi.ng glyphs and entropy-locked registers, with longer holds and less tearing.
The OP art supplies a luminance matte for three organic materials:

- Actual Generative_Art leadplant and decayed-barn vectors, mirrored with
  opposing rotations. Source links and pinned revisions are in
  [assets/bell-organic](assets/bell-organic/README.md).
- A persistent 16,384-agent Physarum simulation on a 384×216 trail field.
  It takes 30 agent steps per second, driven by a fixed 15 Hz update clock.
- Six paintings baked once with HydraPy's dye, drops, ink bleed and crackle
  operators. Their crops, rotations and crossfades move continuously. This is
  animated painted material, not a new fluid simulation.

The drawings change mirror configuration every 23.7 seconds, paint changes
every 31.3 seconds, Physarum parameters follow 33.1 seconds, layer prominence
follows 41.9 seconds and the complementary palette follows 73.9 seconds. Each
has a different phase. These clocks never restart at a music cut, and continuous
opposing rotations avoid repeating a fixed stack of clips. The main cuts and
small exposure pulses still follow the recording. The colour pair travels
between cyan/copper and violet/yellow-green, with dark negative space retained
between pixel cells.

## Render

```bash
python docs/bell_riots_organic.py /path/to/track.flac \
  --art-root /path/to/Generative_Art \
  -o Bell_Riots_Organic.mp4 --size 1920x1080 --fps 30 \
  --cache bell-analysis.npz --assets-cache /tmp/bell-organic-assets \
  --checkpoints render-checkpoints
```

The recording stays local, and the drawings stay in the existing Generative_Art
checkout. No artwork is copied into HydraPy. Install Inkscape or `cairosvg` for the first
SVG rasterization; cached images are reused. Analysis caches are tied to the
recording's SHA-256. The `.cues.json` sidecar records the source trim, shot list,
clock periods, input drawing hashes, seed, dimensions and measured renderer speed.

`--offset 120 --seconds 20` renders an excerpt **within the new edit**. Its offset
is rounded to the nearest output frame; video preroll and audio trim use that
same value. Full preroll reconstructs feedback and all simulation states, so
an excerpt agrees with the corresponding full-render frames at the same FPS.
The edit starts a new simulation, warmed for 12 seconds, rather than trying to
reuse the first video's state after seven minutes.

The default is 1080p30 H.264 at 8 Mbit/s and AAC at 320 kbit/s. Audio is trimmed
without stretching. Video uses the ceiling of duration × FPS to retain the
last audio samples. The full composition runs on a shared GL context; it reads
back only the final RGB frame. Paint is baked once, Physarum stays on the GPU,
and the expensive cellular passes remain at reduced resolutions. Hardware
encoding is probed, with a software fallback. 4K output is accepted but has not
been performance-qualified.
