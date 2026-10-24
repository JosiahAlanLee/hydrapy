# Bell Riots Pt 2

The shorter [organic edit](bell-riots-organic.md) combines the closing six minutes
with Physarum, mirrored botanical/farm drawings and painted film on independent
clocks. This page documents the original full-track version.

A full-track video for Anchor Rite's recording, built around the `pixelOsc`
source and the per-pixel Life-like `ca()` operator. The composition uses ten
visual systems: phase lattice, district field, mirrored glyphs, phase aperture,
cellular loom, rule fronts, orbit contours, bit registers, retained image memory,
and a perspective transit grid.

The supplied track is 795.554694 seconds. Analysis estimates a 136.88 BPM pulse;
the edit has 487 shots on whole- or half-beat boundaries, including sustained
16-beat views. Arrangement landmarks follow the recording's changes in spectral
energy. They are authored for this track; using another track scales those
landmarks rather than claiming to detect its sections automatically.

## Render

```bash
python docs/bell_riots_video.py /path/to/track.flac \
  -o Bell_Riots_Pt_2.mp4 --size 1920x1080 --fps 30 \
  --cache bell-analysis.npz --checkpoints render-checkpoints
```

The default output is H.264 at 8 Mbit/s with AAC at 320 kbit/s. Hardware encoding
is probed, with a software fallback. Audio is re-encoded but never stretched,
rearranged or shortened to a whole number of frames. Video uses `ceil(duration *
fps)` frames so the final samples are retained. A `.cues.json` sidecar records
the recording hash, edit, entropy settings and actual renderer.

Use `--start 224 --seconds 18` for a trimmed test with matching audio, or
`--gallery scenes.jpg` for an overview of the ten systems. Checkpoint images are
occasional source frames saved during the full render, allowing inspection
without making repeated demo movies. `--size 3840x2160` is supported, but this
composition's delivery and verification target is 1080p.

## What drives it

- **Rule fields:** a coarse Brian's Brain field perturbs the rule map and flip
  probability of a finer automaton. Both retain their state throughout the
  track. They run at 15 simulation steps per second, independently of output
  frame rate, at 1/16 and 1/4 output dimensions. Bass affects update probability;
  transients affect fluctuation. Presentation samples cells without filtering.
- **Pixel clocks:** the renderer embeds the registered `pixelOsc` GLSL body.
  Static frequencies run continuously within a shot. Edits can deliberately
  switch density and geometry. Lattice and radial fields stay at output
  resolution rather than being generated as low-resolution video textures.
- **thi.ng cellular:** HydraPy's existing `Cellular` adapter generates a
  retained history with the two public demonstration rules. It is sampled as
  a loom or through an original glyph atlas. Its generation rate is fixed.
- **Entropy locking:** nine seeded `Fx` streams use lock probabilities from
  0.10 to 0.72. Their values are precomputed at half-beats, so seeking the score
  or changing output fps does not consume different random values. The locked
  values control tears, block holds, offsets and the rule bank. The random
  stream repeats while the cell and image histories continue evolving.
- **Orbit field:** the existing rotating-ring Voronoi construction is evaluated
  with 15 site positions computed once per frame on the CPU. The fragment pass
  computes nearest distances and contour bands, avoiding per-pixel trigonometry
  for the sites.

Excerpts replay the cellular simulation from zero and pre-roll two seconds of
image memory. Their initial feedback pixels are not guaranteed bit-identical to
a cut from a complete render. Extract a clip from the final movie when exact
matching is required. Feedback presentation depends on output fps; the cellular
clock does not. Floating-point hash differences can change asynchronous CA
evolution between GPU drivers.

## Provenance

The terminal framing takes its District A / 2024 cues from *Deep Space Nine*'s
“Past Tense” and the recording's title. See [StarTrek.com's episode
discussion](https://www.startrek.com/en-un/news/weight-of-optimism-and-birth-of-the-federation).
The images are generated geometry and cellular state; no episode footage is used.

The cellular and rotating-Voronoi adapters are documented in
[Animated thi.ng sources](TOXI_ANIMATION_SOURCES.md), with upstream attribution
in [NOTICE](../NOTICE). The glyph compositor is original; this is not an
ASCII-SCAPE edition-code port. [Pixel clocks](pixel-oscillators.md) records the
Memo Akten / PixlFlippr inspiration. Entropy locking uses HydraPy's existing
aebrer-inspired implementation (CC0 reference). Bit registers and the video
composition are original, informed by the earlier two-tone study.

The `ca()` dependency came from PR #17, which had been merged into
`perf/everything` rather than `main`. This branch integrates that existing commit
and resolves its import conflict with `pixelOsc`; it does not reimplement the op.

## Checks

```bash
python -m pytest tests/test_bell_riots_video.py tests/test_automata.py \
  tests/test_pixel_oscillators.py -q
```

Tests cover beat-aligned edit boundaries, repeatable entropy controls,
absolute-time audio controls, fixed cellular stepping across sparse/dense
presentation calls, and nonblank output from all ten systems. Full-video QA also
checks decoding, frame continuity, duration and audio alignment.
