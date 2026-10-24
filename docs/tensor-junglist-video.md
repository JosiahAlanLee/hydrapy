# Tensor Junglist

A full-track music video for **Anchor Rite — Tensor Junglist**, from *Pinned*.
The recording is supplied by the user and is not included in this repository.

The first render is 1920×1080 at 30 fps, with H.264 video targeting 6 Mbit/s
and 256 kbit/s AAC stereo. The complete 379.130-second recording is used without
time stretching, rearrangement or gain changes. AAC is a lossy delivery format;
the supplied FLAC remains the audio master.

The edit combines nine scenes: cellular arrays, folded distance geometry,
orbiting Voronoi contours, a projected vector cage, an analytic tunnel,
metaball contours, cellular glyphs, oscillator feedback and a reflective knot.
The score for this recording produces 396 shots. Cuts follow tracked beats;
transient strength controls horizontal tearing, chromatic separation and
blocks retained from the previous video frame. Some geometry shots are held
longer, and the opening and ending carry the artist/title card.

## Render

Install HydraPy's GPU extra and make FFmpeg available on PATH:

```sh
pip install -e '.[gpu]'
python docs/tensor_junglist_video.py '/path/to/Anchor Rite - Pinned - 06 Tensor Junglist.flac' \
  -o Tensor_Junglist.mp4
```

The renderer probes hardware H.264 encoders and falls back to libx264. This
version was rendered on Mesa llvmpipe software OpenGL, using libx264. No compute
shader mode is required. Each frame renders only its selected scene and a
shared post-processing pass; the encoder receives one RGB readback.

Preview an excerpt with its matching audio, or inspect a contact sheet:

```sh
python docs/tensor_junglist_video.py track.flac --start 112 --seconds 16 \
  --cache Tensor_Junglist.analysis.npz -o preview.mp4
python docs/tensor_junglist_video.py track.flac --size 960x540 \
  --cache Tensor_Junglist.analysis.npz --stills contact.jpg -o inspection.mp4
```

`--stills` writes the contact sheet and cue JSON, without a video.
`--size`, `--fps`, `--bitrate` and `--encoder` can be changed for another export.
The default cache filename follows the output basename; supply `--cache` to
share analysis between renders. Its SHA-256 identity prevents reuse for a
different recording. Analysis works in bounded FFT batches using NumPy; the
beat tracker is HydraPy's existing dynamic-programming implementation.

The estimated pulse is about 137.6 BPM, with actual cuts interpolated between
tracked beat times. This is an analysis result, not an assertion about the
recording's production tempo. The accompanying `.cues.json` records every shot,
scene, section and variation. The `.analysis.npz` contains the detected beat
times and control envelopes. Neither file embeds the original audio.

The feedback scene and post-processing retain rendered history. An excerpt
starts with empty post-processing history and a short feedback warm-up, so its
first few frames can differ from the corresponding full-video frames.

## Sources and validation

The cellular scenes use the public **thi.ng cellular** engine adaptation by
Karsten Schmidt already in HydraPy. The Voronoi scene uses HydraPy's adaptation
of the public rotating-Voronoi example by Alberto Massa. These are the sources
documented in [Animated thi.ng adaptations](TOXI_ANIMATION_SOURCES.md) and
[NOTICE](../NOTICE). No C-SCAPE, ASCII-SCAPE or De/Frag edition assets are copied.
The tunnel, scene arrangements, score, typography and glitch composition are
original to this video. The distance and wire scenes use HydraPy's demoscene
operators.

Validation includes compiling and rendering all nine scenes, inspecting their
contact sheet and title frame, exporting a 16-second 1080p excerpt with matching
audio, and decoding that excerpt with FFmpeg. Score tests check deterministic
cuts, beat interpolation, control sampling, complete coverage and registered
operator isolation. The full delivery is checked for duration, dimensions,
frame count, audio/video synchronization, decode errors and unintended blank or
frozen intervals.
