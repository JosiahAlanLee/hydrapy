# Bullfrog demo

[Studio](studio.md) · [Project home](../README.md)

The demo uses **Anchor Rite — Bullfrog**, from *Pinned*. Supply your local audio
file; the music is not distributed in this repository. The supplied track runs
5:36. The renderer uses its actual duration and audio spectrum to drive the piece.

```sh
python -m pip install -e ".[gpu,video]"
python docs/bullfrog_demo.py "/path/to/Bullfrog.flac" -o bullfrog.mp4
```

The default is 1280×720 at 24 fps. ffmpeg is required for FLAC decoding and audio
muxing. The MP4 contains the original stereo track, encoded as AAC. The command
prints the renderer and encoder selected on your machine. Desktop GL rendering can also run
on Mesa software rendering when no GPU is available.

Use `--seconds 20 --size 640x360` for an opening excerpt,
`--contact scenes.png` for a contact sheet, or
`--preview-only --contact scenes.png` to sample all seven scenes without encoding
a movie. A `.cues.json` beside the requested MP4 records the chapter boundaries.

| Track position | Scene | Technique |
| --- | --- | --- |
| 0% | Pressure | Oscillators and feedback |
| 13% | Current | Mathematical marbling |
| 27% | Growth | Neural cellular automata |
| 41% | Network | Physarum trails |
| 57% | Contour | Morphing distance fields |
| 72% | Surface | Normal maps and moving light |
| 86% | Return | Orbiting cells and oscillator interference |

Each transition lasts ten seconds. Both renderers keep advancing throughout the
dissolve; feedback and simulation state remain live on the outgoing scene. Low,
mid and high audio bands drive different visual properties within each scene.
The chapter schedule is composed, not an automatic musical-section detector.

Dissolves and fades run on the GPU with one frame download for encoding. Only
the feedback, NCA and Physarum scenes need simulation warm-up. Output resolution
and simulation settings are preserved. For 1080p/4K timings and a repeatable
benchmark command, see [rendering performance](PERFORMANCE.md).

## Perform a scene in Studio

```sh
HYDRAPY_TRACK="/path/to/Bullfrog.flac" HYDRAPY_DEMO_SCENE=3 \
  hydrapy studio patches/bullfrog.py
```

Scene indices are 0–6. **Motion**, **Audio response**, **Colour**, and **Detail**
are editable performance controls. Without `HYDRAPY_TRACK`, the patch uses slow
synthetic modulation so it can be auditioned immediately. Studio previews the
picture; the finished MP4 is the place to hear it with the track. The full demo's
chapter sequencing and dissolves live in `docs/bullfrog_demo.py`.
