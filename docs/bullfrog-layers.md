# Bullfrog, in layers

[Documentation](README.md) · [Patch](../patches/39_bullfrog_layers.py) · [Renderer](bullfrog_layers.py)

A music video for **Anchor Rite — Bullfrog** (*Pinned*) about how Hydra combines layers. The
picture is built the way the record's cover photograph is: an out-of-focus ground, the burrs, and
silk threads catching the light, each its own chain. The piece adds one way of combining them at a
time, captions the code that does it, and twice takes the stack apart into four panels.

```sh
python docs/bullfrog_layers.py "/path/to/Bullfrog.flac" -o bullfrog_layers.mp4          # 1280x720, 24 fps
python docs/bullfrog_layers.py track.flac --stills 30,95,170,270 --contact sheet.png     # look first
python docs/bullfrog_layers.py track.flac -o part.mp4 --start 160 --seconds 30          # an excerpt
```

The renderer extracts the cover art embedded in the file and uses it as the photograph
(`--cover` for another image). The music and the cover are not in the repository. Without
`HYDRAPY_TRACK` the patch runs on LFOs, so `hydrapy studio patches/39_bullfrog_layers.py` or
`hydrapy live` can audition it.

| Time | Section | What it shows |
| --- | --- | --- |
| 0:00 | ground | `noise().blend(solid())` with `shape(64).repeat()` discs screened on top |
| 0:09 | + silk | `add`: `shape(2)` lines, glinting on the highs through a `noise().thresh()` |
| 0:47 | + burrs | `layer` + `luma`: `osc()` rays tapered by `thresh(gradient())` inside `polar()` (o2) |
| 0:59 | feedback | `src(o1).scale().blend(stack)` |
| 1:20 | exploded | ground, burrs, silk and the stack as four panels with their code |
| 1:48 | modulate | `ground.modulate(src(o2))`: the burrs refract what's behind them |
| 2:15 | mask | `photo.mask(src(o2))`: the photograph seen only through the burrs |
| 2:45 | blend modes | `diff`, `mult`, `overlay`, `screen` against the photograph, one per sixteen beats |
| 3:27 | marbling | `swirl()` and `tine()` inside the feedback loop, on the beat |
| 3:44 | physarum | `Physarum(food=o2)` screened in: the mould grows along the burrs |
| 4:20 | exploded | the new planes, and the display inside itself (`src(o0)` in o0) |
| 4:40 | everything | the whole stack |
| 5:13 | peel | the planes come off in reverse, down to the photograph |

The section times come from a novelty analysis of the track (band energies and stereo width over
eight-second windows) snapped to its beats at 115 bpm; they are written into the patch, so a
different track needs its own. The record drives it through `Beats.from_file` (the beat pulse and
bar phase) and Hydra-style `Audio` fft bands mapped to 0..1 per band.

Outputs: o2 is the burrs; o3 the stack (ground, burrs, silk, photograph, mould); o1 the blend
modes and the feedback; o0 the display, the exploded views and the captions. The captions are
drawn in Python into one `FrameSource`, so the whole overlay costs one texture.
