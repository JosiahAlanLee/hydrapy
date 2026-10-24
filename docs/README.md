# Documentation

[Project home](../README.md) · [Example patches](../patches/) · [Video tour](tour.mp4)

Start with [installation and the first patch](getting-started.md), then choose a
guide for the work you want to make. Run commands from the repository root.

## Guides

| Guide | Contents |
| --- | --- |
| [Getting started](getting-started.md) | Installation, optional dependencies, Python and CLI examples, Hydra compatibility |
| [Signals, feedback and geometry](patching.md) | LFOs, sequences, per-pixel controls, simulations, animated sources, text, SVGs and delay lines |
| [Film, print and drawn sound](film-and-sound.md) | Painted film, image quilting, VHS, Super 8, charcoal strokes, cut paper and optical soundtracks |
| [Grown textures, vectors and listening](ecosystem.md) | Neural cellular automata, physarum, marbling, oscilloscope/laser/plotter output, beat tracking and optical flow |
| [Live control and modular audio](live-io.md) | CV in/out, the painted oscillator, probes, OSC, MIDI, Ableton Link, hot reload and video out |
| [Rendering and export](rendering-and-export.md) | NumPy/GLSL backends, hardware encoding, frame holds, alpha, streaming and audio export, fxhash projects |
| [Demoscene tools](DEMOSCENE.md) | Materials, metaballs, spline wireframes, 3D fields, fractal folds and musical timelines |
| [Animated thi.ng adaptations](TOXI_ANIMATION_SOURCES.md) | Cellular rules, Voronoi fields, curves, vegetation, export limits and source provenance |
| [Pixel clocks](pixel-oscillators.md) | Independent pixel oscillators, two-color moiré, Studio controls and a live browser study |
| [Bell Riots video](bell-riots-video.md) | Full-track edit combining rule fields, pixel clocks, thi.ng cellular patterns and entropy locking |
| [Bell Riots: Unfolding edit](bell-riots-sparse.md) | Closing six minutes with sparse two-layer composition and kick-triggered recursive folds |
| [Bell Riots: counterpoint mix / edit 4](bell-riots-counterpoint.md) | Mix the external Pd score and drive paired visuals from hats, cowbell and FM stabs |

## Reference and development

- [Operator reference](operators.md): every registered operator, argument and default.
- [Development](development.md): adding operators, source layout, testing and regenerating the reference.
- [Audio timing](audio-timing.md): sample-clock behavior, long-duration checks and hardware-testing limits.
- [Performance](PERFORMANCE.md): benchmarks, per-output resolution, `mipblur`, fast noise, phone pages, and what is skipped automatically.

## Watch and reproduce

| Preview | Reproduction script |
| --- | --- |
| [Feature tour](tour.mp4) · [contact sheet](tour_contact.png) | `python docs/tour.py` |
| [Demoscene preview](demoscene_preview.mp4) · [contact sheet](demoscene_preview.png) | `python docs/demoscene_demo.py --out demo.mp4 --contact demo.png` |
| [Drawn soundtrack](ornament_sound.mp4) | Render [patch 25](../patches/25_ornament_sound.py) with audio |
| Animated cellular and geometry sources | `python docs/generative_demo.py --out generative_demo.mp4` |
| [Bullfrog, in layers](bullfrog-layers.md): how Hydra layers, one technique at a time | `python docs/bullfrog_layers.py Bullfrog.flac -o bullfrog_layers.mp4` (your copy of the track) |

See [rendering and export](rendering-and-export.md) for backend and encoder requirements.

- [Studio workspace](studio.md) — editor, live output previews, performance controls, presets, connections and export checks.
- [Bullfrog demo](bullfrog-demo.md) — seven scenes, audio response, and moving crossfades.
- [Bullfrog, in layers](bullfrog-layers.md) — the same track as a lesson in layering: planes added one combinator at a time, captioned, and taken apart on screen.
