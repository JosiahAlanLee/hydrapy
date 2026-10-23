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
| [Live control and modular audio](live-io.md) | CV in/out, the painted oscillator, probes, OSC, MIDI and Ableton Link |
| [Rendering and export](rendering-and-export.md) | NumPy/GLSL backends, hardware encoding, frame holds, alpha, streaming and audio export |
| [Demoscene tools](DEMOSCENE.md) | Materials, metaballs, spline wireframes, 3D fields, fractal folds and musical timelines |
| [Animated thi.ng adaptations](TOXI_ANIMATION_SOURCES.md) | Cellular rules, Voronoi fields, curves, vegetation, export limits and source provenance |

## Reference and development

- [Operator reference](operators.md): every registered operator, argument and default.
- [Development](development.md): adding operators, source layout, testing and regenerating the reference.
- [Audio timing](audio-timing.md): sample-clock behavior, long-duration checks and hardware-testing limits.

## Watch and reproduce

| Preview | Reproduction script |
| --- | --- |
| [Feature tour](tour.mp4) · [contact sheet](tour_contact.png) | `python docs/tour.py` |
| [Demoscene preview](demoscene_preview.mp4) · [contact sheet](demoscene_preview.png) | `python docs/demoscene_demo.py --out demo.mp4 --contact demo.png` |
| [Drawn soundtrack](ornament_sound.mp4) | Render [patch 25](../patches/25_ornament_sound.py) with audio |
| Animated cellular and geometry sources | `python docs/generative_demo.py --out generative_demo.mp4` |

See [rendering and export](rendering-and-export.md) for backend and encoder requirements.
