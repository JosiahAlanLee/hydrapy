# hydrapy

[Hydra](https://hydra.ojack.xyz), Olivia Jack’s video synth, in Python — with
modulation signals, feedback, generative geometry, painted film and drawn sound.
Render with NumPy or GLSL, export a standalone WebGL page, or connect a modular synth.
It also includes **Gorgon**, a CV and audio/video tracker in development, which works with VCV Rack.

[Documentation](docs/README.md) · [Operator reference](docs/operators.md) · [Example patches](patches/) · [Gorgon tracker](#gorgon-a-cv-and-av-tracker)

[![Animated tour of hydrapy effects and example patches](docs/tour.gif)](docs/tour.mp4)

[Watch the feature tour](docs/tour.mp4) · [Demoscene preview](docs/demoscene_preview.mp4)

## Quick start

Python 3.9+:

```sh
git clone https://github.com/JosiahAlanLee/hydrapy.git
cd hydrapy
python -m pip install -e .
```

Save this as `patch.py`:

```python
from hydrapy import *

osc(10, 0.1, 0.8).rotate(Lfo(0.1, lo=0, hi=6.28)).kaleid(5).out(o0)
render(o0)
```

Export a page to open in your browser, or render a GIF:

```sh
hydrapy html patch.py -o patch.html
hydrapy render patch.py -o out.gif --seconds 4 --size 320x180
```

For a live window and GPU/video rendering, install the optional extras:

```sh
python -m pip install -e ".[live,video]"
hydrapy live patch.py
hydrapy render patch.py -o out.mp4 --gpu --seconds 10
```

Open the local workspace with an editor, output previews and performance controls:

```sh
hydrapy studio patches/studio_feedback.py
```

[Studio guide](docs/studio.md) · [Installation and walkthrough](docs/getting-started.md) ·
[Rendering and hardware encoding](docs/rendering-and-export.md)

## Gorgon: a CV and A/V tracker

Gorgon is a tracker inside Studio for sequencing a picture and a modular synth
from one score. Four 64-row patterns hold note lanes and automation lanes.
Python owns the playhead, and every output follows it: the patch's parameters,
the browser audition synth, hardware control voltage and VCV Rack. Gorgon is
the master clock and doesn't follow an external one.

- **Visuals:** automation lanes are bound to a patch's named controls, with
  Program and Auxiliary outputs for separate monitors.
- **CV out:** clock, reset, run, gates, held 1 V/oct pitch, velocity and
  automation are generated per sample in the audio callback. They go out
  through a DC-coupled interface such as an Expert Sleepers ES-3, or through an
  ES-40 with ESX-8GT/ESX-8CV expanders over encoded S/PDIF. PortAudio and native
  JACK/PipeWire are both supported.
- **VCV Rack:** the [Gorgon module](https://github.com/JosiahAlanLee/Gorgon/tree/main/rack)
  follows the playhead on Rack's sample clock. It puts the score on Rack jacks
  and sends CV back into the patch's controls and transport.
- **Audio:** an optical soundtrack read from the picture and a synthesized
  breaks kit are published as JACK/PipeWire ports for Rack to capture.

```sh
hydrapy studio patches/gorgon_corridor.py --tracker
hydrapy studio patches/gorgon_corridor.py --tracker --cv-device "Digi 003" --cv-map 8
```

**Status:** in development. The software is tested in CI, including Chromium,
real JACK and PipeWire daemons, and the Rack module end to end. Physical
interfaces, the ES-40 digital path and end-to-end A/V latency still need
validating on the workstation.

The tracker runs here, in Studio. The project's centre is the
[Gorgon repository](https://github.com/JosiahAlanLee/Gorgon): the Rack module,
the [contract](https://github.com/JosiahAlanLee/Gorgon/tree/main/spec) every
piece implements (score format, bridge API, signal semantics) and the
integration tests. A change to any of those starts there.

[Tracker guide](docs/gorgon-tracker.md) · [Editor walkthrough](docs/gorgon-editor-walkthrough.md) ·
[JACK/PipeWire audio](docs/gorgon-audio.md) · [ES-40 setup](docs/es40.md)

## Explore

| Make | Start with |
| --- | --- |
| Feedback and evolving patterns | [Reaction–diffusion](patches/03_feedback_rd.py) · [CV on every parameter](patches/02_cv_everywhere.py) |
| Painted film and print textures | [Hand-painted film](patches/11_handpainted.py) · [Charcoal animation](patches/23_studie.py) · [Material controls](patches/27_material_controls.py) |
| A picture that plays the modular | [Painted oscillator](patches/26_painted_oscillator.py) · [Live I/O guide](docs/live-io.md) |
| Animated cellular and distance fields | [Cellular](patches/28_cellular.py) · [Morphing fields](patches/30_distance_morph.py) · [thi.ng adaptations](docs/TOXI_ANIMATION_SOURCES.md) |
| Lit surfaces and procedural 3D | [Materials](patches/demoscene_materials.py) · [Fractal folds](patches/demoscene_fractal.py) · [Demoscene guide](docs/DEMOSCENE.md) |
| Linework and plotter geometry | [Text and SVGs](patches/14_text_and_lines.py) · [Spline wireframes and SVG export](docs/DEMOSCENE.md#splines-and-wireframes) · [Plotter SVGs](patches/34_plotter.py) |
| Textures that grow, heal and spread | [Neural cellular automata](patches/32_nca.py) · [Physarum](patches/36_physarum.py) · [Marbling](patches/35_marbling.py) · [Guide](docs/ecosystem.md) |
| An oscilloscope, laser or the beat of a record | [XY beam out](patches/33_vector_out.py) · [Beats and optical flow](patches/37_beats_and_flow.py) · [Guide](docs/ecosystem.md#vectors-out-the-oscilloscope-the-laser-the-plotter) |
| fxhash editions | [A patch that varies per mint](patches/38_fxhash.py) · [Guide](docs/rendering-and-export.md#fxhash-one-patch-every-mint) |

Browse [all patches](patches/) or the [documentation index](docs/README.md) for
film recipes, signals, timing, audio and export details.

## Reference and development

- [Operator reference](docs/operators.md) — arguments, defaults and backend support.
- [Hydra compatibility](docs/getting-started.md#whats-different-from-hydra) — syntax and semantic differences.
- [Development guide](docs/development.md) — source layout, tests and adding an operator.

## License

[AGPL-3.0-or-later](LICENSE), following Hydra. Adapted thi.ng components retain
their Apache-2.0 notices: [NOTICE](NOTICE) · [upstream license](LICENSE-THI-NG-APACHE-2.0.txt).
