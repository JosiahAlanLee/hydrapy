# Pixel clocks

`pixelOsc` is an independent-oscillator source: each cell has a fixed frequency
and phase when its controls are constant. The binary image is calculated directly
from coordinates and time. Apparent moving bands, ripples, and large structures
arise from the spatial phase/frequency fields and their sampling on a pixel grid.

```python
from hydrapy import *

pixelOsc(rate=.04, spread=1.4, density=85, pixel=4,
         geometry=3, phase=.137, threshold=0).out(o0)
```

The source runs on NumPy, OpenGL and the standalone WebGL export. It compiles
into one fragment pass, with no texture samples, neighbor reads or simulation
history. Seeking directly to a time produces the same state as playing to it.
It does not need compute shaders.

## Controls

| Input | Default | Meaning |
| --- | --- | --- |
| `rate` | 0.035 | Shared base frequency, in cycles per second |
| `spread` | 1.6 | Coefficient of the spatial frequency field, in Hz |
| `density` | 90 | Coefficient of the spatial phase field, in cycles |
| `pixel` | 4 | Cell pitch in output pixels; floored and bounded below by 1 |
| `geometry` | 0 | 0 saddle, 1 radial, 2 folded, 3 lattice |
| `phase` | 0.137 | Shared phase offset, in cycles |
| `threshold` | 0 | Compare the sine with this value; normally -1 to 1 |

The frequency at a location is `rate + spread * field(x,y)`. Spread is a
coefficient, not a guaranteed frequency range; the range also depends on the
geometry and aspect ratio. Negative local rates run clocks backwards. Values
outside the normal sine threshold range produce a solid image.

The four pairs of phase/frequency fields are original analytic constructions
using products, radial distance, folds, and a stationary sinusoidal deformation.
Only the clock term changes with time. Each cell evaluates
`sin(2*pi*(density*phaseField + phase + time*(rate + spread*frequencyField)))`.
The result is thresholded to exactly black or white before downstream coloring.

`rate`, `spread`, `density`, `phase` and `threshold` also accept HydraPy signals
or chains. Changing these while running deliberately leaves the fixed-clock
model; a frequency change at a nonzero time can produce a phase jump. The
autonomous demo holds every setting constant throughout.

## Studio and live browser player

```bash
hydrapy studio patches/pixel_oscillators.py --backend gl --size 1920x1080
```

The patch exposes the controls in Studio and uses an off-white / near-black
palette. `make_patch(preset="lattice")` selects one of four examples: `saddle`,
`radial`, `folded`, or `lattice`.

The study renderer can also produce a self-contained HTML player with these
presets, six sliders, pause, restart, time scrubbing, fullscreen and PNG export.
It embeds the registered operator's GLSL body rather than maintaining a second
copy of its equations. WebGL2 is required. The canvas uses the window's native
device-pixel dimensions; resizing can change the aliasing pattern. Saving a PNG
captures only the artwork, without the controls.

## Render a study

```bash
python docs/pixel_oscillator_study.py \
  --out Pixel_Clocks.mp4 --seconds 90 --preset lattice \
  --audio /path/to/track.flac --audio-start 112 \
  --html Pixel_Clocks.html
```

The output is 1920×1080 at 30 fps by default. `--size 3840x2160` changes the
render size; `--fps 60` changes temporal sampling. Hardware encoding is attempted
automatically, falling back to software. Omit `--audio` for a silent study.
With audio, the specified excerpt is muxed as AAC at 256 kbit/s. The recording
does not modulate the visuals. No recording or rendered movie is stored in git.
A JSON sidecar records the controls, dimensions and timing.

To write only the live player from Python:

```python
import sys
from pathlib import Path
sys.path.insert(0, "docs")
from pixel_oscillator_study import live_html
Path("Pixel_Clocks.html").write_text(live_html("lattice"))
```

## Reference and limits

This implementation is inspired by [Memo Akten's description of
PixlFlippr](https://www.memo.tv/projects/2021/pixlflippr/). Its spatial fields
and compositions are new; no edition code or assets were copied. It does not
reproduce PixlFlippr's specific outputs or enforce its seamless loop durations.

These oscillators are uncoupled. Their relative phases drift, but they do not
change their rates in response to one another or synchronize through a shared
mechanical support. Aliasing and moiré are intentional; resizing, smoothing,
compression and temporal sampling can materially change the appearance. The
live player avoids video compression. Hard thresholds can amplify small
floating-point differences between CPU and GPU into occasional flipped cells,
especially with dense fields or long times.

Validation covers cell independence, binary output, cell size, fixed-frequency
periodicity, CPU/GPU agreement, direct seeking, Python round-tripping, signal
inputs and standard HTML export:

```bash
python -m pytest tests/test_pixel_oscillators.py -q
```
