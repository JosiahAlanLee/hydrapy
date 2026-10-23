# Getting started

[Documentation](README.md) · [Project home](../README.md)

## Install

Python 3.9+. Run these commands from the directory where you want the checkout:

```sh
git clone https://github.com/JosiahAlanLee/hydrapy.git
cd hydrapy
python -m pip install -e .
```

The core install includes NumPy, Pillow and imageio for CPU rendering, PNG/GIF output
and standalone HTML export. Run the examples and commands below from the repository root.

Add only the integrations you need:

| Feature | Install from the checkout |
| --- | --- |
| GPU rendering, including headless EGL on Linux | `python -m pip install -e ".[gpu]"` |
| Live window (ModernGL + GLFW) | `python -m pip install -e ".[live]"` |
| MP4 export and compressed audio decoding | `python -m pip install -e ".[video]"` |
| Desktop webcam input | `python -m pip install -e ".[camera]"` |
| CV in/out through a DC-coupled audio interface | `python -m pip install -e ".[cv]"` |
| OSC | `python -m pip install -e ".[osc]"` |
| MIDI | `python -m pip install -e ".[midi]"` |
| Ableton Link | `python -m pip install -e ".[link]"` |

Combine extras, for example `python -m pip install -e ".[live,video]"`,
or install all integrations with `python -m pip install -e ".[all]"`.
The HTML export uses the browser's camera API without OpenCV.

## The sixty-second tour

```python
from hydrapy import *

# 1. Hydra, as you know it. camelCase and snake_case both work.
osc(20, 0.1, 1).modulate(noise(3), 0.3).kaleid(4).out(o0)

# 2. Arrays are Seq, functions are Fn (or just a lambda), bpm is set_bpm.
set_bpm(120)
osc(Seq([10, 20, 40]).fast(2).smooth(), 0.1, Fn(lambda ctx: ctx.time % 1)).out(o1)

# 3. Signals: LFOs, clocks, envelopes, sample & hold, random drift — with arithmetic.
env = Env(Clock(1), attack=0.02, decay=0.5)          # retriggered every beat
osc(Lfo(0.1, lo=5, hi=40), 0.1, SH(4) * 2).scale(1 + env * 0.5).out(o2)

# 4. Per-pixel CV: any float parameter accepts a chain. Hydra can't do this.
osc(noise(2).r() * 30 + 5).rotate(fbm(3) * 6.28).out(o3)

# 5. Feedback ops that need neighbours: Life and reaction–diffusion.
src(o0).rd(feed=0.037, kill=0.06, iterations=8).add(shape(6, 0.05).color(0, 1, 0), 0.1).out(o0)

# 6. Chains combine with operators.
(osc(30) * noise(4) + shape(3).luma() | rings(8) ^ gradient()).out(o1)

render(o0)
```

Render it:

```
hydrapy render patch.py -o out.gif --seconds 6 --fps 30 --size 320x180 --warmup 60
hydrapy render patch.py -o out.mp4 --seconds 60 --size 1280x720 --gpu
hydrapy html   patch.py -o patch.html          # standalone WebGL page (keys: s = save png, f = fullscreen, space = pause)
hydrapy live   patch.py                        # window, needs moderngl + glfw
hydrapy ansi   patch.py                        # in the terminal, half-block ANSI
hydrapy hydra  patch.py --loose                # Hydra JS (drops the ops Hydra lacks)
hydrapy glsl   patch.py                        # the fragment shaders, pass by pass
hydrapy import sketch.js -o patch.py           # Hydra JS -> Python
hydrapy ops                                    # list every op
```

`hydrapy render sketch.js` works too — Hydra sketches are imported on the fly.

## Python API

```python
from hydrapy import *

osc(10, 0.1, 0.8).rotate(Lfo(0.1, lo=0, hi=6.28)).kaleid(5).out(o0)
src(o0).blur(2).edge().palette(0, 0.33, 0.67, Lfo(0.05)).out(o1)
render(o1)

save_gif(render_frames(seconds=4), "out.gif")        # CPU, anywhere
to_html(path="out.html")                              # live WebGL page, no server needed
print(patch_to_hydra(strict=False))                   # paste into hydra.ojack.xyz
```

## What's different from Hydra

| Hydra | hydrapy |
|---|---|
| `osc(10).out(o0)` | same |
| `[1, 2, 3].fast(2).smooth()` | `Seq([1, 2, 3]).fast(2).smooth()` — or just pass a list; identical timing (bpm/60 steps per second, default bpm 30) |
| `() => time * 0.1`, `({time}) => ...` | `Fn(lambda ctx: ctx.time * 0.1)` or a bare lambda; `ctx` has `time, frame, bpm, width, height, mouse` |
| `bpm = 60`, `speed = 2` | `set_bpm(60)`, `speed(2)` |
| `s0.initCam()`, `s0.initImage(url)` | `src(Camera())`, `src(Image("file.png"))`, `src(Video("clip.mp4"))` |
| `a.fft[0]`, `a.setSmooth()` | `a = Audio.from_file("track.wav")` (offline, frame-accurate) or `Audio()` (microphone in the HTML export); `a.fft[i]`, `a.vol`, `a.flux`, `a.set_smooth()`… |
| `render()` shows all four outputs | `render(oN)` shows one; default `o0` |
| `.r()`, `.g()`, `.b()`, `.a()` | same (they return the channel as grey) |
| Hydra's 12 extra-ish GLSL helpers (`_luminance` etc.) | same names, same numbers |

Semantics that are reproduced on purpose: coordinates run 0–1 with y up; coordinate ops apply to the
*coordinate* so later ones act first (`osc().rotate(a).scale(b)` → `osc(rotate(scale(st, b), a))`);
`src(oN)` inside `oN`'s own chain reads the previous frame, inside a later output it reads the current
one; `shape` and `kaleid` keep Hydra's `3.1416`; `color()` with negative values inverts; the
`luminance` weights are Hydra's (0.2125, 0.7154, 0.0721).

What Hydra has that this doesn't (yet): `initScreen`, `setFunction` (write ops in Python instead,
see [Writing an op](development.md#writing-an-op)), the multi-output `render()` grid, `.setResolution` mid-run, and the `p5`/`hydra-synth`
extensions. Hydra's `sum()` returned a float where a vec4 was expected; here it returns grey.
