# HydraPy Studio

[Documentation](README.md) · [Project home](../README.md)

```sh
python -m pip install -e ".[gpu,video]"
hydrapy studio patches/studio_feedback.py
```

Studio opens a local browser workspace and renders the patch in Python. It works with
CPU-only operations, Python textures and the GL backend; it does not substitute an HTML
export for the running patch. With no GL context, `--backend auto` selects CPU. Shader
errors remain visible. `--backend cpu` and `--backend gl` select explicitly.

Run from your checkout to browse `patches/`, or use `--root /path/to/project`. The default
preview is 640×360 at a target 30 fps. Use `--size 960x540 --fps 24` to change it.
`--port 0` chooses a free port; `--no-browser` just prints the local URL. Keep the full
URL, including its session fragment. The server binds only to 127.0.0.1.

![HydraPy Studio workspace](studio/workspace.png)

## Work with a patch

- Search examples by technique or task. Discovery reads source metadata without running
  the examples. Previews are cached after you run a patch. The supplied Studio examples
  include preview images and explicit performance controls.
- Edit Python and **Save & run** (Ctrl/Cmd+Enter). External saves are watched too. Reopen
  reads the file again; a version check prevents overwriting a simultaneous external edit.
- **1–4** selects an output. **Space** pauses image updates, **S** saves a PNG, and **F**
  opens the preview fullscreen. Shortcuts do not fire while typing in a field.
- The status strip shows measured synthesis rate, resolution, backend, clock and hardware
  state. All four output thumbnails remain available. **Restart** clears feedback and
  restarts the patch at zero; a successful same-file edit carries feedback history and the
  named control settings over. Feedback carries over when the backend stays the same;
  stateful source internals such as NCA/Physarum restart.
- The browser interface contains the patch image, not an audio monitor. File-based audio
  analysis advances with the preview clock; hear the soundtrack in the exported video or
  through the patch's explicitly enabled audio routes.

Python remains executable code with your local permissions. Studio runs trusted patches;
its local HTTP token is an access boundary, not a Python sandbox. Only files within the
workspace root can be opened/saved through the interface. A user script's independent
file, process or network side effects cannot be undone by reload recovery.

## Declare performance controls

```python
from hydrapy import *

frequency = Param("Frequency", 12, min=2, max=60, step=.1, unit="cycles")
warp = Param("Distortion", .18, min=0, max=.6, step=.005,
             description="How strongly the noise bends the source.")
osc(frequency, .08, .7).modulate(noise(3), warp).out(o0)
```

A `Param` is a signal, with the same arithmetic as `Lfo` and `Audio` signals. Its slider
bounds are suggestions: numeric entry may exceed them. `scale="log"` requires positive
bounds. `group=` groups controls; names must be unique within a patch.

The inspector can add a sine LFO (rate in Hz) or beat envelope (rate in cycles per beat).
Depth uses the parameter's units. For custom modulation, use `warp.bind(signal, depth)`
in Python. Presets retain base values and modulation settings under `.hydrapy/presets/`.
Changing a base value does not stop its modulation. **Reset base values** deliberately
keeps the modulation configuration.

Construction arguments of signals such as `Lfo` are still literal numbers. For a named
frequency control, `(Time() * motion * 6.283185).sin()` is a signal expression; changing
its frequency changes its phase relationship to the clock. Live Param base values are
captured in standalone HTML exports, while their declared modulation continues.

The operator `Input` metadata also accepts `minimum`, `maximum`, `unit` and `scale` for
custom inspector tooling; these are hints and do not clamp operator evaluation.

## Connections and reload

The Connections panel displays managed CV/OSC/MIDI/Link/video resources, channels, signal
routes and errors. List available devices, then set device names and routes in the patch.
**Enable connections** validates and reloads the patch before opening them. **Disconnect**
reloads into preview mode. Pausing the image does not disconnect hardware.

A candidate's device opens are deferred until the patch loads and its renderer compiles.
Syntax, patch-construction and shader errors retain the running patch and its devices.
A failed device activation attempts to reconnect the previous devices and reports any
failure to do so. Successful reloads briefly reopen hardware; they are not gapless audio
switches. Feedback history is copied; arbitrary external state is not.

What you did in the session survives a reload unless the edit changed it. A control keeps its
slider value if its default in the code is unchanged, and keeps its modulation (none/LFO/beat, or
the depth of a declared `bind()`) if its `bind()` is unchanged; add, remove or change either in
the code and the code wins. The output you chose stays on screen unless the edit changed the
patch's `render()`. "Declared in patch" stays in a control's modulation menu, so a declared bind
can be chosen again after switching to an LFO.

Recording via a patch's own `VideoOut` or file-mode `CvOut` follows that object's lifecycle.
A successful reload closes the previous recording and creates the replacement. Use
unique recording destinations to avoid overwriting previous takes. Studio's separate
**Record video** action blocks patch reloads until recording is stopped.

## Record and export

**Record video** captures the running preview at wall-clock speed, repeating held frames
when rendering falls behind. It records video only. Stop to finish and download the MP4.
For a finished piece with music, use an offline MP4 export and specify an audio file
inside the project, for example `media/track.flac`. Original music and local exports are
ignored by Git. A declared `Soundtrack` is used when no external audio file is selected.

Offline exports use the last successfully running source and a snapshot of its current
controls. They run in a separate process, starting at zero with fresh feedback; live
hardware inputs are disconnected. PNG exports evaluate at the current preview time
with fresh feedback; use **Snapshot** for the exact current image. Export jobs expose
progress, the actual selected encoder, errors and a download button. Exports are saved
under `.hydrapy/exports/`; closing Studio terminates unfinished export jobs.

The compatibility check reports CPU-only operators, Python textures that require baking,
live values that become fixed, sampled Python loops, and browser requirements. This checks
the patch structure, not whether a future browser has the required GPU extensions.

```sh
hydrapy check patches/studio_feedback.py --target html
hydrapy check patches/37_beats_and_flow.py --target html --json
hydrapy ops marble
hydrapy ops --kind texture
```

`check` stages managed devices without opening them. Arbitrary patch code still runs.
MP4 encoding tries available hardware before software; `--gpu`/GL rendering and video
encoding are independent choices. The inspector reports the encoder actually used.

## Example metadata

Patches can declare a literal `STUDIO` dictionary. The browser reads it with `ast.literal_eval`:

```python
STUDIO = {
    "title": "Feedback instrument",
    "description": "A slow feedback field with five performance controls.",
    "tags": ["Feedback", "Sound & rhythm"],
    "preview": "docs/studio/feedback.jpg",
}
```

Without metadata, the browser derives a title, description and task tags from the source.
No node graph or separate project language is required.

For a complete audio-reactive piece with seven scenes and moving crossfades, see
the [Bullfrog demo](bullfrog-demo.md).
