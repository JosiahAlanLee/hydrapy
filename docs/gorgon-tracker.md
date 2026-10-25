# Gorgon tracker: first connected workspace

[Studio](studio.md) · [Performance sessions](performance-sessions.md) · [Editor walkthrough and video capture guide](gorgon-editor-walkthrough.md)

For JACK/PipeWire audio into Rack and the ES-40 polarity selector, see
[Gorgon audio routing](gorgon-audio.md).

Gorgon connects the fullscreen tracker study's video and three automation lanes
to a running HydraPy patch. It is an optional, local Studio workspace, using the
same session token and loopback boundary. **Patch editor & export** opens the
existing editor, connections, recording, snapshots and export tools.

From a checkout with HydraPy installed (`pip install -e ".[gpu,video]"`):

```sh
hydrapy studio patches/gorgon_corridor.py --tracker --backend gl --size 1280x720 --fps 60
# Smaller CPU alternative while a model trains:
hydrapy studio patches/gorgon_corridor.py --tracker --backend cpu --size 320x180 --fps 20
```

Studio also has a **Gorgon tracker** link. Click the visual source card to load a
workspace patch, then bind its named `Param` controls. Loading patch code keeps
Studio's existing execution and hardware staging rules.

## Control surface and displays

- **Space** starts automation or pauses; **F** enters fullscreen; **B** blacks out
  Gorgon's browser monitors. Blackout does not alter rendering, recording,
  snapshots, exports, or an external video device.
- Open **Program** and **Auxiliary**, move each window to its monitor, then press
  **F**. Routing selects `o0`–`o3` independently. Program also selects Studio's
  displayed/recorded output; Auxiliary never changes that selection.
- Four patterns each contain 64 sixteenth-note rows. Select audio cells and use
  the Note / Velocity editor directly below the grid; Enter toggles browser synth
  hits and retains keyboard focus. Piano keys adds QWERTY note entry, octave and
  row advance controls. Shift+click, drag or Shift+arrows select blocks; Copy, Cut,
  Paste, transpose, block velocity and 100-step Undo/Redo edit them. Pattern tools
  renames, copies and clears patterns. Arrangement cards show real event activity.
  Selecting cells or patterns does not seek; Seek to row, Alt+Enter in the grid,
  a double-click on a row number or a card’s ▶ button does. Manual editing turns
  Follow off. Select visual cells, then edit the value, curve or slider.
  Hold changes at the exact key row; Linear and Smooth interpolate between keys.
- Bind Dolly Z, Roll and Feedback to named patch parameters. Missing bindings are
  marked in the selectors; empty bindings disconnect. Lane units are 0–1, degrees,
  and 0–0.9; there is no hidden normalization to a parameter's range. Declared
  modulation continues to add to the automated base value.
- File actions stay available at narrow widths; View provides larger text, a
  collapsible instrument palette and an automation-curve toggle. A failed edit
  preserves a downloadable local recovery copy; Retry checks whether the server
  accepted a reply that was lost. A real revision conflict requires an explicit
  choice of the server score.
- **Save session** writes `.hydrapy/tracker.json` and downloads JSON.
  **Restore saved score** or **Import score** reloads it explicitly. Browser reload
  reconnects to the server score. Edits are validated and revision-checked;
  concurrent edits cannot silently overwrite a newer score. Switching Studio's
  displayed output is not an edit: an open tracker keeps working, and the new
  program output is kept. Restarting Studio
  starts with the initial study until Restore is chosen. Transport and blackout
  are not saved. Unpersisted edits disappear when Studio exits.

The included patch maps Dolly Z to **2D scale**, Roll to rotation and Feedback to
real recursive image memory. This is not Deforum depth reprojection. The left
palette is explicitly **synth demo** audio generated in the control browser.
Archive samples, Stable Audio, RAVE, Pd and model servers are not connected.
Sound starts muted and requires a user gesture.

## Timing, feedback and exports

Python owns the playhead. The browser sends scores after edits (coalesced, with
one edit request in flight), plus explicit transport commands. Automation is
evaluated immediately before existing renderer steps; browser animation timers
do not send per-frame slider updates.

Tracker position is separate from simulation time. Seeking, looping and Stop
move musical position without clearing feedback, replaying hardware hooks or
seeking a Soundtrack. Pause freezes both; disabling automation leaves parameters
at their last base values and does not change renderer pause. Resume rendering
to see automation edits in a paused image. Tempo changes preserve tracker position
and do not rewrite a patch's `bpm`, `speed` or simulation frame rate. The patch can
therefore have an independent internal rhythm. An armed external clock and enabled
tracker automation are mutually exclusive; other connections keep Studio's usual
explicit arming behavior.

With automation enabled, Studio MP4/GIF exports evaluate the visual score from
its beginning (or the selected loop pattern), with fresh feedback history. PNG
export evaluates the current score beat at the current simulation time with fresh
history, matching the existing PNG export model. **Snapshot** saves the actual
current framebuffer. Browser synth audio is not captured; existing audio-file
and Soundtrack export behavior still applies. HTML export requires automation
disabled and retains its static-control behavior.

GPU export normally renders one frame ahead of its readback iterator. Tracker
exports use explicit per-frame render/read steps so keys cannot slip relative to
feedback or probe updates. Other exports keep the original pipeline. Neither
renderer's clock implementation changes.

Closing the controller stops its synth; Python rendering and automation continue,
and output windows connect directly to Studio. Hidden viewers stop requesting
preview frames. The browser synth uses a bounded lookahead scheduler and an
estimated server playhead, not a native sample clock. Browser throttling and
physical audio/video latency still require workstation testing.

The browser keeps its scheduling counter continuous through pattern and song
wraps. Explicit seeks cancel lookahead, including small seeks. Browser, direct
CV and the VCV bridge use the same pre-render beat timestamp: a slow frame does
not add its render time to the browser's clock offset.

## CV out: Gorgon as the master clock

`--cv-device` sends the score out of an audio interface as control voltage,
for example ADAT into an Expert Sleepers ES-3 (DC coupled, ±10 V full scale):

```sh
hydrapy devices                     # find the interface and its channel numbers
hydrapy studio patches/gorgon_corridor.py --tracker --cv-device "Digi 003" --cv-map 8
```

`--cv-map 8` puts the default eight signals on channels 8–15, which should be the
003's ADAT outputs; check the order `hydrapy devices` and your audio system report.
Channel numbers count from 0, as `CvOut` does. Choose signals explicitly with
`--cv-map 8=clock,9=reset,10=gate1,...`.

| Signal | Voltage |
| --- | --- |
| `clock` | 5 V pulses, `--cv-ppqn` per quarter note (default 4: sixteenths), 50% duty |
| `reset` | 5 ms, 5 V on start/resume/seek and at the score or pattern-loop boundary |
| `run` | 5 V while the tracker is playing |
| `gate1`–`gate4` | 5 V for half a row on each audio-lane hit; muted lanes send nothing |
| `pitch1`–`pitch4` | held 1 V/oct pitch; C4 = 0 V by default, follows the key selector |
| `vel1`–`vel4` | last hit's velocity on that lane, 0–5 V, held until the next hit |
| `z`, `feedback` | lane minimum to maximum as 0–5 V |
| `roll` | −45° to 45° as −5 V to 5 V |

The default map is `clock, reset, gate1–gate4, z, roll`, because the ES-3 has
eight outputs; `feedback`, `run` and velocities are opt-in.

Every sample is evaluated in the audio callback from the render thread's latest
playhead, so clock, gates and held key steps land on the exact sample rather than
on a video frame. Pausing or disabling automation stops clock, reset and gates;
lanes hold their value. The CV and the picture share one playhead, so the modular
follows Gorgon's tempo, seeks and loops. Gorgon does not follow an incoming clock.

`--cv-offset-ms` sends the voltages earlier (positive) or later (negative) than
the picture. Set it after the click-and-flash comparison below, using a gate into
a percussion voice as the click.

Notes now accept C0–G9 (sharps), including tracker notation such as `C-4`.
The FM and Sub browser voices play these notes; FM uses a harmonic 2:1 ratio.
Audition previews the selected event. `AIR` is unpitched and leaves pitch CV held;
velocity zero emits no gate or sound. The default score is in D: changing the
key transposes notes by `key - D` in both the pitched browser voices and CV.
`--cv-pitch-zero 48` makes C3 the zero-volt reference instead of C4. Pitch CV holds
between notes and during pause/mute; a seek/start clears its held state before
the next note. Hardware CV does not depend on the browser's Sound toggle.

Startup reset is latched for its full duration at the first available DAC block,
even when the DAC is scheduled ahead. This makes resets reliable but does not
remove the hardware's command latency. Regular clock phase follows the playhead.
Pauses and score edits reach CV when the server handles the command; signals
already queued in the device cannot be recalled. Underflows and callback errors
appear in Output routing; hover for device routes. A stopped/failed render thread
publishes a stopped transport. Normal shutdown sends zero-volt commands before
closing streams, including encoded ESX-8GT gates and ESX-8CV outputs.

See [ES-40 + ESX-8GT / ESX-8CV setup](es40.md) for gate/CV profiles, encoded diagnostics,
and VCV troubleshooting. A profile can combine direct ADAT CV with encoded
S/PDIF pairs in one stream on the same interface. Keep a patch's own `CvOut` on
a separate device: uncoordinated writers can mix or fail to open, depending on
the driver. CV is not included in video/audio exports.

## VCV Rack bridge

The [Gorgon module](https://github.com/JosiahAlanLee/DSP_Emulation/tree/main/vcv/gorgon)
for VCV Rack 2 links Rack to Studio with no configuration. On start Studio writes
`~/.hydrapy/studio.json` (port, session token, root; readable only by you) and
removes it on exit if it is still its own. Set `HYDRAPY_STUDIO_FILE` to use another
path. The module reads that file and polls Studio on loopback.

GorgonBridge 2.0.1 puts the score's clock, gates, held velocities and automation
on Rack jacks, with gates and unipolar lanes at 10 V. Its polyphonic **V/OCT** jack
carries four held pitches (1 V/oct, C4 = 0 V by default; C3 is selectable). Notes,
key transposition, AIR, velocity zero and mute follow direct CV semantics. Reset
is latched on start/resume/seek so a late network update cannot skip the pulse.
The module generates signals per sample from Studio's playhead. It also
sends values back: four CV inputs set the patch's named controls (0–10 V spans
each control's range), and PLAY and RESET drive Studio's transport.

The endpoints are local and need the session token like every other route:

- `GET /api/bridge?revision=N`: the playhead as an anchor (`anchor_beat`,
  `anchor_time`, `clock_time` and `wall`, Studio's monotonic time when it was
  sampled), `enabled`, `playing`, `transport_revision`, `server_now`, the patch's controls, and the score
  `document`. The document is left out when `N` matches the current revision.
- `POST /api/bridge_params` with `{"values": {"Control name": value}}`: sets named
  controls, clamped to their slider range. Controls bound to a tracker lane are
  left alone while automation is on; the reply lists what was applied and what
  was ignored, and why.
- Transport uses the existing `POST /api/tracker_transport`.

`transport_revision` changes on an explicit seek or automation enable change,
independently of the score revision. Ordinary score edits do not restart a
follower's reset pulse or discard held notes. The browser endpoint and transport
replies also include it. The Gorgon integration workflow runs real Chromium
tests and connects the VCV bridge from DSP_Emulation's `main` to this Studio.

## Preview and measurements

Program reuses Studio's bounded preview. Auxiliary has an additional encoder
worker, one pending frame and one latest encoded frame. Readback stays on the
render thread; JPEG and socket writes do not. Buses share readback when both ask
for the selected output in the same iteration. Each viewer requests its next
frame only after decoding/display scheduling. Inactive thumbnails retain their
staggered lower refresh rate. Closed viewers stop demanding captures.

`window.gorgonPreview` exposes up to 300 timing samples per browser: server render,
readback and JPEG durations, plus browser request, body transfer, decode and
display-scheduling durations. Program includes optional GPU timing when enabled
in Studio's preview details. Frame-age estimates exclude first-byte transit;
display scheduling measures the handoff to the canvas animation loop, not physical
scanout or audio latency. The buses have independent sequence cursors.

## Reproducible checks

```sh
python -m pytest -q tests/test_studio_tracker.py tests/test_tracker_cv.py tests/test_es40.py tests/test_studio_bridge.py tests/test_studio.py tests/test_studio_preview.py
python -m pip install playwright
python -m playwright install chromium
HYDRAPY_BROWSER_TESTS=1 python -m pytest -q tests/test_studio_tracker_browser.py tests/test_studio_preview_browser.py
```

Tests cover exact hold boundaries, interpolation, tempo continuity, loop/seek
semantics, validation, revision rejection, persistence, authenticated routes,
independent outputs, and CPU/GL export equivalence with feedback and a held
simulation rate. Existing expensive four-output feedback/audio checks also
exercise Auxiliary demand. Browser tests edit notes with the keyboard, preserve focus and playback position,
copy and transpose blocks, undo pattern operations/imports, exercise narrow
layouts and recover interrupted or rejected edits. They also save, import and reload scores;
open both outputs; exercise blackout; close the controller; and slow decoding
to verify one outstanding preview request per viewer.

Automated GL checks use Mesa llvmpipe and synthetic sources. They do not validate
the Ubuntu physical GPU, multiple monitors, fullscreen permissions, audio-device
latency, ADAT/CV hardware, or model inference contention. On the workstation, test
both outputs at the intended size/fps during training, inspect each timing stage,
and compare an audible click with a visible flash before setting A/V compensation.
