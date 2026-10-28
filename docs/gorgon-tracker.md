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

- **Audio / Visual** are two pages of the same pattern. Audio shows Breaks,
  FM hits, Sub and Texture; Visual shows Dolly Z, Roll, Feedback and Scene,
  with the automation curve and patch bindings. Click a page tab or press
  **Tab while the grid has focus**. The pattern and edit row stay in place,
  each page remembers its lane, and playback continues. **Shift+Tab** leaves
  the grid; Tab in a form keeps normal field navigation. Page choice is remembered
  in that browser tab and does not change the saved score or CV/Rack output.
- **Space** starts automation or pauses; **F** enters fullscreen; **B** blacks out
  Gorgon's browser monitors and the fullscreen output. Blackout does not alter
  rendering, recording, snapshots, exports, or an external video device.
- Open **Program** and **Auxiliary**, move each window to its monitor, then press
  **F**. Routing selects `o0`–`o3` independently. Program also selects Studio's
  displayed/recorded output; Auxiliary never changes that selection.
- The **Hydra monitor** has **o0–o3** buttons for inspecting each output locally;
  **[** and **]** cycle them when you are not typing in a field. **Program** returns
  to following the live feed. The dot marks the output currently on Program.
  This choice stays in the browser tab across reloads and Audio/Visual page changes.
  Use the existing Program and Auxiliary routing controls to change those feeds.
  Inspection uses small, requested previews and leaves playback, recording and the score alone.
- On Linux, **Display › Show Program fullscreen** (**Shift+F**) shows Program on a
  chosen monitor in Studio's own window, drawn from the texture it was rendered
  into at the display's refresh: no readback, JPEG or browser in between. Edits
  are sent at once and applied as they arrive, so an edit or cue is on screen at
  the next refresh (on the workstation's 60 Hz screen, about 15 ms typical and
  22 ms at most from the moment Studio receives it). While it shows, Studio renders
  at the monitor's size (the patch reloads in place; feedback restarts; the old
  size returns when it closes, or after recording stops if a take is in progress)
  and the editor's Program monitor becomes a 480×270
  preview at up to 20 fps, so it never competes with the screen. Keyboard focus
  stays in the editor; work on the Visual page and watch the result on the other
  monitor. The menu remembers the monitor (default: one that isn't the primary).
  It is hidden on other systems and explains itself when Studio has no X display
  or `glfw` (`pip install glfw`). `hydrapy studio … --native-output DP-5` starts
  with it showing; `--no-native-output` renders without a window, as before.
- Four patterns each contain 64 sixteenth-note rows. Select audio cells and use
  the Note / Velocity editor directly below the grid; Enter toggles browser synth
  hits and retains keyboard focus. Piano keys adds QWERTY note entry, octave and
  row advance controls. Shift+click, drag or Shift+arrows select blocks; Copy, Cut,
  Paste, transpose, block velocity and 100-step Undo/Redo edit them. **All** or
  Ctrl/Cmd+A selects only the current page's four lanes. Switching pages clears
  a block selection; Delete and Paste cannot reach hidden lanes. Undo/Redo
  returns to the page of the edit. Pattern tools
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
  concurrent edits cannot silently overwrite a newer score. A score of any version
  holding `NaN` or `Infinity` anywhere, even in a key Studio doesn't know, is
  rejected, since neither is JSON for the browser or Rack. Switching Studio's
  displayed output is not an edit: an open tracker keeps working, and the new
  program output is kept. Restarting Studio
  starts with the initial study until Restore is chosen. Transport and blackout
  are not saved. Unpersisted edits disappear when Studio exits.

The included patch maps Dolly Z to **2D scale**, Roll to rotation and Feedback to
real recursive image memory. This is not Deforum depth reprojection. The left
palette is explicitly **synth demo** audio generated in the control browser.
Archive samples, Stable Audio, RAVE, Pd and model servers are not connected.
Sound starts muted and requires a user gesture.

## Scenes and lane signals

A patch can declare named **scenes**, and the tracker's **SCENE** column picks
which one plays (`patches/gorgon_scenes.py` is a working example):

```python
@scene('drift')
def _():
    return osc(8).rotate(roll)

@scene('fracture')
def _():
    return voronoi(5).kaleid(6).scale(1 + tracker.hit(1, decay=.2))
```

- A scene's function returns the chain for one output: o0 by default, or
  `@scene('glow', o1)`. Each output follows the changes that name its own scenes,
  so one SCENE lane can drive several outputs independently.
- A SCENE cell starts its scene on that row, and it holds until the next change,
  wrapping round the score or the looped pattern. **Fade** (0–64 rows) crossfades
  from the previous scene. A fade never runs past the next change, so a change
  never jumps from a half-finished fade, and a change to the scene already
  showing doesn't fade. One SCENE lane holds one change per row, so two outputs
  can't both change on exactly the same row. Enter on an empty cell writes the scene after the one
  above it, so repeated entries step through the patch's scenes. A name the
  loaded patch doesn't declare is struck through and kept, so swapping patches
  doesn't lose the score.
- Before any change, and outside the tracker (`hydrapy live`, `render`), each
  output shows its first declared scene. Disabling automation keeps the scene
  that is showing, as it does controls.
- Anything sent to an output before its first scene stays underneath, as the base.
  Sending to that output *after* its scenes replaces them, so Studio refuses
  that patch with an error rather than silently dropping the scenes.

The output is the base blended with every scene, each at its weight. A scene
costs nothing while it isn't playing. In GL, a blend at amount 0 is compiled out
of the shader. On the CPU backend, a combine operator at its identity amount
(a blend or modulate at 0, for example) returns its input unchanged without
evaluating the chain it takes. Studio compiles each scene's shader, and each
crossfade the score contains, when the patch loads or the score changes, so a
scene change or fade never compiles while it plays. `switch()` uses the same
mechanism for fxhash mint variants.

Every scene draws into the same output buffer, so feedback (`src(o0)`) and
`delay()` history carry across a scene change. A scene that isn't playing isn't
evaluated, so anything inside it that advances only when drawn holds still until
the scene plays again.

**Lane signals** read the audio lanes per frame, in any operator argument:

| Signal | Value |
| --- | --- |
| `tracker.gate(n)` | 1 while lane n's gate is open (half a row per hit), else 0 |
| `tracker.hit(n, decay=.25)` | 1 at each hit, falling to 0 over `decay` seconds |
| `tracker.note(n)` | the last hit's MIDI note, key-transposed as the CV is (held; 60 to start) |
| `tracker.vel(n)` | the last hit's velocity, 0–1 (held) |
| `tracker.beat()`, `tracker.row()` | the playhead: beat 0–64, sixteenth row 0–63 |

Lanes are numbered 1–4 as in the tracker. A hit fires when the playhead crosses
its row, so it fires once per pass and isn't missed at a low frame rate. Muted
lanes, zero-velocity hits and a stopped tracker send no gates or hits. A seek clears the held note and velocity,
as the CV output does. Video exports evaluate both scenes and lane signals frame
by frame from the score. A static HTML export captures the scene and values
showing when it is exported.

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
| `clock` | 5 V pulses, `--cv-ppqn` per quarter note (default 4: sixteenths), 50% duty; in a version-3 score it counts elapsed beats from the transport's start tick, so it runs steadily through cycle wraps |
| `reset` | 5 ms, 5 V on start/resume/seek and at the score or pattern-loop boundary |
| `run` | 5 V while the tracker is playing |
| `gate1`–`gate6` | 5 V for half a row on each audio-lane hit; muted lanes send nothing |
| `pitch1`–`pitch6` | held 1 V/oct pitch; C4 = 0 V by default, follows the key selector |
| `vel1`–`vel6` | last hit's velocity on that lane, 0–5 V, held until the next hit |
| `z`, `feedback` | lane minimum to maximum as 0–5 V |
| `roll` | −45° to 45° as −5 V to 5 V |

The default map is `clock, reset, gate1–gate4, z, roll`, because the ES-3 has
eight outputs; `feedback`, `run` and velocities are opt-in. Lanes 5 and 6 exist
in version-3 scores (Gorgon's `hydrapy-tracker-3`) and are silent before them.

Every sample is evaluated in the audio callback from the render thread's latest
playhead, so clock, gates and held key steps land on the exact sample rather than
on a video frame. Pausing or disabling automation stops clock, reset and gates;
lanes hold their value. The CV and the picture share one playhead, so the modular
follows Gorgon's tempo, seeks and loops. Gorgon does not follow an incoming clock.
A seek or start reaches the audio callback up to a frame and a block late; if that
is after its row's gate would have closed, the hit it landed on still plays, late,
for up to its usual length. A seek that interrupts an open gate closes it for 1 ms
first, so the hit it lands on triggers again (Gorgon's `spec/cv-semantics.md`).

In a version-3 score the clock is high while `frac(b · ppqn) < 0.5`, where `b` is
the start tick of the current transport segment plus the ticks elapsed since, in
beats, never wrapped at the cycle. It lines up with the cycle beat at every start,
resume and seek, then keeps its period through every cycle wrap, so a cycle that
is not a whole number of clock periods (a looped 5-row pattern at rate 2, or 3 rows
at rate 8, shorter than half a period) neither cuts a pulse short nor merges two.
`reset` still pulses at every wrap. Versions 1 and 2 loop in whole beats, where the
two rules agree.

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

The [Gorgon module](https://github.com/JosiahAlanLee/Gorgon/tree/main/rack)
for VCV Rack 2 links Rack to Studio with no configuration. The endpoints below,
the score format and the signal semantics are specified in the Gorgon
repository's [`spec/`](https://github.com/JosiahAlanLee/Gorgon/tree/main/spec);
this page describes Studio's implementation of them. On start Studio writes
`~/.hydrapy/studio.json` (port, session token, root; readable only by you) and
removes it on exit if it is still its own. Set `HYDRAPY_STUDIO_FILE` to use another
path. The module reads that file and polls Studio on loopback.

GorgonBridge 2.4.0 puts the score's clock, gates, held velocities and automation
on Rack jacks, with gates and unipolar lanes at 10 V. It has one channel per audio
lane: six for a version-3 score (GATE 1–6, and six channels on the polyphonic
**GATES**, **VEL** and **V/OCT** jacks), four for versions 1 and 2, where GATE 5 and 6
stay low. **V/OCT** carries the held pitches (1 V/oct, C4 = 0 V by default; C3 is
selectable). Notes, key transposition, AIR, velocity zero and mute follow direct CV
semantics. Reset
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
tests and connects the Rack module from the Gorgon repository's `main` to this Studio.

### Version 3 scores

Studio also plays Gorgon's score format 3 (`hydrapy-tracker-3`: six lanes, 128
patterns and a chain, per-pattern rates, cell conditions, lane transforms and the
bus layer). Direct CV, the Breaks synth and patch lane signals play it as the spec
says, and so does the editor's audition synth.

In the editor, banks A–H each hold sixteen patterns; the chain under Patterns is
the song order (**+ Add** appends the pattern you are editing, **Edit** takes codes
such as `A01 A02 A02 B01`), and ↻ loops the edited pattern instead. Launching a
pattern that isn't in the chain loops it. **Rate**, **Bar** and **Bars** shape a
pattern (¼× to 8×, up to 128 rows). For the selected audio lane, **Lane** names it,
**Length** makes it shorter than the pattern so it repeats (its repeating rows are
dimmed and edit the cell they play), and **Transforms** reshapes it: `rot k`,
`rev`, `ply n`, `degrade p` and `every a:b (…)`, separated by commas. A note's
**Condition** decides when it plays: `1:4` on the first bar of four, `%50` half
the time, `F` while **FILL** is on, `1ST` on its pattern's first bar, `!` for the
opposite. **Score seed** (Pattern tools) drives every `%` and `degrade`.

A version-1 or 2 score stays in its format while the edits fit it. The first edit
that needs version 3 (a fifth lane, a pattern past A04, a rate, a length, a
condition, a transform, the chain or the seed) upgrades the score, and undo
restores its old format. Until then its patterns' own keys come back unchanged,
including metadata named `rate`, `bar`, `bars` or `lanes`; an upgrade gives those
names their version-3 meanings. An edit that changes nothing, such as Delete on an empty
pattern, leaves the score alone. Three transport actions select a pattern past A04
to play, which only version 3 can hold, so they upgrade the score without an undo
step: launching one, turning ↻ on while editing one, and seeking to a row of one
while ↻ is on. Undo doesn't bring the old format back after these; Reload or
Import does.

- The playhead adds `start`, `elapsed`, `entry`, `fill` and `bus` in integer ticks,
  1920 per quarter note. A seek lands on the nearest tick; Play starts a new
  segment at the playhead without changing `transport_revision`, or continues the
  segment when it lands on the anchor's own tick (a new one would share its key).
- Changing `loop`, the looped pattern, the chain, or any rate, bar or bars that
  moves the chain is a transport change; Studio keeps the playing entry and the
  beats into it where score-v3.md says so.
- `POST /api/tracker_transport` takes `{"fill": true}` for Studio's FILL, stamped at
  the first tick none of Studio's outputs has evaluated. A Rack module sends its
  bus window as `segment`, `sender` and `bus`, never with `beat`, `enabled` or
  `playing`. One module holds the layer at a time; three seconds after its last
  request Studio clears the layer.
- `hydrapy studio --tracker-trace FILE` appends the event trace, one JSON line per
  candidate the picture evaluates and per change to FILL or the bus layer. A bus
  window Studio drops or refuses is a line too, `{"bus": {"window": "dropped" or
  "refused", "reason", "from"}, "received"}`: `dropped` with reason `stale segment`
  for a window of an old segment, `refused` with reason `bus held by another
  module` for one from a module that doesn't hold the layer, and `refused` with
  reason `malformed window` for one that fails validation (still answered with a
  400). `from` is the window's first tick (`null` for a malformed window whose
  `from` isn't a non-negative integer) and `received` the first tick Studio's
  outputs hadn't evaluated when it arrived. Like every line, these carry `transport_revision`
  and `start`.

## Preview and measurements

Program reuses Studio's bounded preview. Auxiliary has an additional encoder
worker, one pending frame and one latest encoded frame. Readback stays on the
render thread; JPEG and socket writes do not. Buses share readback when both ask
for the selected output in the same iteration. Each viewer requests its next
frame only after decoding/display scheduling. Inactive thumbnails retain their
staggered lower refresh rate. Closed viewers stop demanding captures.

Local inspection uses `GET /api/frame?bus=monitor&output=o0` (through `o3`), with
the same session-token and origin checks. Each requested output has an independent,
bounded encoder, created only when first inspected. Captures share a 20 fps budget
and fit within 480×270; GL scales before readback. Multiple controllers can inspect
different outputs. Changing the inspected output never sends a score or transport edit.

`window.gorgonPreview` exposes up to 300 timing samples per browser: server render,
readback and JPEG durations, plus browser request, body transfer, decode and
display-scheduling durations. Program includes optional GPU timing when enabled
in Studio's preview details. Frame-age estimates exclude first-byte transit;
display scheduling measures the handoff to the canvas animation loop, not physical
scanout or audio latency. The buses have independent sequence cursors.

`GET /api/latency` (with the session token) reports every on-screen command —
score edits, transport and blackout cues, control changes — with when Studio
received it, applied it, rendered the frame containing it and, with the fullscreen
output, when that frame flipped. With the fullscreen output it also returns the
last frames' pacing: how late each started, its work from the planned start to the
GPU finishing, the wait for the flip, and skipped refreshes. Neither measurement
includes the monitor's own processing, which needs a light sensor on the screen.

## Reproducible checks

```sh
python -m pytest -q tests/test_studio_tracker.py tests/test_scenes.py tests/test_tracker_cv.py tests/test_es40.py tests/test_studio_bridge.py tests/test_studio.py tests/test_studio_preview.py
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


### External Rack sound controls

The Audio page's selected-cell inspector has **External Rack instrument** controls.
Set a track's machine (0–191) and P1–P8 (raw integers 0–127), then override any
of them on individual hits. Blank fields inherit. A blank track machine uses
the instrument's selected machine; blank track parameters use that machine's
native defaults. A dot marks a hit with locks. Every hit starts from its track
defaults, so a lock never leaks into the next unlocked hit. Clearing a cell
removes its locks; note and velocity changes, transpose and clipboard preserve them.
Undo also restores the score format when undoing the first sound edit.

Editing sound controls upgrades the saved document to `hydrapy-tracker-2`.
Older Gorgon/HydraPy versions reject this format. Untouched legacy scores keep
`hydrapy-tracker-1`. These controls reach an adjacent compatible Rack instrument
through Gorgon's sound link. Browser audition, direct physical CV and visual
lanes retain their existing behavior. P1–P8 labels and machine availability are
instrument-specific; the tracker does not guess a pitch parameter from note names.
