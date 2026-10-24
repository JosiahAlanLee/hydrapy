# Performance sessions: first integration slice

[Documentation](README.md) · [Live I/O](live-io.md)

`hydrapy session` plays finite, editable event clips through named OSC routes.
It can show the arrangement, record dispatch timing, and optionally supervise
explicit child processes. It does not load Torch, a RAVE checkpoint, a GPU
context, or an audio device. The example Pd process is a message monitor.

This is the first control-layer integration for a local workstation. It is not
a finished arranger, live Sardine host, RAVE host, or sample-accurate sequencer.

## Inspect before running

From the repository root:

```sh
python -m pip install -e ".[osc]"
hydrapy session examples/performance/session.json --dry-run
hydrapy session examples/performance/session.json --preview /tmp/session-score.html
```

Open the HTML to see one lane per route/address, event positions, captured
durations, destinations and exact values. This is a static score view; it sends
no messages and plays no audio. Both inspection modes ignore `--launch` and
open no sockets or child processes. Dry-run does not even require python-osc.

## Connect the drawn voice

```sh
hydrapy studio patches/session_oscillator.py --backend cpu --size 320x180 --fps 12
```

In Studio, arm live I/O so the patch's OSC receiver binds to localhost:57130.
Then, in another terminal:

```sh
hydrapy session examples/performance/session.json --repeat 4 --record /tmp/session-timing.jsonl
```

The pitch and grit events change the drawn voice. This patch generates a
`Soundtrack` but does not open a speaker/CV output. CPU preview can be used while
a model trains elsewhere; the session runner itself has no GPU dependency.
An existing render or training process is never discovered, signalled or attached.

To also launch the optional Pd console monitor (requires native Pd on PATH):

```sh
hydrapy session examples/performance/session.json --launch --repeat 4
```

The monitor receives `/voice/gate` on port 57131 with `-noaudio`. Other programs
can use these routes without Pd. UDP has no delivery acknowledgement: successful
sends do not prove a receiver is running. `--lead-in` is a startup allowance,
not a readiness handshake; open/arm external receivers before playback.

## Clip and routing format

`hydrapy-session-1` contains `bpm`, a finite length in quarter-note `beats`,
named `routes`, and `events`. The example is a complete, editable score.

| Field | Meaning |
| --- | --- |
| `beat` | Event onset relative to the start of each clip play |
| `route` | A named host/port from `routes` |
| `address`, `args` | Concrete OSC address and ordered numeric/string arguments |
| `duration` | Captured/display duration; does not create a note-off |
| `kind: "set"` | Apply even when late, including explicit gate releases |
| `kind: "trigger"` | Drop when over `--max-trigger-lateness` seconds late (default .05) |
| route `reset` | Messages sent at session end, interruption, or playback failure |

All events are validated before starting a sender or child. Unknown fields,
missing routes, invalid OSC values and events outside the clip are errors.
Events sharing a beat retain their input order. Resets are best-effort UDP;
an audio host should also have its own mute/panic control.

Scheduling uses absolute monotonic deadlines, so send/processing time is not
added to every subsequent interval. OSC messages are sent when due, **without
bundle timetags**, and receivers such as `OscIn` apply them on receipt. JSONL
records contain intended deadlines, scheduler observations, completed send times,
lateness and dropped triggers (whose send time is null). Lateness measures the
scheduler observation against its deadline. These measurements do not include
network delivery, rendering or audio latency.
State changes catch up after a delay; late trigger bursts are suppressed.

This first version has fixed tempo and a finite repeat count. It has no live
seek, tempo-change or Link follower mode. HydraPy already has `Link` and `OscIn`;
Link beat/phase synchronization must not be confused with shared song position.
A later transport adapter needs explicit seek/reset and latency semantics.

Optional `processes` contain `name`, `argv` and `cwd` (relative to the session
file). They run only with `--launch`, without a shell. A `{python}` argv entry
uses the current Python interpreter. A child exiting stops the session; cleanup
terminates only children/process groups started by this launcher, including on
Ctrl-C, SIGTERM or a closed terminal (SIGHUP). Children run in their own process
groups, so they never receive the terminal's hangup themselves; the runner's
cleanup is what stops them. Further signals during cleanup are ignored, so
resets and the SIGKILL fallback for a child that ignores SIGTERM still finish.
Under `nohup` (or a background job that ignores Ctrl-C) those signals stay
ignored and playback continues. A session file with child commands is
executable project configuration.

## Capture Vortex into an editable clip

Use an existing Sardine environment with this checkout installed:

```sh
python examples/performance/capture_vortex.py /tmp/captured.json
hydrapy session /tmp/captured.json --preview /tmp/captured-score.html
```

The adapter accepts the events from `pattern.query(TimeSpan(...))`; it never
imports Sardine itself or starts its REPL, SuperDirt or clock. `Binding` maps
explicit control fields to OSC arguments. Cycle time is converted to beats
with an explicit `beats_per_cycle` (default 4). Onsets outside the requested
window and fragments of earlier notes are excluded; durations are clipped at
the capture boundary. Continuous patterns require explicit discretization.

Capture evaluates a finite passage once. The JSON is then the editable score;
it is not a promise to round-trip arbitrary edits back into live-coding source.
Sardine remains optional and is not added to HydraPy's default dependencies.

Adapter reference: Sardine's `Event`, `TimeSpan` and `Pattern.query` at
[`5e1e592`](https://github.com/Bubobubobubobubo/sardine/tree/5e1e5920a3dd7f82e7ad04051f877f85ada00cf9/sardine_core/sequences/tidal_parser).

## RAVE boundary and next hardware checks

Training and inference are separate workflows. Nothing here watches a training
directory or starts inference automatically. When a model is ready, export a
**streaming TorchScript model** and inspect its sample rate, channels, methods,
latent dimensions and block requirements before building the native Pd host.

The documented IRCAM live path is Pd/Max `nn~`; its buffering is a better initial
host than calling PyTorch from a Python audio callback. The OSC monitor proves
only control routing, not that Pd/nn~, a particular RAVE model, or ADAT audio
works on a machine. Do not substitute the monitor for an inference host.

The next integration should measure model latency/dropouts alone and under
render load, compare CPU and GPU inference, and define explicit latent/CV
scaling and channel maps. Audio-device ownership and sample-clock buffering
belong in that host. The session scheduler should remain outside the callback.

- [RAVE streaming export and realtime usage](https://github.com/acids-ircam/RAVE#export)
- [nn~ methods and buffer configuration](https://acids-ircam.github.io/nn_tilde/)
- [Sounddevice callback constraints](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)
- [Ableton Link timing model](https://ableton.github.io/link/)

## Tests

```sh
python -m pytest -q tests/test_performance_session.py
```

The tests exercise deadline drift, late trigger/gate-off behavior, shutdown,
real UDP-to-`OscIn` routing, child failure cleanup, capture boundaries and schema
validation. Optional Vortex coverage runs when Sardine is installed. These are
control-layer tests; they do not measure hard realtime audio or GPU contention.
