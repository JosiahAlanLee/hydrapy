# Stress-test diagnostics

Studio can retain a bounded diagnostic timeline alongside a long render and its
audio. It is off by default:

```sh
hydrapy studio patches/gorgon_scenes.py --tracker --backend gl \
  --diagnostics .hydrapy/stress-01.jsonl
```

The file must be **new**; existing files and symlinks are refused. Relative paths
are relative to Studio's `--root`. Files are created with mode 0600. A file is
capped at 32 MiB: a `limit` record and a stderr warning report that diagnostics
have stopped. Rendering and audio continue. Disk errors disable the recorder
and report to stderr instead of propagating into playback.

## Use with Gorgon render/live

In the launch profile, add this to the existing `studio` object:

```json
"args": ["--diagnostics", ".hydrapy/stress.jsonl"]
```

Keep any existing arguments. With a HydraPy revision containing this feature,
Gorgon's existing argument passthrough supports it without a new bridge API or
Rack build:

```sh
gorgon --config rig.json render score.json --output /path/to/new-take --seconds 600
```

The trace is `/path/to/new-take/studio/.hydrapy/stress.jsonl`, next to the copied
workspace. Keep it with `capture.json`, `capture-clock.npz`, the recorded video's
timestamp sidecar and the existing app logs. Each capture uses a new output
directory; if the *source* workspace already contains that diagnostic filename,
choose a fresh name before running, since Gorgon copies the workspace. With
`gorgon start`, use a fresh filename for each session. Removing the argument
disables diagnostics for the comparison run.

## What the file contains

Each complete line is a JSON record. Timestamps use the host's monotonic clock,
shared with the existing recording/capture timestamps. Sort events by
`monotonic` for cross-process comparison; writer batches can interleave records.
The run header carries a run ID, UTC date, requested settings, Python and
HydraPy versions, and the checkout commit/dirty flag when available. Patch and
score events retain SHA-256 identities and the score seed. Actual GL vendor,
renderer and version are recorded after each load: `GL` alone does not prove a
hardware GPU is active. Keep the original patch, score, assets and their versions;
this file is not a replay package or a record of arbitrary patch randomness.

| Record | Contents |
| --- | --- |
| `summary` | Approximately once a second: p50/p95/p99/max render, readback, whole-frame work and frame-start interval; observed encoder depth maximum; audio counters and changes; current Linux RSS; diagnostic losses. |
| `event` | Load/prewarm progress; actual backend; score identity; scene/crossfade membership; transport, recording and device commands; audio holds/resumes; errors and shutdown. |
| `incident` | Up to two seconds / 256 recent numeric frame records when audio counters rise, frame work/interval exceeds both 50 ms and two target periods, the encoder queue reaches 75%, or native deadlines are missed. Detailed snapshots are rate limited to once per five seconds; counters and summaries continue. |
| `closed` | The diagnostic writer drained and closed normally. This does **not** certify a clean render or audio take. |
| `limit` | File-size cap reached; later observations are unavailable. |

The numeric incident columns include frame sequence, simulation time, score
beat and estimated JACK frame position (`-1` when unavailable). JACK position
is extrapolated from the most recent audio anchor, wraps at 32 bits, and is not
an independently measured A/V sync result. Incident audio-counter increases are
observed on the render thread; summaries also sample the existing counters on the
writer so they continue during render stalls. An underrun is localized to the
interval between observations, not to an invented precise callback timestamp.

`render_cpu_ms` is wall time for tracker application and renderer submission,
not GPU-only execution. `readback_ms` sums actual output reads/conversion for
recording and previews; it can include GPU waits. `work_ms` includes both plus
commands/housekeeping (and presentation waits in native mode). These measurements
overlap and must not be added. `interval_ms` measures rendered-frame spacing;
deliberate pauses and loads create gaps too. Use the adjacent state/load/command
events to interpret them. `producer_age_s` can rise during a stall **or while
paused**. Background checkpoints continue when the render thread is blocked,
provided Python can schedule the writer.

Counters are absolute observations. `counter_deltas` is null for the first
summary, and an individual delta is null if its counter reset (for example a
new video recorder). Repeated frames preserve recording duration; they are not
the same thing as an encoder failure. Source underruns, late packets and JACK
xruns are separate: continuous recorded audio blocks can still contain upstream
silence. A source-underrun count of zero does not prove audible musical output.

## Cost and failure behavior

No diagnostic hook runs in the JACK process callback. The render thread copies
numeric observations into a preallocated 4096-frame ring and queues occasional
events (128 pending maximum). It does not write JSON/files, query GPU timers,
wait for a diagnostic lock, or wait for the writer. A full buffer drops diagnostic
records and increments the corresponding loss counter. Scene membership is
checked once per rendered frame; hashing happens only after patch/score changes.
This remains Python soft realtime; the writer still shares CPU and the GIL.

The writer calculates statistics and serializes incident history off the render
thread. It flushes approximately once a second, without `fsync`. Complete lines
normally survive process termination; the last buffered interval or partial line
may be lost. No `closed` record means the trace did not finish normally. Power
loss durability is not guaranteed. Shutdown waits at most two seconds for the
writer. This is a file for timing diagnostics, not per-note tracing; keep
`--tracker-trace` and optional GPU profiling separate when measuring its cost.

## Validate overhead on the rendering machine

The included harness runs the same feedback/scenes patch with MP4 encoding and,
optionally, actual JACK audio. It writes a comparison with frame intervals,
process CPU time, repeated frames and audio errors. Use an existing isolated JACK
server for `--audio`; the script never starts or reconfigures audio services.

```sh
python docs/studio_diagnostics_benchmark.py --output /tmp/diag-ab-1 \
  --seconds 600 --backend gl --size 1280x720 --audio
python docs/studio_diagnostics_benchmark.py --output /tmp/diag-ab-2 \
  --seconds 600 --backend gl --size 1280x720 --audio --order on,off
```

Do not run competing benchmarks/tests simultaneously. Compare tail latency,
worst stalls, queue pressure and audio errors as well as throughput. The harness
does not exercise Rack, physical interfaces or display latency; also repeat the
actual Gorgon/Rack workload with and without the profile argument. Software GL
results do not establish overhead on a discrete GPU.
