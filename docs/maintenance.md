# Maintenance checks

[Development](development.md) · [Audio timing](audio-timing.md) · [Performance sessions](performance-sessions.md)

## Automatic CPU checks

The [CPU checks workflow](../.github/workflows/cpu.yml) runs on pull requests,
pushes to `main`, and manual dispatch. It tests Python 3.9 (the declared minimum)
and 3.13 on Ubuntu, using fresh environments with the `test`, `osc`, `video` and
`vector` extras. The 3.13 job also installs Sardine 0.4.0 and requires Vortex to
import before running its integration test. Node and FFmpeg/ffprobe are required
so browser-runtime assertions and encoded audio/video checks execute. The CPU job
installs Node explicitly with apt rather than relying on the runner image.
`python -m pytest` includes skip reasons by default (`-ra`), including missing
Node, optional Python packages, and opt-in device/browser integrations.

ModernGL, Torch and sounddevice are not installed. GPU and other unavailable
optional integrations remain explicit skips; they are not hardware qualification.
Test failures fail the job. The Actions log lists skip reasons, and each job keeps
JUnit XML and its resolved Python packages for seven days. Only these small reports
are uploaded, not rendered media. Actions are pinned to commits and have read-only
repository permissions. A newer push cancels an older run on the same PR.

Reproduce the CPU environment outside an existing render/training environment:

```sh
python3.13 -m venv /tmp/hydrapy-cpu-check
/tmp/hydrapy-cpu-check/bin/python -m pip install -e '.[test,osc,video,vector]' 'sardine-system==0.4.0'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/hydrapy-cpu-check/bin/python -m pytest -q -ra
```

FFprobe and Node must be on `PATH`; FFmpeg may also come from imageio-ffmpeg.
The encoded A/V tests check H.264/AAC availability, then exercise HydraPy's own
mux path with a one-second video. Missing or unusable codecs/tools skip; a mux
failure or incorrect duration in HydraPy fails the tests. An `IMAGEIO_FFMPEG_EXE`
override remains authoritative.

Ubuntu 22.04's FFmpeg 4.4.2 truncates AAC at the video's decode-time endpoint
when `-shortest` is used with reordered B-frames (one second becomes 0.789002
seconds). HydraPy avoids this by bounding the mux explicitly to the rendered
frame count divided by fps, and padding short audio with silence. Long audio is
cut at the video end. This applies to Python, CLI and Studio offline exports,
including FFmpeg 4.4.2. PCM sample-count and decoded midpoint checks remain intact.

On Linux, Sardine's RtMidi
dependency may need a C++ compiler, `pkg-config` and `libasound2-dev` to build.
For Python 3.9 omit Sardine;
that package requires Python 3.10 or newer. CI resolves compatible dependencies
for each interpreter; `packages.txt` identifies the exact versions in a given run.
The workflow creates status checks; making them mandatory for merges is a separate
repository branch-protection setting.

## Repeatable local GPU check

This command runs only when invoked on the machine being measured. There is no
self-hosted Actions runner or automatic job on the render machine.

In a separate environment, install `.[test,gpu,osc,video,vector]`. Have FFmpeg,
ffprobe and Node on `PATH`, and use the same installed environment for comparisons.
An idle measurement should wait until training/render jobs have finished.

```sh
python tools/check_hardware.py --expect-renderer 'RTX 4090' --label idle-before \
  --output .hydrapy/checks/idle-before
```

The command requires a matching `GL_RENDERER` and rejects known software renderers.
It runs the operator CPU/GPU comparisons, automata, optimized-path equivalence,
render timing and optical-audio timing tests in separate processes. This releases
GL state between groups. A failed test, skipped test, empty collection, timeout or
missing prerequisite makes the check fail. Known failures should remain visible;
do not loosen tolerances or exclude cases just to get a green report.
Automata and patch-brightness tests create and release their GL context in
function-scoped fixtures.
Previously they cached contexts during collection; an earlier audio-timing test
created and released another current context, so the cached object had no current
GL context when reused (`cannot create buffer`). Passing `ctx=` to `GLRenderer`
does not make an old context current. Per-test ownership fixes the isolation
without reordering or skipping tests; renderers are released before their context.

It then benchmarks `osc`, `rd`, `wave` and `nca`, with 30 warm-up frames and 120
measured frames per case at 1920×1080. GPU execution and RGB readback are included.
`--size`, `--frames` and `--warmup` can change this workload; keep them identical
between comparison runs. This is a focused regression check, not the entire GPU
suite, a sustained performance test, or a hardware-encoder test.

The new output directory contains:

| File | Evidence |
| --- | --- |
| `report.json` | Commit, dirty working tree, OS/Python/packages, actual GL renderer/version, tool versions, conditions label, outcomes and checks not performed |
| `*.xml`, `*.log` | Test results, skip reasons, failures and command output for each group |
| `benchmark.json` | Resolution, workload, warm-up, every frame time, median and p95 by case |
| `benchmark.log` | Benchmark output, including errors and incomplete work |

Existing output directories are refused. Reports stay local under the ignored
`.hydrapy/` directory. A zero exit status means the selected correctness tests and
benchmark completed; there is no universal FPS threshold. Compare each case's
median and p95 against an idle baseline on the same device, driver, dependencies
and settings. Record background work in `--label`; the script does not start,
stop or discover training/inference processes. Repeat idle measurements after
the change to distinguish a regression from temporary load or thermal effects.

### Recorded result, 2026-10-25

Commit `67d0897` on the render workstation, idle: Ubuntu 22.04.5, Python 3.11,
GeForce RTX 4090 with driver 580.178.04 (GL 3.3 context), FFmpeg 7.0, Node 26.

| Group | Result |
| --- | --- |
| operators | 141 passed, **2 failed**: `test_cpu_and_gpu_agree[voronoi-chain1]` (mean difference 0.206, limit 0.02) and `[mipblur-chain73]` (0.130) |
| automata | 8 passed |
| equivalence | 98 passed |
| render timing | 42 passed |
| audio timing | 32 passed |

Benchmark, median / p95 per frame: `osc` 4.11 / 4.25 ms, `rd` 0.93 / 0.95 ms,
`wave` 1.05 / 1.12 ms, `nca` 0.91 / 0.92 ms.

The full suite in one process, with the Chromium tests enabled, gave 1137 passed,
7 failed and 63 skipped. The failures are the two operator cases above,
`test_mipblur_is_smooth_cheap_and_the_same_on_both_backends` (maximum difference
0.061, limit 0.01) and the four automata tests. All seven also fail at the
previous `main`, `2d4553e`. `test_studio_jack.py` passed 3/3 against a private
`jackd` dummy server and 3/3 against a private PipeWire 1.0.7 instance.
Audio devices, ADAT/CV, RAVE inference, end-to-end latency and hardware encoding
were not measured.

### Voronoi comparison

Hydra's original Voronoi uses `fract(sin(dot(p, constants)) * 43758.5453)`.
Small implementation differences in `sin` become large changes in the cell
seeds; cross-driver pixel parity is not a valid expectation. On the RTX 4090,
NVIDIA 580.178.04 (GL 3.3), the original CPU/GPU mean image difference is 0.206.
The parity test now injects identical, exactly representable spatial seeds on
both backends and retains its existing pixel tolerances for cell selection,
animation and shading. A separate test exercises the original hash on both
backends for finite, grayscale, opaque, spatially varied and animated output.
The application's Hydra-compatible hash and appearance have not changed.

## Audio, ADAT and model qualification

The automated GPU check uses simulated/file-based audio. It does not open a sound
device, send CV, load RAVE, or measure audible latency. These need an operator-run
test of the actual rig. Add the following observations beside a run's report;
use **not run** for unavailable paths, rather than recording a pass.

1. Record the interface/driver, sample rate, block size, clock master, named
   input/output channels, host version, patch revision and physical loopback route.
   Use the chosen audio channels for the loopback; CV outputs need their own explicit
   voltage/channel calibration. Do not infer that from a successful audio test.
2. With the chosen audio host alone, capture ten minutes of a repeatable test
   signal and timestamped markers through the loopback. Record underrun/overrun
   counts, dropouts, measured marker delay and whether that delay changes over time.
   Listening alone is not a latency or clock-drift measurement.
3. Repeat with the same audio configuration and intended HydraPy performance patch
   running at the intended resolution/frame rate for the same ten minutes. Record
   frame-time tails alongside the audio observations. The short automated benchmark
   above is not a substitute for this sustained concurrent test.
4. Once the RAVE host exists, record the streaming export's SHA-256, sample rate,
   channels, latent dimensions, host block/buffer sizes and CPU/GPU placement.
   Repeat model-only and model-plus-render runs. Keep training separate from the
   idle baseline; test additional load only when that is an intended condition.
5. Exercise start, stop, panic/reset and disconnect/reconnect on the intended
   performance host. Confirm output levels return to the expected state and no
   owned worker remains. Record failures and the exact operation that caused them.

There is not yet a RAVE audio host or an automated physical-loopback measurement
in this repository. This protocol defines the missing evidence without treating
the OSC monitor or software timing tests as that evidence.
