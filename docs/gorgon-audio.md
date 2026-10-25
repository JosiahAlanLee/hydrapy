# Gorgon audio routing: JACK, PipeWire and Rack

The tracker can publish four listening-audio ports independently of its CV
output. Rack's Gorgon module version 2.1 adds four corresponding audio jacks.
The HTTP bridge still carries the score, transport and control values; PCM goes
through the audio graph.

## Start and connect

1. Install the optional client: `pip install -e ".[jack]"`. PortAudio CV needs
   `.[cv]`; native JACK CV also uses `.[jack]`. A running JACK server is required; Studio does
   not start or reconfigure one.
2. With native JACK, start Studio and Rack normally. With PipeWire, use its
   JACK compatibility layer, for example `pw-jack hydrapy studio
   patches/gorgon_corridor.py --tracker` and `pw-jack /path/to/Rack`. Both must
   connect to the same server/user session.
3. In the tracker, open **Output routing → JACK / PipeWire audio** and click
   **Enable JACK audio ports**. The panel lists the actual port names, server
   sample rate, period, buffer lead and error counters. Levels and enable state
   are saved in `.hydrapy/outputs.json` and restored on the next Studio launch.
4. In Rack, right-click **Gorgon** and choose its audio driver, capture device
   and input channel group. Select JACK and a device/group exposing four input
   channels. In your JACK/PipeWire patchbay, connect the four HydraPy ports to
   those four Rack capture channels in the order below. Device choices depend
   on the JACK driver; use the actual port names shown by the patchbay.
5. Patch Gorgon's audio jacks into a Rack mixer or processing chain, then an
   ordinary Audio output module for listening. Enable browser **Sound** only
   when you also want the independent browser audition synth.

| Studio port suffix | Capture channel | Gorgon jack |
| --- | --- | --- |
| `optical_L` | 1 | OPT L |
| `optical_R` | 2 | OPT R |
| `breaks_L` | 3 | BRK L |
| `breaks_R` | 4 | BRK R |

JACK may suffix duplicate client names, so use the full names displayed in
Studio rather than assuming a fixed `HydraPy:` prefix. Rack saves its audio
device selection with the patch; the external patchbay connections follow the
session manager's normal persistence rules.

## What produces the sound

**Optical** reads the first two `Soundtrack` probes declared in the running
patch. The first feeds L and the second R; one is duplicated to both channels;
no Soundtrack produces silence. Soundtracks retain their existing image-based
waveform and held-picture behavior. A patch's main displayed output alone does
not automatically become audio. For example, add `Soundtrack(o0)` to a patch
that draws a waveform; a uniform image with AC removal produces no sustained
tone. Optical audio continues with rendering when tracker automation is off.

**Breaks** renders the first audio lane as a synthesized kit: C-2 is a kick,
D-2 a snare, other notes a hat, with the score's velocity and mute. It uses the
same kit roles as browser audition, with a deterministic native waveform/noise
implementation. It is not recorded breaks, a sample slicer, or any of the other
three browser synth voices. Both Breaks ports carry the same mono kit. The
backend continues when the controller closes and is independent of browser
Sound. Pause silences both audio pairs. A musical seek clears queued Breaks
events and tails; it leaves the renderer/Soundtrack timeline intact.

## Timing and levels

Audio is prepared by a worker against JACK sample positions, then copied in
the process callback. The worker never synthesizes in the process callback.
This remains Python soft realtime: CPU/GPU contention can cause dropouts. The
**Buffer lead** setting trades latency for scheduling margin (2–8 periods).
The displayed lead is that software buffer alone, not total device or display
latency. At 48 kHz/256 frames, 2 periods add about 10.7 ms before any subsequent
graph/device latency. Increase it if the source-underrun counter grows.
Late blocks are discarded rather than replayed late. Server disconnects and
producer failures appear in the panel; **Retry JACK audio** reconnects.

JACK uses floating-point PCM in −1…1. Rack converts that to −10…10 V; the
default optical gain 0.5 limits a full-scale waveform to ±5 V. The separate
**CV output levels** menu offers 10 V (existing Rack default) or 5 V (matching
direct CV) for gates, clock, reset, run, velocity, Z and feedback. Pitch stays
1 V/oct, roll stays ±5 V, and audio scaling is independent of that choice.
**OFFSET** adjusts Rack's score/CV timing; it does not move received PCM.
Measure end-to-end audio/video latency on the actual workstation.

## ES-40 data stays on its own route

The encoded ES-40 S/PDIF pair is not listening audio and does not pass through
these four ports, gains or Rack's PCM resampler. Keep that route at its required
rate, with no gain/mix/resample/dither. Its per-pair **Normal / Invert left /
Invert right / Invert both** selector is also in Output routing; see
[the polarity details and full-scale limitation](es40.md#polarity-control-in-the-tracker).

For native JACK CV, use `--cv-backend jack`. Direct CV can be launched with
`--cv-device GorgonCV --cv-backend jack --cv-map 0`; here `GorgonCV` is the
JACK client name. A digital profile can set `"backend": "jack"` in a device
entry (or inherit the CLI choice). For example:

```json
{
  "format": "hydrapy-cv-1",
  "devices": [{
    "backend": "jack",
    "device": "GorgonDigital",
    "samplerate": 48000,
    "es40": [{
      "name": "ES-40 gates",
      "channels": [0, 1],
      "header": 1,
      "expander": "8gt",
      "map": {"1": "gate1", "2": "gate2", "3": "gate3", "4": "gate4"}
    }]
  }]
}
```

Start with `pw-jack hydrapy studio patches/gorgon_corridor.py --tracker
--cv-profile cv.json`. Output routing lists the dedicated CV port names.
Connect `GorgonDigital:out_00` and `out_01` directly to the interface's physical
S/PDIF playback pair at unity. These are JACK client channel indices, not
automatic hardware channel assignments. Connect any separate direct CV ports
to their physical DC-coupled outputs. An explicit ES-40 rate must match the
server; opening refuses a mismatch rather than resampling. A runtime rate
change latches zero-voltage commands and reports an error until restart.
Float transport preserves every signed 24-bit encoded word exactly before
the external graph. The optional polarity correction retains its documented
full-scale limitation. Keep the interface at the supported 44.1/48 kHz rate.

CV retains its existing callback evaluator, now paced by JACK periods; this is
also Python soft realtime. Shutdown sends actual zero-voltage encoder commands
and lets downstream clients consume them before disconnecting. Polarity and
PCM transport tests do not establish a bit-perfect physical S/PDIF route.

## Verification

`tests/test_studio_audio.py` checks scheduling/chunk invariance, pause/seek,
channel isolation, mono duplication and reconnect behavior with simulated
callbacks. `tests/test_output_settings.py` checks real encoded words and saved
settings/rollback. The Chromium suite checks the actual controls. These do not
require JACK. To run the real four-channel loopback against a running server:

```sh
HYDRAPY_JACK_TESTS=1 python -m pytest -q -s tests/test_studio_jack.py
# PipeWire compatibility variant:
HYDRAPY_JACK_TESTS=1 pw-jack python -m pytest -q -s tests/test_studio_jack.py
```

Linux CI runs isolated native JACK/dummy and PipeWire daemons separately,
checking the audio loopback, encoded gate/CV words, live polarity changes,
separate direct CV and zero-voltage shutdown. Rack
tests exercise its real Port/Device API with a deterministic capture device,
including 44.1/48/96 kHz conversion. Neither test measures a physical interface,
the ES-40 digital path, or real display latency. Cross-repo CI uses explicit
peer commits; manual workflow inputs can check another ref without changing
the default baseline.
