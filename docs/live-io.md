# Live control and modular audio

[Documentation](README.md) · [Project home](../README.md)

## The modular, OSC, MIDI and Link

`hydrapy.live` turns outside control into ordinary signals, so anything below plugs into any
parameter. Live sources only move in real time (`hydrapy live`); for a finished render, record them
and use the file forms.

**Control voltage in (ES-6, ES-8, any DC-coupled interface).** An ES-6 arrives as DC-coupled audio
inputs on the ADAT channels of your interface; `hydrapy devices` lists devices and channels.

```python
cv = CvIn(device="ES-8", channels=[8, 9, 10, 11], volts_per_fs=10)   # Expert Sleepers: ±10 V = full scale
cv.cv[0]                       # volts           cv.uni(0)  0..1 for 0..+10 V      cv.bi(1)  -1..1 for ±5 V
cv.volt_oct(2)                 # Hz from 1 V/oct
gate = cv.trig(3, threshold=1.0, length=0.6)      # phase since the last rising edge
osc(cv.uni(0).map(6, 50)).scale(1 + Env(gate, 0.02, 1.0)).out(o0)
cv.audio(2).fft[1]             # an audio-rate channel through the same Bark analyser as Audio
```

`CvIn.from_file("performance.wav", channels=[0, 1, 2, 3])` gives the same signals from a
multichannel recording, sampled at each frame's time: record the CV in your DAW alongside the audio,
then `hydrapy render` at any resolution with no dropouts and the envelopes landing on the gates.

**Control voltage out (ES-3).** Values become voltages, ramped across each audio block:

```python
out = CvOut(device="ES-8", channels=[8, 9])
out.send(0, Probe(o0, 0.5, 0.5, 0.08, "luma", lo=0, hi=8))   # the centre's brightness, 0..8 V
out.send(1, Lfo(0.25, "tri", -5, 5))                          # a plain LFO in volts
```

`Probe(output, x, y, size, channel, lo, hi)` reads a region back after every frame (one frame of
latency), on both backends; that's the image as a sequencer. `CvOut(file="cv.wav", fps=30)` writes
the voltages as a multichannel WAV during an offline render instead, to play from the DAW into the
ES-3 next to the video.

**Audio out: the painted oscillator.** `out.send(k, Soundtrack(o0, x=0.5, gain=0.5))` sends a drawn
soundtrack ([how it works](film-and-sound.md#fischinger-cut-paper-and-the-drawn-soundtrack)) down an ES-3 channel at audio rate — a VCO made of charcoal, ±5 V, to be put
through a filter — while the other channels carry CVs, and the ES-6 brings pitch, wobble and grit
back in. `patches/26_painted_oscillator.py` is that loop: a `wave` stroke standing upright down the
middle of the frame, the column through it read as the waveform, 1 V/oct from `wave(2 ** pitch * 2)`
(`pitch` a `CvIn` channel in volts); a damped `harmonograph` as a second, rougher voice. The file
mode records the audio channels too.

**OSC.** `OscIn(57120).value("/synth/freq", default, lo, hi, smooth)`, `.trig("/kick")`;
`OscOut(host, port).send("/addr", signal_or_probe)`. VCV Rack's cvOSCcv makes this a second CV path
without hardware.

**MIDI.** `Midi("port name")`: `.cc(74)` 0..1, `.note(60)` velocity while held, `.gate(60)` a
trigger, `.bend`, and `.clock` as a transport that follows MIDI clock. Needs `mido python-rtmidi`.

**Ableton Link.** `default_patch().transport = Link(bpm=120)`; `ctx.time` then runs on the session's
beat grid and `bpm` follows the session tempo, so every `Clock`, `Seq` and `Env` lands on the shared
beat and `Clock(1)` is the Link beat phase. Needs `aalink`. A transport is any object with `now()`
and `bpm`; `MidiClock` is the other one supplied.

`patches/18_modular.py` is the whole loop: CV in, envelopes from gates, two probes back out, Link
as the clock, with file and synthetic fallbacks so it runs anywhere. Calibrate with a known voltage:
the ES-6/ES-3 are nominally ±10 V at full scale, but check one channel against a meter and set
`volts_per_fs` accordingly. Note the browser export can't see any of this (pages freeze live values
at what they were when exported); the GL window is the live venue.

See [rendering and export](rendering-and-export.md) for recording and frame holds,
and [audio timing](audio-timing.md) for sample-clock behavior and verification.

## Hot reload and video out

[Studio](studio.md) adds a code editor, output thumbnails, controls, connection status and export jobs.
The desktop window supports **1–4** for outputs, **Space** for pause, **S** for a screenshot and
**F** for fullscreen. Its title shows playback, measured display rate, resolution, clock, connections
and rejected edits. Studio starts with managed hardware connections disconnected.


`hydrapy live patch.py` watches the file: save it and the patch re-runs in place. The outputs'
current frames carry over, so feedback keeps its history and the clock keeps running; an edit that
raises prints the traceback and the old patch keeps playing. Managed devices are staged without opening them while the new patch is loaded and compiled.
Only a validated replacement switches connections. If activation fails, the previous device
objects are reopened; any reconnection error is reported. A successful reload can briefly
interrupt hardware streams and restarts stateful sources such as NCA/Physarum. Arbitrary Python
side effects cannot be rolled back. `Param` values, modulation and the chosen output carry over
unless the edit changed their declaration in the code ([details](studio.md#connections-and-reload)).
`close_resources()` closes the current connections from Python. `--no-watch` turns it off.

```python
VideoOut("virtualcam")               # a webcam other software can pick: OBS, Zoom, Resolume, TouchDesigner (needs pyvirtualcam)
VideoOut("udp://127.0.0.1:1234")     # an MPEG-TS stream (also srt://, rtp://, tcp://; rtmp:// as FLV) — an OBS Media Source, VLC
VideoOut("set.mkv")                  # record the set as you play it
```

`VideoOut` reads the displayed picture back each frame and pipes it to pyvirtualcam (the OBS
Virtual Camera on macOS/Windows, v4l2loopback on Linux) or ffmpeg. NDI, Syphon and Spout aren't
built in: bring either route into OBS and publish from there.
