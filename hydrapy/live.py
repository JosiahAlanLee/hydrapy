"""Live I/O: control voltage in and out (Expert Sleepers ES-6 / ES-3 or any DC-coupled interface),
OSC, MIDI, Ableton Link, and probes that turn pixels back into signals.

Everything here produces ordinary signals, so it plugs into any parameter:

    cv = CvIn(device="ES-8", channels=[8, 9, 10, 11])        # ES-6 arrives as audio inputs
    osc(cv.uni(0).map(5, 60), 0.1, cv.bi(1) * 2).kaleid(cv.uni(2).map(2, 8) + 1)
       .scale(1 + Env(cv.trig(3), 0.02, 0.6)).out(o0)        # a gate on channel 3 retriggers an envelope

    out = CvOut(device="ES-8", channels=[8, 9])               # to the ES-3
    out.send(0, Probe(o0, 0.5, 0.5, 0.1))                     # the brightness of the centre, as a voltage
    out.send(1, Lfo(0.25, "tri", -5, 5))                      # a plain LFO, in volts

    default_patch().transport = Link()                        # beat clock from Ableton Link

Live sources only move in real time (``hydrapy live``); for a finished render, record the CV as a
multichannel WAV and use ``CvIn.from_file`` — frame-accurate, any resolution, no dropouts.

Optional dependencies: sounddevice (CV), python-osc (OSC), mido + python-rtmidi (MIDI), aalink (Link).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from .core import Output, Patch, default_patch
from .signals import Audio, Signal, _num


# -----------------------------------------------------------------------------
# a signal whose value is set from outside
# -----------------------------------------------------------------------------

class LiveValue(Signal):
    """A float set from another thread (audio callback, OSC server, MIDI port), read per frame.

    ``smooth`` is a one-pole lag in seconds; ``lo``/``hi`` clip.  Exports (HTML, Hydra) freeze it at
    its current value, since a browser page can't see your hardware.
    """

    def __init__(self, name: str = "live", default: float = 0.0, lo: Optional[float] = None,
                 hi: Optional[float] = None, smooth: float = 0.0):
        self.name = name
        self._raw = float(default)
        self._out = float(default)
        self._t = None
        self.lo, self.hi, self.smooth = lo, hi, float(smooth)

    def set(self, v: float):
        self._raw = float(v)

    def value(self, ctx):
        v = self._raw
        if self.lo is not None or self.hi is not None:
            v = min(max(v, self.lo if self.lo is not None else v), self.hi if self.hi is not None else v)
        if self.smooth > 0:
            if self._t is None:                 # first read: start from the live value, not the default
                self._out, self._t = v, ctx.time
                return v
            dt = max(ctx.time - self._t, 0.0)
            a = 1.0 - math.exp(-dt / self.smooth) if dt > 0 else 0.0
            self._out += (v - self._out) * a
            self._t = ctx.time
            return self._out
        return v

    def js(self):
        return _num(self._out if self.smooth > 0 else self._raw)

    def __repr__(self):
        return f"<LiveValue {self.name}={self._raw:.3f}>"


class TrigPhase(Signal):
    """Phase 0..1 since the last trigger, over ``length`` seconds (then it sits at 1): feed it to ``Env``."""

    def __init__(self, name: str = "trig", length: float = 1.0):
        self.name = name
        self.length = float(length)
        self.last: Optional[float] = None      # wall-clock time of the last rising edge
        self.count = 0

    def fire(self, when: Optional[float] = None):
        self.last = time.monotonic() if when is None else when
        self.count += 1

    def value(self, ctx):
        if self.last is None:
            return 1.0
        return min((time.monotonic() - self.last) / self.length, 1.0)

    def js(self):
        return "1.0"


# -----------------------------------------------------------------------------
# transports (what ctx.time and ctx.bpm follow in `hydrapy live`)
# -----------------------------------------------------------------------------

class Transport:
    """Wall clock. ``now()`` is seconds since start; ``bpm`` is the patch's."""

    def __init__(self, bpm: float = 30.0):
        self.t0 = time.monotonic()
        self.bpm = float(bpm)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def close(self):
        pass


class Link(Transport):
    """Ableton Link: ``now()`` runs on the session's beat grid (time = beat * 60 / tempo), so every
    Clock, Seq and Env lands on the shared beat, and ``bpm`` follows the session tempo.

    ``pip install aalink``.  ``quantum`` is the bar length in beats.  Reading ``.beat``, ``.phase``
    and ``.peers`` is fine from the render thread.
    """

    def __init__(self, bpm: float = 120.0, quantum: float = 4.0, start_stop_sync: bool = False):
        import asyncio
        try:
            import aalink
        except ImportError as e:
            raise RuntimeError("Link needs the aalink package: pip install aalink") from e
        super().__init__(bpm)
        try:
            self._link = aalink.Link(bpm)
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._link = aalink.Link(bpm, self._loop)
        self._link.quantum = quantum
        self._link.start_stop_sync_enabled = start_stop_sync
        self._link.enabled = True

    @property
    def tempo(self) -> float:
        return float(self._link.tempo)

    @tempo.setter
    def tempo(self, v: float):
        self._link.tempo = float(v)

    @property
    def bpm(self) -> float:
        return float(self._link.tempo)

    @bpm.setter
    def bpm(self, v: float):
        if hasattr(self, "_link"):
            self._link.tempo = float(v)

    @property
    def beat(self) -> float:
        return float(self._link.beat)

    @property
    def phase(self) -> float:
        return float(self._link.phase)

    @property
    def peers(self) -> int:
        return int(self._link.num_peers)

    @property
    def playing(self) -> bool:
        return bool(self._link.playing)

    def now(self) -> float:
        return float(self._link.beat) * 60.0 / max(float(self._link.tempo), 1e-6)

    def close(self):
        self._link.enabled = False


# -----------------------------------------------------------------------------
# OSC
# -----------------------------------------------------------------------------

class OscIn:
    """Receive OSC floats as signals: ``osc_in.value("/synth/freq", default=10, lo=1, hi=100)``.

    A message with several floats fills ``/addr/0``, ``/addr/1``…; bundles are unpacked.  Needs
    python-osc.  VCV Rack (cvOSCcv), TouchDesigner, SuperCollider and Max all speak this.
    """

    def __init__(self, port: int = 57120, host: str = "0.0.0.0"):
        try:
            from pythonosc import dispatcher, osc_server
        except ImportError as e:
            raise RuntimeError("OscIn needs python-osc: pip install python-osc") from e
        self.values: dict[str, LiveValue] = {}
        self.trigs: dict[str, TrigPhase] = {}
        self.latest: dict[str, list] = {}
        d = dispatcher.Dispatcher()
        d.set_default_handler(self._handle)
        self.server = osc_server.ThreadingOSCUDPServer((host, port), d)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _handle(self, address, *args):
        self.latest[address] = list(args)
        nums = [a for a in args if isinstance(a, (int, float))]
        if address in self.trigs:
            self.trigs[address].fire()
        if not nums:
            return
        if address in self.values:
            self.values[address].set(nums[0])
        for i, v in enumerate(nums):
            key = f"{address}/{i}"
            if key in self.values:
                self.values[key].set(v)

    def value(self, address: str, default: float = 0.0, lo: Optional[float] = None, hi: Optional[float] = None,
              smooth: float = 0.0) -> LiveValue:
        if address not in self.values:
            self.values[address] = LiveValue(address, default, lo, hi, smooth)
        return self.values[address]

    def trig(self, address: str, length: float = 1.0) -> TrigPhase:
        """Any message to ``address`` fires a trigger; the result is a phase for ``Env``."""
        if address not in self.trigs:
            self.trigs[address] = TrigPhase(address, length)
        return self.trigs[address]

    def close(self):
        self.server.shutdown()


class OscOut:
    """Send signals (or probes) as OSC floats every frame: ``out.send("/hydrapy/centre", Probe(o0))``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 57121):
        try:
            from pythonosc import udp_client
        except ImportError as e:
            raise RuntimeError("OscOut needs python-osc: pip install python-osc") from e
        self.client = udp_client.SimpleUDPClient(host, port)
        self.routes: list[tuple[str, Signal]] = []
        register_hook(self.tick)

    def send(self, address: str, signal) -> "OscOut":
        from .signals import as_signal
        self.routes.append((address, as_signal(signal)))
        return self

    def tick(self, ctx, renderer=None):
        for address, sig in self.routes:
            v = sig(ctx)
            self.client.send_message(address, float(v) if math.isfinite(v) else 0.0)


# -----------------------------------------------------------------------------
# MIDI
# -----------------------------------------------------------------------------

class Midi:
    """MIDI in as signals: ``m.cc(74)`` 0..1, ``m.note(60)`` velocity 0..1 while held, ``m.gate(60)``
    a trigger phase, ``m.clock`` a transport following MIDI clock.  Needs mido + python-rtmidi;
    ``port`` is a name or a substring of one (``Midi.ports()`` lists them).
    """

    def __init__(self, port: Optional[str] = None, channel: Optional[int] = None):
        try:
            import mido
        except ImportError as e:
            raise RuntimeError("Midi needs mido and python-rtmidi: pip install mido python-rtmidi") from e
        names = mido.get_input_names()
        if port is None:
            if not names:
                raise RuntimeError("no MIDI input ports")
            name = names[0]
        else:
            matches = [n for n in names if port.lower() in n.lower()]
            if not matches:
                raise RuntimeError(f"no MIDI input matching {port!r}; have {names}")
            name = matches[0]
        self.channel = channel
        self._cc: dict[int, LiveValue] = {}
        self._note: dict[int, LiveValue] = {}
        self._gate: dict[int, TrigPhase] = {}
        self.any_note = TrigPhase("note", 1.0)
        self.last_note = LiveValue("last_note", 60.0)
        self.bend = LiveValue("bend", 0.0, -1.0, 1.0)
        self.clock = MidiClock()
        self.port = mido.open_input(name, callback=self._handle)

    @staticmethod
    def ports() -> list[str]:
        import mido
        return mido.get_input_names()

    def _handle(self, msg):
        if self.channel is not None and getattr(msg, "channel", None) not in (None, self.channel):
            return
        if msg.type == "control_change":
            if msg.control in self._cc:
                self._cc[msg.control].set(msg.value / 127.0)
        elif msg.type == "note_on" and msg.velocity > 0:
            self.last_note.set(msg.note)
            self.any_note.fire()
            if msg.note in self._note:
                self._note[msg.note].set(msg.velocity / 127.0)
            if msg.note in self._gate:
                self._gate[msg.note].fire()
        elif msg.type in ("note_off", "note_on"):
            if msg.note in self._note:
                self._note[msg.note].set(0.0)
        elif msg.type == "pitchwheel":
            self.bend.set(msg.pitch / 8192.0)
        elif msg.type in ("clock", "start", "stop", "continue", "songpos"):
            self.clock.handle(msg)

    def cc(self, number: int, default: float = 0.0, smooth: float = 0.0) -> LiveValue:
        if number not in self._cc:
            self._cc[number] = LiveValue(f"cc{number}", default, 0.0, 1.0, smooth)
        return self._cc[number]

    def note(self, number: int) -> LiveValue:
        if number not in self._note:
            self._note[number] = LiveValue(f"note{number}", 0.0, 0.0, 1.0)
        return self._note[number]

    def gate(self, number: int, length: float = 1.0) -> TrigPhase:
        if number not in self._gate:
            self._gate[number] = TrigPhase(f"gate{number}", length)
        return self._gate[number]

    def close(self):
        self.port.close()


class MidiClock(Transport):
    """A transport driven by MIDI clock (24 ppqn): beats counted from the last start/songpos."""

    def __init__(self):
        super().__init__(120.0)
        self.ticks = 0
        self._last_tick: Optional[float] = None
        self._tick_times: list[float] = []
        self.running = False

    def handle(self, msg):
        now = time.monotonic()
        if msg.type == "clock":
            self.ticks += 1
            self._tick_times.append(now)
            if len(self._tick_times) > 48:
                del self._tick_times[:-48]
            if len(self._tick_times) >= 24:
                span = self._tick_times[-1] - self._tick_times[0]
                if span > 0:
                    self.bpm = 60.0 * (len(self._tick_times) - 1) / 24.0 / span
        elif msg.type in ("start",):
            self.ticks = 0
            self.running = True
        elif msg.type == "songpos":
            self.ticks = msg.pos * 6
        elif msg.type == "stop":
            self.running = False
        elif msg.type == "continue":
            self.running = True

    def now(self) -> float:
        return (self.ticks / 24.0) * 60.0 / max(self.bpm, 1e-6)


# -----------------------------------------------------------------------------
# control voltage
# -----------------------------------------------------------------------------

def audio_devices() -> str:
    """The audio devices sounddevice can see (what to pass as ``device``)."""
    import sounddevice as sd
    return str(sd.query_devices())


class CvIn:
    """Control voltages as signals, from a DC-coupled interface (an ES-6 feeding ADAT, an ES-8, a
    Motu/RME with DC inputs) or from a multichannel WAV recorded from one.

    ``channels`` are zero-based interface input channels.  ``volts_per_fs`` is the voltage at full
    scale; the Expert Sleepers modules are ±10 V, so 10.  Signals:

    * ``cv[k]`` volts;  ``uni(k)`` 0..1 for 0..+10 V;  ``bi(k)`` -1..1 for ±5 V;  ``volt_oct(k)``
      Hz from 1 V/oct (C4 at 0 V by default)
    * ``gate(k, threshold)`` 0/1;  ``trig(k, threshold, length)`` phase since the last rising edge
    * ``audio(k)`` an ``Audio`` analyser on an audio-rate channel (``.fft[i]``, ``.vol``, ``.flux``)
    """

    def __init__(self, device=None, channels: Sequence[int] = (0, 1), samplerate: Optional[int] = None,
                 blocksize: int = 256, volts_per_fs: float = 10.0, smooth: float = 0.005, _file=None):
        self.channels = list(channels)
        self.volts_per_fs = float(volts_per_fs)
        self.smooth = float(smooth)
        self.cv = [LiveValue(f"cv{k}", 0.0, smooth=smooth) for k in self.channels]
        self._gate_state = [False] * len(self.channels)
        self._trigs: dict[int, list[tuple[float, TrigPhase]]] = {}
        self._audio: dict[int, Audio] = {}
        self._file = _file
        self.stream = None
        self.samplerate = samplerate
        if _file is None:
            import sounddevice as sd
            if samplerate is None:
                dev = sd.query_devices(device, "input")
                samplerate = int(dev["default_samplerate"])
            self.samplerate = int(samplerate)
            self.stream = sd.InputStream(device=device, channels=max(self.channels) + 1, samplerate=self.samplerate,
                                         blocksize=blocksize, dtype="float32", callback=self._callback)
            self.stream.start()

    # -- live ---------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        self.process(indata, time.monotonic())

    def process(self, block: np.ndarray, now: Optional[float] = None):
        """Feed a (frames, channels) float32 block (the audio callback does this)."""
        now = time.monotonic() if now is None else now
        for i, k in enumerate(self.channels):
            col = block[:, k]
            v = float(col.mean()) * self.volts_per_fs
            self.cv[i].set(v)
            if i in self._trigs:
                for threshold, trig in self._trigs[i]:
                    above = col * self.volts_per_fs > threshold
                    if above.any() and (not self._gate_state[i] or (above & ~np.roll(above, 1)).sum() > 1):
                        trig.fire(now)
                    self._gate_state[i] = bool(above[-1])
            if i in self._audio:
                self._audio[i].feed(col)

    # -- signals -----------------------------------------------------------------
    def uni(self, k: int, smooth: Optional[float] = None) -> Signal:
        return (self.cv[k] / 10.0).clip(0.0, 1.0)

    def bi(self, k: int) -> Signal:
        return (self.cv[k] / 5.0).clip(-1.0, 1.0)

    def volt_oct(self, k: int, base_hz: float = 261.63) -> Signal:
        from .signals import Fn
        cv = self.cv[k]
        return Fn(lambda ctx: base_hz * 2.0 ** cv(ctx))

    def gate(self, k: int, threshold: float = 1.0) -> Signal:
        from .signals import Fn
        cv = self.cv[k]
        return Fn(lambda ctx: 1.0 if cv(ctx) > threshold else 0.0)

    def trig(self, k: int, threshold: float = 1.0, length: float = 1.0) -> TrigPhase:
        t = TrigPhase(f"trig{k}", length)
        self._trigs.setdefault(k, []).append((float(threshold), t))
        return t

    def audio(self, k: int, bins: int = 4) -> Audio:
        if k not in self._audio:
            a = Audio(None, self.samplerate or 44100, bins=bins)
            self._audio[k] = a
        return self._audio[k]

    # -- offline ---------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str, channels: Optional[Sequence[int]] = None, volts_per_fs: float = 10.0,
                  window: float = 0.002) -> "CvFile":
        return CvFile(path, channels, volts_per_fs, window)

    def close(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close()


class CvFile(CvIn):
    """CV recorded as a multichannel WAV (or anything ffmpeg decodes): the same signals, sampled at
    the frame's time, so an offline render follows a recorded performance exactly."""

    def __init__(self, path: str, channels: Optional[Sequence[int]] = None, volts_per_fs: float = 10.0,
                 window: float = 0.002):
        data, sr = load_channels(path)
        self.data = data            # (frames, channels) float32
        self.sr = sr
        chans = list(channels) if channels is not None else list(range(data.shape[1]))
        super().__init__(channels=chans, samplerate=sr, volts_per_fs=volts_per_fs, smooth=0.0, _file=path)
        self.window = float(window)
        self.cv = [FileCv(self, i) for i in range(len(chans))]

    def sample(self, k: int, t: float, window: Optional[float] = None) -> float:
        w = self.window if window is None else window
        c = self.channels[k]
        n = self.data.shape[0]
        i0 = int((t - w / 2) * self.sr)
        i1 = int((t + w / 2) * self.sr) + 1
        i0, i1 = max(i0, 0), min(max(i1, i0 + 1), n)
        if i0 >= n:
            return 0.0
        return float(self.data[i0:i1, c].mean()) * self.volts_per_fs

    def trig(self, k: int, threshold: float = 1.0, length: float = 1.0) -> Signal:
        from .signals import Fn
        c = self.channels[k]
        above = self.data[:, c] * self.volts_per_fs > threshold
        edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        edge_t = edges / self.sr

        def phase(ctx):
            i = np.searchsorted(edge_t, ctx.time, side="right") - 1
            if i < 0:
                return 1.0
            return min((ctx.time - edge_t[i]) / length, 1.0)
        return Fn(phase)

    def gate(self, k: int, threshold: float = 1.0) -> Signal:
        from .signals import Fn
        return Fn(lambda ctx: 1.0 if self.sample(k, ctx.time) > threshold else 0.0)

    def audio(self, k: int, bins: int = 4) -> Audio:
        if k not in self._audio:
            self._audio[k] = Audio(self.data[:, self.channels[k]].copy(), self.sr, bins=bins)
        return self._audio[k]

    def duration(self) -> float:
        return self.data.shape[0] / self.sr

    def close(self):
        pass


class FileCv(Signal):
    def __init__(self, f: CvFile, k: int):
        self.f, self.k = f, k

    def value(self, ctx):
        return self.f.sample(self.k, ctx.time)

    def js(self):
        return "0.0"


def load_channels(path: str):
    """(frames, channels) float32 in -1..1, and the sample rate. WAV natively, anything else via ffmpeg."""
    import shutil, subprocess, wave
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            a = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) ^ 0x800000) - 0x800000
            a = a.astype(np.float32) / 8388608.0
        elif sw == 1:
            a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise ValueError("unsupported WAV sample width")
        return a.reshape(-1, ch), sr
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg is needed to decode non-WAV audio")
    probe = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=channels,sample_rate", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    sr, ch = 44100, 2
    if probe:
        parts = probe.split(",")
        sr, ch = int(parts[0]), int(parts[1])
    out = subprocess.run([ff, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, ch).copy(), sr


class CvOut:
    """Signals out as control voltages: ``out.send(k, signal)`` sets interface output channel
    ``channels[k]`` to the signal's value in volts, every frame, ramped across the audio block so the
    modular sees a smooth voltage.  The ES-3 is ±10 V at full scale (``volts_per_fs=10``).

    ``CvOut(file="cv.wav", ...)`` records the same voltages to a multichannel WAV instead (call
    ``close()`` when the render is done) to play back from a DAW into the ES-3 alongside the video.
    """

    def __init__(self, device=None, channels: Sequence[int] = (0, 1), samplerate: Optional[int] = None,
                 blocksize: int = 256, volts_per_fs: float = 10.0, file: Optional[str] = None, fps: float = 30.0):
        self.channels = list(channels)
        self.volts_per_fs = float(volts_per_fs)
        self.routes: dict[int, Signal] = {}
        self._target = np.zeros(len(self.channels), dtype=np.float32)
        self._current = np.zeros(len(self.channels), dtype=np.float32)
        self.file = file
        self.fps = float(fps)
        self._frames: list[np.ndarray] = []
        self.stream = None
        self.samplerate = int(samplerate or 48000)
        if file is None:
            import sounddevice as sd
            if samplerate is None:
                dev = sd.query_devices(device, "output")
                self.samplerate = int(dev["default_samplerate"])
            self.stream = sd.OutputStream(device=device, channels=max(self.channels) + 1, samplerate=self.samplerate,
                                          blocksize=blocksize, dtype="float32", callback=self._callback)
            self.stream.start()
        register_hook(self.tick)

    def send(self, k: int, signal) -> "CvOut":
        from .signals import as_signal
        self.routes[k] = as_signal(signal)
        return self

    def tick(self, ctx, renderer=None):
        for k, sig in self.routes.items():
            v = sig(ctx)
            self._target[k] = (float(v) if math.isfinite(v) else 0.0) / self.volts_per_fs
        if self.file is not None:
            self._frames.append(self._target.copy())

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0.0)
        ramp = np.linspace(0.0, 1.0, frames, endpoint=False, dtype=np.float32)[:, None]
        block = self._current[None, :] + (self._target - self._current)[None, :] * ramp
        for i, ch in enumerate(self.channels):
            outdata[:, ch] = np.clip(block[:, i], -1.0, 1.0)
        self._current[:] = self._target

    def close(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close()
        if self.file is not None and self._frames:
            import wave
            per = max(int(round(self.samplerate / self.fps)), 1)
            frames = np.array(self._frames, dtype=np.float32)                 # (nframes, nchan)
            prev = np.vstack([frames[:1], frames[:-1]])
            ramp = np.linspace(0.0, 1.0, per, endpoint=False, dtype=np.float32)
            sig = prev[:, None, :] + (frames - prev)[:, None, :] * ramp[None, :, None]   # (nframes, per, nchan)
            sig = np.clip(sig.reshape(-1, frames.shape[1]), -1.0, 1.0)
            nchan = max(self.channels) + 1
            out = np.zeros((sig.shape[0], nchan), dtype=np.float32)
            for i, ch in enumerate(self.channels):
                out[:, ch] = sig[:, i]
            with wave.open(self.file, "wb") as w:
                w.setnchannels(nchan); w.setsampwidth(2); w.setframerate(self.samplerate)
                w.writeframes((out * 32767).astype("<i2").tobytes())
            self._frames = []


# -----------------------------------------------------------------------------
# probes: pixels back into signals
# -----------------------------------------------------------------------------

class Probe(Signal):
    """The mean of a region of an output, read back after each frame: a pixel as a voltage.

    ``x``/``y`` 0..1 (y up), ``size`` as a fraction of the height; ``channel`` 'luma', 'r', 'g', 'b'
    or 'a'; ``lo``/``hi`` rescale 0..1 to volts or whatever you need.  One frame of latency: the
    value drives the *next* frame (and the voltage leaving the ES-3).
    """

    def __init__(self, output: Optional[Output] = None, x: float = 0.5, y: float = 0.5, size: float = 0.05,
                 channel: str = "luma", lo: float = 0.0, hi: float = 1.0, smooth: float = 0.0, patch: Optional[Patch] = None):
        p = patch or default_patch()
        self.output = output or p.o0
        self.x, self.y, self.size, self.channel = float(x), float(y), float(size), channel
        self.lo, self.hi = float(lo), float(hi)
        self.live = LiveValue(f"probe({self.output.name})", 0.0, smooth=smooth)
        p.probes.append(self)

    def region(self, width: int, height: int):
        h = max(int(self.size * height), 1)
        w = h
        x0 = int(self.x * width - w / 2)
        y0 = int(self.y * height - h / 2)
        x0 = min(max(x0, 0), max(width - w, 0))
        y0 = min(max(y0, 0), max(height - h, 0))
        return x0, y0, w, h

    def update(self, rgba: np.ndarray):
        """Feed the region's pixels, (h, w, 4) float, any row order."""
        if self.channel == "luma":
            v = float((rgba[..., 0] * 0.2125 + rgba[..., 1] * 0.7154 + rgba[..., 2] * 0.0721).mean())
        else:
            v = float(rgba[..., "rgba".index(self.channel)].mean())
        self.live.set(v)

    def value(self, ctx):
        return self.lo + (self.hi - self.lo) * self.live(ctx)

    def js(self):
        return _num(self.lo + (self.hi - self.lo) * self.live._raw)


class Soundtrack(Probe):
    """An optical soundtrack the patch draws for itself.

    A vertical strip of an output (``x`` across, ``width`` of the width) is read every frame, top
    to bottom, and its luminance is the waveform for the next 1/``fps`` of a second — variable-density
    sound-on-film, the way Fischinger's *Ornament Sound* and the Soviet graphical-sound films
    (Avraamov, Sholpo's Variophone, Yankovsky) drew their music onto the track.  Horizontal stripes
    at *n* cycles per frame height sound at *n × fps* Hz: ``osc(10).rotate(1.5708)`` at 24 fps is a
    240 Hz tone; a stripe pattern scrolling up is a continuous waveform, a still picture repeats
    once a frame.  ``ac`` removes the DC (a one-pole high-pass at 20 Hz), ``gain`` scales ±1.

    Render, then ``wav(path)``, or ``save_mp4(frames, path, fps, audio=track)``.  The HTML export
    plays it live (``readPixels`` of the strip into a ScriptProcessor); the GL window is silent
    unless you pipe ``samples`` somewhere yourself.  As a Probe it is also a signal: the strip's
    mean brightness, one frame late.
    """

    def __init__(self, output: Optional[Output] = None, x: float = 0.98, width: float = 0.01, fps: float = 24.0,
                 sr: int = 44100, gain: float = 1.0, ac: bool = True, patch: Optional[Patch] = None):
        super().__init__(output, x=x, y=0.5, size=1.0, channel="luma", patch=patch)
        self.width, self.fps, self.sr, self.gain, self.ac = float(width), float(fps), int(sr), float(gain), bool(ac)
        self.chunks: list[np.ndarray] = []
        self._hp_x = 0.0
        self._hp_y = 0.0
        self._hp_a = math.exp(-2.0 * math.pi * 20.0 / self.sr)

    def region(self, width: int, height: int):
        w = max(int(round(self.width * width)), 1)
        x0 = int(self.x * width - w / 2)
        x0 = min(max(x0, 0), max(width - w, 0))
        return x0, 0, w, height

    def update(self, rgba: np.ndarray):
        col = (rgba[..., 0] * 0.2125 + rgba[..., 1] * 0.7154 + rgba[..., 2] * 0.0721).mean(axis=1)[::-1]  # top to bottom
        self.live.set(float(col.mean()))
        self.chunks.append(self._frame_audio(np.asarray(col, dtype=np.float64)))

    def _frame_audio(self, col: np.ndarray) -> np.ndarray:
        n = int(round(self.sr / self.fps))
        h = len(col)
        x = (col - 0.5) * 2.0 * self.gain
        y = np.interp((np.arange(n) + 0.5) * h / n - 0.5, np.arange(h), x)
        if self.ac:
            a, px, py = self._hp_a, self._hp_x, self._hp_y
            out = np.empty(n)
            for i in range(n):          # one-pole high-pass, state carried between frames
                py = y[i] - px + a * py
                px = y[i]
                out[i] = py
            self._hp_x, self._hp_y = px, py
            y = out
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    @property
    def duration(self) -> float:
        return sum(len(c) for c in self.chunks) / self.sr

    def clear(self):
        self.chunks = []
        self._hp_x = self._hp_y = 0.0

    def wav(self, path: str) -> str:
        import wave
        data = (np.clip(self.samples, -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(data)
        return path

    def spec(self) -> dict:
        return {"output": self.output.name, "x": self.x, "width": self.width, "fps": self.fps, "gain": self.gain, "ac": self.ac}


# -----------------------------------------------------------------------------
# per-frame hooks (CvOut.tick, OscOut.tick, your own)
# -----------------------------------------------------------------------------

_HOOKS: list[Callable] = []


def register_hook(fn: Callable):
    """``fn(ctx, renderer)`` runs after every frame in ``hydrapy live`` and the GL/CPU renderers."""
    if fn not in _HOOKS:
        _HOOKS.append(fn)
    return fn


def run_hooks(ctx, renderer=None):
    for fn in list(_HOOKS):
        try:
            fn(ctx, renderer)
        except Exception as e:  # a failing hook shouldn't stop the frames
            import sys
            print(f"hydrapy hook {getattr(fn, '__qualname__', fn)!r} failed: {e}", file=sys.stderr)
            _HOOKS.remove(fn)


def clear_hooks():
    _HOOKS.clear()
