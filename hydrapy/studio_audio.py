"""Four named audio ports on native JACK or PipeWire's JACK implementation.

Only a bounded packet copy runs in the process callback. A producer evaluates
audio against JACK sample positions ahead of that callback; late packets are
discarded by timestamp. This is Python soft realtime, not a hard-RT guarantee.
The encoded ES-40 stream never passes through these listening-audio ports.
"""
from __future__ import annotations

from collections import deque
import math
import threading
import time

import numpy as np

from .live import AudioQueue, Soundtrack
from . import tracker_v3
from .studio_tracker import note_midi
from .tracker_cv import TrackerCV, snapshot as playhead

PORTS = ('optical_L', 'optical_R', 'breaks_L', 'breaks_R')
DEFAULTS = dict(enabled=False, optical_gain=.5, breaks_gain=.4, lead_blocks=2)


def settings(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('Audio settings use enabled, optical_gain, breaks_gain and lead_blocks')
    result = dict(DEFAULTS, **values)
    if type(result['enabled']) is not bool:
        raise ValueError('Audio enabled must be boolean')
    for key in ('optical_gain', 'breaks_gain'):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError('Audio levels must be finite numbers between 0 and 1')
    if type(result['lead_blocks']) is not int or not 2 <= result['lead_blocks'] <= 8:
        raise ValueError('Audio buffer lead must be 2–8 JACK periods')
    return result


def drum(note):
    """The Breaks kit voice for a note, by pitch: C-2 (MIDI 36) kick, D-2 (38) snare, else hat."""
    midi = note_midi(note)
    return 'kick' if midi == 36 else 'snare' if midi == 38 else 'hat'


class BreaksSynth:
    """The tracker Breaks lane as a deterministic kick/snare/hat synth.

    This is the existing synthesized kit, not a sample slicer. Notes select kick
    (C-2), snare (D-2), or hat (other notes, and AIR) by pitch, so C2 is the kick too;
    velocity and mute follow the score.
    Voice envelopes and phase use seconds on the audio clock, not video frames.
    A version-3 score plays lane 1's candidates that fire, ``ply`` repeats included;
    a mute stops new hits but lets one that has started ring out.
    """
    def __init__(self):
        self.timeline = TrackerCV({0: 'clock'}, open_stream=False)
        self.reset()

    def reset(self):
        self.transport = None
        self.row = None
        self.moving = False
        self.voices = []
        self.serial = 0
        self.segment = None
        self.follower = tracker_v3.Follower()

    def render(self, snapshot, start, frames, samplerate):
        out = np.zeros(frames, np.float32)
        if not snapshot or not frames:
            return out
        moving = snapshot['enabled'] and snapshot['playing']
        d = snapshot['document']
        v3 = d['format'] == tracker_v3.FORMAT
        transport = snapshot['transport_revision']
        times = start + np.arange(frames) / samplerate
        live = self.transport is not None and (transport != self.transport or (moving and not self.moving))
        if transport != self.transport or moving != self.moving:
            self.voices.clear()
            self.row = None
        if v3 and (snapshot['transport_revision'], snapshot['start']['tick']) != self.segment:
            # A transport change heard as it starts plays the anchor's row, as a version-2
            # seek does; otherwise (joining, or a segment without a transport change)
            # start from the current tick, so no hit already heard sounds again.
            u = tracker_v3.elapsed(snapshot, times[:1])[0]
            self.segment = (snapshot['transport_revision'], snapshot['start']['tick'])
            self.follower.begin(-7680 if live else math.floor(u), self.segment)
        self.transport, self.moving = transport, moving
        if not moving:
            return out
        if v3:
            self._candidates(snapshot, times)
        else:
            self._rows(d, snapshot, times)
        active = []
        for onset, kind, velocity, seed in self.voices:
            age = np.rint((times - onset) * samplerate) / samplerate
            length = .3 if kind == 'kick' else .17 if kind == 'snare' else .06
            selected = (age >= 0) & (age < length)
            if selected.any():
                t = age[selected]
                if kind == 'kick':
                    k = math.log(44 / 145) / .13
                    early = np.minimum(t, .13)
                    phase = 145 * np.expm1(k * early) / k + 44 * np.maximum(t - .13, 0)
                    signal = np.sin(2 * np.pi * phase) * .95 * np.exp(math.log(.001 / .95) * t / .29)
                else:
                    # Stateless noise indexed by sample age: chunking cannot change the sound.
                    n = np.rint(t * samplerate).astype(np.uint64)
                    def noise(x):
                        x = x * np.uint64(747796405) + np.uint64((seed + 1) * 2891336453 % (2**64))
                        x = ((x >> np.uint64(16)) ^ x) * np.uint64(277803737)
                        return ((x >> np.uint64(16)) & np.uint64(65535)).astype(float) / 32767.5 - 1
                    high = (noise(n + np.uint64(1)) - noise(n)) * .5
                    snare = kind == 'snare'
                    level, decay = (.44, .15) if snare else (.18, .04)
                    signal = high * level * np.exp(math.log(.001 / level) * t / decay)
                out[selected] += (signal * velocity).astype(np.float32)
            if times[-1] - onset < length:
                active.append((onset, kind, velocity, seed))
        self.voices = active[-32:]
        return np.clip(out, -1, 1)

    def _rows(self, d, snapshot, times):
        """Start a voice for each version-1/2 row with a hit on the Breaks lane."""
        if d['mutes'][0]:
            self.voices.clear()
        beat, _ = self.timeline.beats(snapshot, times)
        absolute_rows = np.floor(beat * 4 + 1e-9).astype(int)
        rows = absolute_rows % 256
        changes = np.flatnonzero(np.r_[rows[0] != self.row, rows[1:] != rows[:-1]])
        for index in changes:
            row = int(rows[index])
            hit = d['patterns'][row // 64]['audio'][0][row % 64]
            if hit and hit['vel'] > 0 and not d['mutes'][0]:
                # Reconstruct the onset when attaching in the middle of a row.
                elapsed = max(0., beat[index] * 4 - absolute_rows[index]) * 15 / d['bpm']
                self.voices.append((times[index] - elapsed, drum(hit['note']), hit['vel'] / 127, self.serial))
                self.serial += 1
        self.row = int(rows[-1])

    def evaluated(self, segment):
        """The first tick of version-3 ``segment`` the synth hasn't evaluated, or None."""
        return self.follower.evaluated(segment)

    def _candidates(self, snapshot, times):
        """Start a voice for each version-3 candidate of lane 1 that fires, at its own tick."""
        u = tracker_v3.elapsed(snapshot, times[[0, -1]])
        seconds = 60 / snapshot['document']['bpm'] / tracker_v3.TICKS
        for c in self.follower.advance(snapshot, u[1]):
            if c['lane'] == 0 and c['fires']:
                self.voices.append((times[0] + (c['at'] - u[0]) * seconds, drum(c['note']), c['vel'] / 127, self.serial))
                self.serial += 1
        self.follower.forget(u[0])


class StudioAudio:
    def __init__(self, *, client_factory=None):
        self.config = dict(DEFAULTS)
        self.client_factory = client_factory
        self.client = None
        self.ports = []
        self.sources = ()
        self.source_names = []
        self.snapshot = None
        self.error = ''
        self.xruns = self.underruns = self.late_packets = 0
        self.samplerate = self.blocksize = 0
        self.epoch = 0
        self.anchor = None
        self.packets = deque()
        self.pending = None
        self.thread = None
        self.running = False
        self.synth = BreaksSynth()
        self._patch = None
        self._raw_frame = 0
        self._wrap = 0
        self._first_frame = None

    def attach(self, patch):
        self._patch = patch
        old = self.sources
        tracks = [p for p in patch.probes if isinstance(p, Soundtrack) or
                  (hasattr(p, 'listeners') and hasattr(p, 'sr') and hasattr(p, 'rate'))][:2]
        self.sources = tuple(AudioQueue(track, self.samplerate or 48000) for track in tracks) if self.client is not None else ()
        self.source_names = [getattr(getattr(t, 'output', None), 'name', 'Optical source') for t in tracks]
        for source in old:
            source.close()

    def publish(self, tracker, clock_time, wall, playing):
        self.snapshot = playhead(tracker, clock_time, wall, playing)

    def evaluated(self, segment):
        return self.synth.evaluated(segment)

    def configure(self, values):
        prepared = settings(values)
        restart = bool(prepared['enabled']) != (bool(self.client) and self.running)
        old = self.config
        self.config = prepared
        try:
            if restart:
                if prepared['enabled']:
                    self.start()
                else:
                    self.stop()
            elif not prepared['enabled'] and self.client is not None:
                self.stop()
        except Exception:
            self.config = old
            raise
        return self.describe()

    def start(self):
        self.stop()
        self.error = ''
        try:
            if self.client_factory is None:
                try:
                    import jack
                except ImportError as exc:
                    raise RuntimeError('Install HydraPy with the jack extra: pip install -e ".[jack]"') from exc
                factory = jack.Client
            else:
                factory = self.client_factory
            client = factory('HydraPy', no_start_server=True)
            self.client = client
            self.samplerate, self.blocksize = int(client.samplerate), int(client.blocksize)
            self.ports = [client.outports.register(name) for name in PORTS]
            client.set_process_callback(self._process)
            client.set_samplerate_callback(self._sample_rate)
            client.set_blocksize_callback(self._block_size)
            client.set_xrun_callback(self._xrun)
            client.set_shutdown_callback(self._shutdown)
            if self._patch is not None:
                self.attach(self._patch)
            self.running = True
            self.synth.reset()
            self.thread = threading.Thread(target=self._produce, daemon=True, name='gorgon-audio-producer')
            self.thread.start()
            client.activate()
        except Exception as exc:
            self.error = str(exc)
            self.stop()
            raise ValueError(f'JACK audio could not start: {exc}') from exc

    def _sample_rate(self, samplerate):
        self.samplerate = int(samplerate)
        self.epoch += 1
        self.anchor = None

    def _block_size(self, frames):
        self.blocksize = int(frames)
        self.epoch += 1
        self.anchor = None

    def _xrun(self, delayed):
        self.xruns += 1

    def _shutdown(self, status, reason):
        self.error = 'JACK server stopped: ' + str(reason)
        self.running = False

    def _process(self, frames):
        """No synthesis, rendering, network I/O or waiting for the producer here."""
        arrays = [p.get_array() for p in self.ports]
        for out in arrays:
            out.fill(0)
        raw = int(self.client.last_frame_time)
        if raw < self._raw_frame and self._raw_frame - raw > 2**31:
            self._wrap += 2**32
        self._raw_frame = raw
        at = raw + self._wrap
        wall = time.monotonic() - self.client.frames_since_cycle_start / self.samplerate
        self.anchor = (at, wall, frames, self.epoch)
        if self._first_frame is None:
            self._first_frame = at + frames * self.config['lead_blocks']
        end, covered = at + frames, 0
        current = self.snapshot
        while True:
            if self.pending is None:
                try:
                    self.pending = self.packets.popleft()
                except IndexError:
                    break
            epoch, start, transport, samples = self.pending
            stop = start + len(samples)
            if epoch != self.epoch or stop <= at:
                self.pending = None
                self.late_packets += 1
                continue
            if start >= end:
                break
            first, last = max(start, at), min(stop, end)
            a, b = first - at, last - at
            offset = first - start
            if current and current['playing']:
                for channel in range(4):
                    if channel >= 2 and (not current['enabled'] or transport != current['transport_revision']):
                        continue
                    arrays[channel][a:b] = samples[offset:offset + b - a, channel]
            covered += b - a
            if stop <= end:
                self.pending = None
            else:
                break
        if current and current['playing'] and at >= self._first_frame and covered < frames:
            self.underruns += 1

    def _produce(self):
        next_frame, epoch, old_rate = None, None, None
        try:
            while self.running:
                anchor = self.anchor
                if anchor is None:
                    time.sleep(.001)
                    continue
                frame, wall, count, generation = anchor
                rate = self.samplerate
                if generation != epoch:
                    epoch, next_frame = generation, None
                    self.synth.reset()
                if old_rate != rate:
                    for source in self.sources:
                        if source.samplerate != rate:
                            with source._lock:
                                source.samplerate = rate
                                source._reset()
                    old_rate = rate
                lead = count * self.config['lead_blocks']
                if next_frame is None or next_frame < frame:
                    next_frame = frame + lead
                if next_frame >= frame + lead + count or len(self.packets) >= 8:
                    time.sleep(.001)
                    continue
                snapshot = self.snapshot
                source_time = wall + (next_frame - frame - lead) / rate
                block = np.zeros((count, 4), np.float32)
                sources = self.sources
                for channel in range(2):
                    if sources:
                        source = sources[min(channel, len(sources) - 1)]
                        # A single mono source must be pulled once then duplicated.
                        if channel == 1 and len(sources) == 1:
                            block[:, 1] = block[:, 0]
                        else:
                            block[:, channel] = source.pull(count) * self.config['optical_gain']
                breaks = self.synth.render(snapshot, source_time, count, rate) * self.config['breaks_gain']
                block[:, 2] = block[:, 3] = breaks
                np.nan_to_num(block, copy=False, nan=0., posinf=0., neginf=0.)
                np.clip(block, -1., 1., out=block)
                if generation == self.epoch:
                    revision = snapshot['transport_revision'] if snapshot else -1
                    self.packets.append((generation, next_frame, revision, block))
                next_frame += count
        except Exception as exc:
            self.error = f'Audio producer stopped: {type(exc).__name__}: {exc}'
            self.running = False

    def stop(self):
        self.running = False
        client = self.client
        if client is not None:
            try:
                client.deactivate()
            except Exception as exc:
                self.error = self.error or str(exc)
            finally:
                try:
                    client.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
        self.client = None
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        old, self.sources = self.sources, ()
        for source in old:
            source.close()
        self.packets.clear()
        self.pending = self.anchor = None
        self.ports = []
        self._raw_frame = self._wrap = 0
        self._first_frame = None

    def close(self):
        self.stop()
        for source in self.sources:
            source.close()
        self.sources = ()

    def describe(self):
        return dict(config=dict(self.config), active=bool(self.client) and self.running, error=self.error,
                    client=getattr(self.client, 'name', ''), ports=[p.name for p in self.ports],
                    samplerate=self.samplerate, blocksize=self.blocksize, xruns=self.xruns,
                    underruns=self.underruns, late_packets=self.late_packets,
                    buffer_lead_ms=1000 * self.config['lead_blocks'] * self.blocksize / max(1, self.samplerate),
                    sources=list(self.source_names), rhythm='Synthesized Breaks kit (not sample playback)')
