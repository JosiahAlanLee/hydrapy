"""Gorgon's tracker as control voltage: clock, reset, pitched voices and automation lanes.

The tracker is the master clock. Every output sample is evaluated in the audio
callback from the render thread's latest playhead snapshot, so clock pulses,
gates and held automation steps land on the exact sample, not on a video frame.
The video side is unchanged: it keeps evaluating the same score per frame.

Typical use is an interface's ADAT output into an Expert Sleepers ES-3, which is
DC coupled and ±10 V at full scale (``volts_per_fs=10``). Channels are interface
channel numbers counted from 0, as ``hydrapy devices`` and ``CvOut`` use them.

Signals a channel can carry:

``clock``           pulses at ``ppqn`` per quarter note (default 4 = sixteenths), 50% duty
``reset``           a trigger on start/resume/seek and at the score/pattern loop boundary
``run``             high while the tracker is playing
``gate1``-``gate6`` the audio lanes: high for ``gate`` of a row when a hit is there,
                    kept whole when a seek or start is heard late (``_landing_gate``),
                    and retriggered (1 ms low) when a seek interrupts an open gate
``pitch1``-``pitch6`` held 1 V/oct pitch, C4 = 0 V by default, with key transposition
``vel1``-``vel6``   the last hit's velocity on that lane, 0..``high`` volts, held
``z`` ``roll`` ``feedback``  the automation lanes, scaled to their volt ranges

Lanes 5 and 6 exist in version-3 scores (Gorgon's ``hydrapy-tracker-3``); with an
older score they stay silent. A version-3 score plays ``tracker_v3.evaluate()``'s
candidates: a ``ply`` sub-step gets its own gate, and conditions, transforms and
mutes are judged once, at each candidate's tick.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import tracker_v3
from .studio_tracker import LANES, note_midi

SIGNALS = ('clock', 'reset', 'run', *(f'gate{i}' for i in range(1, 7)), *(f'vel{i}' for i in range(1, 7)), *(f'pitch{i}' for i in range(1, 7)), *LANES)
# The ES-3 has eight outputs. Feedback, run and velocities are opt-in.
DEFAULT_SIGNALS = ('clock', 'reset', 'gate1', 'gate2', 'gate3', 'gate4', 'z', 'roll')
# Volts at each lane's minimum and maximum. Roll is bipolar, the others unipolar.
LANE_VOLTS = {'z': (0., 5.), 'roll': (-5., 5.), 'feedback': (0., 5.)}


def parse_map(text):
    """``"8=clock,9=reset,10=gate1"`` → ``{8: 'clock', 9: 'reset', 10: 'gate1'}``.

    A bare number (``"8"``) starts the default eight signals at that channel.
    """
    text = str(text).strip()
    if text.isdigit():
        first = int(text)
        return {first + i: name for i, name in enumerate(DEFAULT_SIGNALS)}
    mapping = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        channel, sep, name = part.partition('=')
        channel, name = channel.strip(), name.strip()
        if not sep or not channel.isdigit() or name not in SIGNALS:
            raise ValueError(f'CV map entries look like 8=clock; signals are {", ".join(SIGNALS)}')
        if int(channel) in mapping:
            raise ValueError(f'Channel {channel} is mapped twice')
        mapping[int(channel)] = name
    if not mapping:
        raise ValueError('The CV map is empty')
    return mapping


def lane_values(pattern, lane, rows):
    """``value_at`` for an array of fractional rows (same hold/linear/smooth rules)."""
    keys = pattern['keys'][lane]
    rows = np.asarray(rows, dtype=np.float64)
    if not keys:
        return np.full(rows.shape, LANES[lane][2], dtype=np.float64)
    r = np.array([k['r'] for k in keys], dtype=np.float64)
    v = np.array([k['v'] for k in keys], dtype=np.float64)
    if len(keys) == 1:
        return np.full(rows.shape, v[0])
    # Index of the key at or before each row, matching value_at's ``row < b['r']``.
    i = np.clip(np.searchsorted(r, rows, side='right') - 1, 0, len(keys) - 2)
    t = np.clip((rows - r[i]) / (r[i + 1] - r[i]), 0., 1.)
    ease = pattern['easing'][lane]
    if ease == 'hold':
        t = np.where(rows >= r[i + 1], 1., 0.)
    elif ease == 'smooth':
        t = t * t * (3 - 2 * t)
    out = v[i] + (v[i + 1] - v[i]) * t
    out = np.where(rows <= r[0], v[0], out)
    return np.where(rows >= r[-1], v[-1], out)


def snapshot(tracker, clock_time, wall, playing):
    """The playhead a follower reads, at monotonic ``wall``; version 3 adds its anchor, ticks, FILL and bus."""
    result = dict(document=tracker.document, enabled=tracker.enabled, anchor_beat=tracker.anchor_beat,
                  anchor_time=tracker.anchor_time, clock_time=clock_time, wall=wall, playing=bool(playing),
                  transport_revision=tracker.transport_revision)
    if tracker.document['format'] == tracker_v3.FORMAT:
        result.update(start=tracker.start, elapsed=tracker.elapsed, fill=tracker.fill, bus=tracker.bus,
                      layers=getattr(tracker, 'layers', None))
    return result


class TrackerCV:
    """Render the tracker score as voltages.

    ``publish(...)`` hands over the playhead from the render thread; the audio
    callback (or ``render`` in tests and offline use) reads only that snapshot.
    ``offset_ms`` moves the voltages earlier (positive) or later (negative)
    relative to the playhead, for audio/video compensation.
    """

    def __init__(self, mapping, device=None, samplerate=None, blocksize=256, volts_per_fs=10.,
                 ppqn=4, gate=.5, high=5., reset_ms=5., offset_ms=0., pitch_zero=60, open_stream=True,
                 backend='portaudio'):
        if backend not in ('portaudio', 'jack'):
            raise ValueError('CV backend must be portaudio or jack')
        self.backend = backend
        self.mapping = dict(sorted(mapping.items()))
        if not self.mapping or any(type(c) is not int or c < 0 for c in self.mapping) or any(s not in SIGNALS for s in self.mapping.values()):
            raise ValueError('CV channels are non-negative integers mapped to known signals')
        for value, label, low, top in ((ppqn, 'ppqn', 1, 96), (gate, 'Gate length', .01, 1.),
                                       (high, 'Gate voltage', .1, 10.), (reset_ms, 'Reset trigger', .5, 100.),
                                       (pitch_zero, 'Zero-volt MIDI note', 0, 127),
                                       (volts_per_fs, 'Volts per full scale', .1, 100.), (offset_ms, 'Offset', -500., 500.)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= top:
                raise ValueError(f'{label} must be between {low} and {top}')
        self.ppqn, self.gate, self.high = float(ppqn), float(gate), float(high)
        self.reset_s, self.offset_s, self.volts_per_fs = reset_ms / 1000, offset_ms / 1000, float(volts_per_fs)
        self.samplerate = int(samplerate or 48000)
        if self.samplerate <= 0 or type(blocksize) is not int or blocksize < 24:
            raise ValueError('Use a positive sample rate and an audio block of at least 24 frames')
        self.snapshot = None
        self._velocity = np.zeros(6)
        self._pitch = np.zeros(6)
        self.pitch_zero = pitch_zero
        self._transport = None
        self._segment = None            # version 3: (transport_revision, anchor tick)
        self._moving = False
        self._reset_remaining = 0
        self._landing_remaining = 0
        self._landing_row = None
        self._stood = None
        self._gate_was = np.zeros(6, bool)
        self._retrigger = np.zeros(6, int)
        self._follower = tracker_v3.Follower()
        self._landing_lanes = np.zeros(6, int)        # version 3: landing-gate samples left, per lane
        self._landing_span = [(0, 0)] * 6
        self._silencing = False
        self._silenced = False
        self._silence_frames = 1
        self._silence_sent = 0
        self.status = ''
        self.underflows = 0
        self._stream_offset = None
        self.stream = None
        self.error = ''
        if open_stream:
            self._open_stream(device, samplerate, blocksize)

    def _open_stream(self, device, samplerate, blocksize, *, channels=None, dtype='float32'):
        if self.backend == 'jack':
            from .jack_cv import JackCVStream
            self.stream = JackCVStream(self, device, samplerate, channels or max(self.mapping) + 1, dtype)
            try:
                self.stream.start()
            except BaseException:
                self.stream.close()
                self.stream = None
                raise
            return
        import sounddevice as sd
        if samplerate is None:
            self.samplerate = int(sd.query_devices(device, 'output')['default_samplerate'])
        self.stream = sd.OutputStream(device=device, channels=channels or max(self.mapping) + 1,
                                      samplerate=self.samplerate, blocksize=blocksize, dtype=dtype,
                                      clip_off=True, dither_off=True, callback=self._callback)
        try:
            self.stream.start()
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def describe(self):
        return dict(map={str(k): v for k, v in self.mapping.items()}, samplerate=self.samplerate,
                    backend=self.backend, ports=[p.name for p in getattr(self.stream, 'ports', [])],
                    error=self.error, status=self.status, underflows=self.underflows)

    # -- render thread --------------------------------------------------------
    def publish(self, tracker, clock_time, wall, playing):
        """Called at render boundaries: the playhead at monotonic ``wall``."""
        published = snapshot(tracker, clock_time, wall, playing)
        if published['document']['format'] != tracker_v3.FORMAT:
            published['cells'] = self._cells(tracker.document)
        self.snapshot = published                   # whole, for the audio callback

    def evaluated(self, segment):
        """The first tick of version-3 ``segment`` this output has not evaluated, or None."""
        return self._follower.evaluated(segment)

    # -- evaluation -----------------------------------------------------------
    def _cells(self, document):
        """Hit presence, velocity and pitch as (pattern, lane, row) arrays, cached per score."""
        cached = getattr(self, '_cell_cache', None)
        if cached is not None and cached[0] is document:
            return cached[1:]
        present = np.zeros((4, 4, 64), dtype=bool)
        velocity = np.zeros((4, 4, 64))
        pitch = np.full((4, 4, 64), np.nan)
        for p, pattern in enumerate(document['patterns']):
            for lane, cells in enumerate(pattern['audio']):
                for r, hit in enumerate(cells):
                    if hit is not None:
                        present[p, lane, r], velocity[p, lane, r] = hit['vel'] > 0, hit['vel']
                        midi = note_midi(hit['note'])
                        if midi is not None:
                            pitch[p, lane, r] = (midi + document['key'] - 2 - self.pitch_zero) / 12
        self._cell_cache = (document, present, velocity, pitch)
        return present, velocity, pitch

    def beats(self, snapshot, monotonic):
        """Tracker beats (0..64) and loop start at an array of monotonic times."""
        d = snapshot['document']
        playing = snapshot['playing']
        clock = snapshot['clock_time'] + np.where(playing, monotonic - snapshot['wall'], 0.)
        elapsed = clock - snapshot['anchor_time'] if snapshot['enabled'] else np.zeros_like(monotonic)
        raw = snapshot['anchor_beat'] + elapsed * d['bpm'] / 60
        start, length = (d['pattern'] * 16, 16) if d['loop'] else (0, 64)
        beat = start + np.mod(raw - start, length)
        # Guard float wrap: x % length can round to exactly length.
        return np.minimum(beat, np.nextafter(start + length, 0)), start

    def render(self, frames, start, snapshot=None):
        """``frames`` samples of voltages (in volts) from monotonic time ``start``.

        Returns an array shaped ``(frames, len(mapping))`` in mapping order.
        """
        snapshot = snapshot or self.snapshot
        out = np.zeros((frames, len(self.mapping)))
        if snapshot is None or not frames:
            return out
        if snapshot['document']['format'] == tracker_v3.FORMAT:
            return self._render_v3(frames, start, snapshot, out)
        d = snapshot['document']
        moving = snapshot['playing'] and snapshot['enabled']
        transport = snapshot['transport_revision']
        times = start + self.offset_s + np.arange(frames) / self.samplerate
        beat, loop_start = self.beats(snapshot, times)
        pattern_index = (beat // 16).astype(int)
        row_pos = np.mod(beat, 16) * 4                      # fractional sixteenth row, 0..64
        row = np.minimum(row_pos.astype(int), 63)
        row_phase = row_pos - row
        changed = transport != self._transport
        if changed or (moving and not self._moving):
            self._reset_remaining = round(self.reset_s * self.samplerate) if moving else 0
            self._velocity.fill(0)
            self._pitch.fill(0)
            self._landing_remaining = 0
            # A gate the change interrupts closes for 1 ms, so a hit it lands on rises again.
            self._retrigger = np.where(self._gate_was & moving, round(.001 * self.samplerate), 0)
            if moving:
                landing = snapshot['anchor_beat'] if changed or self._stood is None else self._stood
                self._landing_gate(landing, float(beat[0]), d['bpm'])
        if not moving:
            self._reset_remaining = self._landing_remaining = 0
            self._retrigger[:] = 0
            self._stood = float(beat[-1])
        self._transport, self._moving = transport, moving
        reset = np.arange(frames) < self._reset_remaining
        self._reset_remaining = max(0, self._reset_remaining - frames)
        landing = np.zeros(frames, bool)
        if self._landing_remaining:
            landing = (np.arange(frames) < self._landing_remaining) & (pattern_index * 64 + row == self._landing_row)
            # Leaving the landing row ends it for good.
            self._landing_remaining = self._landing_remaining - frames if landing[-1] else 0
        present, velocity, pitch = snapshot['cells']
        hits, held_velocity, held_pitch = {}, {}, {}
        # Evaluate held signals once per lane, not once per physical route.
        for lane in range(4):
            on = present[pattern_index, lane, row] & moving & (not d['mutes'][lane])
            hits[lane] = on & ((row_phase < self.gate) | landing) & (np.arange(frames) >= self._retrigger[lane])
            self._gate_was[lane] = hits[lane][-1]
            for source, memory, target in ((velocity, self._velocity, held_velocity),
                                           (pitch, self._pitch, held_pitch)):
                values = source[pattern_index, lane, row]
                seen = on & np.isfinite(values)
                index = np.maximum.accumulate(np.where(seen, np.arange(frames), -1))
                held = np.where(index >= 0, values[np.maximum(index, 0)], memory[lane])
                memory[lane] = held[-1]
                target[lane] = held
        for lane in (4, 5):                                 # version 3's lanes, silent here
            hits[lane] = np.zeros(frames, bool)
            held_velocity[lane], held_pitch[lane] = np.full(frames, self._velocity[lane]), np.full(frames, self._pitch[lane])
        self._retrigger = np.maximum(0, self._retrigger - frames)
        for column, (channel, name) in enumerate(self.mapping.items()):
            if name == 'clock':
                value = (np.mod(beat * self.ppqn, 1.) < .5) & moving
                out[:, column] = value * self.high
            elif name == 'reset':
                window = self.reset_s * d['bpm'] / 60
                out[:, column] = (reset | ((beat - loop_start) < window)) & moving
                out[:, column] *= self.high
            elif name == 'run':
                out[:, column] = self.high if moving else 0.
            elif name.startswith('gate'):
                out[:, column] = hits[int(name[4]) - 1] * self.high
            elif name.startswith('vel'):
                out[:, column] = held_velocity[int(name[3]) - 1] / 127 * self.high
            elif name.startswith('pitch'):
                out[:, column] = held_pitch[int(name[5]) - 1]
            else:
                low, high = LANES[name][:2]
                volts_low, volts_high = LANE_VOLTS[name]
                values = np.empty(frames)
                for p in np.unique(pattern_index):
                    sel = pattern_index == p
                    values[sel] = lane_values(d['patterns'][p], name, row_pos[sel])
                out[:, column] = volts_low + (values - low) / (high - low) * (volts_high - volts_low)
        return out

    def _landing_gate(self, landing, beat, bpm):
        """Start the gates of the row a transport change landed on (Gorgon spec/cv-semantics.md).

        Heard late, a seek or start still plays the hits it landed on, for the length they had
        from the landing, starting now; heard after the gate window closed, it lost them. The
        gate ends 1 ms before the row does, so the next hit still rises.
        """
        def row_of(b):
            position = (b % 16) * 4
            return int(b // 16) * 64 + min(int(position), 63), position - int(position)
        landing_row, landing_phase = row_of(landing)
        current_row, phase = row_of(beat)
        if landing_phase >= self.gate or current_row != landing_row:
            return
        seconds = min(self.gate - landing_phase, 1 - phase) * 15 / bpm - .001
        self._landing_remaining, self._landing_row = max(0, round(seconds * self.samplerate)), landing_row

    def _render_v3(self, frames, start, snapshot, out):
        """``render`` for a version-3 score: the signals of the candidates that fire (cv-semantics.md)."""
        d = snapshot['document']
        moving = snapshot['playing'] and snapshot['enabled']
        segment = (snapshot['transport_revision'], snapshot['start']['tick'])
        times = start + self.offset_s + np.arange(frames) / self.samplerate
        u = tracker_v3.elapsed(snapshot, times)
        beat, entry, row_pos = tracker_v3.position(d, snapshot['start']['tick'], u)
        # A transport change is a new transport_revision or moving becoming true (cv-semantics.md).
        # A new segment can also come without one, when a score changes format in chain mode:
        # that restarts evaluation but sends no reset.
        changed = snapshot['transport_revision'] != self._transport
        new_segment = segment != self._segment
        if new_segment:
            # Heard as it starts, from the anchor's row; only the first playhead this
            # output ever sees is joined late, from its own tick (evaluator.md#evaluate).
            self._follower.begin(-tracker_v3.TICKS if self._transport is not None else math.floor(u[0]), segment)
            self._segment = segment
        if changed or (moving and not self._moving):
            self._reset_remaining = round(self.reset_s * self.samplerate) if moving else 0
            self._velocity.fill(0)
            self._pitch.fill(0)
            self._landing_lanes.fill(0)
            # A gate the change interrupts closes for 1 ms, so a hit it lands on rises again.
            self._retrigger = np.where(self._gate_was & moving, round(.001 * self.samplerate), 0)
            if moving:
                self._follower.advance(snapshot, u[0])
                landing = 0. if changed or new_segment or self._stood is None else self._stood
                self._landing_v3(landing, float(u[0]), d['bpm'])
        if not moving:
            self._reset_remaining = 0
            self._landing_lanes.fill(0)
            self._retrigger[:] = 0
            self._stood = float(u[-1])
        self._transport, self._moving = snapshot['transport_revision'], moving
        if moving:
            self._follower.advance(snapshot, u[-1])
        reset = np.arange(frames) < self._reset_remaining
        self._reset_remaining = max(0, self._reset_remaining - frames)
        hits, held_velocity, held_pitch = {}, {}, {}
        for lane in range(6):
            gate = np.zeros(frames, bool)
            velocity, pitch = np.full(frames, np.nan), np.full(frames, np.nan)
            for c in self._follower.candidates:
                if c['lane'] == lane and c['fires'] and c['at'] <= u[-1] and c['at'] + c['duration'] > u[0]:
                    gate |= (u >= c['at']) & (u < c['at'] + self.gate * c['duration'])
                    span = (u >= c['at']) & (u < c['at'] + c['duration'])
                    midi = note_midi(c['note'])
                    velocity[span] = c['vel']
                    pitch[span] = np.nan if midi is None else (midi + d['key'] - 2 - self.pitch_zero) / 12
            if self._landing_lanes[lane]:
                low, high = self._landing_span[lane]
                landing = (np.arange(frames) < self._landing_lanes[lane]) & (u >= low) & (u < high)
                # Leaving the landing candidate's span ends it for good.
                self._landing_lanes[lane] = self._landing_lanes[lane] - frames if landing[-1] else 0
                gate |= landing
            hits[lane] = gate & moving & (np.arange(frames) >= self._retrigger[lane])
            self._gate_was[lane] = hits[lane][-1]
            for values, memory, target in ((velocity, self._velocity, held_velocity), (pitch, self._pitch, held_pitch)):
                seen = np.isfinite(values) & moving
                index = np.maximum.accumulate(np.where(seen, np.arange(frames), -1))
                held = np.where(index >= 0, values[np.maximum(index, 0)], memory[lane])
                memory[lane] = held[-1]
                target[lane] = held
        self._retrigger = np.maximum(0, self._retrigger - frames)
        self._follower.forget(u[0])
        for column, (channel, name) in enumerate(self.mapping.items()):
            if name == 'clock':
                out[:, column] = ((np.mod(beat * self.ppqn, 1.) < .5) & moving) * self.high
            elif name == 'reset':
                out[:, column] = ((reset | (beat < self.reset_s * d['bpm'] / 60)) & moving) * self.high
            elif name == 'run':
                out[:, column] = self.high if moving else 0.
            elif name.startswith('gate'):
                out[:, column] = hits[int(name[4]) - 1] * self.high
            elif name.startswith('vel'):
                out[:, column] = held_velocity[int(name[3]) - 1] / 127 * self.high
            elif name.startswith('pitch'):
                out[:, column] = held_pitch[int(name[5]) - 1]
            else:
                low, high = LANES[name][:2]
                volts_low, volts_high = LANE_VOLTS[name]
                values = np.full(frames, LANES[name][2])
                entries = tracker_v3.timeline(d).entries
                for e in np.unique(entry):
                    pattern = d['patterns'][entries[e]]
                    if pattern is not None:
                        sel = entry == e
                        values[sel] = lane_values(pattern, name, row_pos[sel])
                out[:, column] = volts_low + (values - low) / (high - low) * (volts_high - volts_low)
        return out

    def _landing_v3(self, landing, now, bpm):
        """``_landing_gate`` for version 3: the candidate's span plays the part of the row.

        Per lane, the firing candidate whose span holds the landing tick starts a gate when the
        landing is inside its gate fraction and this sample is still inside its span
        (cv-semantics.md#landing-gate, evaluator.md's vector V6).
        """
        for c in self._follower.candidates:
            end = c['at'] + c['duration']
            if c['fires'] and c['at'] <= landing < end and c['at'] <= now < end:
                phase = (landing - c['at']) / c['duration']
                if phase < self.gate:
                    ticks = min(self.gate - phase, 1 - (now - c['at']) / c['duration']) * c['duration']
                    seconds = ticks * 60 / bpm / tracker_v3.TICKS - .001
                    self._landing_lanes[c['lane']] = max(0, round(seconds * self.samplerate))
                    self._landing_span[c['lane']] = (c['at'], end)

    # -- audio thread ---------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0.)
        try:
            if status:
                self.status = str(status)
                self.underflows += int(bool(getattr(status, 'output_underflow', False)))
            now = time.monotonic()
            dac = getattr(time_info, 'outputBufferDacTime', 0.) or 0.
            current = getattr(time_info, 'currentTime', 0.) or 0.
            if getattr(time_info, 'monotonic_start', None) is not None:
                start = time_info.monotonic_start
            elif dac and current:
                # Map the stream clock onto monotonic time; keep the smallest
                # offset seen, which is the one least delayed by scheduling.
                offset = now - current
                self._stream_offset = offset if self._stream_offset is None else min(self._stream_offset, offset)
                start = dac + self._stream_offset
            else:
                start = now + (self.stream.latency if self.stream is not None else 0.)
            volts = np.zeros((frames, len(self.mapping))) if self._silencing else self.render(frames, start)
            self._write_volts(outdata, volts)
            if self._silencing:
                self._silence_sent += frames
                self._silenced = self._silence_sent >= self._silence_frames
        except Exception as exc:    # never raise into PortAudio; report instead
            self.error = f'{type(exc).__name__}: {exc}'
            outdata.fill(0)
            # Encoded devices retain the last CV until they receive a new DAC
            # command. On evaluation failure send actual zero-volt commands.
            try:
                self._write_volts(outdata, np.zeros((frames, len(self.mapping))))
            except Exception:
                outdata.fill(0)

    def _write_volts(self, outdata, volts):
        for column, channel in enumerate(self.mapping):
            outdata[:, channel] = np.clip(volts[:, column] / self.volts_per_fs, -1., 1.)

    def close(self):
        if self.stream is not None:
            # Send zero *volts* before draining: an encoded stream's PCM silence
            # is not a command to clear an expander's held DAC outputs.
            self._silencing = True
            deadline = time.monotonic() + .25
            while not self._silenced and self.stream.active and time.monotonic() < deadline:
                time.sleep(.002)
            stream, self.stream = self.stream, None
            try:
                stream.stop()
            finally:
                stream.close()
