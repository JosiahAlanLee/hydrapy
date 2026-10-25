"""Gorgon's tracker as control voltage: clock, reset, gates and automation lanes.

The tracker is the master clock. Every output sample is evaluated in the audio
callback from the render thread's latest playhead snapshot, so clock pulses,
gates and held automation steps land on the exact sample, not on a video frame.
The video side is unchanged: it keeps evaluating the same score per frame.

Typical use is an interface's ADAT output into an Expert Sleepers ES-3, which is
DC coupled and ±10 V at full scale (``volts_per_fs=10``). Channels are interface
channel numbers counted from 0, as ``hydrapy devices`` and ``CvOut`` use them.

Signals a channel can carry:

``clock``           pulses at ``ppqn`` per quarter note (default 4 = sixteenths), 50% duty
``reset``           a short trigger at the start of the score, or of the looped pattern
``run``             high while the tracker is playing
``gate1``-``gate4`` the four audio lanes: high for ``gate`` of a row when a hit is there
``vel1``-``vel4``   the last hit's velocity on that lane, 0..``high`` volts, held
``z`` ``roll`` ``feedback``  the automation lanes, scaled to their volt ranges
"""
from __future__ import annotations

import math
import time

import numpy as np

from .studio_tracker import LANES

SIGNALS = ('clock', 'reset', 'run', *(f'gate{i}' for i in range(1, 5)), *(f'vel{i}' for i in range(1, 5)), *LANES)
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


class TrackerCV:
    """Render the tracker score as voltages.

    ``publish(...)`` hands over the playhead from the render thread; the audio
    callback (or ``render`` in tests and offline use) reads only that snapshot.
    ``offset_ms`` moves the voltages earlier (positive) or later (negative)
    relative to the playhead, for audio/video compensation.
    """

    def __init__(self, mapping, device=None, samplerate=None, blocksize=256, volts_per_fs=10.,
                 ppqn=4, gate=.5, high=5., reset_ms=5., offset_ms=0., open_stream=True):
        self.mapping = dict(sorted(mapping.items()))
        if any(type(c) is not int or c < 0 for c in self.mapping) or any(s not in SIGNALS for s in self.mapping.values()):
            raise ValueError('CV channels are non-negative integers mapped to known signals')
        for value, label, low, top in ((ppqn, 'ppqn', 1, 96), (gate, 'Gate length', .01, 1.),
                                       (high, 'Gate voltage', .1, 10.), (reset_ms, 'Reset trigger', .5, 100.),
                                       (volts_per_fs, 'Volts per full scale', .1, 100.), (offset_ms, 'Offset', -500., 500.)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= top:
                raise ValueError(f'{label} must be between {low} and {top}')
        self.ppqn, self.gate, self.high = float(ppqn), float(gate), float(high)
        self.reset_s, self.offset_s, self.volts_per_fs = reset_ms / 1000, offset_ms / 1000, float(volts_per_fs)
        self.samplerate = int(samplerate or 48000)
        self.snapshot = None
        self._velocity = np.zeros(4)
        self._stream_offset = None
        self.stream = None
        self.error = ''
        if open_stream:
            import sounddevice as sd
            if samplerate is None:
                self.samplerate = int(sd.query_devices(device, 'output')['default_samplerate'])
            self.stream = sd.OutputStream(device=device, channels=max(self.mapping) + 1, samplerate=self.samplerate,
                                          blocksize=blocksize, dtype='float32', callback=self._callback)
            self.stream.start()

    # -- render thread --------------------------------------------------------
    def publish(self, tracker, clock_time, wall, playing):
        """Called after each render iteration: the playhead at monotonic ``wall``."""
        self.snapshot = dict(document=tracker.document, enabled=tracker.enabled,
                             anchor_beat=tracker.anchor_beat, anchor_time=tracker.anchor_time,
                             clock_time=clock_time, wall=wall, playing=bool(playing))

    # -- evaluation -----------------------------------------------------------
    def _cells(self, document):
        """Hit presence and velocity as (pattern, lane, row) arrays, cached per score."""
        cached = getattr(self, '_cell_cache', None)
        if cached is not None and cached[0] is document:
            return cached[1], cached[2]
        present = np.zeros((4, 4, 64), dtype=bool)
        velocity = np.zeros((4, 4, 64))
        for p, pattern in enumerate(document['patterns']):
            for lane, cells in enumerate(pattern['audio']):
                for r, hit in enumerate(cells):
                    if hit is not None:
                        present[p, lane, r], velocity[p, lane, r] = True, hit['vel']
        self._cell_cache = (document, present, velocity)
        return present, velocity

    def beats(self, snapshot, monotonic):
        """Tracker beats (0..64) and loop start at an array of monotonic times."""
        d = snapshot['document']
        playing = snapshot['playing']
        clock = snapshot['clock_time'] + ((monotonic - snapshot['wall']) if playing else 0.)
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
        if snapshot is None:
            return out
        d = snapshot['document']
        moving = snapshot['playing'] and snapshot['enabled']
        times = start + self.offset_s + np.arange(frames) / self.samplerate
        beat, loop_start = self.beats(snapshot, times)
        pattern_index = (beat // 16).astype(int)
        row_pos = np.mod(beat, 16) * 4                      # fractional sixteenth row, 0..64
        row = np.minimum(row_pos.astype(int), 63)
        row_phase = row_pos - row
        present, velocity = self._cells(d)
        hits = {}
        for lane in range(4):
            on = present[pattern_index, lane, row]
            vel = np.where(on, velocity[pattern_index, lane, row], np.nan)
            if d['mutes'][lane] or not moving:
                on = np.zeros(frames, dtype=bool)
                vel = np.full(frames, np.nan)
            hits[lane] = (on & (row_phase < self.gate), vel)
        for column, (channel, name) in enumerate(self.mapping.items()):
            if name == 'clock':
                value = (np.mod(beat * self.ppqn, 1.) < .5) & moving
                out[:, column] = value * self.high
            elif name == 'reset':
                window = self.reset_s * d['bpm'] / 60
                out[:, column] = ((beat - loop_start) < window) & moving
                out[:, column] *= self.high
            elif name == 'run':
                out[:, column] = self.high if moving else 0.
            elif name.startswith('gate'):
                out[:, column] = hits[int(name[4]) - 1][0] * self.high
            elif name.startswith('vel'):
                lane = int(name[3]) - 1
                vel = hits[lane][1]
                # Hold the last velocity seen, across blocks too.
                seen = ~np.isnan(vel)
                index = np.maximum.accumulate(np.where(seen, np.arange(frames), -1))
                held = np.where(index >= 0, vel[np.maximum(index, 0)], self._velocity[lane])
                if seen.any():
                    self._velocity[lane] = held[-1]
                out[:, column] = held / 127 * self.high
            else:
                low, high = LANES[name][:2]
                volts_low, volts_high = LANE_VOLTS[name]
                values = np.empty(frames)
                for p in np.unique(pattern_index):
                    sel = pattern_index == p
                    values[sel] = lane_values(d['patterns'][p], name, row_pos[sel])
                out[:, column] = volts_low + (values - low) / (high - low) * (volts_high - volts_low)
        return out

    # -- audio thread ---------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0.)
        try:
            now = time.monotonic()
            dac = getattr(time_info, 'outputBufferDacTime', 0.) or 0.
            current = getattr(time_info, 'currentTime', 0.) or 0.
            if dac and current:
                # Map the stream clock onto monotonic time; keep the smallest
                # offset seen, which is the one least delayed by scheduling.
                offset = now - current
                self._stream_offset = offset if self._stream_offset is None else min(self._stream_offset, offset)
                start = dac + self._stream_offset
            else:
                start = now + (self.stream.latency if self.stream is not None else 0.)
            volts = self.render(frames, start)
            for column, channel in enumerate(self.mapping):
                outdata[:, channel] = np.clip(volts[:, column] / self.volts_per_fs, -1., 1.)
        except Exception as exc:    # never raise into PortAudio; report instead
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        if self.stream is not None:
            stream, self.stream = self.stream, None
            stream.stop()
            stream.close()
