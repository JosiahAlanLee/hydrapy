"""Opt-in Studio flight recorder. No hooks run on the audio process callback.

The render thread is the sole producer. Numeric frame records use a preallocated
single-producer/single-consumer ring: publish an index only after filling its row,
release a row only after copying it. A full ring drops diagnostics, never waits.
JSON, statistics, incident history and file I/O belong to the writer thread.
This is Python soft realtime, not an allocation-free or hard-RT guarantee.
"""
from __future__ import annotations

from array import array
from collections import deque
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import uuid


FORMAT = 'hydrapy-diagnostics-1'
COLUMNS = ('monotonic', 'frame', 'simulation_time', 'beat', 'render_cpu_ms',
           'readback_ms', 'work_ms', 'interval_ms', 'target_fps', 'encoder_depth',
           'repeated_frames', 'xruns', 'underruns', 'late_packets', 'suspensions',
           'jack_frame_estimate', 'native_missed')
COUNTERS = ('repeated_frames', 'xruns', 'underruns', 'late_packets', 'suspensions')


def environment():
    """Run once on the writer, never during a render or audio callback."""
    result = dict(python=platform.python_version(), platform=platform.platform(), pid=os.getpid())
    try:
        result['hydrapy'] = importlib.metadata.version('hydrapy')
    except importlib.metadata.PackageNotFoundError:
        result['hydrapy'] = 'uninstalled checkout'
    root = Path(__file__).resolve().parent.parent
    if (root / '.git').exists():
        try:
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                      text=True, timeout=2, check=True).stdout.strip()
            result.update(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '-uno')))
        except (OSError, subprocess.SubprocessError):
            result['commit'] = None
    return result


def memory_bytes():
    """Current process RSS on Linux, unavailable elsewhere (not peak RSS)."""
    try:
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(p50=values[(len(values) - 1) // 2], p95=values[int((len(values) - 1) * .95)],
                p99=values[int((len(values) - 1) * .99)], maximum=values[-1])


class Diagnostics:
    def __init__(self, path, metadata=None, *, audio=None, capacity=4096, event_capacity=128,
                 max_bytes=32 * 1024 * 1024, interval=1.):
        if capacity < 1 or event_capacity < 1 or max_bytes < 4096 or interval <= 0:
            raise ValueError('Invalid diagnostic recorder limits')
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse existing files, including symlinks; don't destroy an earlier take.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stream = os.fdopen(fd, 'w', encoding='utf-8')
        self.rows = [array('d', [0.]) * len(COLUMNS) for _ in range(capacity)]
        self.read_index = self.write_index = 0
        self.events = deque()
        self.event_capacity, self.max_bytes, self.interval = event_capacity, max_bytes, interval
        self.dropped_frames = self.dropped_events = 0
        self.error, self.disabled, self.finished = '', False, False
        self.stop = threading.Event()
        self.origin = time.monotonic()
        self.last_frame = None
        self.frame_count = 0
        self.metadata = dict(metadata or {})
        self.audio = audio
        self.run_id = uuid.uuid4().hex
        self.bytes_written = 0
        self.history = deque(maxlen=256)
        self.window = deque(maxlen=4096)
        self.previous_counters = None
        self.summary_counters = None
        self.last_incident = -math.inf
        self.incidents = 0
        self.thread = threading.Thread(target=self._run, name='hydrapy-diagnostics', daemon=True)
        self.thread.start()

    def frame(self, now, simulation, beat, render_ms, readback_ms, work_ms, fps,
              depth, repeats, audio, missed=0):
        """Render-thread-only; counters are observations, not audio-callback timestamps."""
        if self.disabled or self.stop.is_set():
            return
        previous, self.last_frame = self.last_frame, now
        self.frame_count += 1
        if self.write_index - self.read_index >= len(self.rows):
            self.dropped_frames += 1
            return
        row = self.rows[self.write_index % len(self.rows)]
        row[0], row[1], row[2], row[3] = now, self.frame_count, simulation, beat
        row[4], row[5], row[6] = render_ms, readback_ms, work_ms
        row[7] = 0. if previous is None else (now - previous) * 1000
        row[8], row[9], row[10] = fps, depth, repeats
        row[11], row[12], row[13], row[14] = audio.xruns, audio.underruns, audio.late_packets, audio.suspensions
        stamp = audio.jack_frame_at(now)
        row[15], row[16] = -1 if stamp is None else stamp, missed
        self.write_index += 1

    def event(self, name, **values):
        """Rare render-thread markers only; no score bodies, pixels, PCM or credentials."""
        if self.disabled or self.stop.is_set():
            return
        if len(self.events) >= self.event_capacity:
            self.dropped_events += 1
            return
        self.events.append(dict(kind='event', event=name, monotonic=time.monotonic(), **values))

    def _emit(self, record):
        line = json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n'
        size = len(line.encode('utf-8'))
        if self.bytes_written + size > self.max_bytes - 1024:
            self.stream.write(json.dumps(dict(kind='limit', monotonic=time.monotonic(),
                                             max_bytes=self.max_bytes, complete=False)) + '\n')
            raise OSError('diagnostic file size limit reached; recording diagnostics stopped')
        self.stream.write(line)
        self.bytes_written += size

    def _observe(self, row):
        self.history.append(row)
        self.window.append(row)
        now = row[0]
        while self.history and self.history[0][0] < now - 2:
            self.history.popleft()
        counters = tuple(row[COLUMNS.index(k)] for k in COUNTERS)
        reasons = []
        if self.previous_counters is not None:
            reasons.extend(k for k, a, b in zip(COUNTERS, counters, self.previous_counters) if a > b)
        self.previous_counters = counters
        budget = 1000 / max(1, row[8])
        if row[6] > max(50, 2 * budget) or row[7] > max(50, 2 * budget):
            reasons.append('frame_stall')
        if row[9] >= 135:  # 75% of Recorder's 180-frame queue
            reasons.append('encoder_backlog')
        if row[16] > 0:
            reasons.append('native_deadline')
        if reasons:
            self.incidents += 1
            # Continuous overload must not turn logging into another load source.
            if now - self.last_incident >= 5:
                self._emit(dict(kind='incident', monotonic=now, reasons=reasons,
                                columns=COLUMNS, frames=list(self.history)))
                self.last_incident = now

    def _summary(self, now, final=False):
        rows = list(self.window)
        self.window.clear()
        last = self.history[-1] if self.history else None
        counters = None if last is None else {k: last[COLUMNS.index(k)] for k in COUNTERS}
        if self.audio is not None:
            # Observe existing scalar counters even when the renderer stalls or is paused.
            counters = dict(counters or {}, **{k: getattr(self.audio, k) for k in COUNTERS if k != 'repeated_frames'})
            counters.setdefault('repeated_frames', 0)
        before, self.summary_counters = self.summary_counters, counters
        deltas = None if before is None or counters is None else {
            k: counters[k] - before[k] if counters[k] >= before[k] else None for k in COUNTERS}
        self._emit(dict(kind='summary', monotonic=now, final=final, frames=len(rows),
                        frame_total=self.frame_count, rss_bytes=memory_bytes(),
                        producer_age_s=None if self.last_frame is None else max(0, now - self.last_frame),
                        timing_ms={key: distribution([r[COLUMNS.index(key)] for r in rows])
                                   for key in ('render_cpu_ms', 'readback_ms', 'work_ms', 'interval_ms')},
                        encoder_depth_max=max((r[9] for r in rows), default=0),
                        counters=counters, counter_deltas=deltas,
                        audio_error=None if self.audio is None else self.audio.error,
                        incidents=self.incidents, dropped_frames=self.dropped_frames,
                        dropped_events=self.dropped_events))

    def _run(self):
        try:
            with self.stream:
                self._emit(dict(kind='run', format=FORMAT, run_id=self.run_id, monotonic=self.origin,
                                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                environment=environment(), config=self.metadata,
                                frame_capacity=len(self.rows), event_capacity=self.event_capacity,
                                max_bytes=self.max_bytes))
                self.stream.flush()
                checkpoint = time.monotonic()
                while True:
                    # Bound each drain, even if the producer keeps submitting frames/events.
                    end = self.write_index
                    while self.read_index < end:
                        row = list(self.rows[self.read_index % len(self.rows)])
                        self.read_index += 1
                        self._observe(row)
                    for _ in range(len(self.events)):
                        self._emit(self.events.popleft())
                    now = time.monotonic()
                    stopping = self.stop.is_set()
                    if stopping or now - checkpoint >= self.interval:
                        self._summary(now, final=stopping)
                        self.stream.flush()  # crash checkpoints; no fsync on the render thread
                        checkpoint = now
                    if stopping:
                        self._emit(dict(kind='closed', monotonic=now, complete=True,
                                        dropped_frames=self.dropped_frames, dropped_events=self.dropped_events))
                        break
                    self.stop.wait(.1)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.disabled = True
            print(f'HydraPy diagnostics: {self.error}', file=sys.stderr)
        finally:
            self.disabled = True
            self.finished = True

    def close(self, timeout=2.):
        """Call after the producer stops; slow/broken storage cannot hold shutdown forever."""
        self.stop.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.error = 'diagnostic writer did not finish before shutdown timeout'
        return self.finished and not self.error
