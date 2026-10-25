"""Hardware-free callback cost: PYTHONPATH=. python tools/benchmark_tracker_cv.py."""
import json
import platform
import time
import types

import numpy as np

from hydrapy.es40 import ES40CV
from hydrapy.studio_tracker import Tracker

tracker = Tracker()
for pattern in tracker.document['patterns']:
    for lane in range(4):
        pattern['audio'][lane] = [{'note': ['C-3', 'D#4', 'G-4', 'A#5'][(row + lane) % 4], 'vel': 100} for row in range(64)]
    for lane, bounds in [('z', (0, 1)), ('roll', (-45, 45)), ('feedback', (0, .9))]:
        pattern['keys'][lane] = [{'r': row, 'v': bounds[row % 2]} for row in range(64)]
tracker.enabled = True
cv = ES40CV([
    dict(channels=[0, 1], header=1, map={str(i + 1): s for i, s in enumerate(['pitch1', 'gate1', 'pitch2', 'gate2', 'clock', 'reset', 'z', 'roll'])}),
    dict(channels=[2, 3], header=5, map={str(i + 1): s for i, s in enumerate(['pitch3', 'gate3', 'pitch4', 'gate4', 'vel1', 'vel2', 'feedback', 'run'])}),
], samplerate=48000, open_stream=False)
cv.publish(tracker, 0., time.monotonic(), True)
out = np.zeros((256, 4), np.int32)
timing = types.SimpleNamespace(currentTime=0., outputBufferDacTime=0.)
durations = []
for block in range(2200):
    start = time.perf_counter()
    cv._callback(out, len(out), timing, None)
    elapsed = (time.perf_counter() - start) * 1000
    if block >= 200:
        durations.append(elapsed)
assert not cv.error, cv.error
print(json.dumps(dict(python=platform.python_version(), blocks=len(durations), frames=256,
                     samplerate=48000, budget_ms=256/48,
                     callback_ms=dict(zip(['p50', 'p95', 'p99', 'max'], map(float, np.percentile(durations, [50, 95, 99, 100]))))), indent=2))
