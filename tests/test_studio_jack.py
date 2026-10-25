"""Opt-in loopback over real JACK or PipeWire-JACK, with no audio hardware.

HYDRAPY_JACK_TESTS=1 requires an already-running server; failures do not skip.
CI runs the same test once with jackd's dummy backend and once with PipeWire.
"""
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy.studio_audio import StudioAudio
from hydrapy.studio_tracker import Tracker

pytestmark = pytest.mark.skipif(os.getenv('HYDRAPY_JACK_TESTS') != '1', reason='requires a real JACK server')


def test_real_ports_preserve_four_channels_pause_and_reconnect():
    import jack
    audio = StudioAudio()
    tracks = [SimpleNamespace(listeners=[], sr=48000, rate=30, _generation=0,
                              output=SimpleNamespace(name=f'o{i}')) for i in range(2)]
    audio.attach(SimpleNamespace(probes=tracks))
    tracker = Tracker()
    tracker.enabled = True
    for p in tracker.document['patterns']:
        p['audio'] = [[None] * 64 for _ in range(4)]
        for row in range(0, 64, 4):
            p['audio'][0][row] = dict(note='C-2', vel=100)
    wall = time.monotonic()
    audio.publish(tracker, 0., wall, True)
    receiver = jack.Client('gorgon_audio_test', no_start_server=True)
    ports = [receiver.inports.register(f'capture_{i + 1}') for i in range(4)]
    captured = []
    @receiver.set_process_callback
    def receive(frames):
        if len(captured) < 4000:
            captured.append(np.column_stack([p.get_array() for p in ports]))
    try:
        receiver.activate()
        audio.configure(dict(enabled=True, optical_gain=.25, breaks_gain=.5, lead_blocks=4))
        # Push after startup: the same AudioQueues used by a real Soundtrack.
        for track, value in zip(tracks, [.4, -.8]):
            track.listeners[0](np.full(4096, value, np.float32))
        def connect():
            for source, sink in zip(audio.ports, ports): receiver.connect(source.name, sink.name)
        connect()
        time.sleep(.8)
        samples = np.concatenate(captured[-max(1, int(.4 * receiver.samplerate / receiver.blocksize)):])
        audible = np.abs(samples[:, 0]) > .01
        assert audible.sum() > len(samples) * .8, audio.describe()
        np.testing.assert_allclose(samples[audible, :2], np.tile([.1, -.2], (audible.sum(), 1)), atol=2e-5)
        assert np.max(np.abs(samples[:, 2])) > .01, audio.describe()
        np.testing.assert_array_equal(samples[:, 2], samples[:, 3])
        # Pause silences even packets already queued for future JACK frames.
        audio.publish(tracker, .8, time.monotonic(), False)
        time.sleep(max(.08, 4 * receiver.blocksize / receiver.samplerate + .03))
        assert not np.concatenate(captured[-3:]).any()
        audio.configure(dict(enabled=False))
        assert not any(t.listeners for t in tracks)
        audio.configure(dict(enabled=True))
        connect()
        audio.publish(tracker, 0., time.monotonic(), True)
        for track in tracks: track.listeners[0](np.full(4096, .2, np.float32))
        time.sleep(.15)
        assert np.max(np.abs(np.concatenate(captured[-5:])[:, :2])) > .05
        print(audio.describe())
    finally:
        audio.close()
        receiver.deactivate()
        receiver.close()
