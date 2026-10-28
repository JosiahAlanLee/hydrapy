"""Audio clock, channel isolation and lifecycle without an installed JACK server."""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy.studio_audio import BreaksSynth, PORTS, StudioAudio, settings
from hydrapy.studio_tracker import Tracker


class FakeClient:
    samplerate, blocksize, last_frame_time, frames_since_cycle_start = 48000, 128, 0, 0
    name = 'HydraPy'
    def __init__(self, name, **kw):
        assert kw == dict(no_start_server=True)
        self.ports = []
        self.outports = SimpleNamespace(register=self.register)
        self.closed = False
    def register(self, name):
        buffer = np.zeros(self.blocksize, np.float32)
        port = SimpleNamespace(name='HydraPy:' + name, get_array=lambda: buffer)
        self.ports.append(port)
        return port
    def __getattr__(self, name):
        if name.startswith('set_') and name.endswith('_callback'):
            return lambda fn: setattr(self, name[4:-9], fn)
        raise AttributeError(name)
    def activate(self): pass
    def deactivate(self): pass
    def close(self): self.closed = True


def snapshot():
    tracker = Tracker()
    tracker.enabled = True
    tracker.document = copy.deepcopy(tracker.document)
    tracker.document['bpm'] = 120
    for p in tracker.document['patterns']:
        p['audio'] = [[None] * 64 for _ in range(4)]
    for r, note in [(0, 'C-2'), (2, 'D-2'), (3, 'F#2'), (4, 'C-2')]:
        tracker.document['patterns'][0]['audio'][0][r] = dict(note=note, vel=100)
    audio = StudioAudio()
    audio.publish(tracker, 0., 100., True)
    return audio.snapshot


@pytest.mark.parametrize('bad', [dict(enabled=1), dict(optical_gain=float('nan')),
    dict(breaks_gain=2), dict(lead_blocks=True), dict(lead_blocks=1), dict(device='x')])
def test_settings_reject_invalid_values(bad):
    with pytest.raises(ValueError): settings(bad)


def test_breaks_uses_sample_clock_across_arbitrary_callback_chunks():
    state = snapshot()
    whole = BreaksSynth().render(state, 100., 48000, 48000)
    synth = BreaksSynth()
    blocks, offset = [], 0
    while offset < 48000:
        count = min(137, 48000 - offset)
        blocks.append(synth.render(state, 100. + offset / 48000, count, 48000))
        offset += count
    assert np.max(np.abs(whole)) > .2
    assert np.isfinite(whole).all()
    np.testing.assert_allclose(np.concatenate(blocks), whole, atol=2e-6)
    state['document']['mutes'][0] = True
    assert not synth.render(state, 101., 128, 48000).any()
    state['document']['mutes'][0] = False
    state['playing'] = False
    assert not synth.render(state, 101., 128, 48000).any()
    state['playing'] = True
    state['transport_revision'] += 1
    # Seek to an empty row: no tail from the prior transport.
    assert not synth.render(state, 102.01, 128, 48000).any()


def test_callback_copies_only_current_timestamps_and_never_replays_after_seek(monkeypatch):
    audio = StudioAudio(client_factory=FakeClient)
    monkeypatch.setattr(audio, '_produce', lambda: None)
    audio.snapshot = snapshot()
    audio.configure(dict(enabled=True))
    try:
        assert [p.name for p in audio.ports] == ['HydraPy:' + p for p in PORTS]
        values = np.tile([.1, -.2, .3, -.4], (128, 1)).astype(np.float32)
        audio.packets.append((audio.epoch, 0, 0, values))
        audio._process(128)
        np.testing.assert_allclose([p.get_array()[0] for p in audio.ports], values[0])
        audio.snapshot['transport_revision'] += 1
        audio.client.last_frame_time = 128
        audio.packets.append((audio.epoch, 128, 0, values))
        audio._process(128)
        assert not audio.ports[2].get_array().any()  # old rhythmic event
        assert audio.ports[0].get_array().any()     # optical follows renderer, not musical seek
        audio.snapshot['playing'] = False
        audio.client.last_frame_time = 256
        audio.packets.append((audio.epoch, 256, 1, values))
        audio._process(128)
        assert not any(p.get_array().any() for p in audio.ports)
        audio._block_size(256)
        audio.packets.append((audio.epoch - 1, 384, 1, values))
        audio.client.last_frame_time = 384
        audio._process(128)
        assert audio.late_packets == 1
    finally:
        client = audio.client
        audio.close()
    assert client.closed and not audio.ports and not audio.packets


def test_mono_optical_is_pulled_once_duplicated_and_detached_on_disable():
    audio = StudioAudio(client_factory=FakeClient)
    track = SimpleNamespace(listeners=[], sr=48000, rate=30, _generation=0, output=SimpleNamespace(name='o2'))
    audio.attach(SimpleNamespace(probes=[track]))
    assert not track.listeners  # disabled ports do not accumulate a queue
    audio.snapshot = snapshot()
    audio.configure(dict(enabled=True, optical_gain=.5))
    try:
        ramp = np.arange(1600, dtype=np.float32) / 1600
        track.listeners[0](ramp)
        audio._process(128)  # establish a JACK sample/wall-clock anchor
        deadline = time.monotonic() + 2
        while not audio.packets and time.monotonic() < deadline:
            time.sleep(.002)
        _, _, _, packet = audio.packets[0]
        np.testing.assert_array_equal(packet[:, 0], packet[:, 1])
        np.testing.assert_allclose(packet[:, 0], ramp[:128] * .5)
        assert audio.describe()['sources'] == ['o2']
    finally:
        audio.configure(dict(enabled=False))
    assert not track.listeners


def test_server_disconnect_can_retry_and_start_failure_preserves_disabled_config(monkeypatch):
    audio = StudioAudio(client_factory=FakeClient)
    monkeypatch.setattr(audio, '_produce', lambda: None)
    audio.configure(dict(enabled=True))
    first = audio.client
    audio._shutdown(0, 'test server exited')
    assert not audio.describe()['active']
    audio.configure(dict(enabled=True))
    assert first.closed and audio.client is not first and audio.describe()['active']
    audio.close()
    def fail(*a, **kw): raise RuntimeError('no running server')
    audio.client_factory = fail
    audio.config['enabled'] = False
    with pytest.raises(ValueError, match='no running server'):
        audio.configure(dict(enabled=True))
    assert not audio.config['enabled'] and not audio.client


def test_breaks_kit_hears_equivalent_note_spellings_alike():
    """C2 is C-2 (the kick) and D2 is D-2 (the snare): scores written outside the editor, imported or
    sent over the API keep their spelling, and must not turn the kick and snare into hats."""
    canonical = snapshot()
    other = copy.deepcopy(canonical)
    for row in other['document']['patterns'][0]['audio'][0]:
        if row:
            row['note'] = row['note'].replace('-', '')       # C-2 -> C2, D-2 -> D2; F#2 stays
    a = BreaksSynth().render(canonical, 100., 48000, 48000)
    b = BreaksSynth().render(other, 100., 48000, 48000)
    np.testing.assert_array_equal(a, b)


def test_xy_streams_feed_the_optical_ports():
    from hydrapy import XY, Lissajous, Patch
    from hydrapy.live import clear_hooks
    clear_hooks()
    patch = Patch(size=(16, 9))
    xy = XY(Lissajous(3, 2), patch=patch)
    audio = StudioAudio(client_factory=FakeClient)
    audio.attach(patch)
    audio.snapshot = snapshot()
    audio.configure(dict(enabled=True))
    try:
        assert not xy.x.listeners and not xy.y.listeners          # played by the audio thread, not queued
        assert audio.describe()['sources'] == [xy.x.label, xy.y.label]
        left, right = audio.sources
        assert left.voice is right.voice and not left.pull(64).any()   # silent until a frame has drawn
    finally:
        audio.configure(dict(enabled=False))
        xy.close(); clear_hooks()
