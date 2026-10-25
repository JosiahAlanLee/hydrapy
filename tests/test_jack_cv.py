"""Native CV transport preserves all 24-bit words and never resamples ES-40 data."""
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy.es40 import ES40CV, pack_es40
from hydrapy.studio_tracker import Tracker
from hydrapy.tracker_cv import TrackerCV
from test_studio_audio import FakeClient


class CVClient(FakeClient):
    blocksize = 256
    opened = []
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.opened.append(self)


@pytest.fixture
def clients(monkeypatch):
    CVClient.opened.clear()
    monkeypatch.setitem(sys.modules, 'jack', SimpleNamespace(Client=CVClient))
    return CVClient.opened


def test_every_signed24_word_survives_jack_float_conversion(clients):
    output = ES40CV([dict(channels=[0, 1], header=1, map={'1': 'pitch1'})], backend='jack', open_stream=False)
    # All 256 high-byte values plus independently varying lower header bytes.
    headers = np.random.default_rng(17).integers(0, 256, (256, 5), dtype=np.int32)
    headers[:, 0] = np.arange(256)
    pcm = pack_es40(headers)
    output._callback = lambda buffer, *args: np.copyto(buffer, pcm)
    output._open_stream('GorgonDigital', 48000, 256, channels=2, dtype='int32')
    try:
        output.stream._process(256)
        received = np.column_stack([p.get_array() for p in output.stream.ports])
        restored = np.rint(received.astype(np.float64) * 2147483648).astype(np.int64)
        np.testing.assert_array_equal(restored, pcm)
        assert output.describe()['backend'] == 'jack'
        assert len(output.describe()['ports']) == 2
    finally:
        output.stream._active = False  # fake server does not schedule shutdown callbacks
        output.close()
    assert clients[0].closed


def test_native_direct_cv_scaling_and_rate_change_clears_outputs(clients):
    tracker = Tracker(); tracker.enabled = True
    output = TrackerCV({1: 'run'}, device='GorgonCV', backend='jack', samplerate=48000)
    try:
        output.publish(tracker, 0., time.monotonic(), True)
        output.stream._process(256)
        assert not output.stream.ports[0].get_array().any()
        assert np.all(output.stream.ports[1].get_array() == .5)
        output.stream.client.samplerate = 96000
        output.stream._process(256)
        assert not output.stream.ports[1].get_array().any()
        assert 'sample rate changed' in output.error
    finally:
        output.stream._active = False
        output.close()


def test_es40_refuses_server_rate_mismatch_before_activation(clients):
    output = ES40CV([dict(channels=[0, 1], header=1, map={'1': 'pitch1'})], backend='jack', open_stream=False)
    with pytest.raises(ValueError, match='no resampling'):
        output._open_stream('GorgonDigital', 44100, 256, channels=2, dtype='int32')
    assert clients[0].closed and output.stream is None
