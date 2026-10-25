"""Protocol checks use the manufacturer's C++ encoder and an independent decoder."""
import copy
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from hydrapy.es40 import Encoder8CV, ES40CV, diagnostic_wav, load_profile, pack_es40
from hydrapy.studio_tracker import Tracker, note_midi, validate


def unpack(pcm, header):
    words = (pcm.astype(np.int64) >> 8) & 0xffffff
    left, right = words[:, 0], words[:, 1]
    return [left >> 16, (left >> 8) & 255, right >> 16, (right >> 8) & 255,
            (left & 240) | ((right >> 4) & 15)][header - 1]


def decode(data):
    """Decode packets by marker bits, not encoder phase: (frame, jack, DAC code)."""
    low = middle = None
    result = []
    for frame, byte in enumerate(data):
        byte = int(byte)
        if byte & 128:
            low, middle = byte & 31, None
        elif byte < 32:
            middle = byte
        elif low is not None and middle is not None:
            jack = ((byte >> 2) & 3) + (4 if byte & 64 else 0)
            result.append((frame, jack, low | middle << 5 | (byte & 3) << 10))
            low = middle = None
    return result


def unit(pair=(2, 3), header=1, mapping=None):
    return dict(channels=list(pair), header=header, map=mapping or {'1': 'pitch1', '2': 'gate1'})


def test_manufacturer_reference_all_headers_and_arbitrary_block_boundaries():
    fixture = json.loads(Path(__file__).with_name('fixtures').joinpath('es40-reference.json').read_text())
    expected = np.array(fixture['rows'], np.int64)
    volts = (((np.arange(101)[:, None] * 103 + np.arange(8) * 719) % 7000 - 3500) / 384).astype(np.float32)
    for lengths in ([101], [1, 1, 1, 20, 2, 1, 64, 11]):
        encoder = Encoder8CV()
        chunks, start = [], 0
        for frames in lengths:
            chunks.append(encoder.encode(volts[start:start + frames])); start += frames
        data = np.concatenate(chunks)
        assert np.array_equal(data, expected[:, 0])
        for header in range(5):
            headers = np.zeros((101, 5), np.int32)
            headers[:, header] = data
            packed = pack_es40(headers)
            assert np.array_equal(packed.astype(np.int64) >> 8, expected[:, 1 + header * 2:3 + header * 2])
            assert np.array_equal(unpack(packed, header + 1), data)


def test_all_header_bytes_survive_packing_together():
    headers = np.random.default_rng(12).integers(0, 256, (1000, 5), dtype=np.int32)
    packed = pack_es40(headers)
    assert packed.dtype == np.int32
    for h in range(5):
        assert np.array_equal(unpack(packed, h + 1), headers[:, h])


def test_dac_calibration_clipping_and_packet_sample_hold():
    volts = np.tile([0., 1., -1., 5., -5., 20., -20., 1/12], (48, 1))
    volts[1:3, 0] = 5.  # changes after first packet starts must not tear its value
    packets = decode(Encoder8CV().encode(volts))
    assert [code for _, _, code in packets[:8]] == [2048, 2432, 1664, 3968, 128, 4095, 0, 2080]
    assert [jack for _, jack, _ in packets] == list(range(8)) * 2


def test_two_es40_units_and_direct_cv_share_one_device_without_crosstalk():
    cv = ES40CV([unit((2, 3), 1), unit((6, 7), 5, {'1': 'pitch2', '3': 'gate2'})],
                direct='0=clock', open_stream=False)
    values = np.zeros((96, len(cv.mapping)))
    for name, value in [('pitch1', 1.), ('pitch2', -1.), ('gate1', 5.), ('clock', 5.)]:
        values[:, cv.columns[name]] = value
    out = np.zeros((96, 8), np.int32)
    cv._write_volts(out, values)
    first = decode(unpack(out[:, [2, 3]], 1))
    second = decode(unpack(out[:, [6, 7]], 5))
    assert [p[2] for p in first[:3]] == [2432, 3968, 2048]
    assert [p[2] for p in second[:3]] == [1664, 2048, 2048]
    assert np.all(out[:, 0] == 2**30)
    assert not out[:, [1, 4, 5]].any()


def test_callback_failure_sends_zero_volts_not_pcm_silence():
    cv = ES40CV([unit()], open_stream=False)
    cv.snapshot = {'broken': True}
    out = np.zeros((256, 4), np.int32)
    cv._callback(out, 256, types.SimpleNamespace(), None)
    assert cv.error
    packets = decode(unpack(out[:, [2, 3]], 1))
    assert len(packets) >= 8 and all(code == 2048 for _, _, code in packets)


@pytest.mark.parametrize('bad', [unit((2, 2)), unit(header=0), unit(header=True), unit(mapping={'0': 'pitch1'}),
                                  unit(mapping={'1': 'unknown'}), unit((-1, 2))])
def test_invalid_hardware_routes_are_rejected(bad):
    with pytest.raises(ValueError):
        ES40CV([bad], open_stream=False)


def test_overlapping_routes_and_wrong_sample_rate_are_rejected():
    with pytest.raises(ValueError, match='share'):
        ES40CV([unit(), unit((3, 4))], open_stream=False)
    with pytest.raises(ValueError, match='share'):
        ES40CV([unit()], direct='2=clock', open_stream=False)
    with pytest.raises(ValueError, match='48000'):
        ES40CV([unit()], samplerate=96000, open_stream=False)


def test_profile_validates_before_opening_and_closes_on_start_failure(tmp_path, monkeypatch):
    opened = []
    class Stream:
        active = False
        def __init__(self, **options):
            self.options = options; self.closed = False; opened.append(self)
        def start(self):
            if len(opened) == 2:
                raise RuntimeError('device busy')
        def stop(self): pass
        def close(self): self.closed = True
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(OutputStream=Stream))
    devices = [dict(device=1, samplerate=48000, es40=[unit()]), dict(device=2, samplerate=48000, es40=[unit(header=9)])]
    path = tmp_path/'cv.json'
    path.write_text(json.dumps(dict(format='hydrapy-cv-1', devices=devices)))
    with pytest.raises(ValueError): load_profile(path)
    assert not opened
    devices[1]['es40'][0]['header'] = 2
    path.write_text(json.dumps(dict(format='hydrapy-cv-1', devices=devices)))
    with pytest.raises(RuntimeError, match='device busy'): load_profile(path)
    assert len(opened) == 2 and all(s.closed for s in opened)
    assert all(s.options['dtype'] == 'int32' and s.options['dither_off'] for s in opened)


def test_diagnostic_wav_has_each_jack_one_volt_then_returns_to_zero(tmp_path):
    path = tmp_path/'diagnostic.wav'
    diagnostic_wav(path, header=5)
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 3, 48000, 18*48000)
        for second in range(18):
            raw = np.frombuffer(wav.readframes(48000), np.uint8).reshape(-1, 3)
            padded = np.zeros((len(raw), 4), np.uint8); padded[:, 1:] = raw
            pcm = padded.tobytes()
            packets = decode(unpack(np.frombuffer(pcm, '<i4').reshape(-1, 2), 5))
            last_cycle = {jack: code for _, jack, code in packets[-8:]}
            target = (second - 1) // 2 if 1 <= second <= 16 and second % 2 else None
            assert last_cycle == {j: 2432 if j == target else 2048 for j in range(8)}


@pytest.mark.parametrize('note,midi', [('C0', 12), ('C-4', 60), ('C4', 60), ('D#4', 63), ('A-4', 69), ('G9', 127), ('AIR', None)])
def test_notes(note, midi):
    assert note_midi(note) == midi


@pytest.mark.parametrize('note', ['H4', 'C#-4', 'G#9', 'B#4', 'E#4', [], None])
def test_invalid_notes(note):
    d = copy.deepcopy(Tracker().document)
    d['patterns'][0]['audio'][0][0] = {'note': note, 'vel': 100}
    with pytest.raises(ValueError, match='Note'):
        validate(d)


def test_coherent_cycle_applies_pitch_before_gate_even_at_a_split_packet():
    volts = np.zeros((72, 8))
    volts[1:, 0] = 1.
    volts[1:, 1] = 5.
    encoder = Encoder8CV(coherent=True)
    data = np.concatenate([encoder.encode(volts[:2]), encoder.encode(volts[2:25]), encoder.encode(volts[25:])])
    packets = decode(data)
    pitch = next(frame for frame, jack, code in packets if jack == 0 and code == 2432)
    gate = next(frame for frame, jack, code in packets if jack == 1 and code == 3968)
    assert (pitch, gate) == (26, 29)


def test_shutdown_clears_all_latched_dacs_before_stopping_stream():
    cv = ES40CV([unit()], open_stream=False)
    initial = np.ones((2, len(cv.mapping))) * 5
    cv._write_volts(np.zeros((2, 4), np.int32), initial)  # stop halfway through a packet
    captured = []
    class Stream:
        latency = 0.
        stopped = closed = False
        @property
        def active(self):
            block = np.zeros((32, 4), np.int32)
            cv._callback(block, 32, types.SimpleNamespace(), None)
            captured.append(block)
            return True
        def stop(self): self.stopped = True
        def close(self): self.closed = True
    stream = cv.stream = Stream()
    cv.close()
    packets = decode(unpack(np.concatenate(captured)[:, [2, 3]], 1))
    assert {jack: code for _, jack, code in packets[-8:]} == {j: 2048 for j in range(8)}
    assert stream.stopped and stream.closed and cv.stream is None
