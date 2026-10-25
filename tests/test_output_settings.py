"""Real ES-40 encoding and Studio persistence, including failed settings writes."""
import json

import numpy as np
import pytest

from hydrapy.es40 import ES40CV, flip_pcm24, pack_es40
from hydrapy.studio import Session


def cv():
    return ES40CV([dict(name='ES-40', channels=[2, 3], header=1, expander='8gt', map={'1': 'gate1'})],
                  direct='0=clock', open_stream=False)


@pytest.mark.parametrize('mode,columns', [('normal', []), ('invert_left', [0]),
                                        ('invert_right', [1]), ('invert_both', [0, 1])])
def test_polarity_is_applied_after_encoding_to_selected_channels_only(mode, columns):
    rng = np.random.default_rng(12)
    words = rng.integers(-8388607, 8388607, (500, 2), dtype=np.int32)
    out, clips = flip_pcm24(words << 8, mode)
    expected = words.copy()
    expected[:, columns] *= -1
    np.testing.assert_array_equal(out >> 8, expected)
    restored, _ = flip_pcm24(out, mode)
    np.testing.assert_array_equal(restored >> 8, words)
    assert clips == 0


def test_full_scale_endpoint_saturates_and_reports_loss_instead_of_wrapping():
    encoded = pack_es40(np.array([[128, 0, 0, 0, 0]], np.int32))
    out, clips = flip_pcm24(encoded, 'invert_left')
    assert (out >> 8).tolist() == [[8388607, 0]] and clips == 1


def test_route_polarity_compensates_gate1_and_leaves_direct_cv_unchanged():
    output = cv()
    route = output.polarity_routes()[0]['id']
    assert output.set_polarity(route, 'invert_left') == 'normal'
    values = np.full((128, len(output.mapping)), 5.)
    buffer = np.zeros((128, 4), np.int32)
    output._write_volts(buffer, values)
    assert np.all(buffer[:, 0] == 2**30)  # ordinary clock CV has no polarity correction
    # Simulated downstream polarity reversal restores the exact gate-1-only byte.
    assert np.all((-buffer[:, 2].astype(np.int64) >> 24) & 255 == 1)
    assert not buffer[:, [1, 3]].any()
    with pytest.raises(ValueError): output.set_polarity(route, 'other')
    with pytest.raises(ValueError): output.set_polarity('unknown pair', 'normal')


def test_gui_settings_survive_restart_and_failed_write_rolls_back(tmp_path, monkeypatch):
    (tmp_path/'p.py').write_text('from hydrapy import *\nsolid(0).out(o0)\n')
    output = cv()
    route = output.polarity_routes()[0]['id']
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), cv=output)
    try:
        state = session.call('cv_polarity', route=route, mode='invert_left')
        assert state['polarity_routes'][0]['mode'] == 'invert_left'
        session.call('jack_audio', enabled=False, optical_gain=.25, breaks_gain=.3, lead_blocks=4)
        path = tmp_path/'.hydrapy/outputs.json'
        saved = path.read_text()
        def fail(*a, **kw): raise OSError('disk full')
        monkeypatch.setattr('hydrapy.studio.atomic_text', fail)
        with pytest.raises(ValueError, match='disk full'):
            session.call('cv_polarity', route=route, mode='invert_both')
        with pytest.raises(ValueError, match='disk full'):
            session.call('jack_audio', enabled=False, optical_gain=.9)
        assert output.polarity_routes()[0]['mode'] == 'invert_left'
        assert session.audio.config['optical_gain'] == .25
        assert path.read_text() == saved
    finally:
        session.close()
    monkeypatch.undo()
    output = cv()
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), cv=output)
    try:
        assert output.polarity_routes()[0]['mode'] == 'invert_left'
        assert session.audio.config['optical_gain'] == .25
        assert session.audio.config['lead_blocks'] == 4
    finally:
        session.close()
    # One invalid entry prevents partially applying a settings document.
    data = json.loads(saved)
    data['polarity']['bad pair'] = 'invalid mode'
    path.write_text(json.dumps(data))
    output = cv()
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), cv=output)
    try:
        assert output.polarity_routes()[0]['mode'] == 'normal'
    finally:
        session.close()
