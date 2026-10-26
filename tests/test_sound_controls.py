import copy

import pytest

from hydrapy.sound_controls import resolve_sound
from hydrapy.studio_tracker import Tracker, validate


def score():
    d = copy.deepcopy(Tracker().document)
    d['format'] = 'hydrapy-tracker-2'
    d['sounds'] = [{'machine': None, 'params': {}} for _ in range(4)]
    return d


def test_each_hit_resolves_from_fresh_defaults_and_round_trips():
    d = score()
    d['sounds'][0] = {'machine': 16, 'params': {'p2': 50}}
    d['patterns'][0]['audio'][0][0] = {'note': 'C-4', 'vel': 90, 'locks': {'machine': 48, 'p1': 0, 'p2': 127}}
    d['patterns'][0]['audio'][0][1] = {'note': 'C-4', 'vel': 90}
    assert validate(d) == d
    assert resolve_sound(d, 0, 0, 0) == {'machine': 48, 'params': {'p1': 0, 'p2': 127}}
    assert resolve_sound(d, 0, 0, 1) == {'machine': 16, 'params': {'p2': 50}}
    result = resolve_sound(d, 0, 0, 0)
    result['params']['p2'] = 1
    assert d['sounds'][0]['params']['p2'] == 50
    t = Tracker()
    t.replace(d, 0., t.revision)
    assert t.document == d


@pytest.mark.parametrize('locks', [None, [], {'wat': 1}, {'p0': 1}, {'p1': True},
                                 {'p1': -1}, {'p1': 128}, {'p1': 1.5}, {'p1': float('nan')},
                                 {'p1': float('inf')}, {'machine': None}, {'machine': 192}])
def test_invalid_locks_rejected(locks):
    d = score()
    d['patterns'][0]['audio'][0][0] = {'note': 'C-4', 'vel': 90, 'locks': locks}
    with pytest.raises(ValueError):
        validate(d)


def test_version_boundary_and_sparse_defaults():
    d = score()
    assert validate(d)['sounds'][0] == {'machine': None, 'params': {}}
    d['format'] = 'hydrapy-tracker-1'
    with pytest.raises(ValueError):
        validate(d)
    del d['sounds']
    d['patterns'][0]['audio'][0][0] = {'note': 'C-4', 'vel': 90, 'locks': {}}
    with pytest.raises(ValueError):
        validate(d)
    del d['patterns'][0]['audio'][0][0]['locks']
    assert 'sounds' not in validate(d)


@pytest.mark.parametrize('sound', [{}, {'machine': None, 'params': {'machine': 1}},
                                  {'machine': False, 'params': {}}, {'machine': 0, 'params': {'p1': None}},
                                  {'machine': 0, 'params': {}, 'extra': 1}])
def test_invalid_track_defaults_rejected(sound):
    d = score(); d['sounds'][0] = sound
    with pytest.raises(ValueError):
        validate(d)
