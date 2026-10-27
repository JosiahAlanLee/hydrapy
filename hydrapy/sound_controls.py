"""Version-2 per-hit controls for external Rack instruments.

Unspecified controls deliberately stay unspecified: the instrument, not the
tracker, owns each machine's native defaults. No state is carried between hits.
"""
from __future__ import annotations

import copy
import math

PARAMS = frozenset(f'p{i}' for i in range(1, 9))


def controls(value, *, machine=False):
    if not isinstance(value, dict):
        raise ValueError('Sound controls must be an object')
    allowed = PARAMS | ({'machine'} if machine else set())
    for key, raw in value.items():
        high = 191 if key == 'machine' else 127
        if key not in allowed or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError('Unknown or nonnumeric sound control')
        if not math.isfinite(raw) or raw != int(raw) or not 0 <= raw <= high:
            raise ValueError(f'{key} must be an integer from 0 to {high}')
    return value


def validate_sounds(document):
    """Validate the reserved fields without upgrading a legacy score."""
    count = {'hydrapy-tracker-2': 4, 'hydrapy-tracker-3': 6}.get(document['format'])
    if count is None:
        if 'sounds' in document:
            raise ValueError('Sound defaults require hydrapy-tracker-2')
        return
    sounds = document.get('sounds')
    if not isinstance(sounds, list) or len(sounds) != count:
        raise ValueError(f'{"Four" if count == 4 else "Six"} track sound defaults are required')
    for sound in sounds:
        if not isinstance(sound, dict) or set(sound) != {'machine', 'params'}:
            raise ValueError('A track sound has machine and params')
        if sound['machine'] is not None:
            controls({'machine': sound['machine']}, machine=True)
        controls(sound['params'])


def resolve_sound(document, pattern, lane, row):
    """Fresh defaults + hit locks; None machine means use the instrument's choice."""
    base = document.get('sounds', [{'machine': None, 'params': {}}] * 4)[lane]
    result = copy.deepcopy(base)
    hit = document['patterns'][pattern]['audio'][lane][row]
    for key, value in (hit or {}).get('locks', {}).items():
        if key == 'machine':
            result['machine'] = value
        else:
            result['params'][key] = value
    return result
