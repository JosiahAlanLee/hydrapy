"""Studio's musical playhead and small, validated automation score.

The render thread owns this object. Moving the playhead never seeks a renderer,
replays hooks, or clears feedback. Browser audio is deliberately outside it.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

LANES = {'z': (0., 1., .35), 'roll': (-45., 45., 0.), 'feedback': (0., .9, .25)}

def note_midi(note):
    """Tracker naturals use a separator: C-4 is C4 (MIDI 60), not C minus 4.

    Accept C0–G9, including sharps; AIR is an unpitched event.
    """
    if note == 'AIR':
        return None
    match = re.fullmatch(r'([A-G])([-#]?)([0-9])', note) if isinstance(note, str) else None
    if match:
        letter, accidental, octave = match.groups()
        midi = 12 * (int(octave) + 1) + {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[letter] + (accidental == '#')
        if midi <= 127 and not (accidental == '#' and letter in 'BE'):
            return midi
    raise ValueError('Note must be C0–G9 (sharps allowed), or AIR')


def number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be a finite number between {low} and {high}')
    return value


def validate(document):
    """Copy before validation so accepted scores cannot change behind the renderer."""
    d = copy.deepcopy(document)
    if not isinstance(d, dict) or d.get('format') != 'hydrapy-tracker-1':
        raise ValueError('Expected a hydrapy-tracker-1 score')
    number(d.get('bpm'), 60, 200, 'Tempo')
    if d.get('key') not in (0, 2, 5, 7):
        raise ValueError('Choose C, D, F or G minor')
    if type(d.get('pattern')) is not int or not 0 <= d['pattern'] < 4:
        raise ValueError('Choose pattern 0–3')
    for name in ('loop', 'sound'):
        if type(d.get(name)) is not bool:
            raise ValueError(f'{name} must be boolean')
    if not isinstance(d.get('mutes'), list) or len(d['mutes']) != 4 or any(type(m) is not bool for m in d['mutes']):
        raise ValueError('Four audio mute states are required')
    if not isinstance(d.get('modes'), dict) or set(d['modes']) != {'program', 'aux'} or any(v not in ('o0', 'o1', 'o2', 'o3') for v in d['modes'].values()):
        raise ValueError('Route program and auxiliary to o0–o3')
    bindings = d.get('bindings')
    if not isinstance(bindings, dict) or set(bindings) != set(LANES) or any(not isinstance(v, str) or len(v) > 120 for v in bindings.values()):
        raise ValueError('Three named parameter bindings are required; use an empty name to disconnect')
    named = [v for v in bindings.values() if v]
    if len(named) != len(set(named)):
        raise ValueError('Bind each parameter to only one lane')
    if not isinstance(d.get('patterns'), list) or len(d['patterns']) != 4:
        raise ValueError('Four patterns are required')
    for p in d['patterns']:
        if not isinstance(p, dict) or not isinstance(p.get('name'), str) or not 1 <= len(p['name']) <= 40:
            raise ValueError('Pattern names must contain 1–40 characters')
        if not isinstance(p.get('audio'), list) or len(p['audio']) != 4:
            raise ValueError('Four audio lanes are required')
        for lane in p['audio']:
            if not isinstance(lane, list) or len(lane) != 64:
                raise ValueError('Audio lanes contain 64 rows')
            for hit in lane:
                if hit is not None:
                    if not isinstance(hit, dict):
                        raise ValueError('Expected a note and velocity')
                    note_midi(hit.get('note'))
                    number(hit.get('vel'), 0, 127, 'Velocity')
        if not isinstance(p.get('keys'), dict) or set(p['keys']) != set(LANES) or not isinstance(p.get('easing'), dict):
            raise ValueError('Three automation lanes are required')
        for name, (lo, hi, _) in LANES.items():
            if p['easing'].get(name) not in ('hold', 'linear', 'smooth'):
                raise ValueError('Interpolation must be hold, linear or smooth')
            keys = p['keys'][name]
            if not isinstance(keys, list) or len(keys) > 64:
                raise ValueError('At most 64 keys per lane')
            previous = -1
            for key in keys:
                if not isinstance(key, dict) or type(key.get('r')) is not int or not previous < key['r'] <= 63:
                    raise ValueError('Key rows must be unique, sorted integers from 0 to 63')
                number(key.get('v'), lo, hi, 'Key value')
                previous = key['r']
    return d


def value_at(pattern, lane, row):
    keys = pattern['keys'][lane]
    if not keys:
        return LANES[lane][2]
    if row <= keys[0]['r']:
        return keys[0]['v']
    for a, b in zip(keys, keys[1:]):
        if row < b['r']:
            t = (row - a['r']) / (b['r'] - a['r'])
            ease = pattern['easing'][lane]
            if ease == 'hold':
                t = 0.
            elif ease == 'smooth':
                t = t * t * (3 - 2 * t)
            return a['v'] + (b['v'] - a['v']) * t
    return keys[-1]['v']


class Tracker:
    def __init__(self, document=None):
        if document is None:
            document = json.loads(Path(__file__).with_name('tracker').joinpath('demo.json').read_text())
        self.document = validate(document)
        self.revision = 0
        # Last revision written by a score edit. Studio's own program-output
        # syncs advance ``revision`` but not this, so they are never conflicts.
        self.edit_revision = 0
        self.enabled = False
        self.transport_revision = 0
        self.anchor_time = self.anchor_beat = 0.

    def normalize(self, beat):
        start, length = (self.document['pattern'] * 16, 16) if self.document['loop'] else (0, 64)
        return start + (beat - start) % length

    def beat(self, clock_time):
        elapsed = clock_time - self.anchor_time if self.enabled else 0.
        return self.normalize(self.anchor_beat + elapsed * self.document['bpm'] / 60)

    def seek(self, beat, clock_time, *, discontinuity=True):
        self.anchor_beat, self.anchor_time = self.normalize(beat), clock_time
        if discontinuity:
            self.transport_revision += 1

    def replace(self, document, clock_time, revision):
        """Apply a score edit based on ``revision``.

        An edit is a conflict only if another score edit landed after its base.
        If just Studio's program-output syncs happened since, the edit is
        accepted and the server's current program output is kept, because the
        editor could not have known about that change.
        """
        if type(revision) is not int or not self.edit_revision <= revision <= self.revision:
            raise ValueError('Score changed in another window. Reload before editing.')
        prepared = validate(document)
        if revision != self.revision:
            prepared['modes']['program'] = self.document['modes']['program']
        self._commit(prepared, clock_time)
        self.edit_revision = self.revision

    def sync_program(self, output, clock_time):
        """Record Studio's displayed output without invalidating open editors."""
        if self.document['modes']['program'] == output:
            return
        document = copy.deepcopy(self.document)
        document['modes']['program'] = output
        self._commit(validate(document), clock_time)

    def _commit(self, prepared, clock_time):
        beat = self.beat(clock_time)
        self.document = prepared
        self.revision += 1
        self.seek(beat, clock_time, discontinuity=False)  # tempo changes preserve position

    def apply(self, patch, clock_time):
        if not self.enabled:
            return
        beat = self.beat(clock_time)
        pattern, row = self.document['patterns'][int(beat // 16)], beat % 16 * 4
        for lane, name in self.document['bindings'].items():
            if name in patch.controls:
                patch.controls[name].set(value_at(pattern, lane, row))

    def snapshot(self, clock_time, playing):
        return dict(document=self.document, revision=self.revision, enabled=self.enabled,
                    beat=self.beat(clock_time), playing=playing and self.enabled,
                    transport_revision=self.transport_revision)

    def frames(self, patch, *, seconds, fps, size, backend, start=0., beat=0.):
        """Apply automation at the render boundary, not at a pipelined JPEG yield.

        GL's normal export iterator renders one frame ahead. Changing parameters
        outside that iterator would therefore apply the next key one frame late.
        Tracker exports use explicit steps; ordinary exports keep their pipeline.
        """
        if backend == 'gl':
            from .backend_gl import GLRenderer as Renderer
        else:
            from .backend_np import Renderer
        from .backend_np import to_uint8
        from .live import clear_recordings
        renderer = Renderer(patch, size=size, fps=fps)
        self.seek(beat, start)
        try:
            if hasattr(renderer, 'prewarm'):
                renderer.prewarm([start + i / fps for i in range(round(seconds * fps))])
            clear_recordings()
            for probe in patch.probes:
                if hasattr(probe, 'clear'):
                    probe.clear()
            for i in range(round(seconds * fps)):
                t = start + i / fps
                self.apply(patch, t)
                renderer.step(t)
                yield renderer.read() if hasattr(renderer, 'read') else to_uint8(renderer.displayed(), patch.composite_alpha)
        finally:
            if hasattr(renderer, 'release'):
                renderer.release()
