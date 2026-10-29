"""Studio's musical playhead and small, validated automation score.

The render thread owns this object. Moving the playhead never seeks a renderer,
replays hooks, or clears feedback. Browser audio is deliberately outside it.
"""
from __future__ import annotations

import bisect
import copy
import json
import math
import re
from pathlib import Path

from . import tracker_v3
from .sound_controls import controls, validate_sounds

LANES = {'z': (0., 1., .35), 'roll': (-45., 45., 0.), 'feedback': (0., .9, .25)}
SCENE_NAME = re.compile(r'[A-Za-z0-9_-]{1,24}')

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
    if isinstance(document, dict) and document.get('format') == tracker_v3.FORMAT:
        return tracker_v3.validate(document)
    d = copy.deepcopy(document)
    if not isinstance(d, dict) or d.get('format') not in ('hydrapy-tracker-1', 'hydrapy-tracker-2'):
        raise ValueError('Expected a hydrapy-tracker-1, hydrapy-tracker-2 or hydrapy-tracker-3 score')
    validate_sounds(d)
    number(d.get('bpm'), 60, 200, 'Tempo')
    # bool is a numeric subclass in Python, but JSON false is not the key C.
    # The Rack parser and score schema distinguish booleans from numbers too.
    number(d.get('key'), 0, 7, 'Key')
    if d.get('key') not in (0, 2, 5, 7):
        raise ValueError('Choose C, D, F or G minor')
    if type(d.get('pattern')) is not int or not 0 <= d['pattern'] < 4:
        raise ValueError('Choose pattern 0–3')
    for name in ('loop', 'sound'):
        if type(d.get(name)) is not bool:
            raise ValueError(f'{name} must be boolean')
    if not isinstance(d.get('mutes'), list) or len(d['mutes']) != 4 or any(type(m) is not bool for m in d['mutes']):
        raise ValueError('Four audio mute states are required')
    validate_outputs(d)
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
                    if 'locks' in hit:
                        if d['format'] != 'hydrapy-tracker-2':
                            raise ValueError('Sound locks require hydrapy-tracker-2')
                        controls(hit['locks'], machine=True)
                    note_midi(hit.get('note'))
                    number(hit.get('vel'), 0, 127, 'Velocity')
        validate_automation(p, 64)
    tracker_v3.finite(d)        # in the unknown keys a score keeps too, as version 3 (score-v3.md)
    return d


def validate_outputs(d):
    """Monitor routes and parameter bindings, which every format shares."""
    if not isinstance(d.get('modes'), dict) or set(d['modes']) != {'program', 'aux'} or any(v not in ('o0', 'o1', 'o2', 'o3') for v in d['modes'].values()):
        raise ValueError('Route program and auxiliary to o0–o3')
    bindings = d.get('bindings')
    if not isinstance(bindings, dict) or set(bindings) != set(LANES) or any(not isinstance(v, str) or len(v) > 120 for v in bindings.values()):
        raise ValueError('Three named parameter bindings are required; use an empty name to disconnect')
    named = [v for v in bindings.values() if v]
    if len(named) != len(set(named)):
        raise ValueError('Bind each parameter to only one lane')


def validate_automation(p, rows):
    """A pattern's keys, easing and scenes, for a pattern of ``rows`` rows (64 before version 3)."""
    if not isinstance(p.get('keys'), dict) or set(p['keys']) != set(LANES) or not isinstance(p.get('easing'), dict):
        raise ValueError('Three automation lanes are required')
    for name, (lo, hi, _) in LANES.items():
        if p['easing'].get(name) not in ('hold', 'linear', 'smooth'):
            raise ValueError('Interpolation must be hold, linear or smooth')
        keys = p['keys'][name]
        if not isinstance(keys, list) or len(keys) > rows:
            raise ValueError(f'At most {rows} keys per lane')
        previous = -1
        for key in keys:
            if not isinstance(key, dict) or type(key.get('r')) is not int or not previous < key['r'] < rows:
                raise ValueError(f'Key rows must be unique, sorted integers from 0 to {rows - 1}')
            number(key.get('v'), lo, hi, 'Key value')
            previous = key['r']
    # The SCENE lane is optional, so scores saved before it existed still load.
    scenes = p.setdefault('scenes', [])
    if not isinstance(scenes, list) or len(scenes) > rows:
        raise ValueError(f'At most {rows} scene changes per pattern')
    previous = -1
    for key in scenes:
        if not isinstance(key, dict) or type(key.get('r')) is not int or not previous < key['r'] < rows:
            raise ValueError(f'Scene rows must be unique, sorted integers from 0 to {rows - 1}')
        if not isinstance(key.get('name'), str) or not SCENE_NAME.fullmatch(key['name']):
            raise ValueError('Scene names are 1-24 letters, digits, - or _')
        key.setdefault('fade', 0)
        if type(key['fade']) is not int or not 0 <= key['fade'] <= rows:
            raise ValueError(f'A scene fade is 0-{rows} rows')
        if set(key) != {'r', 'name', 'fade'}:
            raise ValueError('A scene change has a row, a name and a fade')
        previous = key['r']


def cycle_beats(document):
    """The playing cycle's length in beats: a pattern or the song, or version 3's chain."""
    if document['format'] == tracker_v3.FORMAT:
        return tracker_v3.timeline(document).cycle / tracker_v3.TICKS
    return 16 if document['loop'] else 64


def scene_events(document):
    """Scene changes in the playing cycle as (beat, name, fade in beats), in order.

    In version 3 each chain entry contributes its pattern's changes, at that
    pattern's own rate, so a pattern played twice changes scene twice.
    """
    if document['format'] == tracker_v3.FORMAT:
        t = tracker_v3.timeline(document)
        events = []
        for start, p in zip(t.starts, t.entries):
            pattern = document['patterns'][p]
            if pattern is not None:
                rows = 4 * pattern['rate']
                events += [(start / tracker_v3.TICKS + key['r'] / rows, key['name'], key['fade'] / rows)
                           for key in pattern.get('scenes', [])]
        return events
    patterns = [document['pattern']] if document['loop'] else range(4)
    return [(p * 16 + key['r'] / 4, key['name'], key['fade'] / 4)
            for p in patterns for key in document['patterns'][p].get('scenes', [])]


def scene_fades(document, scenes):
    """The (from, to) scene pairs the score crossfades between, per output."""
    pairs = set()
    for index in {value[0] for value in scenes.values()}:
        events = [e for e in scene_events(document) if e[1] in scenes and scenes[e[1]][0] == index]
        for k, (_, name, fade) in enumerate(events):
            before = events[k - 1][1]
            if fade > 0 and before != name:
                pairs.add((before, name))
    return sorted(pairs)


def scene_weights(document, beat, scenes):
    """How much of each patch scene shows at ``beat`` (name -> 0..1).

    ``scenes`` is the patch's name -> (output index, ...) in declaration order.
    Each output follows the changes that name its own scenes; a change holds
    until the next one and wraps round the cycle. An output with no change in
    the cycle shows its first declared scene. A fade crossfades from the scene
    before over ``fade`` rows, smoothstepped; it never runs past the next change
    (so a later change never jumps from a half-finished fade), and a change to
    the scene already showing doesn't fade at all.
    """
    order = {name: k for k, name in enumerate(scenes)}
    length = cycle_beats(document)
    weights = {name: 0. for name in scenes}
    for index in sorted({value[0] for value in scenes.values()}):
        mine = [name for name, value in scenes.items() if value[0] == index]
        events = [e for e in scene_events(document) if e[1] in mine]
        if not events:
            weights[mine[0]] = 1.
            continue
        # The change most recently passed, going back round the cycle.
        position = min(range(len(events)), key=lambda k: (beat - events[k][0]) % length)
        when, name, fade = events[position]
        weights[name] = 1.
        elapsed = (beat - when) % length
        gap = (events[(position + 1) % len(events)][0] - when) % length or length
        fade = min(fade, gap)
        before = events[position - 1][1]
        if fade > 0 and elapsed < fade and before != name:
            t = elapsed / fade
            t = t * t * (3 - 2 * t)
            if order[name] > order[before]:
                weights[before] = 1.               # blended later in the chain: fade the newcomer in
                weights[name] = t
            else:
                weights[before] = 1. - t           # blended earlier: fade the outgoing one out over it
    return weights


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


class LaneState:
    """The audio lanes as the patch sees them, per frame: what TrackerCV sends as CV.

    Gates follow TrackerCV (half a row, muted lanes and zero velocity silent).
    A hit starts when the playhead crosses a row holding one, so every row
    passed since the last frame counts, even at a low frame rate, and a hit
    that has been muted or deleted never fires again. Note and velocity hold
    the last hit; a seek or a new transport clears them.

    There are six lanes; 5 and 6 stay silent before version 3. A version-3 score
    (``update_v3``) plays the candidates that fire, judged once at their tick.
    """
    GATE = .5

    def __init__(self):
        self.transport = None
        self.note = [60.] * 6
        self.vel = [0.] * 6
        self.since = [None] * 6     # seconds since each lane's latest hit
        self.last_row = self.last_beat = None
        self.started = [None] * 6   # version 3: the tick of each lane's latest hit
        self.segment = None
        self.moving = False
        self.new = []               # version 3: the candidates the last update evaluated
        self._follower = tracker_v3.Follower()

    def evaluated(self, segment):
        """The first tick of version-3 ``segment`` not yet evaluated, or None."""
        return self._follower.evaluated(segment)

    @staticmethod
    def _hit(document, row, lane):
        hit = document['patterns'][row // 64]['audio'][lane][row % 64]
        return hit if hit is not None and hit['vel'] > 0 and not document['mutes'][lane] else None

    def _hold(self, document, lane, hit):
        midi = note_midi(hit['note'])
        if midi is not None:
            self.note[lane] = float(midi + document['key'] - 2)
        self.vel[lane] = hit['vel'] / 127

    def update(self, document, beat, moving, transport):
        if transport != self.transport:
            self.__init__()
            self.transport = transport
        first, rows = (document['pattern'] * 64, 64) if document['loop'] else (0, 256)
        length, seconds = rows / 4, 60 / document['bpm']
        current = min(int(beat * 4), first + rows - 1)
        rowpos = beat % 16 * 4
        gate = [0.] * 6
        if not moving:
            self.last_row = self.last_beat = None
            return dict(gate=gate, since=[None] * 6, note=list(self.note), vel=list(self.vel),
                        beat=beat, row=float(current % 64))
        if self.last_row is None:
            crossed = [current]
        else:
            count = min((current - self.last_row) % rows, rows)
            crossed = [first + (self.last_row - first + k) % rows for k in range(1, count + 1)]
        elapsed = (beat - self.last_beat) % length if self.last_beat is not None else 0.
        for lane in range(4):
            started = next((r for r in reversed(crossed) if self._hit(document, r, lane)), None)
            if started is not None:
                self._hold(document, lane, self._hit(document, started, lane))
                self.since[lane] = ((beat - started / 4) % length) * seconds
            elif self.since[lane] is not None:
                self.since[lane] += elapsed * seconds
            hit = self._hit(document, current, lane)
            if hit is not None:
                self._hold(document, lane, hit)          # an edit to the playing cell shows at once
                gate[lane] = 1. if rowpos - int(rowpos) < self.GATE else 0.
        self.last_row, self.last_beat = current, beat
        return dict(gate=gate, since=list(self.since), note=list(self.note), vel=list(self.vel),
                    beat=beat, row=float(current % 64))

    def update_v3(self, snapshot, u, moving):
        """``update`` for a version-3 score, at elapsed tick ``u`` of the playhead ``snapshot``."""
        d = snapshot['document']
        segment = (snapshot['transport_revision'], snapshot['start']['tick'])
        if segment != self.segment:
            # A transport change (a new transport_revision, or Play) heard as it starts plays the
            # anchor's row and clears the holds; a new segment without one (a change of format)
            # starts from the current tick.
            live = self.transport is not None and (snapshot['transport_revision'] != self.transport
                                                   or moving and not self.moving)
            if live or self.transport is None:
                self.__init__()
            self.transport, self.segment, self.started = snapshot['transport_revision'], segment, [None] * 6
            self._follower.begin(-7680 if live else math.floor(u), segment)
        self.moving = moving
        beat, _, row = tracker_v3.position(d, snapshot['start']['tick'], [u])
        gate = [0.] * 6
        if not moving:
            self.new = []
            return dict(gate=gate, since=[None] * 6, note=list(self.note), vel=list(self.vel),
                        beat=float(beat[0]), row=float(int(row[0])))
        self.new = self._follower.advance(snapshot, u)
        for c in self.new:
            if c['fires']:
                midi = note_midi(c['note'])
                if midi is not None:
                    self.note[c['lane']] = float(midi + d['key'] - 2)
                self.vel[c['lane']], self.started[c['lane']] = c['vel'] / 127, c['at']
        gap = .001 * d['bpm'] / 60 * tracker_v3.TICKS
        previous = [None] * 6
        for c in self._follower.candidates:
            if c['fires'] and c['at'] <= u:
                lane = c['lane']
                start = c['at'] + (gap if tracker_v3.retriggers(previous[lane], c) else 0)
                gate[lane] = float(start <= u < c['at'] + tracker_v3.gate_duration(c, self.GATE))
                previous[lane] = c
        self._follower.forget(u - gap)
        seconds = 60 / d['bpm'] / tracker_v3.TICKS
        self.since = [None if at is None else (u - at) * seconds for at in self.started]
        return dict(gate=gate, since=list(self.since), note=list(self.note), vel=list(self.vel),
                    beat=float(beat[0]), row=float(int(row[0])))


def _timeline(document):
    """Version 3's timeline for any score: versions 1 and 2 as they map onto version 3."""
    if document['format'] == tracker_v3.FORMAT:
        t = tracker_v3.timeline(document)
        return t.loop, list(t.entries), list(t.starts)
    entries = [document['pattern']] if document['loop'] else [0, 1, 2, 3]
    return document['loop'], entries, [k * 16 * tracker_v3.TICKS for k in range(len(entries) + 1)]


def _carry(before, after, entry, offset):
    """The new anchor's cycle tick for score-v3.md's transport changes.

    ``before`` and ``after`` are timelines (loop, entries, starts); the playhead
    was ``offset`` ticks into entry ``entry`` of ``before``.
    """
    loop, entries, starts = after
    length = [b - a for a, b in zip(starts, starts[1:])]
    if loop != before[0]:
        if loop:            # the selected pattern: if it was playing, keep the place in it
            return offset % length[0] if before[1][entry] == entries[0] else 0
        if before[1][0] in entries:     # the first chain entry that plays the looped pattern
            e = entries.index(before[1][0])
            return starts[e] + offset % length[e]
        return 0
    if loop:                # a new looped pattern, or a new length: keep the elapsed beats, wrapped
        return offset % length[0]
    if entry < len(entries) and entries[entry] == before[1][entry]:
        return starts[entry] + offset % length[entry]
    return 0


class Tracker:
    """Studio's playhead and score.

    Versions 1 and 2 anchor a beat. Version 3 (Gorgon's evaluator.md) plays
    segments: ``start`` is the anchor ``{tick, entry}``, ``elapsed`` the whole
    ticks played at ``anchor_time``, ``fill`` Studio's FILL transitions and
    ``bus`` the layer of FILL and lane mutes a Rack module writes (bus-io.md).
    """
    LEASE = 3.          # seconds without a window before the bus layer is cleared
    FOLD = 10.          # how far behind the frontier transitions may be folded, in seconds
    FOLD_AFTER = 32     # fold a list once it holds more transitions than this

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
        self.start, self.elapsed = {'tick': 0, 'entry': None}, 0
        # FILL and the bus layer, shared live with Studio's outputs (tracker_v3.Layers).
        self.layers = tracker_v3.Layers()
        self.lease = None           # [sender, monotonic time of its last request]
        self.trace = None           # a text file that takes JSONL trace lines, or None
        self._lanes = LaneState()
        if self.v3:
            self._segment(0, 0.)

    @property
    def v3(self):
        return self.document['format'] == tracker_v3.FORMAT

    @property
    def fill(self):
        return self.layers.fill

    @fill.setter
    def fill(self, transitions):
        self.layers.fill = transitions

    @property
    def bus(self):
        return self.layers.bus

    @bus.setter
    def bus(self, layer):
        self.layers.bus = layer

    @property
    def segment(self):
        return self.transport_revision, self.start['tick']

    def normalize(self, beat):
        if self.v3:
            return beat % cycle_beats(self.document)
        start, length = (self.document['pattern'] * 16, 16) if self.document['loop'] else (0, 64)
        return start + (beat - start) % length

    def ticks(self, clock_time, bpm=None):
        """Version 3's elapsed ticks ``u`` at ``clock_time``, a real number."""
        if not self.enabled:
            return float(self.elapsed)
        return self.elapsed + (clock_time - self.anchor_time) * (bpm or self.document['bpm']) / 60 * tracker_v3.TICKS

    def beat(self, clock_time):
        if self.v3:
            cycle = tracker_v3.timeline(self.document).cycle
            return (self.start['tick'] + self.ticks(clock_time)) % cycle / tracker_v3.TICKS
        elapsed = clock_time - self.anchor_time if self.enabled else 0.
        return self.normalize(self.anchor_beat + elapsed * self.document['bpm'] / 60)

    def seek(self, beat, clock_time, *, discontinuity=True):
        if self.v3:
            if not discontinuity:
                self._reanchor(clock_time, self.document['bpm'])
                return
            # A seek lands on the nearest tick (evaluator.md#ticks).
            with self.layers.lock:
                self.transport_revision += 1
                self._segment(math.floor(beat * tracker_v3.TICKS + .5) % tracker_v3.timeline(self.document).cycle,
                              clock_time)
            return
        self.anchor_beat, self.anchor_time = self.normalize(beat), clock_time
        if discontinuity:
            self.transport_revision += 1

    def play(self, clock_time):
        """Studio started playing: version 3 starts a segment at the tick nearest the playhead.

        Play doesn't change ``transport_revision``; ``start`` tells the segments apart.
        So a Play that lands on the anchor's own tick (a pause right after a seek, or
        after whole cycles) continues the segment instead: a new one would share its key.
        """
        self._lanes.moving = False          # the patch's lane signals hear the resume too
        if self.v3 and self.enabled:
            tick = round(self.start['tick'] + self.ticks(clock_time)) % tracker_v3.timeline(self.document).cycle
            if tick != self.start['tick']:
                with self.layers.lock:
                    self._segment(tick, clock_time)

    def _segment(self, tick, clock_time):
        """Start a segment at cycle ``tick``, with FILL and the bus layer in the state last set.

        That includes a change stamped at a tick the old segment never reached, such as FILL
        pressed while paused: it was in force for everything Studio hadn't yet evaluated.
        """
        def carried(transitions):
            return [{'at': 0, 'on': transitions[-1]['on']}]
        with self.layers.lock:
            self.fill = carried(self.fill)
            self.bus = {'fill': carried(self.bus['fill']), 'mutes': [carried(m) for m in self.bus['mutes']]}
            self.start, self.elapsed = tracker_v3.anchor(self.document, tick), 0
            self.anchor_time, self.anchor_beat = clock_time, tick / tracker_v3.TICKS
            self.layers.segment = self.segment

    def _reanchor(self, clock_time, bpm):
        """Keep the segment for a tempo change: ``elapsed`` becomes the whole tick most recently
        reached at ``bpm``, and ``anchor_time`` the clock time at which the playhead reached it."""
        if self.enabled:
            u = self.ticks(clock_time, bpm)
            self.elapsed = math.floor(u)
            self.anchor_time = clock_time - (u - self.elapsed) * 60 / bpm / tracker_v3.TICKS
        else:
            self.anchor_time = clock_time
        cycle = tracker_v3.timeline(self.document).cycle
        self.anchor_beat = (self.start['tick'] + self.elapsed) % cycle / tracker_v3.TICKS

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
        old = self.document
        if old['format'] != tracker_v3.FORMAT and prepared['format'] != tracker_v3.FORMAT:
            beat = self.beat(clock_time)
            self.document = prepared
            self.revision += 1
            self.seek(beat, clock_time, discontinuity=False)  # tempo changes preserve position
            return
        # Where the playhead is, as an entry and the ticks into it, on the old score's timeline.
        before, after = _timeline(old), _timeline(prepared)
        if self.v3:
            tick = (self.start['tick'] + self.ticks(clock_time)) % before[2][-1]
        else:
            tick = (self.beat(clock_time) - (16 * old['pattern'] if old['loop'] else 0)) * tracker_v3.TICKS
        entry = bisect.bisect_right(before[2], tick) - 1
        moved = before != after                     # score-v3.md's transport changes
        # A looped pattern's beats count from the song's start in versions 1 and 2 and from the
        # pattern's in version 3, so changing format there moves the beat origin too.
        changes_format = old['format'] != prepared['format']
        transport = moved or changes_format and old['loop']
        tick = _carry(before, after, entry, tick - before[2][entry]) if moved else tick
        tick = round(tick) % after[2][-1]
        bpm = old['bpm']
        with self.layers.lock:
            self.document = prepared
            self.revision += 1
            if transport:
                self.transport_revision += 1
            if not self.v3:                         # back to version 1 or 2: a beat anchor
                beat = tick / tracker_v3.TICKS + (16 * prepared['pattern'] if prepared['loop'] else 0)
                self.anchor_beat, self.anchor_time = self.normalize(beat), clock_time
            elif transport or changes_format:
                self._segment(tick, clock_time)
            else:
                self._reanchor(clock_time, bpm)     # an ordinary edit: tempo changes keep the place

    # -- FILL and the bus layer (version 3) -----------------------------------
    def set_fill(self, on, stamp):
        """Studio's FILL key: on or off from ``stamp``, the first tick Studio's outputs haven't
        evaluated. Call it holding ``layers.lock``, from finding the stamp on."""
        if self.fill[-1]['on'] == on:
            return                                  # reapplying the state last set is not a change
        self.fill = [t for t in self.fill if t['at'] <= stamp] + [{'at': stamp, 'on': on}]
        self._trace({'fill': {'at': stamp, 'on': on}, 'received': stamp})

    def apply_window(self, segment, sender, window, wall, received):
        """Apply a Rack module's window of the bus layer (bus-io.md#bus-in).

        ``received`` is the first tick Studio's outputs haven't evaluated; call it holding
        ``layers.lock`` from finding that on. Returns False for a window of a segment that
        is no longer current, which is dropped.
        """
        if not isinstance(sender, str) or not re.fullmatch(r'[0-9a-f]{16}', sender):
            raise ValueError('sender is 16 hexadecimal digits')
        first = window.get('from') if isinstance(window, dict) else None
        try:
            lists = self._window_lists(window, first)
        except ValueError:
            # Traced like a window refused for the lease (evaluator.md#trace); still a 400.
            traced = first if type(first) is int and first >= 0 else None
            self._trace({'bus': {'window': 'refused', 'reason': 'malformed window', 'from': traced},
                         'received': received})
            raise
        self.expire(wall, received)
        if self.lease and self.lease[0] != sender:
            self._trace({'bus': {'window': 'refused', 'reason': 'bus held by another module', 'from': first},
                         'received': received})
            raise ValueError('bus held by another module')
        if segment != {'transport_revision': self.transport_revision, 'start': self.start}:
            if self.lease:
                self.lease[1] = wall
            self._trace({'bus': {'window': 'dropped', 'reason': 'stale segment', 'from': first}, 'received': received})
            return False
        self.lease = [sender, wall]
        old = [self.bus['fill'], *self.bus['mutes']]
        new = [[t for t in kept if t['at'] < first] + [dict(t) for t in given] for kept, given in zip(old, lists)]
        for k, (was, now) in enumerate(zip(old, new)):
            seen = {(t['at'], t['on']) for t in was if t['at'] >= first}
            for t in now:
                if t['at'] >= first and (t['at'], t['on']) not in seen and tracker_v3.state_at(was, t['at']) != t['on']:
                    change = {'fill': t} if k == 0 else {'mute': dict(t, lane=k - 1)}
                    self._trace({'bus': change, 'received': received})
        self.bus = {'fill': new[0], 'mutes': new[1:]}
        return True

    @staticmethod
    def _window_lists(window, first):
        """A bus window's seven transition lists, fill first; ValueError if it is malformed."""
        if type(first) is not int or first < 0 or set(window) != {'from', 'fill', 'mutes'} or \
                not isinstance(window['mutes'], list) or len(window['mutes']) != 6:
            raise ValueError('A bus window is {from, fill, mutes}: a tick and seven lists')
        lists = [window['fill'], *window['mutes']]
        for transitions in lists:
            if not isinstance(transitions, list) or not transitions or any(
                    not isinstance(t, dict) or set(t) != {'at', 'on'} or type(t['at']) is not int or type(t['on']) is not bool
                    for t in transitions):
                raise ValueError('Each bus list holds {at, on} transitions')
            ats = [t['at'] for t in transitions]
            if ats[0] != first or ats != sorted(ats):
                raise ValueError('Each bus list starts at the window\'s from and is sorted')
        return lists

    def expire(self, wall, stamp):
        """End a lease 3 s after its sender's last request: the layer clears from ``stamp``."""
        if self.lease and wall - self.lease[1] >= self.LEASE:
            sender, self.lease = self.lease[0], None
            def cleared(transitions):
                kept = [t for t in transitions if t['at'] <= stamp]
                return kept + [{'at': stamp, 'on': False}] if tracker_v3.state_at(kept, stamp) else kept
            self.bus = {'fill': cleared(self.bus['fill']), 'mutes': [cleared(m) for m in self.bus['mutes']]}
            self._trace({'bus': {'lease': 'ended', 'sender': sender}, 'received': stamp})

    def fold(self, stamp):
        """Fold long transition lists at least ``FOLD`` seconds behind ``stamp`` (bus-io.md#transport-endpoint)."""
        horizon = math.floor(stamp - self.FOLD * self.document['bpm'] / 60 * tracker_v3.TICKS)
        def folded(transitions):
            if len(transitions) <= self.FOLD_AFTER or horizon <= 0 or transitions[1]['at'] > horizon:
                return transitions
            return [{'at': 0, 'on': tracker_v3.state_at(transitions, horizon)}] + [t for t in transitions if t['at'] > horizon]
        self.fill = folded(self.fill)
        self.bus = {'fill': folded(self.bus['fill']), 'mutes': [folded(m) for m in self.bus['mutes']]}

    def housekeep(self, wall, stamp):
        """Once a frame: end a lapsed lease and fold long lists."""
        if self.v3:
            self.expire(wall, stamp)
            self.fold(stamp)

    def evaluated(self):
        """The first tick of the current segment the patch's lane signals haven't evaluated, or None."""
        return self._lanes.evaluated(self.segment)

    def _trace(self, record):
        if self.trace is not None:
            self.trace.write(tracker_v3.trace_line(record, self.transport_revision, self.start) + '\n')

    # -- per frame ------------------------------------------------------------
    def apply(self, patch, clock_time):
        if self.v3:
            return self._apply_v3(patch, clock_time)
        beat = self.beat(clock_time)
        # Lane signals for the patch (scenes.TrackerSignal): gates stop while disabled.
        patch.tracker_frame = self._lanes.update(self.document, beat, self.enabled, self.transport_revision)
        if not self.enabled:
            return
        pattern, row = self.document['patterns'][int(beat // 16)], beat % 16 * 4
        for lane, name in self.document['bindings'].items():
            if name in patch.controls:
                patch.controls[name].set(value_at(pattern, lane, row))
        # Disabling automation leaves the last scene showing, as it does controls.
        if patch.scenes:
            patch.scene_weights = scene_weights(self.document, beat, patch.scenes)

    def _apply_v3(self, patch, clock_time):
        u = self.ticks(clock_time)
        playhead = dict(document=self.document, transport_revision=self.transport_revision, start=self.start,
                        fill=self.fill, bus=self.bus, layers=self.layers)
        patch.tracker_frame = self._lanes.update_v3(playhead, u, self.enabled)
        for c in self._lanes.new:
            self._trace(c)
        if not self.enabled:
            return
        beat, entry, row = tracker_v3.position(self.document, self.start['tick'], [u])
        pattern = self.document['patterns'][tracker_v3.timeline(self.document).entries[entry[0]]]
        for lane, name in self.document['bindings'].items():
            if name in patch.controls:
                patch.controls[name].set(value_at(pattern, lane, row[0]) if pattern else LANES[lane][2])
        if patch.scenes:
            patch.scene_weights = scene_weights(self.document, float(beat[0]), patch.scenes)

    def playhead(self):
        """Version 3's playhead fields (evaluator.md#playhead-fields); none before version 3."""
        if not self.v3:
            return {}
        t = tracker_v3.timeline(self.document)
        return dict(start=self.start, elapsed=self.elapsed, fill=self.fill, bus=self.bus,
                    entry=None if t.loop else t.entry((self.start['tick'] + self.elapsed) % t.cycle))

    def snapshot(self, clock_time, playing):
        # The editor also gets version 3's elapsed ticks now, to place its own playhead.
        extra = dict(self.playhead(), ticks=self.ticks(clock_time)) if self.v3 else {}
        return dict(document=self.document, revision=self.revision, enabled=self.enabled,
                    beat=self.beat(clock_time), playing=playing and self.enabled,
                    transport_revision=self.transport_revision, **extra)

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
            if hasattr(renderer, 'prewarm_scenes'):
                renderer.prewarm_scenes(scene_fades(self.document, patch.scenes))
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
