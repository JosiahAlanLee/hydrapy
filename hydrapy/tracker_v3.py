"""Score format 3 (``hydrapy-tracker-3``): validation, the version-1/2 mapping and ``evaluate()``.

Gorgon's spec is the contract: ``score-v3.md`` (shape and timeline),
``evaluator.md`` (ticks, candidates, conditions, chance, FILL and the trace),
``transforms.md`` and ``bus-io.md`` (the bus layer). ``evaluate()`` is pure:
the same score, transport and interval always give the same candidates, and
splitting the interval anywhere gives the same set. Followers play the
candidates that fire; the others say why they don't.

Times are integer ticks, 1920 per quarter note. Elapsed ticks ``u`` count from
the transport anchor and don't wrap; the playhead is at cycle tick
``(anchor + u) mod cycle``.
"""
from __future__ import annotations

import bisect
import copy
import json
import math
import threading

import numpy as np

from . import studio_tracker         # read at call time: studio_tracker imports this module
from .sound_controls import controls, validate_sounds

FORMAT = 'hydrapy-tracker-3'
TICKS = 1920                    # per quarter note
LANE_COUNT = 6
PATTERN_COUNT = 128
RATES = (.25, .5, 1, 2, 4, 8)
PROB, DEGRADE = 0x70726f62, 0x64656772      # "prob" and "degr", chance's salts
M32 = 0xffffffff
CONDITIONS = {'every': ('a', 'b'), 'prob': ('p',), 'fill': (), 'first': ()}
LEAVES = {'rot': ('k',), 'rev': (), 'ply': ('n',), 'degrade': ('p',)}


def code(index):
    """A pattern's display code: 0 is A01, 17 is B02, 127 is H16."""
    return 'ABCDEFGH'[index // 16] + f'{index % 16 + 1:02d}'


def empty_pattern(index):
    """What a null pattern plays as, and exactly what an editor writes on its first edit."""
    return {'name': code(index), 'rate': 1, 'bar': 16, 'bars': 4,
            'lanes': [{'len': 64, 'cells': [None] * 64, 'transforms': []} for _ in range(LANE_COUNT)],
            'keys': {name: [] for name in studio_tracker.LANES},
            'easing': {name: 'smooth' for name in studio_tracker.LANES}, 'scenes': []}


# -- validation ---------------------------------------------------------------

def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer from {low} to {high}')
    return value


def _fields(value, kind, allowed, label):
    if set(value) - {'kind'} != set(allowed):
        raise ValueError(f'{label} {kind} takes {", ".join(allowed) or "no fields"}')


def _condition(cond):
    kind = cond.get('kind') if isinstance(cond, dict) else None
    if not isinstance(kind, str) or kind not in CONDITIONS:
        raise ValueError('A condition is every, prob, fill or first')
    if 'not' in cond and type(cond['not']) is not bool:
        raise ValueError('A condition\'s not is true or false')
    _fields({k: v for k, v in cond.items() if k != 'not'}, kind, CONDITIONS[kind], 'Condition')
    if kind == 'every':
        _integer(cond['a'], 1, 8, 'every a')
        if _integer(cond['b'], 1, 8, 'every b') < cond['a']:
            raise ValueError('every a:b needs a ≤ b')
    elif kind == 'prob':
        _integer(cond['p'], 1, 99, 'A probability')


def _transform(node, depth):
    """Check one node; return the product of the ``ply`` counts in it."""
    kind = node.get('kind') if isinstance(node, dict) else None
    if kind == 'every':
        if depth == 1:
            raise ValueError('every nests at most three deep')
        _fields(node, kind, ('a', 'b', 'inner'), 'Transform')
        _integer(node['a'], 1, 8, 'every a')
        if _integer(node['b'], 1, 8, 'every b') < node['a']:
            raise ValueError('every a:b needs a ≤ b')
        return _transform(node['inner'], depth - 1)
    if not isinstance(kind, str) or kind not in LEAVES:
        raise ValueError('A transform is rot, rev, ply, degrade or every')
    _fields(node, kind, LEAVES[kind], 'Transform')
    if kind == 'rot':
        _integer(node['k'], -127, 127, 'rot')
    elif kind == 'degrade':
        _integer(node['p'], 1, 99, 'degrade')
    elif kind == 'ply':
        return _integer(node['n'], 2, 4, 'ply')
    return 1


def _pattern(p):
    if not isinstance(p, dict) or not isinstance(p.get('name'), str) or not 1 <= len(p['name']) <= 40:
        raise ValueError('Pattern names must contain 1–40 characters')
    if isinstance(p.get('rate'), bool) or p.get('rate') not in RATES:
        raise ValueError('Rate must be ¼, ½, 1, 2, 4 or 8')
    rows = _integer(p.get('bar'), 1, 32, 'Rows per bar') * _integer(p.get('bars'), 1, 8, 'Bars')
    if rows > 128:
        raise ValueError('A pattern has at most 128 rows (bar × bars)')
    if not isinstance(p.get('lanes'), list) or len(p['lanes']) != LANE_COUNT:
        raise ValueError('Six lanes are required')
    for lane in p['lanes']:
        if not isinstance(lane, dict):
            raise ValueError('A lane has len, cells and transforms')
        length = _integer(lane.get('len'), 1, rows, 'Lane length')
        if not isinstance(lane.get('cells'), list) or len(lane['cells']) != length:
            raise ValueError('A lane has one cell per row of its length')
        for hit in lane['cells']:
            if hit is not None:
                if not isinstance(hit, dict):
                    raise ValueError('Expected a note and velocity')
                studio_tracker.note_midi(hit.get('note'))
                studio_tracker.number(hit.get('vel'), 0, 127, 'Velocity')
                if 'locks' in hit:
                    controls(hit['locks'], machine=True)
                if 'cond' in hit:
                    _condition(hit['cond'])
        transforms = lane.get('transforms')
        if not isinstance(transforms, list) or len(transforms) > 8:
            raise ValueError('A lane has at most eight transforms; [] for none')
        product = 1
        for node in transforms:
            product *= _transform(node, 3)
        if product > 4:
            raise ValueError('The ply counts in a lane multiply to at most 4')
    studio_tracker.validate_automation(p, rows)


def validate(document):
    """Copy, then check everything ``score-v3.md`` and the schema require."""
    d = copy.deepcopy(document)
    if not isinstance(d, dict) or d.get('format') != FORMAT:
        raise ValueError('Expected a hydrapy-tracker-3 score')
    validate_sounds(d)
    studio_tracker.number(d.get('bpm'), 60, 200, 'Tempo')
    studio_tracker.number(d.get('key'), 0, 7, 'Key')
    if d['key'] not in (0, 2, 5, 7):
        raise ValueError('Choose C, D, F or G minor')
    _integer(d.get('pattern'), 0, PATTERN_COUNT - 1, 'The selected pattern')
    for name in ('loop', 'sound'):
        if type(d.get(name)) is not bool:
            raise ValueError(f'{name} must be boolean')
    chain = d.get('chain')
    if not isinstance(chain, list) or not 1 <= len(chain) <= 256:
        raise ValueError('The chain has 1–256 entries')
    for entry in chain:
        _integer(entry, 0, PATTERN_COUNT - 1, 'A chain entry')
    _integer(d.get('seed'), 0, M32, 'Seed')
    if not isinstance(d.get('mutes'), list) or len(d['mutes']) != LANE_COUNT or any(type(m) is not bool for m in d['mutes']):
        raise ValueError('Six lane mute states are required')
    lanes = d.get('lanes')
    if not isinstance(lanes, list) or len(lanes) != LANE_COUNT or any(
            not isinstance(lane, dict) or not isinstance(lane.get('name'), str) or len(lane['name']) > 24 for lane in lanes):
        raise ValueError('Six lane names of at most 24 characters are required')
    studio_tracker.validate_outputs(d)
    if not isinstance(d.get('patterns'), list) or len(d['patterns']) != PATTERN_COUNT:
        raise ValueError('128 patterns are required; null is an empty one')
    for p in d['patterns']:
        if p is not None:
            _pattern(p)
    finite(d)
    return d


def finite(document):
    """score-v3.md: non-finite numbers are rejected everywhere, in the unknown keys a score keeps too.

    Version-1 and 2 scores follow the same rule (studio_tracker.validate).
    """
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('A score holds only finite numbers (no NaN or Infinity)')
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def upgrade(document):
    """The version-3 score a valid version-1 or 2 score maps onto, playing the same song (score-v3.md)."""
    d = copy.deepcopy(document)
    sounds = d.get('sounds') or [{'machine': None, 'params': {}} for _ in range(4)]
    d.update(format=FORMAT, chain=[0, 1, 2, 3], seed=0, mutes=d['mutes'] + [False, False],
             lanes=[{'name': f'Lane {n}'} for n in range(1, LANE_COUNT + 1)],
             sounds=sounds + [{'machine': None, 'params': {}} for _ in range(2)])
    for p in d['patterns']:
        cells = p.pop('audio') + [[None] * 64 for _ in range(2)]
        p.update(rate=1, bar=16, bars=4, lanes=[{'len': 64, 'cells': c, 'transforms': []} for c in cells])
    d['patterns'] += [None] * (PATTERN_COUNT - 4)
    return validate(d)


# -- chance -------------------------------------------------------------------

def lowbias32(x):
    x &= M32
    x ^= x >> 16
    x = x * 0x7feb352d & M32
    x ^= x >> 15
    x = x * 0x846ca68b & M32
    return x ^ x >> 16


def chance(seed, salt, pattern, lane, row, bar_count, sub):
    """evaluator.md's 32-bit hash of the seed and an occurrence. Its value mod 100 is the draw."""
    h = lowbias32(seed ^ salt)
    for value in (pattern, lane, row, bar_count & M32, sub):
        h = lowbias32(h ^ value)
    return h


# -- the timeline -------------------------------------------------------------

def geometry(pattern):
    """``(W, rows, bar, bars)``: ticks per row and the shape of a pattern; None is the empty pattern."""
    if pattern is None:
        return 480, 64, 16, 4
    return TICKS // round(4 * pattern['rate']), pattern['bar'] * pattern['bars'], pattern['bar'], pattern['bars']


class Timeline:
    """The cycle in ticks: its entries, their starts ``S`` and its length ``C`` (score-v3.md)."""

    def __init__(self, score):
        self.loop = score['loop']
        self.entries = [score['pattern']] if self.loop else list(score['chain'])
        self.starts = [0]
        for p in self.entries:
            w, rows, _, _ = geometry(score['patterns'][p])
            self.starts.append(self.starts[-1] + w * rows)
        self.cycle = self.starts[-1]
        n = len(self.entries)
        # How many entries just before each one play its pattern too, going back
        # round the cycle. None when every entry does: then a visit never ends.
        self.runs = None
        if any(p != self.entries[0] for p in self.entries):
            self.runs = []
            for i, p in enumerate(self.entries):
                run = 0
                while self.entries[(i - run - 1) % n] == p:
                    run += 1
                self.runs.append(run)

    def entry(self, tick):
        """The entry containing cycle tick ``tick``."""
        return bisect.bisect_right(self.starts, tick) - 1


_timeline_cache = None


def timeline(score):
    """The score's ``Timeline``, cached for the score object last asked about.

    Scores are treated as immutable, as Studio's are: an edit is a new document.
    """
    global _timeline_cache
    cached = _timeline_cache            # one read, so another thread can't swap it midway
    if cached is None or cached[0] is not score:
        cached = _timeline_cache = (score, Timeline(score))
    return cached[1]


def anchor(score, tick):
    """The transport anchor ``{tick, entry}`` for cycle tick ``tick``, ``entry`` None when looping."""
    t = timeline(score)
    return {'tick': tick, 'entry': None if t.loop else t.entry(tick)}


# -- FILL and the bus layer ---------------------------------------------------

class Transitions:
    """An ``[{at, on}…]`` list: on at ``u`` if the last transition at or before ``u`` is on.

    Of two at the same tick the later counts, and before the anchor (``u < 0``)
    the list has the anchor's state.
    """

    def __init__(self, transitions=None):
        transitions = transitions or [{'at': 0, 'on': False}]
        self.ats = [t['at'] for t in transitions]
        self.ons = [t['on'] for t in transitions]

    def __call__(self, u):
        return self.ons[max(bisect.bisect_right(self.ats, max(u, 0)) - 1, 0)]


def state_at(transitions, u):
    return Transitions(transitions)(u)


class _Layers:
    """Studio's FILL and the bus layer, merged by OR (bus-io.md)."""

    def __init__(self, transport):
        bus = transport.get('bus') or {}
        self.fill = (Transitions(transport.get('fill')), Transitions(bus.get('fill')))
        self.mutes = [Transitions(m) for m in bus.get('mutes') or [None] * LANE_COUNT]

    def fill_on(self, u):
        return self.fill[0](u) or self.fill[1](u)


# -- evaluate() ---------------------------------------------------------------

def condition_code(cond):
    """``a:b``, ``%p``, ``F`` or ``1ST``, with ``!`` when negated."""
    text = {'every': lambda: f"{cond['a']}:{cond['b']}", 'prob': lambda: f"%{cond['p']}",
            'fill': lambda: 'F', 'first': lambda: '1ST'}[cond['kind']]()
    return ('!' if cond.get('not') else '') + text


def _judge(cond, seed, pattern, lane, row, bar_count, fill):
    """Whether a condition holds on an occurrence, and its deciding value."""
    kind = cond['kind']
    if kind == 'every':
        holds, why = bar_count % cond['b'] == cond['a'] - 1, f"bar {bar_count} mod {cond['b']} = {bar_count % cond['b']}"
    elif kind == 'prob':
        h = chance(seed, PROB, pattern, lane, row, bar_count, 0)
        holds, why = h % 100 < cond['p'], f'chance {h} mod 100 = {h % 100}'
    elif kind == 'fill':
        holds, why = fill, f"fill {'on' if fill else 'off'}"
    else:
        holds, why = bar_count == 0, f'bar {bar_count}'
    return holds != bool(cond.get('not')), why


def _leaves(transforms, bar_count):
    """The active leaves in order, each with its node number and JSON Pointer (transforms.md, step 1)."""
    leaves, count = [], 0

    def walk(node, pointer, active):
        nonlocal count
        number_ = count
        count += 1
        if node['kind'] == 'every':
            walk(node['inner'], pointer + '/inner', active and bar_count % node['b'] == node['a'] - 1)
        elif active:
            leaves.append((node, number_, pointer))
    for k, node in enumerate(transforms):
        walk(node, f'/{k}', True)
    return leaves


def _row(out, score, layers, u0, u1, play, entry, p, row, row_at, w, bar_count, fill):
    """Append the candidates of one pattern row that start in ``[u0, u1)`` and end after the anchor."""
    for lane_index, lane in enumerate(score['patterns'][p]['lanes']):
        leaves = _leaves(lane['transforms'], bar_count)
        length = lane['len']
        lane_row = cell = row % length
        for node, _, _ in reversed(leaves):
            if node['kind'] == 'rot':
                cell = (cell - node['k']) % length
            elif node['kind'] == 'rev':
                cell = length - 1 - cell
        hit = lane['cells'][cell]
        if hit is None or hit['vel'] <= 0:
            continue
        events = [(0, 1, None)]                 # (sub-step, of, why a degrade dropped it)
        for node, number_, pointer in leaves:
            if node['kind'] == 'ply':
                n = node['n']
                events = [(s * n + q, m * n, dropped) for s, m, dropped in events for q in range(n)]
            elif node['kind'] == 'degrade':
                kept = []
                for s, m, dropped in events:
                    if dropped is None:
                        h = chance(score['seed'], DEGRADE + number_, p, lane_index, row, bar_count, s)
                        if h % 100 >= node['p']:
                            dropped = f'degrade {pointer}: chance {h} mod 100 = {h % 100}'
                    kept.append((s, m, dropped))
                events = kept
        cond = hit.get('cond')
        holds, cond_why = _judge(cond, score['seed'], p, lane_index, row, bar_count, fill) if cond else (True, None)
        via = [pointer for _, _, pointer in leaves]
        for s, m, dropped in events:
            duration = w // m
            at = row_at + s * duration
            if not u0 <= at < u1 or at + duration <= 0:
                continue
            if score['mutes'][lane_index]:
                fires, why = False, 'muted (score)'
            elif layers.mutes[lane_index](at):
                fires, why = False, 'muted (bus)'
            elif not holds:
                fires, why = False, cond_why
            elif dropped:
                fires, why = False, dropped
            else:
                fires, why = True, cond_why
            out.append({'id': f'{play}.{lane_index}.{row}' + (f'.{s}' if m > 1 else ''), 'at': at,
                        'lane': lane_index, 'duration': duration, 'note': hit['note'], 'vel': hit['vel'],
                        'locks': copy.deepcopy(hit['locks']) if 'locks' in hit else None, 'fires': fires, 'why': why,
                        'origin': {'play': play, 'entry': entry, 'pattern': p, 'row': row, 'lane_row': lane_row,
                                   'cell': cell, 'sub': s, 'of': m, 'bar': bar_count,
                                   'cond': condition_code(cond) if cond else None, 'via': list(via)}})


def evaluate(score, transport, u0, u1):
    """Every candidate with ``u0 ≤ at < u1`` that ends after the anchor, sorted by ``at``, then lane.

    ``transport`` is ``{tick, fill, bus: {fill, mutes}}``: the anchor's cycle
    tick, Studio's FILL transitions and the bus layer. Missing lists are off.
    The anchor's row comes back with a negative ``at`` when the anchor is
    inside it; earlier rows never do.
    """
    t = timeline(score)
    a = transport['tick']
    if type(a) is not int or not 0 <= a < t.cycle:
        raise ValueError('The anchor is a cycle tick')
    if type(u0) is not int or type(u1) is not int:
        raise ValueError('Evaluate whole ticks')
    layers = _Layers(transport)
    n, first = len(t.entries), t.entry(a)
    # Start at the row holding tick max(u0, 0): earlier rows end before u0 or at the anchor.
    cycles, position = divmod(a + max(u0, 0), t.cycle)
    i = t.entry(position)
    play = cycles * n + i - first
    play_start = cycles * t.cycle + t.starts[i] - a
    w = geometry(score['patterns'][t.entries[i]])[0]
    row = (max(u0, 0) - play_start) // w
    out = []
    while play_start < u1:
        p = t.entries[i]
        w, rows, bar, bars = geometry(score['patterns'][p])
        visit = play if t.runs is None else min(play, t.runs[i])
        for row in range(row, rows):
            row_at = play_start + row * w
            if row_at >= u1:
                break
            if score['patterns'][p] is not None:
                _row(out, score, layers, u0, u1, play, None if t.loop else i, p, row, row_at, w,
                     visit * bars + row // bar, layers.fill_on(row_at))
        play += 1
        i += 1
        if i == n:
            i, cycles = 0, cycles + 1
        play_start, row = cycles * t.cycle + t.starts[i] - a, 0
    out.sort(key=lambda c: (c['at'], c['lane']))
    return out


def trace_line(record, transport_revision, start):
    """One line of the JSONL trace: a candidate, or a change to FILL or the bus layer, with its segment."""
    return json.dumps({'transport_revision': transport_revision, 'start': start, **record})


# -- followers ----------------------------------------------------------------

def transport(snapshot):
    """``evaluate()``'s transport from a version-3 playhead snapshot."""
    return {'tick': snapshot['start']['tick'], 'fill': snapshot.get('fill'), 'bus': snapshot.get('bus')}


def elapsed(snapshot, monotonic):
    """Elapsed ticks ``u`` (real) at monotonic times, from a version-3 playhead (evaluator.md#playhead-fields).

    ``u`` moves while the tracker is playing and enabled, as the beat does.
    """
    monotonic = np.asarray(monotonic, dtype=np.float64)
    if not snapshot['enabled']:
        return np.full(monotonic.shape, float(snapshot['elapsed']))
    clock = snapshot['clock_time'] + np.where(snapshot['playing'], monotonic - snapshot['wall'], 0.)
    return snapshot['elapsed'] + (clock - snapshot['anchor_time']) * snapshot['document']['bpm'] / 60 * TICKS


def position(score, start, u):
    """Where elapsed ticks ``u`` (an array) fall in the cycle.

    Returns the cycle beat, each sample's entry and its fractional row position
    (score-v3.md#the-timeline), for signals and automation.
    """
    t = timeline(score)
    tick = np.mod(start + np.asarray(u, dtype=np.float64), t.cycle)
    tick = np.minimum(tick, np.nextafter(float(t.cycle), 0))      # x % C can round to C
    entry = np.searchsorted(t.starts, tick, side='right') - 1
    rows = np.array([geometry(score['patterns'][p])[0] for p in t.entries], dtype=np.float64)
    return tick / TICKS, entry, (tick - np.asarray(t.starts, dtype=np.float64)[entry]) / rows[entry]


def off():
    return [{'at': 0, 'on': False}]


class Layers:
    """Studio's FILL and bus layer, shared live with Studio's own outputs.

    Studio stamps a change with the first tick none of its outputs has evaluated
    (evaluator.md#ticks), so its outputs and the record agree. An output takes
    the current lists and claims the ticks it is about to evaluate under
    ``lock``, and Studio finds the stamp and changes the lists under the same
    lock, so no output judges a tick at or after a stamp under the old state.
    ``segment`` is the (transport_revision, anchor tick) the lists belong to.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.segment = None
        self.fill = off()
        self.bus = {'fill': off(), 'mutes': [off() for _ in range(LANE_COUNT)]}


class Follower:
    """One follower's candidates for the current segment.

    Each tick is evaluated once per segment, so a FILL or bus change that
    arrives after a candidate was judged never changes it (evaluator.md's late
    transitions: nothing is replayed). Candidates are kept until they end.
    """

    def __init__(self):
        self.frontier = None        # the first tick not yet evaluated, None before a segment
        self.segment = None
        self.candidates = []

    def begin(self, tick, segment=None):
        """Start ``segment``, evaluating from ``tick``: at or before the anchor's row when the
        follower hears of the segment as it starts, its own current tick when it joins late."""
        self.frontier, self.segment, self.candidates = tick, segment, []

    def evaluated(self, segment):
        """The first tick of ``segment`` not yet evaluated, or None for another segment."""
        return self.frontier if segment == self.segment else None

    def advance(self, snapshot, u):
        """Evaluate through real tick ``u`` and return the candidates that are new.

        With Studio's live ``layers`` in the snapshot, this reads FILL and the bus
        layer from them and claims the ticks before evaluating, under their lock.
        """
        top = math.floor(u) + 1
        if self.frontier is None or top <= self.frontier:
            return []
        t = transport(snapshot)
        layers = snapshot.get('layers')
        if layers is not None:
            with layers.lock:
                if layers.segment == self.segment:
                    t['fill'], t['bus'] = layers.fill, layers.bus
                first, self.frontier = self.frontier, top
        else:
            first, self.frontier = self.frontier, top
        new = evaluate(snapshot['document'], t, first, top)
        self.candidates += new
        return new

    def forget(self, u):
        """Drop candidates that ended at or before tick ``u``."""
        self.candidates = [c for c in self.candidates if c['at'] + c['duration'] > u]

