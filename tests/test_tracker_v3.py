"""Score format 3: Gorgon's hand-checked vectors, evaluate()'s invariants, validation and the v1/v2 mapping."""
import copy
import json
import random
import time

import pytest

from hydrapy.studio_tracker import Tracker
from hydrapy.tracker_v3 import (PROB, TICKS, Timeline, _Layers, _row, chance, code, empty_pattern, evaluate,
                                geometry, lowbias32, state_at, trace_line, upgrade, validate)

C4 = {'note': 'C-4', 'vel': 100}


def blank(**fields):
    """A valid version-3 score with every pattern empty, looping A01."""
    d = upgrade(Tracker().document)
    d.update({'patterns': [None] * 128, 'chain': [0], 'pattern': 0, 'loop': True, 'seed': 0, **fields})
    return d


def pattern(bars=1, rate=1, bar=16, lens=None):
    p = empty_pattern(0)
    p.update(rate=rate, bar=bar, bars=bars)
    p['lanes'] = [{'len': n, 'cells': [None] * n, 'transforms': []} for n in lens or [bar * bars] * 6]
    return p


def hit(cond=None, note='C-4'):
    return dict(C4, note=note, **({'cond': cond} if cond else {}))


def every(a, b, **extra):
    return dict(kind='every', a=a, b=b, **extra)


def run(score, transport, u0, u1):
    """Evaluate a validated copy, checking invariant 2 (splitting the interval anywhere) on the way."""
    score = validate(score)
    whole = evaluate(score, transport, u0, u1)
    cuts = sorted({c['at'] for c in whole} | {c['at'] + 1 for c in whole} | {u0 + (u1 - u0) // 3})
    edges = [u0] + [c for c in cuts if u0 < c < u1] + [u1]
    assert [c for x, y in zip(edges, edges[1:]) for c in evaluate(score, transport, x, y)] == whole
    assert len({c['id'] for c in whole}) == len(whole)
    return whole


def table(candidates, *fields):
    def value(c, f):
        return c['origin'][f] if f in c['origin'] and f not in c else c[f]
    return [tuple(value(c, f) for f in fields) for c in candidates]


def test_chance_check_values():
    assert lowbias32(1) == 1753845952 and lowbias32(0xffffffff) == 1734902346
    assert [chance(0x1234, PROB, 0, 1, 12, bar, 0) for bar in range(6)] == [
        1936593078, 1284988505, 1732901356, 3891962147, 2080199420, 1655901407]


def test_codes_and_the_empty_pattern():
    assert [code(i) for i in (0, 17, 127)] == ['A01', 'B02', 'H16']
    d = blank()
    d['patterns'][5] = empty_pattern(5)
    assert validate(d)['patterns'][5] == empty_pattern(5)
    assert geometry(None) == geometry(empty_pattern(5)) == (480, 64, 16, 4)


def test_v1_a_seven_row_lane_against_a_sixteen_row_bar():
    p = pattern(bars=2, lens=[7, 32, 32, 32, 32, 32])
    p['lanes'][0]['cells'][0] = hit(every(1, 2))
    got = run(blank(patterns=[p] + [None] * 127), {'tick': 0}, 0, 16 * TICKS)
    assert table(got, 'at', 'id', 'row', 'bar', 'fires') == [
        (0, '0.0.0', 0, 0, True), (3360, '0.0.7', 7, 0, True), (6720, '0.0.14', 14, 0, True),
        (10080, '0.0.21', 21, 1, False), (13440, '0.0.28', 28, 1, False), (15360, '1.0.0', 0, 2, True),
        (18720, '1.0.7', 7, 2, True), (22080, '1.0.14', 14, 2, True), (25440, '1.0.21', 21, 3, False),
        (28800, '1.0.28', 28, 3, False)]


def v2_pattern():
    p = pattern()
    p['lanes'][1]['cells'][4] = hit(every(1, 3))
    p['lanes'][1]['cells'][12] = hit({'kind': 'prob', 'p': 50})
    return p


def test_v2_every_and_probability():
    got = run(blank(seed=0x1234, patterns=[v2_pattern()] + [None] * 127), {'tick': 0}, 0, 24 * TICKS)
    assert table(got, 'at', 'id', 'bar', 'cond', 'why', 'fires') == [
        (1920, '0.1.4', 0, '1:3', 'bar 0 mod 3 = 0', True),
        (5760, '0.1.12', 0, '%50', 'chance 1936593078 mod 100 = 78', False),
        (9600, '1.1.4', 1, '1:3', 'bar 1 mod 3 = 1', False),
        (13440, '1.1.12', 1, '%50', 'chance 1284988505 mod 100 = 5', True),
        (17280, '2.1.4', 2, '1:3', 'bar 2 mod 3 = 2', False),
        (21120, '2.1.12', 2, '%50', 'chance 1732901356 mod 100 = 56', False),
        (24960, '3.1.4', 3, '1:3', 'bar 3 mod 3 = 0', True),
        (28800, '3.1.12', 3, '%50', 'chance 3891962147 mod 100 = 47', True),
        (32640, '4.1.4', 4, '1:3', 'bar 4 mod 3 = 1', False),
        (36480, '4.1.12', 4, '%50', 'chance 2080199420 mod 100 = 20', True),
        (40320, '5.1.4', 5, '1:3', 'bar 5 mod 3 = 2', False),
        (44160, '5.1.12', 5, '%50', 'chance 1655901407 mod 100 = 7', True)]


def test_v3_a_seek_mid_bar_restarts_the_visit():
    p = pattern(bars=2)
    p['lanes'][2]['cells'][0] = p['lanes'][2]['cells'][14] = hit({'kind': 'first'})
    p['lanes'][3]['cells'][20] = hit({'kind': 'first', 'not': True})
    got = run(blank(patterns=[p] + [None] * 127), {'tick': 5760}, -TICKS, 16 * TICKS)
    assert table(got, 'at', 'id', 'row', 'bar', 'cond', 'fires') == [
        (960, '0.2.14', 14, 0, '1ST', True), (3840, '0.3.20', 20, 1, '!1ST', True),
        (9600, '1.2.0', 0, 2, '1ST', False), (16320, '1.2.14', 14, 2, '1ST', False),
        (19200, '1.3.20', 20, 3, '!1ST', True), (24960, '2.2.0', 0, 4, '1ST', False)]


def test_v4_fill_arriving_mid_bar():
    p = pattern()
    for row in (8, 12):
        p['lanes'][4]['cells'][row] = hit({'kind': 'fill'})
        p['lanes'][5]['cells'][row] = hit({'kind': 'fill', 'not': True})
    fill = [{'at': 0, 'on': False}, {'at': 4800, 'on': True}, {'at': 11520, 'on': False}]
    got = run(blank(patterns=[p] + [None] * 127), {'tick': 0, 'fill': fill}, 0, 8 * TICKS)
    assert table(got, 'at', 'lane', 'row', 'fires') == [
        (3840, 4, 8, False), (3840, 5, 8, True), (5760, 4, 12, True), (5760, 5, 12, False),
        (11520, 4, 8, False), (11520, 5, 8, True), (13440, 4, 12, False), (13440, 5, 12, True)]


def chained():
    a02 = pattern(rate=2, bars=2)
    a02['lanes'][0]['cells'][0] = hit(every(1, 4))
    a02['lanes'][1]['cells'][16] = hit(every(2, 4))
    patterns = [None] * 128
    patterns[0], patterns[1], patterns[16] = pattern(bars=4), a02, pattern(rate=.5, bar=8, bars=4)
    return blank(loop=False, chain=[0, 1, 1, 16], patterns=patterns)


def test_v5_two_plays_one_visit():
    got = run(chained(), {'tick': 0}, 0, 80 * TICKS)
    assert Timeline(validate(chained())).starts == [0, 30720, 38400, 46080, 76800]
    assert table(got, 'at', 'id', 'entry', 'row', 'bar', 'cond', 'fires') == [
        (30720, '1.0.0', 1, 0, 0, '1:4', True), (34560, '1.1.16', 1, 16, 1, '2:4', True),
        (38400, '2.0.0', 2, 0, 2, '1:4', False), (42240, '2.1.16', 2, 16, 3, '2:4', False),
        (107520, '5.0.0', 1, 0, 0, '1:4', True), (111360, '5.1.16', 1, 16, 1, '2:4', True),
        (115200, '6.0.0', 2, 0, 2, '1:4', False), (119040, '6.1.16', 2, 16, 3, '2:4', False)]


def test_v6_the_anchors_row_and_a_muted_lane():
    p = pattern()
    for row in (3, 4, 5):
        p['lanes'][0]['cells'][row] = hit()
    p['lanes'][1]['cells'][4] = hit()
    p['lanes'][1]['transforms'] = [{'kind': 'ply', 'n': 4}]
    p['lanes'][2]['cells'][4] = hit()
    p['lanes'][2]['cells'][8] = hit({'kind': 'first'})
    score = blank(mutes=[False, False, True, False, False, False], patterns=[p] + [None] * 127)
    got = run(score, {'tick': 2040}, -TICKS, 5640)
    assert table(got, 'at', 'id', 'duration', 'fires', 'why') == [
        (-120, '0.0.4', 480, True, None), (-120, '0.2.4', 480, False, 'muted (score)'),
        (0, '0.1.4.1', 120, True, None), (120, '0.1.4.2', 120, True, None), (240, '0.1.4.3', 120, True, None),
        (360, '0.0.5', 480, True, None), (1800, '0.2.8', 480, False, 'muted (score)')]
    later = run(score, {'tick': 2280}, -TICKS, 0)
    assert ('0.0.4', -360) in [(c['id'], c['at']) for c in later]


def test_v7_conditions_are_judged_once_per_row_of_a_ratchet():
    p = v2_pattern()
    p['lanes'][1]['transforms'] = [{'kind': 'ply', 'n': 4}]
    got = run(blank(seed=0x1234, patterns=[p] + [None] * 127), {'tick': 0}, 0, 12 * TICKS)
    rows = {}
    for c in got:
        assert c['duration'] == 120 and c['id'] == f"{c['origin']['play']}.1.{c['origin']['row']}.{c['origin']['sub']}"
        rows.setdefault((c['origin']['play'], c['origin']['row']), []).append((c['at'], c['why'], c['fires']))
    expected = [(0, 4, 1920, 'bar 0 mod 3 = 0', True), (0, 12, 5760, 'chance 1936593078 mod 100 = 78', False),
                (1, 4, 9600, 'bar 1 mod 3 = 1', False), (1, 12, 13440, 'chance 1284988505 mod 100 = 5', True),
                (2, 4, 17280, 'bar 2 mod 3 = 2', False), (2, 12, 21120, 'chance 1732901356 mod 100 = 56', False)]
    assert sorted(rows) == [(play, row) for play, row, *_ in expected]
    for play, row, at, why, fires in expected:
        assert rows[play, row] == [(at + 120 * s, why, fires) for s in range(4)]


def test_v8_degrade_before_and_after_ply():
    p = pattern()
    for lane, transforms in ((2, [{'kind': 'ply', 'n': 4}, {'kind': 'degrade', 'p': 50}]),
                             (3, [{'kind': 'degrade', 'p': 50}, {'kind': 'ply', 'n': 4}])):
        p['lanes'][lane]['cells'][0] = p['lanes'][lane]['cells'][8] = hit()
        p['lanes'][lane]['transforms'] = transforms
    got = run(blank(seed=0x1234, patterns=[p] + [None] * 127), {'tick': 0}, 0, 8 * TICKS)
    lane2 = {(0, 0): [(2286100646, 1), (385726237, 1), (370582005, 1), (246513302, 1)],
             (0, 8): [(2303062892, 0), (565417095, 0), (146505197, 0), (3613849973, 0)],
             (1, 0): [(922554248, 1), (517400578, 0), (1910746187, 0), (2456862593, 0)],
             (1, 8): [(2572734288, 0), (1657527929, 1), (3140429349, 1), (2465367129, 1)]}
    lane3 = {(0, 0): (3038833604, 1), (0, 8): (3396531612, 1), (1, 0): (4023513783, 0), (1, 8): (1919793396, 0)}
    for c in got:
        o = c['origin']
        assert o['via'] == ['/0', '/1'] and c['at'] == o['play'] * 7680 + o['row'] * 480 + o['sub'] * 120
        if c['lane'] == 2:
            draw, fires = lane2[o['play'], o['row']][o['sub']]
            pointer = '/1'
        else:
            draw, fires = lane3[o['play'], o['row']]
            pointer = '/0'
        assert c['fires'] == bool(fires)
        assert c['why'] == (None if fires else f'degrade {pointer}: chance {draw} mod 100 = {draw % 100}')
    assert len(got) == 32


def test_v9_rot_rev_and_every_on_a_short_lane():
    p = pattern(bars=2, lens=[6, 32, 32, 32, 32, 32])
    for cell, note in ((0, 'C-4'), (1, 'D-4'), (2, 'E-4'), (5, 'G-4')):
        p['lanes'][0]['cells'][cell] = hit(note=note)
    p['lanes'][0]['transforms'] = [{'kind': 'rot', 'k': 1}, every(2, 2, inner={'kind': 'rev'})]
    got = run(blank(patterns=[p] + [None] * 127), {'tick': 0}, 0, 8 * TICKS)
    assert all(c['fires'] for c in got)
    early, late = ['/0'], ['/0', '/1/inner']
    expected = [(0, 0, 5, 'G-4', early), (1, 1, 0, 'C-4', early), (2, 2, 1, 'D-4', early), (3, 3, 2, 'E-4', early),
                (6, 0, 5, 'G-4', early), (7, 1, 0, 'C-4', early), (8, 2, 1, 'D-4', early), (9, 3, 2, 'E-4', early),
                (12, 0, 5, 'G-4', early), (13, 1, 0, 'C-4', early), (14, 2, 1, 'D-4', early), (15, 3, 2, 'E-4', early),
                (16, 4, 0, 'C-4', late), (17, 5, 5, 'G-4', late),
                (20, 2, 2, 'E-4', late), (21, 3, 1, 'D-4', late), (22, 4, 0, 'C-4', late), (23, 5, 5, 'G-4', late),
                (26, 2, 2, 'E-4', late), (27, 3, 1, 'D-4', late), (28, 4, 0, 'C-4', late), (29, 5, 5, 'G-4', late)]
    assert table(got, 'row', 'lane_row', 'cell', 'note', 'via') == expected
    assert [c['at'] for c in got] == [row * 480 for row, *_ in expected]
    assert [c['origin']['bar'] for c in got] == [row // 16 for row, *_ in expected]


def off(*changes, start=False):
    return [{'at': 0, 'on': start}] + [{'at': at, 'on': on} for at, on in changes]


def test_v10_studio_fill_and_the_bus_layer_merge_by_or():
    p = pattern()
    for row in (4, 8, 12):
        p['lanes'][0]['cells'][row] = hit()
    p['lanes'][1]['cells'][4] = hit()
    for row in (0, 4, 8, 12):
        p['lanes'][2]['cells'][row] = hit({'kind': 'fill'})
    mutes = [off((2880, True), (4800, False)), off(start=True)] + [off()] * 4
    transport = {'tick': 0, 'fill': off((5760, True)),
                 'bus': {'fill': off((1920, True), (3840, False)), 'mutes': mutes}}
    score = blank(mutes=[False, True, False, False, False, False], patterns=[p] + [None] * 127)
    assert table(run(score, transport, 0, 7680), 'at', 'id', 'fires', 'why') == [
        (0, '0.2.0', False, 'fill off'), (1920, '0.0.4', True, None), (1920, '0.1.4', False, 'muted (score)'),
        (1920, '0.2.4', True, 'fill on'), (3840, '0.0.8', False, 'muted (bus)'), (3840, '0.2.8', False, 'fill off'),
        (5760, '0.0.12', True, None), (5760, '0.2.12', True, 'fill on')]


@pytest.mark.parametrize('stamp, plays', [(479, False), (480, False), (481, True)])
def test_v11_a_mute_stamped_on_a_hits_tick_applies_to_it(stamp, plays):
    p = pattern()
    p['lanes'][2]['cells'][1] = hit()
    mutes = [off()] * 2 + [off((stamp, True))] + [off()] * 3
    [c] = run(blank(patterns=[p] + [None] * 127), {'tick': 0, 'bus': {'mutes': mutes}}, 0, 7680)
    assert (c['fires'], c['why']) == (plays, None if plays else 'muted (bus)')


def test_v12_feedback_mutes_the_row_after_each_hit():
    p = pattern()
    for row in range(16):
        p['lanes'][2]['cells'][row] = hit()
    for row in (0, 4, 8, 12):
        p['lanes'][0]['cells'][row] = hit()
    lane2 = off(*[(480 * r + 1 + 480 * k, k == 0) for r in (0, 4, 8, 12) for k in (0, 1)])
    got = run(blank(patterns=[p] + [None] * 127), {'tick': 0, 'bus': {'mutes': [off(), off(), lane2] + [off()] * 3}}, 0, 7680)
    assert [c['origin']['row'] for c in got if not c['fires']] == [1, 5, 9, 13]
    assert all(c['why'] == 'muted (bus)' for c in got if not c['fires']) and len(got) == 20


def test_transition_lists():
    transitions = [{'at': 0, 'on': True}, {'at': 10, 'on': False}, {'at': 10, 'on': True}, {'at': 20, 'on': False}]
    assert [state_at(transitions, u) for u in (-5, 0, 9, 10, 19, 20)] == [True, True, True, True, True, False]
    assert state_at(None, 100) is False
    layers = _Layers({'tick': 0, 'fill': off(), 'bus': {'fill': off((5, True))}})
    assert not layers.fill_on(4) and layers.fill_on(5)


def test_trace_line_shape():
    p = v2_pattern()
    score = validate(blank(seed=0x1234, patterns=[p] + [None] * 127))
    [c] = [c for c in evaluate(score, {'tick': 0}, 0, 24 * TICKS) if c['id'] == '1.1.12']
    line = json.loads(trace_line(c, 3, {'tick': 0, 'entry': None}))
    assert list(line) == ['transport_revision', 'start', 'id', 'at', 'lane', 'duration', 'note', 'vel', 'locks',
                          'fires', 'why', 'origin']
    assert line['origin'] == {'play': 1, 'entry': None, 'pattern': 0, 'row': 12, 'lane_row': 12, 'cell': 12,
                              'sub': 0, 'of': 1, 'bar': 1, 'cond': '%50', 'via': []}
    assert (line['at'], line['duration'], line['why']) == (13440, 480, 'chance 1284988505 mod 100 = 5')


# -- invariants on random scores ----------------------------------------------

def random_node(rng, depth):
    kind = rng.choice(['rot', 'rev', 'degrade', 'ply'] + ['every'] * (depth > 1))
    if kind == 'every':
        b = rng.randint(1, 8)
        return every(rng.randint(1, b), b, inner=random_node(rng, depth - 1))
    return {'kind': kind, **{'rot': {'k': rng.randint(-127, 127)}, 'rev': {}, 'degrade': {'p': rng.randint(1, 99)},
                             'ply': {'n': rng.randint(2, 4)}}[kind]}


def ply_product(nodes):
    product = 1
    for node in nodes:
        product *= node['n'] if node['kind'] == 'ply' else ply_product([node['inner']]) if node['kind'] == 'every' else 1
    return product


def random_score(rng):
    patterns = [None] * 128
    used = rng.sample(range(128), rng.randint(1, 4))
    for index in used:
        bar = rng.choice([3, 4, 7, 8, 12, 16])
        p = pattern(rate=rng.choice([.25, .5, 1, 2, 4, 8]), bar=bar, bars=rng.randint(1, min(8, 128 // bar)))
        rows = p['bar'] * p['bars']
        for lane in p['lanes']:
            lane['len'] = rng.randint(1, rows)
            lane['cells'] = [None] * lane['len']
            for k in range(lane['len']):
                if rng.random() < .4:
                    cond = None
                    if rng.random() < .4:
                        kind = rng.choice(['every', 'prob', 'fill', 'first'])
                        b = rng.randint(1, 8)
                        cond = {'kind': kind, **{'every': {'a': rng.randint(1, b), 'b': b},
                                                 'prob': {'p': rng.randint(1, 99)}}.get(kind, {})}
                        if rng.random() < .3:
                            cond['not'] = True
                    lane['cells'][k] = dict(hit(cond), vel=rng.choice([0, 1, 100, 127]))
            while True:
                lane['transforms'] = [random_node(rng, 3) for _ in range(rng.randint(0, 4))]
                if ply_product(lane['transforms']) <= 4:
                    break
        patterns[index] = p
    chain = [rng.choice(used + [rng.randrange(128)]) for _ in range(rng.randint(1, 5))]
    return validate(blank(loop=rng.random() < .5, pattern=rng.choice(used), chain=chain, seed=rng.randrange(1 << 32),
                          mutes=[rng.random() < .15 for _ in range(6)], patterns=patterns))


def random_transitions(rng, span):
    changes = sorted((rng.randrange(span), rng.random() < .5) for _ in range(rng.randint(0, 4)))
    return off(*changes, start=rng.random() < .3)


def brute_force(score, transport, u0, u1):
    """Walk every play from the anchor's, row by row, counting each visit from its plays."""
    t = Timeline(score)
    layers = _Layers(transport)
    a, n = transport['tick'], len(t.entries)
    first = t.entry(a)
    out, play = [], 0
    while True:
        cycles, i = divmod(first + play, n)
        start = cycles * t.cycle + t.starts[i] - a
        if start >= u1:
            break
        p = t.entries[i]
        visit = 0
        while visit < play and t.entries[(first + play - visit - 1) % n] == p:
            visit += 1
        w, rows, bar, bars = geometry(score['patterns'][p])
        for row in range(rows):
            if score['patterns'][p] is not None and start + (row + 1) * w > 0:
                _row(out, score, layers, u0, u1, play, None if t.loop else i, p, row, start + row * w, w,
                     visit * bars + row // bar, layers.fill_on(start + row * w))
        play += 1
    return sorted(out, key=lambda c: (c['at'], c['lane']))


def test_random_scores_keep_the_invariants():
    rng = random.Random(20261027)
    total = 0
    for _ in range(150):
        score = random_score(rng)
        cycle = Timeline(score).cycle
        transport = {'tick': rng.randrange(cycle), 'fill': random_transitions(rng, 2 * cycle)}
        if rng.random() < .7:
            transport['bus'] = {'fill': random_transitions(rng, 2 * cycle),
                                'mutes': [random_transitions(rng, 2 * cycle) for _ in range(6)]}
        u0, u1 = -8 * TICKS, rng.randrange(cycle, 3 * cycle)
        whole = evaluate(score, transport, u0, u1)
        assert whole == brute_force(score, transport, u0, u1)
        edges = [u0] + sorted({rng.randrange(u0, u1) for _ in range(6)}) + [u1]
        assert [c for x, y in zip(edges, edges[1:]) for c in evaluate(score, transport, x, y)] == whole
        assert len({c['id'] for c in whole}) == len(whole)
        assert [(c['at'], c['lane']) for c in whole] == sorted((c['at'], c['lane']) for c in whole)
        for c in whole:
            w = geometry(score['patterns'][c['origin']['pattern']])[0]
            assert c['at'] + c['duration'] > 0 and c['duration'] * c['origin']['of'] == w and c['origin']['of'] <= 4
            if c['at'] < 0:                       # only the anchor's row comes back early
                assert c['origin']['play'] == 0 and c['at'] - c['origin']['sub'] * c['duration'] > -w
            if not c['fires']:
                assert c['why']
        for c in rng.sample(whole, min(8, len(whole))):     # stable identity
            assert evaluate(score, transport, c['at'], c['at'] + 1) == [x for x in whole if x['at'] == c['at']]
        total += len(whole)
    assert total > 10000


def test_evaluating_late_in_a_long_session_is_cheap_and_periodic():
    p = pattern(bars=2, lens=[7, 32, 5, 32, 32, 32])
    p['lanes'][0]['cells'][0] = p['lanes'][2]['cells'][3] = hit()
    p['lanes'][3]['transforms'] = [{'kind': 'rot', 'k': -3}, {'kind': 'ply', 'n': 2}]
    p['lanes'][3]['cells'][30] = hit()
    score = validate(blank(patterns=[p] + [None] * 127))
    length, plays = 32 * 480, 10 ** 8
    began = time.perf_counter()
    late = evaluate(score, {'tick': 960}, plays * length, (plays + 1) * length)
    assert time.perf_counter() - began < .5        # walking 10⁸ plays would take minutes

    def shifted(c):
        play = c['origin']['play'] - plays
        return dict(c, at=c['at'] - plays * length, id='.'.join([str(play)] + c['id'].split('.')[1:]),
                    origin=dict(c['origin'], play=play, bar=c['origin']['bar'] - 2 * plays))
    assert [shifted(c) for c in late] == evaluate(score, {'tick': 960}, 0, length)


# -- validation and the version-1/2 mapping -----------------------------------

def rich():
    """A valid score that uses every field, for the rejection cases to break."""
    d = chained()
    lane = d['patterns'][1]['lanes'][2]
    lane['len'] = 5
    lane['cells'] = [dict(C4, cond={'kind': 'prob', 'p': 30, 'not': True}, locks={'p1': 5, 'machine': 3}, x_note='kept'),
                     None, dict(C4, vel=0), None, hit({'kind': 'fill'})]
    lane['transforms'] = [{'kind': 'rot', 'k': -2}, every(1, 4, inner=every(2, 2, inner={'kind': 'ply', 'n': 2})),
                          {'kind': 'degrade', 'p': 50}, {'kind': 'rev'}, {'kind': 'ply', 'n': 2}]
    d['patterns'][1]['keys']['z'] = [{'r': 0, 'v': .5}, {'r': 31, 'v': 1}]
    d['patterns'][1]['scenes'] = [{'r': 3, 'name': 'a', 'fade': 32}]
    d['x_editor'] = {'kept': True}
    d['patterns'][1]['x_pattern'] = 1
    d['lanes'][0]['x_lane'] = 1
    return d


def test_a_rich_score_validates_and_keeps_unknown_keys():
    d = validate(rich())
    assert d['x_editor'] == {'kept': True} and d['patterns'][1]['x_pattern'] == 1
    assert d['patterns'][1]['lanes'][2]['cells'][0]['x_note'] == 'kept'
    assert evaluate(d, {'tick': 0}, 0, Timeline(d).cycle)


@pytest.mark.parametrize('change', [
    lambda d: d.update(format='hydrapy-tracker-2'),
    lambda d: d.update(seed=-1),
    lambda d: d.update(seed=1 << 32),
    lambda d: d.update(seed=True),
    lambda d: d.update(seed=1.),
    lambda d: d.update(pattern=128),
    lambda d: d.update(chain=[]),
    lambda d: d.update(chain=[0] * 257),
    lambda d: d.update(chain=[128]),
    lambda d: d.update(chain=[True]),
    lambda d: d.update(mutes=[False] * 4),
    lambda d: d.update(lanes=[{'name': 'x'}] * 5),
    lambda d: d['lanes'].__setitem__(0, {'name': 'x' * 25}),
    lambda d: d['lanes'].__setitem__(0, {}),
    lambda d: d.update(sounds=d['sounds'][:4]),
    lambda d: d.update(patterns=d['patterns'][:127]),
    lambda d: d['patterns'].__setitem__(3, {}),
    lambda d: d['patterns'][1].update(rate=3),
    lambda d: d['patterns'][1].update(rate=True),
    lambda d: d['patterns'][1].update(rate=float('nan')),
    lambda d: d['patterns'][1].update(bar=0),
    lambda d: d['patterns'][1].update(bar=17, bars=8),
    lambda d: d['patterns'][1].update(bars=9),
    lambda d: d['patterns'][1].update(name=''),
    lambda d: d['patterns'][1]['lanes'].pop(),
    lambda d: d['patterns'][1]['lanes'][0].update(len=33),
    lambda d: d['patterns'][1]['lanes'][0].update(len=0),
    lambda d: d['patterns'][1]['lanes'][0]['cells'].pop(),
    lambda d: d['patterns'][1]['lanes'][0].pop('transforms'),
    lambda d: d['patterns'][1]['lanes'][2]['transforms'].append({'kind': 'rev'}) or
              d['patterns'][1]['lanes'][2]['transforms'].extend([{'kind': 'rev'}] * 3),
    lambda d: d['patterns'][1]['lanes'][2]['transforms'].append({'kind': 'ply', 'n': 2}),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'ply', 'n': 5}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'ply', 'n': 2}, {'kind': 'ply', 'n': 3}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'rot', 'k': 128}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'rot'}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'rev', 'k': 1}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'degrade', 'p': 100}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': 'swing'}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[{'kind': ['rev']}]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[every(3, 2, inner={'kind': 'rev'})]),
    lambda d: d['patterns'][1]['lanes'][0].update(transforms=[every(1, 2)]),
    lambda d: d['patterns'][1]['lanes'][0].update(
        transforms=[every(1, 2, inner=every(1, 2, inner=every(1, 2, inner={'kind': 'rev'})))]),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'prob', 'p': 0}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'prob', 'p': 50, 'a': 1}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'every', 'a': 2, 'b': 1}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'every', 'a': 1}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'fill', 'not': 1}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(cond={'kind': 'pre'}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(locks={'p9': 1}),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(vel=128),
    lambda d: d['patterns'][1]['lanes'][2]['cells'][0].update(note='H-4'),
    lambda d: d['patterns'][1]['keys']['z'].append({'r': 32, 'v': .5}),
    lambda d: d['patterns'][1]['scenes'][0].update(fade=33),
    lambda d: d['patterns'][1]['scenes'][0].update(r=32),
])
def test_validation_rejects(change):
    d = rich()
    change(d)
    with pytest.raises(ValueError):
        validate(d)


@pytest.mark.parametrize('where', [
    lambda d, x: d.update(x_note=x),                                          # an unknown key at the top
    lambda d, x: d.update(x_editor={'zoom': [1, x]}),                         # nested in one
    lambda d, x: d['patterns'][1].update(x_pattern=x),                        # in a pattern
    lambda d, x: d['patterns'][1]['lanes'][2]['cells'][0].update(x_note=x),   # in a hit
    lambda d, x: d['lanes'][0].update(x_lane=x),
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_validation_rejects_non_finite_numbers_in_unknown_keys(where, value):
    """score-v3.md: unknown keys are kept, but non-finite numbers are rejected everywhere (#22)."""
    d = rich()
    where(d, value)
    with pytest.raises(ValueError, match='finite'):
        validate(d)
    with pytest.raises(ValueError, match='finite'):                           # what Studio parses from a body
        validate(json.loads(json.dumps(d)))


def test_validation_does_not_change_its_argument():
    d = rich()
    del d['patterns'][1]['scenes'][0]['fade']
    before = copy.deepcopy(d)
    assert validate(d)['patterns'][1]['scenes'][0]['fade'] == 0
    assert d == before


@pytest.mark.parametrize('fmt', ['hydrapy-tracker-1', 'hydrapy-tracker-2'])
def test_versions_1_and_2_map_onto_the_same_song(fmt):
    old = copy.deepcopy(Tracker().document)
    old['format'] = fmt
    old['mutes'] = [False, True, False, False]
    if fmt == 'hydrapy-tracker-2':
        old['sounds'] = [{'machine': k, 'params': {'p1': k}} for k in range(4)]
        old['patterns'][2]['audio'][1][6] = {'note': 'D-4', 'vel': 90, 'locks': {'p2': 7}}
    old['patterns'][3]['audio'][2][5] = {'note': 'C-4', 'vel': 0}
    new = upgrade(old)
    assert new['format'] == 'hydrapy-tracker-3' and new['chain'] == [0, 1, 2, 3] and new['seed'] == 0
    assert new['mutes'] == old['mutes'] + [False, False] and [lane['name'] for lane in new['lanes']][5] == 'Lane 6'
    assert len(new['sounds']) == 6 and new['sounds'][4] == {'machine': None, 'params': {}}
    assert new['sounds'][4] is not new['sounds'][5] and new['patterns'][0]['lanes'][4]['cells'] is not new['patterns'][0]['lanes'][5]['cells']
    assert new['patterns'][4:] == [None] * 124 and 'audio' not in new['patterns'][0]
    assert new['patterns'][0]['keys'] == old['patterns'][0]['keys'] and new['version'] == old['version']
    assert old['format'] == fmt and 'audio' in old['patterns'][0]
    expected = sorted((p * 30720 + row * 480, lane, h['note'], not old['mutes'][lane], h.get('locks'))
                      for p, pat in enumerate(old['patterns']) for lane, cells in enumerate(pat['audio'])
                      for row, h in enumerate(cells) if h is not None and h['vel'] > 0)
    got = evaluate(new, {'tick': 0}, 0, 64 * TICKS)
    assert [(c['at'], c['lane'], c['note'], c['fires'], c['locks']) for c in got] == expected

@pytest.mark.parametrize('length', [.25, .333, 1, 2.5, 16])
def test_cell_lengths_validate_and_set_tick_duration(length):
    p = pattern()
    p['lanes'][0]['cells'][0] = dict(hit(), len=length)
    d = blank(patterns=[p] + [None] * 127)
    assert run(d, {'tick': 0}, 0, 1)[0]['duration'] == length * 480


@pytest.mark.parametrize('length', [True, False, None, '2', 0, .249, 16.01, float('nan'), float('inf')])
def test_invalid_cell_lengths(length):
    p = pattern()
    p['lanes'][0]['cells'][0] = dict(hit(), len=length)
    with pytest.raises(ValueError):
        validate(blank(patterns=[p] + [None] * 127))


def test_explicit_lengths_are_base_rows_unsplit_by_ply_and_clip_to_pattern():
    p = pattern(rate=2)
    p['lanes'][0]['cells'][14] = dict(hit(), len=2.5)
    p['lanes'][0]['transforms'] = [{'kind': 'ply', 'n': 2}]
    candidates = run(blank(patterns=[p] + [None] * 127), {'tick': 0}, 14 * 240, 16 * 240)
    assert [(c['at'], c['duration']) for c in candidates] == [(3360, 480), (3480, 360)]


def test_seek_recovers_a_long_note_from_more_than_four_rows_back():
    p = pattern()
    p['lanes'][0]['cells'][0] = dict(hit(), len=12)
    d = blank(patterns=[p] + [None] * 127)
    candidates = run(d, {'tick': 10 * 480}, -7680, 1)
    assert [(c['at'], c['duration']) for c in candidates] == [(-4800, 5760)]
    # No tie survives a pattern boundary, even when the same pattern loops.
    assert evaluate(d, {'tick': 0}, -7680, 0) == []
