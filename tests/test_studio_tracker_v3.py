"""Studio plays version 3: segments in ticks, score-v3.md's transport changes, FILL and the bus layer."""
import copy
import json
import time

import pytest

from hydrapy.studio import Session
from hydrapy.studio_tracker import Tracker, scene_weights
from hydrapy.tracker_cv import TrackerCV
from hydrapy.tracker_v3 import TICKS, upgrade, validate
from test_tracker_v3 import blank, hit, off, pattern

PATCH = "from hydrapy import *\nz = Param('Dolly Z', .35)\nsolid(z, 0, 0).out(o0)\n"
SENDER = '4f1c9a7e2b6d0835'


def chain(**fields):
    """score-v3.md's examples: A01 (16 beats), A02 (4 beats at rate 2), B01 (16 beats at rate ½); C = 40 beats."""
    patterns = [None] * 128
    patterns[0], patterns[1], patterns[16] = pattern(bars=4), pattern(rate=2, bars=2), pattern(rate=.5, bar=8, bars=4)
    return validate(blank(loop=False, chain=[0, 1, 1, 16], patterns=patterns, bpm=120, **fields))


def playing(document, beat=0., clock=0.):
    tracker = Tracker(document)
    tracker.enabled = True
    tracker.seek(beat, clock)
    return tracker


def edited(tracker, clock, change):
    document = copy.deepcopy(tracker.document)
    change(document)
    revision = tracker.transport_revision
    tracker.replace(document, clock, tracker.revision)
    return tracker.transport_revision - revision


def test_a_seek_lands_on_the_nearest_tick_and_starts_a_segment():
    t = playing(chain(), 21.5 + .4 / TICKS)
    assert t.start == {'tick': 41280, 'entry': 2} and t.elapsed == 0 and t.transport_revision == 1
    assert t.beat(0.) == 21.5 and t.beat(.5) == 22.5                 # 120 BPM: two beats a second
    assert t.ticks(.5) == 1920 and t.playhead()['entry'] == 2
    t.seek(39.9999, 1.)
    assert t.start['tick'] == 0                                      # rounds to C, which wraps to 0


def test_a_tempo_change_keeps_the_segment_and_the_place():
    t = playing(chain(), 3.)
    start, revision = dict(t.start), t.transport_revision
    assert edited(t, .3, lambda d: d.update(bpm=90)) == 0
    assert t.start == start and t.transport_revision == revision
    assert t.elapsed == 1152 and t.anchor_time == pytest.approx(.3)   # 0.3 s at 120 BPM: exactly 1152 ticks
    assert t.beat(.3) == pytest.approx(3.6) and t.beat(.3 + 2 / 3) == pytest.approx(4.6)


@pytest.mark.parametrize('beat, change, expected', [
    # loop false -> true: the selected pattern, keeping the place in it when it was playing
    (21.5, lambda d: d.update(loop=True, pattern=1), 1.5 * TICKS),
    (21.5, lambda d: d.update(loop=True, pattern=0), 0),
    # loop true -> false: the first entry that plays the looped pattern, keeping the place
    (None, lambda d: d.update(loop=False), 16 * TICKS + 1.5 * TICKS),
    # a new looped pattern keeps the elapsed beats, wrapped into its length
    (None, lambda d: d.update(pattern=16), 1.5 * TICKS),
    # a chain edit keeps the entry if it still plays the same pattern, else beat 0
    (21.5, lambda d: d.update(chain=[0, 1, 1]), 20 * TICKS + 1.5 * TICKS),
    (21.5, lambda d: d.update(chain=[0, 1, 16]), 0),
    # the playing pattern's rate: the same entry, the elapsed beats wrapped into its new length
    (23.5, lambda d: d['patterns'][1].update(rate=4), (16 + 2 + 1.5) * TICKS),        # A02 now lasts 2 beats
    # a pattern elsewhere in the chain changing length moves every later entry: the same entry
    (5., lambda d: d['patterns'].__setitem__(16, pattern(rate=.5, bar=8, bars=2)), 5 * TICKS),
    (25., lambda d: d['patterns'].__setitem__(0, pattern(bars=2)), (8 + 4 + 4 + 1) * TICKS),   # B01 now starts at 16
])
def test_transport_changes_follow_score_v3(beat, change, expected):
    score = chain()
    if beat is None:                               # looping A02 at 1.5 beats in
        score = validate(dict(score, loop=True, pattern=1))
        beat = 1.5
    t = playing(score, beat)
    assert edited(t, 0., change) == 1
    assert t.start['tick'] == expected and t.elapsed == 0


@pytest.mark.parametrize('change', [
    lambda d: d['patterns'][0]['lanes'][0]['cells'].__setitem__(3, hit()),
    lambda d: d['patterns'][0]['lanes'][0].update(len=5, cells=[None] * 5),
    lambda d: d.update(pattern=16),                # selecting a pattern doesn't move the chain
    lambda d: d['patterns'].__setitem__(40, pattern(rate=8)),
    lambda d: d.update(seed=99, mutes=[True] * 6),
])
def test_ordinary_edits_keep_the_segment(change):
    t = playing(chain(), 21.5)
    start = dict(t.start)
    assert edited(t, .25, change) == 0
    assert t.start == start and t.beat(.25) == pytest.approx(22.)


def test_upgrading_keeps_the_song_but_moves_a_looped_patterns_beat_origin():
    old = copy.deepcopy(Tracker().document)
    t = playing(old, 40.5)
    revision = t.transport_revision
    t.replace(upgrade(old), 0., t.revision)
    assert t.transport_revision == revision and t.beat(0.) == 40.5      # the chain plays the same song
    assert t.start['tick'] == 40.5 * TICKS and t.v3
    old['loop'], old['pattern'] = True, 2
    t = playing(old, 40.5)
    revision = t.transport_revision
    t.replace(upgrade(old), 0., t.revision)
    assert t.transport_revision == revision + 1 and t.beat(0.) == 8.5   # beats count from A03's start
    t.replace(old, 0., t.revision)                                      # undo restores the old format
    assert t.transport_revision == revision + 2 and t.beat(0.) == 40.5 and not t.v3


def test_play_starts_a_segment_without_a_new_revision():
    t = playing(chain(), 3.)
    revision = t.transport_revision
    t.play(.26)
    assert t.transport_revision == revision and t.start['tick'] == round(3.52 * TICKS) and t.elapsed == 0


def test_fill_is_stamped_and_carried_into_the_next_segment(tmp_path):
    t = playing(chain())
    t.trace = (tmp_path / 'trace.jsonl').open('w')
    t.set_fill(True, 4800)
    t.set_fill(True, 5000)                                             # already on: not a change
    assert t.fill == off((4800, True))
    t.set_fill(False, 11520)
    t.seek(30., 3.)                                                    # 3 s at 120 BPM: tick 11520, off
    assert t.fill == off()
    t.set_fill(True, 10)
    t.seek(10., 6.)
    assert t.fill == [{'at': 0, 'on': True}]
    t.trace.close()
    lines = [json.loads(line) for line in (tmp_path / 'trace.jsonl').read_text().splitlines()]
    assert [line['fill'] for line in lines] == [{'at': 4800, 'on': True}, {'at': 11520, 'on': False}, {'at': 10, 'on': True}]
    assert lines[0]['received'] == 4800 and lines[0]['start'] == {'tick': 0, 'entry': 0}


def window(start, fill=None, mutes=None):
    return {'from': start, 'fill': fill or [{'at': start, 'on': False}],
            'mutes': mutes or [[{'at': start, 'on': False}] for _ in range(6)]}


def segment(t):
    return {'transport_revision': t.transport_revision, 'start': t.start}


def test_bus_windows_replace_from_their_tick_and_the_lease_keeps_one_owner():
    t = playing(chain())
    mutes = [[{'at': 4800, 'on': False}] for _ in range(6)]
    mutes[2] = [{'at': 4800, 'on': False}, {'at': 4801, 'on': True}]
    assert t.apply_window(segment(t), SENDER, window(4800, mutes=mutes), 10., 4810)
    assert t.bus['mutes'][2] == off((4800, False), (4801, True))       # the window's lists, from its tick
    # A window restates: a repeat, or one that starts earlier, leaves one record.
    assert t.apply_window(segment(t), SENDER, window(4800, mutes=mutes), 10.1, 4820)
    assert t.apply_window(segment(t), SENDER, window(0, fill=[{'at': 0, 'on': False}, {'at': 960, 'on': True}]), 10.2, 4830)
    assert t.bus['mutes'][2] == off() and t.bus['fill'] == off((960, True))
    with pytest.raises(ValueError, match='held by another module'):
        t.apply_window(segment(t), 'ffffffffffffffff', window(4800), 11., 4840)
    assert not t.apply_window(dict(segment(t), transport_revision=0), SENDER, window(0), 11., 4840)   # stale: dropped
    t.housekeep(13.9, 4990)                                            # any request from the holder renews it
    assert t.lease is not None
    t.housekeep(14., 5000)                                             # 3 s after the holder's last request
    assert t.lease is None and t.bus['fill'] == off((960, True), (5000, False))
    assert t.apply_window(segment(t), 'ffffffffffffffff', window(5000), 14., 5010)


def test_a_window_refused_for_the_lease_is_traced(tmp_path):
    """evaluator.md#trace: a window Studio refuses is a line, like a dropped one (#23)."""
    t = playing(chain())
    t.trace = (tmp_path / 'trace.jsonl').open('w')
    assert t.apply_window(segment(t), SENDER, window(4800), 10., 4810)
    with pytest.raises(ValueError, match='held by another module'):
        t.apply_window(segment(t), 'ffffffffffffffff', window(4800), 11., 4840)
    assert not t.apply_window(dict(segment(t), transport_revision=0), SENDER, window(0), 11., 4850)
    t.trace.close()
    lines = [json.loads(line) for line in (tmp_path / 'trace.jsonl').read_text().splitlines()]
    windows = [line for line in lines if 'window' in line.get('bus', {})]
    assert windows == [
        {'transport_revision': 1, 'start': t.start, 'received': 4840,
         'bus': {'window': 'refused', 'reason': 'bus held by another module', 'from': 4800}},
        {'transport_revision': 1, 'start': t.start, 'received': 4850,
         'bus': {'window': 'dropped', 'reason': 'stale segment', 'from': 0}},
    ]


@pytest.mark.parametrize('broken', [
    lambda w: w.update({'from': -1}),
    lambda w: w.update({'from': 1.5}),
    lambda w: w.pop('fill'),
    lambda w: w['mutes'].pop(),
    lambda w: w.update(fill=[]),
    lambda w: w.update(fill=[{'at': 4801, 'on': True}]),             # must start at from
    lambda w: w.update(fill=[{'at': 4800, 'on': True}, {'at': 4790, 'on': False}]),
    lambda w: w.update(fill=[{'at': 4800, 'on': 1}]),
    lambda w: w.update(extra=True),
])
def test_malformed_windows_are_refused(broken):
    t = playing(chain())
    w = window(4800)
    broken(w)
    with pytest.raises(ValueError):
        t.apply_window(segment(t), SENDER, w, 0., 0)
    with pytest.raises(ValueError):
        t.apply_window(segment(t), 'not hex', window(4800), 0., 0)


@pytest.mark.parametrize('broken, first', [
    (lambda w: w.update({'from': -1}), None),
    (lambda w: w.update({'from': 1.5}), None),
    (lambda w: w.update({'from': True}), None),
    (lambda w: w.update(fill=[{'at': 4801, 'on': True}]), 4800),
    (lambda w: w.update(extra=True), 4800),
    (lambda w: [w.clear(), w.update(fill=None)], None),
])
def test_a_malformed_window_is_traced_as_refused(tmp_path, broken, first):
    """evaluator.md#trace: a window that fails validation is a refused line too; the reply is still a 400."""
    t = playing(chain())
    t.trace = (tmp_path / 'trace.jsonl').open('w')
    w = window(4800)
    broken(w)
    with pytest.raises(ValueError):
        t.apply_window(segment(t), SENDER, w, 10., 4840)
    with pytest.raises(ValueError):
        t.apply_window(segment(t), SENDER, [], 10., 4850)                # not even an object
    t.trace.close()
    lines = [json.loads(line) for line in (tmp_path / 'trace.jsonl').read_text().splitlines()]
    assert lines == [
        {'transport_revision': 1, 'start': t.start, 'received': 4840,
         'bus': {'window': 'refused', 'reason': 'malformed window', 'from': first}},
        {'transport_revision': 1, 'start': t.start, 'received': 4850,
         'bus': {'window': 'refused', 'reason': 'malformed window', 'from': None}},
    ]


def test_folding_keeps_what_plays_from_the_horizon_on():
    t = playing(chain())
    transitions = off(*[(k * 100, k % 2 == 1) for k in range(1, 60)])
    t.apply_window(segment(t), SENDER, window(0, fill=transitions), 0., 0)
    t.fold(5900 + 10 * 2 * TICKS)                                     # 10 s at 120 BPM behind the frontier
    folded = t.bus['fill']
    assert len(folded) < len(transitions) and folded[0]['at'] == 0
    from hydrapy.tracker_v3 import state_at
    assert all(state_at(folded, u) == state_at(transitions, u) for u in range(5900, 7000))


def test_lanes_automation_and_scenes_follow_the_timeline(tmp_path):
    a, b = pattern(), pattern(rate=2)
    a['keys']['z'] = [{'r': 0, 'v': .1}, {'r': 8, 'v': .9}]
    a['easing']['z'] = 'linear'
    a['scenes'] = [{'r': 0, 'name': 'a', 'fade': 0}]
    b['scenes'] = [{'r': 8, 'name': 'b', 'fade': 4}]                  # 4 rows of A02 are half a beat
    a['lanes'][5]['cells'][4] = hit(note='A-4')
    score = validate(blank(bpm=120, loop=False, chain=[0, 1], patterns=[a, b] + [None] * 126,
                           bindings={'z': 'Dolly Z', 'roll': '', 'feedback': ''}))
    t = playing(score)
    t.trace = (tmp_path / 't.jsonl').open('w')
    control = type('Control', (), {'set': lambda self, v: setattr(self, 'value', v)})()
    patch = type('Patch', (), {'controls': {'Dolly Z': control}, 'scenes': {'a': (0,), 'b': (0,)},
                               'tracker_frame': None, 'scene_weights': None})()
    t.apply(patch, .25)                                                # beat .5: row 2 of A01
    assert control.value == pytest.approx(.3) and patch.tracker_frame['row'] == 2.
    t.apply(patch, .6)                                                 # past row 4: lane 6's hit
    assert patch.tracker_frame['note'][5] == 69 and patch.tracker_frame['gate'][5] == 0.
    assert scene_weights(score, 5.25, patch.scenes)['b'] == pytest.approx(.5)   # halfway into b's fade
    assert scene_weights(score, 5.75, patch.scenes) == {'a': 0., 'b': 1.}
    t.trace.close()
    [line] = [json.loads(x) for x in (tmp_path / 't.jsonl').read_text().splitlines()]
    assert line['id'] == '0.5.4' and line['transport_revision'] == 1 and line['fires']


@pytest.fixture
def studio(tmp_path):
    (tmp_path / 'p.py').write_text(PATCH)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30, trace=tmp_path / 'trace.jsonl')
    try:
        p = pattern()
        p['lanes'][0]['cells'][0] = hit({'kind': 'fill'})
        score = blank(bpm=120, patterns=[p] + [None] * 127)
        session.call('tracker_score', document=score, revision=session.tracker.revision)
        yield session
    finally:
        session.close()


def test_studio_publishes_the_version_3_playhead(studio):
    studio.call('tracker_transport', enabled=True, playing=False, beat=1.)
    time.sleep(.1)
    with studio.lock:
        bridge = dict(studio.bridge_state)
    assert bridge['start'] == {'tick': 1920, 'entry': None} and bridge['elapsed'] == 0
    assert bridge['fill'] == off() and bridge['bus']['mutes'] == [off()] * 6 and bridge['entry'] is None
    with pytest.raises(ValueError, match='below the cycle'):
        studio.call('tracker_transport', beat=4.)                      # A01 is 16 rows: 4 beats
    reply = studio.call('tracker_transport', fill=True)
    assert reply['fill'] == off((0, True))                             # nothing evaluated yet: stamped 0


def test_studio_takes_rack_windows_and_refuses_mixed_requests(studio):
    studio.call('tracker_transport', enabled=True, playing=True, beat=0.)
    time.sleep(.15)
    reply = studio.call('tracker_transport', playing=False)
    seg = {'transport_revision': reply['transport_revision'], 'start': reply['start']}
    reply = studio.call('tracker_transport', segment=seg, sender=SENDER, bus=window(0, fill=[{'at': 0, 'on': True}]))
    assert reply['bus']['fill'] == [{'at': 0, 'on': True}] and reply['transport_revision'] == seg['transport_revision']
    with pytest.raises(ValueError, match='without beat'):
        studio.call('tracker_transport', segment=seg, sender=SENDER, bus=window(0), beat=1.)
    with pytest.raises(ValueError, match='held by another module'):
        studio.call('tracker_transport', segment=seg, sender='0123456789abcdef', bus=window(0))
    reply = studio.call('tracker_transport', playing=True)             # Play: a segment of its own
    assert reply['transport_revision'] == seg['transport_revision'] and reply['start'] != seg['start']
    assert reply['bus']['fill'] == [{'at': 0, 'on': True}]             # carried into it
    reply = studio.call('tracker_transport', segment=seg, sender=SENDER, bus=window(0))
    assert reply['bus']['fill'] == [{'at': 0, 'on': True}]             # the old segment's window: dropped


def test_studio_refuses_fill_and_windows_for_older_scores(tmp_path):
    (tmp_path / 'p.py').write_text(PATCH)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        with pytest.raises(ValueError, match='hydrapy-tracker-3'):
            session.call('tracker_transport', fill=True)
        with pytest.raises(ValueError, match='hydrapy-tracker-3'):
            session.call('tracker_transport', segment={}, sender=SENDER, bus=window(0))
    finally:
        session.close()


def test_direct_cv_plays_studios_version_3_playhead(studio):
    cv = TrackerCV({0: 'gate1'}, samplerate=48000, open_stream=False)
    studio.call('tracker_transport', enabled=True, playing=False, beat=0., fill=True)
    clock = studio.tracker.anchor_time
    cv.publish(studio.tracker, clock, 100., False)
    cv.render(1, 100.)
    cv.publish(studio.tracker, clock, 100., True)
    volts = cv.render(4800, 100.)
    assert volts[0, 0] == 5. and volts[3000, 0] == 0.                  # row 0's F, judged with FILL on
    assert studio.tracker.trace is not None


# -- regressions from the review of #64/#65 -------------------------------------

def test_older_scores_keep_their_song_relative_beats_on_the_transport(tmp_path):
    (tmp_path / 'p.py').write_text(PATCH)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        document = copy.deepcopy(session.tracker.document)
        document.update(loop=True, pattern=2)
        session.call('tracker_score', document=document, revision=session.tracker.revision)
        for request in ({'playing': False}, {'enabled': True}, {'blackout': True}, {'beat': 32.}):
            session.call('tracker_transport', **request)
        assert session.tracker.beat(session.clock_time) == 32.
    finally:
        session.close()


def test_fill_set_while_paused_holds_into_the_next_segment():
    t = playing(chain(), 3.)
    t.set_fill(True, round(.3 * 3840) + 7)           # stamped past where the playhead paused
    t.play(.3)
    assert t.fill == [{'at': 0, 'on': True}]
    t.set_fill(False, 50)
    t.seek(5., .3)
    assert t.fill == off()


def test_a_chain_upgrade_undone_and_redone_keeps_fill():
    old = copy.deepcopy(Tracker().document)
    t = playing(old, 3.)
    new = upgrade(old)
    t.replace(new, 0., t.revision)
    t.set_fill(True, 0)
    t.replace(old, 0., t.revision)
    t.replace(new, 0., t.revision)
    assert t.fill == [{'at': 0, 'on': True}]


def test_play_on_the_anchors_own_tick_continues_the_segment():
    p = pattern()
    t = playing(validate(blank(bpm=120, patterns=[p] + [None] * 127)))
    start, elapsed = dict(t.start), t.elapsed
    t.play(2.)                                        # exactly one 7680-tick cycle at 120 BPM
    assert t.start == start and t.elapsed == elapsed and t.ticks(2.5) == 9600


def test_studios_outputs_read_fill_live_from_the_stamp():
    # The CV has evaluated up to its frontier; Studio stamps FILL there. The next block,
    # rendered before any new snapshot, must already judge with FILL on.
    p = pattern()
    p['lanes'][0]['cells'][2] = hit({'kind': 'fill'})
    t = playing(validate(blank(bpm=120, patterns=[p] + [None] * 127)))
    cv = TrackerCV({0: 'gate1'}, samplerate=48000, open_stream=False)
    cv.publish(Tracker(), 0., 100., False)            # an earlier playhead, so this one is heard live
    cv.render(1, 100.)
    cv.publish(t, 0., 100., True)
    first = cv.render(6000, 100.)                     # row 0, to tick 480
    with t.layers.lock:
        stamp = cv.evaluated(t.segment)
        t.set_fill(True, stamp)
    later = cv.render(12000, 100. + 6000 / 48000)
    assert stamp == 480 and not first.any() and later[6000, 0] == 5.  # row 2, at tick 960


def test_patch_lane_signals_hear_play_as_a_transport_change():
    p = pattern()
    p['lanes'][0]['cells'][0] = hit()
    t = playing(validate(blank(bpm=120, patterns=[p] + [None] * 127)))
    patch = type('Patch', (), {'controls': {}, 'scenes': {}, 'tracker_frame': None, 'scene_weights': None})()
    t.apply(patch, 0.)
    t.apply(patch, .01)                               # inside row 0's gate
    assert patch.tracker_frame['gate'][0] == 1.
    t.play(.02)                                       # paused there, then Play: a new segment
    t.apply(patch, .02)
    assert patch.tracker_frame['gate'][0] == 1. and patch.tracker_frame['since'][0] is not None


def test_bus_window_replies_keep_studios_own_error(studio):
    reply = studio.call('tracker_transport', enabled=True, playing=False, beat=0.)
    seg = {'transport_revision': reply['transport_revision'], 'start': reply['start']}
    studio.error = 'Render paused: something'
    studio.call('tracker_transport', segment=seg, sender=SENDER, bus=window(0))
    assert studio.error == 'Render paused: something'
    with pytest.raises(ValueError) as refused:
        studio.call('tracker_transport', segment=seg, sender='0123456789abcdef', bus=window(0))
    assert str(refused.value) == 'bus held by another module' and studio.error == 'Render paused: something'
