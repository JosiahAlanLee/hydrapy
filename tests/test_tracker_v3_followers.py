"""Version-3 followers: TrackerCV, the Breaks synth and the patch's lane signals play evaluate()'s candidates."""
import copy
import random
import types

import numpy as np
import pytest

from hydrapy.studio_audio import BreaksSynth
from hydrapy.studio_tracker import LaneState, Tracker
from hydrapy.tracker_cv import TrackerCV
from hydrapy.tracker_v3 import TICKS, Timeline, anchor, upgrade, validate
from test_tracker_v3 import blank, hit, off, pattern

SR = 48000
ROW = SR // 8                       # a rate-1 row at 120 BPM, in samples
EVERY = {0: 'clock', 1: 'reset', 2: 'run', 3: 'gate1', 4: 'gate2', 5: 'gate3', 6: 'gate4', 7: 'vel1', 8: 'pitch2',
         9: 'z', 10: 'roll', 11: 'feedback', 12: 'vel3', 13: 'pitch4', 14: 'pitch1', 15: 'vel4'}


def v3(document, tick=0, elapsed=0, revision=0, fill=None, bus=None, enabled=True):
    """The playhead a version-3 Studio publishes, as TrackerCV.publish reads it."""
    d = validate(document)
    cycle = Timeline(d).cycle
    return types.SimpleNamespace(document=d, enabled=enabled, transport_revision=revision,
                                 anchor_beat=(tick + elapsed) % cycle / TICKS, anchor_time=0., start=anchor(d, tick),
                                 elapsed=elapsed, fill=fill or off(), bus=bus or {'fill': off(), 'mutes': [off()] * 6})


def render(tracker, seconds, mapping=EVERY, delay=0., cv=None, playing=True, live=True, **options):
    if cv is None:
        cv = TrackerCV(mapping, samplerate=SR, open_stream=False, **options)
        if live:    # follow an earlier, stopped segment first, so this one is heard as it starts
            cv.publish(types.SimpleNamespace(**dict(vars(tracker), transport_revision=tracker.transport_revision - 1)),
                       0., 100., False)
            cv.render(1, 100.)
    cv.publish(tracker, 0., 100., playing)              # playhead at clock 0 == monotonic 100
    return cv, cv.render(int(seconds * SR), 100. + delay)


def rising(column):
    high = column > 0
    return np.flatnonzero(high[1:] & ~high[:-1]) + 1


def v2_score(seed):
    """A random version-2 score: hits, AIR, zero velocity, mutes, keys and a loop setting."""
    rng = random.Random(seed)
    d = copy.deepcopy(Tracker().document)
    d.update(bpm=rng.choice([97, 120, 138, 171]), key=rng.choice([0, 2, 5, 7]), loop=rng.random() < .5,
             pattern=rng.randrange(4), mutes=[rng.random() < .2 for _ in range(4)])
    for p in d['patterns']:
        p['audio'] = [[{'note': rng.choice(['C-2', 'D-4', 'AIR', 'G#5']), 'vel': rng.choice([0, 40, 127])}
                       if rng.random() < .3 else None for _ in range(64)] for _ in range(4)]
        p['easing'] = {lane: rng.choice(['hold', 'linear', 'smooth']) for lane in p['easing']}
    return d


@pytest.mark.parametrize('seed', range(6))
def test_an_upgraded_score_sends_the_same_voltages(seed):
    old = v2_score(seed)
    new = upgrade(old)
    rng = random.Random(seed)
    for _ in range(3):
        beat = rng.uniform(0, 16 if old['loop'] else 64)
        if old['loop']:
            beat = beat % 16 + 16 * old['pattern']       # version 2 counts a loop's beats from the song's start
        tick = round(beat * TICKS) % Timeline(new).cycle if not old['loop'] else round((beat - 16 * old['pattern']) * TICKS)
        legacy = Tracker(old)
        legacy.enabled = True
        legacy.seek(tick / TICKS + (16 * old['pattern'] if old['loop'] else 0), 0.)
        for delay in (0., .03, .2):
            _, want = render(legacy, 1.5, delay=delay)
            _, got = render(v3(new, tick, revision=legacy.transport_revision), 1.5, delay=delay)
            # Automation is real-valued: its row position comes from ticks here, from the beat there.
            assert np.array_equal(got[:, :9], want[:, :9]) and np.array_equal(got[:, 12:], want[:, 12:]), (beat, delay)
            assert np.allclose(got[:, 9:12], want[:, 9:12], rtol=0, atol=1e-9)


def test_lanes_five_and_six_play_in_version_3_only():
    p = pattern()
    p['lanes'][4]['cells'][1] = hit(note='A-4')
    p['lanes'][5]['cells'][2] = dict(hit(note='C-3'), vel=127)
    _, volts = render(v3(blank(bpm=120, patterns=[p] + [None] * 127)), .5,
                      {0: 'gate5', 1: 'gate6', 2: 'pitch5', 3: 'vel6'})
    assert list(rising(volts[:, 0])) == [ROW] and list(rising(volts[:, 1])) == [2 * ROW]
    assert volts[ROW, 2] == (69 + 2 - 2 - 60) / 12 and volts[-1, 3] == 5.
    legacy = Tracker()
    legacy.enabled = True
    _, volts = render(legacy, .5, {0: 'gate5', 1: 'pitch6'})
    assert not volts.any()


def test_each_ply_sub_step_gets_its_own_gate():
    p = pattern()
    p['lanes'][0]['cells'][2] = hit()
    p['lanes'][0]['transforms'] = [{'kind': 'ply', 'n': 4}]
    _, volts = render(v3(blank(bpm=120, patterns=[p] + [None] * 127)), .5, {0: 'gate1'})
    quarter = ROW // 4
    assert list(rising(volts[:, 0])) == [2 * ROW + k * quarter for k in range(4)]
    assert np.count_nonzero(volts[:, 0]) == 4 * (quarter // 2)


def test_a_bus_mute_that_arrives_during_a_hit_lets_it_finish():
    p = pattern()
    p['lanes'][0]['cells'][1] = p['lanes'][0]['cells'][3] = hit()
    bus = {'fill': off(), 'mutes': [off((480 + 60, True))] + [off()] * 5}
    _, volts = render(v3(blank(bpm=120, patterns=[p] + [None] * 127), bus=bus), .5, {0: 'gate1'})
    assert list(rising(volts[:, 0])) == [ROW]                # row 3 is muted
    assert np.count_nonzero(volts[:, 0]) == ROW // 2         # row 1's gate is whole


def test_a_late_fill_change_does_not_replay_what_was_already_judged():
    p = pattern()
    p['lanes'][0]['cells'][2] = p['lanes'][0]['cells'][6] = hit({'kind': 'fill'})
    score = blank(bpm=120, patterns=[p] + [None] * 127)
    cv, first = render(v3(score), .3, {0: 'gate1'})          # through row 2, judged with FILL off
    late = v3(score, fill=off((0, True)))                    # FILL on from the anchor, heard late
    cv.publish(late, 0., 100., True)
    rest = cv.render(int(.7 * SR), 100. + .3)
    volts = np.concatenate([first[:, 0], rest[:, 0]])
    assert list(rising(volts)) == [6 * ROW]                   # row 2 stays silent; row 6 sees the new FILL


def test_a_seek_heard_late_still_plays_the_candidate_it_landed_on():
    # evaluator.md's V6: the anchor at row 4, phase 1/4. Lane 0 has a cell on row 4.
    p = pattern()
    p['lanes'][0]['cells'][4] = hit()
    score = blank(bpm=120, patterns=[p] + [None] * 127)
    cv, _ = render(v3(score, revision=1), .01, {0: 'gate1'})
    ms = SR // 1000
    for revision, delay in ((2, 0.), (3, .03)):              # heard at once, then 30 ms late
        tracker = v3(score, tick=2040, revision=revision)
        _, volts = render(tracker, .2, {0: 'gate1'}, delay=delay, cv=cv)
        high = np.count_nonzero(volts[:, 0])
        if delay == 0.:
            assert volts[0, 0] == 5. and high == ROW // 4      # its own gate covers the landing gate
        else:
            assert high == ROW // 4 - ms                     # a quarter row, from when it was heard


def test_a_follower_that_joins_late_starts_at_its_own_tick():
    p = pattern()
    p['lanes'][0]['cells'][0] = p['lanes'][0]['cells'][4] = hit()
    score = blank(bpm=120, patterns=[p] + [None] * 127)
    _, volts = render(v3(score), .6, {0: 'gate1'}, delay=.01, live=False)     # first heard 10 ms into row 0
    assert volts[0, 0] == 0. and list(rising(volts[:, 0])) == [4 * ROW - int(.01 * SR)]


def test_reset_pulses_at_every_wrap_of_an_odd_length_cycle():
    p = pattern(bar=7)                                       # 7 rows: 1.75 beats, 0.875 s at 120 BPM
    _, volts = render(v3(blank(bpm=120, patterns=[p] + [None] * 127)), 3., {0: 'reset', 1: 'clock'})
    assert list(rising(volts[:, 0])) == [round(.875 * SR * k) for k in (1, 2, 3)]
    assert volts[round(.875 * SR), 1] == 5.                  # 7 whole clock periods: a pulse starts at the wrap


CLOCK = SR // 8                     # one clock period at 120 BPM and 4 PPQN, in samples


@pytest.mark.parametrize('p, cycle', [
    (pattern(rate=8, bar=3), 180),                           # 3/32 beat: shorter than half a clock period
    (pattern(rate=2, bar=5), 1200),                          # 5 rows of 1/8 beat: 2.5 clock periods
])
def test_clock_runs_steadily_through_cycle_wraps(p, cycle):
    tracker = v3(blank(bpm=120, patterns=[p] + [None] * 127))
    assert Timeline(tracker.document).cycle == cycle
    _, volts = render(tracker, 2., {0: 'clock', 1: 'reset'})
    assert len(rising(volts[:, 1])) >= 3                     # several wraps, each still resets
    high = volts[:, 0] > 0
    edges = np.flatnonzero(high[1:] != high[:-1]) + 1
    assert high[0] and list(edges) == [k * CLOCK // 2 for k in range(1, len(edges) + 1)]
    assert len(edges) == 2 * SR // (CLOCK // 2) - 1


@pytest.mark.parametrize('tick', [0, 7 * TICKS // 4 + 100])
def test_a_whole_beat_cycle_clocks_as_it_did_with_the_cycle_beat(tick):
    from hydrapy import tracker_v3
    score = blank(bpm=120, patterns=[pattern(bars=2)] + [None] * 127)        # 8 beats
    cv, volts = render(v3(score, tick=tick), 9., {0: 'clock'})
    times = 100. + np.arange(len(volts)) / SR
    u = tracker_v3.elapsed(cv.snapshot, times)
    beat = tracker_v3.position(cv.snapshot['document'], tick, u)[0]
    assert np.array_equal(volts[:, 0], (np.mod(beat * 4, 1.) < .5) * 5.)


def test_automation_follows_each_entrys_own_rate_and_keys():
    a, b = pattern(), pattern(rate=2)
    a['keys']['z'] = [{'r': 0, 'v': 0.}, {'r': 15, 'v': 1.}]
    a['easing']['z'] = 'linear'
    b['keys']['z'] = [{'r': 0, 'v': .2}]
    score = blank(bpm=120, loop=False, chain=[0, 1], patterns=[a, b] + [None] * 126)
    _, volts = render(v3(score), 3.5, {0: 'z'})
    assert volts[0, 0] == 0. and volts[15 * ROW, 0] == 5.               # rows 0 and 15 of A01
    assert volts[16 * ROW, 0] == pytest.approx(1.)                      # A02 from beat 4: z .2 of 0–5 V
    assert volts[3 * SR - 1, 0] == pytest.approx(1.) and volts[3 * SR, 0] == 0.        # A02 lasts 2 beats


def test_the_breaks_synth_plays_lane_one_candidates_at_their_ticks():
    p = pattern()
    p['lanes'][0]['cells'][2] = hit(note='C-2')
    p['lanes'][0]['transforms'] = [{'kind': 'ply', 'n': 2}]
    p['lanes'][0]['cells'][5] = dict(hit(note='D-2'), vel=0)
    tracker = v3(blank(bpm=120, patterns=[p] + [None] * 127))
    cv = TrackerCV({0: 'clock'}, samplerate=SR, open_stream=False)
    cv.publish(tracker, 0., 100., True)
    synth = BreaksSynth()
    started = []
    for k in range(40):
        before = {v[3] for v in synth.voices}
        synth.render(cv.snapshot, 100. + k * 512 / SR, 512, SR)
        started += [round((v[0] - 100.) * SR) for v in synth.voices if v[3] not in before]
    assert started == [2 * ROW, 2 * ROW + ROW // 2]


def test_patch_lane_signals_count_every_hit_between_frames():
    p = pattern()
    p['lanes'][5]['cells'][1] = hit(note='A-4')
    p['lanes'][5]['cells'][2] = dict(hit(note='C-5'), vel=127)
    p['lanes'][2]['cells'][3] = hit()
    score = blank(bpm=120, mutes=[False, False, True, False, False, False], patterns=[p] + [None] * 127)
    snapshot = dict(vars(v3(score)), playing=True)
    lanes = LaneState()
    lanes.update_v3(snapshot, 0., True)
    frame = lanes.update_v3(snapshot, 2.5 * 480, True)      # one frame later, past rows 1 and 2
    assert frame['note'][5] == 72 + 2 - 2 and frame['vel'][5] == 1. and frame['gate'][5] == 0.
    assert frame['since'][5] == pytest.approx(.5 * 480 * 60 / 120 / TICKS)
    assert frame['row'] == 2. and frame['beat'] == pytest.approx(2.5 / 4)
    frame = lanes.update_v3(snapshot, 3.2 * 480, True)
    assert frame['gate'][2] == 0. and frame['since'][2] is None     # muted in the score
    assert lanes.update_v3(dict(snapshot, transport_revision=1), 0., True)['note'][5] == 60.


def test_a_new_segment_without_a_transport_change_sends_no_reset():
    # A chain-mode score upgraded to version 3 while playing gets a segment of its own with
    # the same transport_revision: evaluation restarts, but nothing resets or plays twice.
    p = pattern()
    p['lanes'][0]['cells'][2] = hit(note='C-2')
    score = blank(bpm=120, patterns=[p] + [None] * 127)
    mapping = {0: 'reset', 1: 'gate1', 2: 'vel1'}
    cv, before = render(v3(score, revision=4), .26, mapping)            # into row 2's gate
    synth = BreaksSynth()
    synth.render(cv.snapshot, 100., 1, SR)
    synth.render(cv.snapshot, 100. + 1 / SR, int(.26 * SR), SR)
    moved = v3(score, tick=round(.26 * 2 * TICKS), revision=4)         # the same place, as a new anchor
    moved.anchor_time = .26
    cv.publish(moved, .26, 100.26, True)
    after = cv.render(int(.1 * SR), 100.26)
    assert not after[:, 0].any() and after[0, 1] == 5. and after[0, 2] == before[-1, 2]
    voices = len(synth.voices)
    synth.render(cv.snapshot, 100.26, int(.1 * SR), SR)
    assert len(synth.voices) == voices                                 # row 2's kick isn't heard again


def test_publish_hands_the_callback_a_whole_snapshot(monkeypatch):
    legacy = Tracker()
    cv = TrackerCV({0: 'gate1'}, samplerate=SR, open_stream=False)
    cv.publish(legacy, 0., 100., True)
    before = cv.snapshot
    seen = []
    real = cv._cells
    monkeypatch.setattr(cv, '_cell_cache', None)
    monkeypatch.setattr(cv, '_cells', lambda d: seen.append(cv.snapshot is before) or real(d))
    cv.publish(legacy, 0., 100., True)
    assert seen == [True] and 'cells' in cv.snapshot and cv.snapshot is not before
