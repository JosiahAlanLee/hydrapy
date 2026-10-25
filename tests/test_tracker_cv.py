"""Tracker CV: sample-exact clock, gates and lanes, derived from the same score as the video."""
import copy
import time
import types

import numpy as np
import pytest

from hydrapy.studio import Session
from hydrapy.studio_tracker import Tracker, value_at
from hydrapy.tracker_cv import DEFAULT_SIGNALS, TrackerCV, lane_values, parse_map

SR = 48000


def empty_score(bpm=120):
    d = copy.deepcopy(Tracker().document)
    d['bpm'] = bpm
    for p in d['patterns']:
        p['audio'] = [[None] * 64 for _ in range(4)]
    return d


def playing(document, beat=0.):
    tracker = Tracker(document)
    tracker.enabled = True
    tracker.seek(beat, 0.)
    return tracker


def run(mapping, tracker, seconds, playing=True, **options):
    cv = TrackerCV(mapping, samplerate=SR, open_stream=False, **options)
    cv.publish(tracker, 0., 100., playing)        # playhead at clock 0 == monotonic 100
    return cv, cv.render(int(seconds * SR), 100.)


def rising(column):
    high = column > 0
    return np.flatnonzero(high[1:] & ~high[:-1]) + 1


def test_clock_and_reset_are_sample_exact_at_the_score_tempo():
    tracker = playing(empty_score(bpm=120))       # 2 beats/s, sixteenths every 0.125 s
    _, volts = run({0: 'clock', 1: 'reset', 2: 'run'}, tracker, 1.)
    edges = rising(volts[:, 0])
    assert volts[0, 0] == 5.
    assert np.array_equal(edges, np.arange(1, 8) * SR // 8)
    assert np.all(volts[:, 2] == 5.)
    reset = np.flatnonzero(volts[:, 1])
    assert reset[0] == 0 and abs(len(reset) - round(.005 * SR)) <= 1   # 5 ms, within a sample


def test_gates_follow_hits_mutes_and_gate_length():
    d = empty_score(bpm=120)
    d['patterns'][0]['audio'][0][2] = {'note': 'C-2', 'vel': 127}
    d['patterns'][0]['audio'][1][2] = {'note': 'D-2', 'vel': 127}
    d['mutes'][1] = True
    _, volts = run({0: 'gate1', 1: 'gate2'}, playing(d), .5, gate=.25)
    row = SR // 8                                  # one sixteenth at 120 BPM
    on = np.flatnonzero(volts[:, 0])
    assert on[0] == 2 * row and on[-1] == 2 * row + row // 4 - 1
    assert not volts[:, 1].any()


def test_paused_or_disabled_tracker_sends_no_clock_or_gates():
    d = empty_score()
    d['patterns'][0]['audio'][0][0] = {'note': 'C-2', 'vel': 100}
    tracker = playing(d)
    _, paused = run({0: 'clock', 1: 'gate1', 2: 'z'}, tracker, .1, playing=False)
    assert not paused[:, :2].any()
    tracker.enabled = False
    _, disabled = run({0: 'clock', 1: 'gate1', 2: 'z'}, tracker, .1)
    assert not disabled[:, :2].any()
    assert np.allclose(disabled[:, 2], .35 * 5)   # lanes hold their value


@pytest.mark.parametrize('ease', ['hold', 'linear', 'smooth'])
def test_lane_values_match_the_video_evaluator(ease):
    p = copy.deepcopy(Tracker().document['patterns'][1])
    p['keys']['roll'] = [{'r': 3, 'v': -40.}, {'r': 10, 'v': 20.}, {'r': 50, 'v': 45.}]
    p['easing']['roll'] = ease
    rows = np.linspace(0, 63.99, 2000)
    assert np.allclose(lane_values(p, 'roll', rows), [value_at(p, 'roll', r) for r in rows])


def test_held_key_steps_on_its_exact_sample_and_scales_to_volts():
    d = empty_score(bpm=120)
    d['patterns'][0]['keys']['roll'] = [{'r': 0, 'v': -45.}, {'r': 4, 'v': 45.}]
    d['patterns'][0]['easing']['roll'] = 'hold'
    _, volts = run({0: 'roll'}, playing(d), 1.)
    step = 4 * SR // 8
    assert volts[step - 1, 0] == -5. and volts[step, 0] == 5.


def test_velocity_holds_across_blocks_and_loop_reset_uses_the_pattern_start():
    d = empty_score(bpm=120)
    d['patterns'][2]['audio'][3][0] = {'note': 'A-4', 'vel': 64}
    d['loop'], d['pattern'] = True, 2
    tracker = playing(d, beat=32)
    cv = TrackerCV({0: 'vel4', 1: 'reset'}, samplerate=SR, open_stream=False)
    cv.publish(tracker, 0., 100., True)
    first = cv.render(SR // 4, 100.)
    second = cv.render(SR // 4, 100.25)            # no hit in this block
    assert first[-1, 0] == pytest.approx(64 / 127 * 5)
    assert np.allclose(second[:, 0], 64 / 127 * 5)
    assert first[0, 1] == 5.


def test_offset_and_extrapolation_from_the_last_snapshot():
    tracker = playing(empty_score(bpm=120))
    cv = TrackerCV({0: 'clock'}, samplerate=SR, open_stream=False, offset_ms=10)
    cv.publish(tracker, 2., 100., True)             # clock 2 s at monotonic 100
    beats, _ = cv.beats(cv.snapshot, np.array([100., 100.5]))
    assert np.allclose(beats, [4., 5.])
    volts = cv.render(SR // 8, 100. - .01)          # 10 ms early: lands on beat 4
    assert volts[0, 0] == 5. and rising(volts[:, 0]).size == 0


def test_parse_map_defaults_and_errors():
    assert parse_map('8') == {8 + i: s for i, s in enumerate(DEFAULT_SIGNALS)}
    assert parse_map('2=clock, 3=reset') == {2: 'clock', 3: 'reset'}
    for bad in ('', '1=clock,1=reset', '1=volume', 'clock'):
        with pytest.raises(ValueError):
            parse_map(bad)


def test_callback_scales_to_full_scale_and_reports_instead_of_raising():
    tracker = playing(empty_score(bpm=60))         # clock stays high for the first 125 ms
    cv = TrackerCV({1: 'clock', 3: 'roll'}, samplerate=SR, open_stream=False)
    cv.publish(tracker, 0., time.monotonic(), True)  # no DAC time: the callback uses "now"
    out = np.ones((64, 4), np.float32)
    cv._callback(out, 64, types.SimpleNamespace(outputBufferDacTime=0., currentTime=0.), None)
    assert np.allclose(out[:, 1], .5) and np.allclose(out[:, [0, 2]], 0.)
    cv.snapshot = {'broken': True}
    cv._callback(out, 64, types.SimpleNamespace(), None)
    assert cv.error and not out.any()


def test_studio_publishes_its_playhead_to_cv(tmp_path):
    (tmp_path / 'p.py').write_text("from hydrapy import *\nsolid(1,0,0).out(o0)\n")
    cv = TrackerCV({0: 'clock'}, samplerate=SR, open_stream=False)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30, cv=cv)
    try:
        session.call('tracker_transport', enabled=True, playing=True, beat=8)
        deadline = time.monotonic() + 5
        while not (cv.snapshot and cv.snapshot['enabled']) and time.monotonic() < deadline:
            time.sleep(.01)                         # published after the next render iteration
        snap = cv.snapshot
        assert snap['enabled'] and snap['playing']
        beats, _ = cv.beats(snap, np.array([time.monotonic()]))
        assert 8 <= beats[0] < 9
        assert session.state is not None
        assert session.tracker_state['cv']['map'] == {'0': 'clock'}
    finally:
        session.close()
    assert cv.stream is None


def test_reset_survives_dac_lookahead_and_does_not_retrigger_on_publish(monkeypatch):
    cv = TrackerCV({0: 'reset'}, samplerate=SR, open_stream=False)
    tracker = playing(empty_score())
    cv.publish(tracker, 0., 100., True)
    monkeypatch.setattr('hydrapy.tracker_cv.time.monotonic', lambda: 100.)
    out = np.zeros((256, 1), np.float32)
    cv._callback(out, 256, types.SimpleNamespace(currentTime=10., outputBufferDacTime=10.02), None)
    assert np.count_nonzero(out[:, 0]) == 240  # full 5 ms despite 20 ms DAC lead
    cv.publish(tracker, .005, 100.005, True)
    cv._callback(out, 256, types.SimpleNamespace(currentTime=10., outputBufferDacTime=10.026), None)
    assert not out.any()
    tracker.seek(0., .005)
    cv.publish(tracker, .005, 100.005, True)
    cv._callback(out, 256, types.SimpleNamespace(currentTime=10., outputBufferDacTime=10.03), None)
    assert np.count_nonzero(out[:, 0]) == 240


def test_duplicate_velocity_and_pitch_routes_are_identical_across_a_hit():
    d = empty_score()
    d['patterns'][0]['audio'][0][1] = {'note': 'D#4', 'vel': 100}
    cv = TrackerCV({0: 'vel1', 1: 'vel1', 2: 'pitch1', 3: 'pitch1'}, samplerate=SR, open_stream=False)
    cv.publish(playing(d), 0., 100., True)
    volts = cv.render(256, 100.124)
    assert np.array_equal(volts[:, 0], volts[:, 1])
    assert np.array_equal(volts[:, 2], volts[:, 3])
    assert volts[0, 0] == volts[0, 2] == 0.
    assert volts[-1, 0] == pytest.approx(100 / 127 * 5)
    assert volts[-1, 2] == pytest.approx(3 / 12)


def test_pitch_octaves_key_transposition_air_mute_and_zero_velocity():
    d = empty_score()
    d['key'] = 7  # G minor = +5 semitones from the D-based score
    for row, note in enumerate(('C-4', 'C-5', 'AIR', 'D-5')):
        d['patterns'][0]['audio'][1][row] = {'note': note, 'vel': 0 if row == 3 else 100}
    cv, volts = run({0: 'pitch2', 1: 'gate2'}, playing(d), .5, pitch_zero=48)
    assert volts[0, 0] == pytest.approx(17 / 12)
    assert volts[6000, 0] == pytest.approx(29 / 12)
    assert np.allclose(volts[12000:, 0], 29 / 12)  # AIR and velocity 0 hold pitch
    assert not volts[18000:, 1].any()
    d['mutes'][1] = True
    cv.publish(playing(d), 0., 101., True)
    muted = cv.render(256, 101.)
    assert not muted[:, 1].any()


def test_pause_freezes_automation_and_holds_pitch_without_clock():
    d = empty_score()
    d['patterns'][0]['audio'][0][0] = {'note': 'A-4', 'vel': 80}
    tracker = playing(d)
    cv, volts = run({0: 'pitch1', 1: 'clock', 2: 'roll'}, tracker, .25)
    cv.publish(tracker, .25, 100.25, False)
    paused = cv.render(256, 101.)
    assert np.allclose(paused[:, 0], .75)
    assert not paused[:, 1].any()
    assert len(np.unique(paused[:, 2])) == 1


def test_callback_records_underflows():
    cv = TrackerCV({0: 'clock'}, open_stream=False)
    status = types.SimpleNamespace(output_underflow=True)
    cv._callback(np.zeros((256, 1), np.float32), 256, types.SimpleNamespace(), status)
    assert cv.underflows == 1 and cv.describe()['status']
