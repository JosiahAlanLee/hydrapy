import time

import pytest

from hydrapy.studio_native import Pacer


def test_pacer_budget_is_a_percentile_not_one_stall():
    p = Pacer(margin=.0015)
    for _ in range(239):
        p.record(.001)
    p.record(.021)  # one garbage collection
    assert p.budget(1 / 60) == pytest.approx(.0025)


def test_pacer_widens_on_a_miss_and_narrows_slowly():
    p = Pacer(margin=.0015)
    for _ in range(60):
        p.record(.001)
    p.record(.001, missed=True)
    assert p.margin == pytest.approx(.0025)
    for _ in range(299):
        p.record(.001)
    assert p.margin == pytest.approx(.0025)
    p.record(.001)
    assert p.margin == pytest.approx(.00225)


def test_pacer_starts_the_frame_its_budget_before_the_next_flip():
    p = Pacer(margin=.0015)
    for _ in range(60):
        p.record(.002)
    period = 1 / 60
    assert p.start_at(10., period) == pytest.approx(10. + period - .0035)
    assert p.budget(.003) == pytest.approx(.002)  # never more than a refresh less a millisecond, nor under 2 ms


def test_studio_times_on_screen_commands(tmp_path):
    from hydrapy.studio import Session
    patch = tmp_path / 'p.py'
    patch.write_text('from hydrapy import *\nosc(4).out()\n')
    s = Session(tmp_path, 'p.py', backend='cpu', size=(64, 36), fps=60)
    try:
        assert s.edit_delay_ms == 120
        s.call('tracker_transport', enabled=True, playing=True, beat=0)
        s.call('tracker_transport', blackout=True, _mark='cue')
        s.call('source', path='p.py')  # not on screen: not timed
        time.sleep(.2)
        report = s.latency_report()
        assert report['measured_to'] == 'render' and report['native'] is None
        cue = [r for r in report['samples'] if r['mark'] == 'cue']
        assert len(cue) == 1 and cue[0]['action'] == 'tracker_transport'
        assert cue[0]['queue_ms'] >= 0 and cue[0]['total_ms'] >= cue[0]['queue_ms']
        assert all(r['action'] != 'source' for r in report['samples'])
        assert s.tracker_state['edit_delay_ms'] == 120
    finally:
        s.close()
