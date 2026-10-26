import os
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


def test_without_a_display_the_display_menu_says_why(tmp_path, monkeypatch):
    from hydrapy.studio import Session
    monkeypatch.delenv('DISPLAY', raising=False)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nosc(4).out()\n')
    s = Session(tmp_path, 'p.py', backend='cpu', size=(64, 36), native='available')
    try:
        time.sleep(.1)
        info = s.tracker_state['native']
        assert info['available'] is False and info['active'] is False and 'display' in info['note'].lower()
        with pytest.raises(ValueError, match='display'):
            s.call('native_output', enabled=True, monitor='window')
        assert s.edit_delay_ms == 120
    finally:
        s.close()


@pytest.mark.skipif(not os.environ.get('HYDRAPY_TEST_NATIVE'),
                    reason='opens a window on $DISPLAY; set HYDRAPY_TEST_NATIVE=1 (use a virtual display)')
def test_display_menu_shows_and_hides_the_program_window(tmp_path):
    from hydrapy.studio import Session
    (tmp_path / 'p.py').write_text('from hydrapy import *\nosc(4).out()\n')
    s = Session(tmp_path, 'p.py', backend='gl', size=(64, 36), native='available')
    try:
        deadline = time.monotonic() + 3
        while 'native' not in s.tracker_state and time.monotonic() < deadline:
            time.sleep(.02)  # published by the render loop after the patch loads
        assert s.tracker_state['native']['available'] and not s.tracker_state['native']['active']
        info = s.call('native_output', enabled=True, monitor='window', match_size=False)
        assert info['active'] and info['where'] == 'window' and s.size == (64, 36)
        assert s.edit_delay_ms == 0
        s.call('tracker_transport', enabled=True, playing=True, beat=0)
        s.call('tracker_transport', blackout=True, _mark='cue')
        time.sleep(.3)
        report = s.latency_report()
        assert report['measured_to'] == 'flip' and report['frames']
        assert [r for r in report['samples'] if r['mark'] == 'cue'][0]['to_flip_ms'] is not None
        assert not s.call('native_output', enabled=False)['active']
        # Matching the window's size reloads the patch at 1280x720 and gives the old size back after.
        s.call('native_output', enabled=True, monitor='window')
        assert s.size == (1280, 720) and s.renderer.size == (1280, 720)
        # The editor's Program monitor is now a small confidence preview of it.
        from PIL import Image
        import io
        frame = s.preview.frame(s.preview.output, -1, timeout=3, thumbnails=False)
        frame = s.preview.frame(s.preview.output, frame['sequence'], timeout=3, thumbnails=False)
        assert Image.open(io.BytesIO(frame['data'])).size == (480, 270)
        s.call('native_output', enabled=False)
        assert s.size == (64, 36) and s.renderer.size == (64, 36)
        assert (tmp_path / '.hydrapy' / 'outputs.json').read_text().count('"window"') == 1
    finally:
        s.close()


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
