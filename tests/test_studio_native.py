import os
import threading
import time

import pytest

from hydrapy.studio_native import Pacer


@pytest.fixture
def recording_output(tmp_path, monkeypatch):
    """Exercise Studio's real render loop without an X display or a video encoder."""
    from hydrapy.studio import Session

    class Window:
        active, where = False, None
        size, refresh, period = (96, 54), 60., 1 / 60

        def __init__(self):
            self.closed = threading.Event()
            self.presented = 0

        def target_size(self, where):
            return self.size if where == 'window' else (128, 72)

        def show(self, where):
            self.active, self.where = True, where

        def hide(self):
            self.active, self.where = False, None

        def poll(self):
            closed = self.closed.is_set()
            self.closed.clear()
            return closed

        def present(self, *args, **kwargs):
            now = time.monotonic()
            self.presented += 1
            return now, now, False

        def describe(self):
            return dict(available=True, active=self.active, where=self.where)

        def close(self):
            pass

    recordings = []

    class Recorder:
        def __init__(self, path, fps):
            self.path, self.frames, self.closed = path, [], False
            self.fail = threading.Event()
            self.close_error = False
            self.finishing, self.finished = threading.Event(), None   # set `finished` to hold close()
            recordings.append(self)

        def push(self, frame, at=None):
            if self.fail.is_set():
                raise RuntimeError('test encoder failed')
            self.frames.append(frame.copy())

        def close(self):
            self.finishing.set()
            if self.finished is not None:
                self.finished.wait(5)
            self.closed = True
            if self.close_error:
                raise RuntimeError('test encoder close failed')

    monkeypatch.setattr('hydrapy.studio_native.unavailable', lambda: None)
    monkeypatch.setattr('hydrapy.studio_native.NativeOutput', Window)
    monkeypatch.setattr('hydrapy.studio.Recorder', Recorder)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nosc(4).out()\n')
    s = Session(tmp_path, 'p.py', backend='cpu', size=(64, 36), native='available')
    try:
        yield s, s.native, recordings
    finally:
        s.close()


def wait_for(session, condition):
    deadline = time.monotonic() + 3
    while not condition():
        assert session.thread.is_alive(), 'Studio render thread stopped'
        assert time.monotonic() < deadline, 'Studio did not reach the expected state'
        time.sleep(.01)


@pytest.mark.parametrize('close_from', ['menu', 'window'])
def test_closing_output_while_recording_keeps_frames_and_restores_size_on_stop(recording_output, close_from):
    s, window, recordings = recording_output
    s.call('native_output', enabled=True, monitor='window')
    s.call('tracker_transport', enabled=True, playing=True, beat=0)
    s.call('record')
    recording = recordings[-1]
    wait_for(s, lambda: len(recording.frames) >= 2)
    if close_from == 'menu':
        assert not s.call('native_output', enabled=False)['active']
    else:
        window.closed.set()  # the desktop's close event, handled outside the command exception guard
        wait_for(s, lambda: not window.active)
    count, clock = len(recording.frames), s.clock_time
    wait_for(s, lambda: len(recording.frames) >= count + 2)
    assert s.thread.is_alive() and not recording.closed and s.recorder is recording
    assert s.clock_time > clock and s.tracker.enabled
    assert s.size == s.renderer.size == (96, 54) and s.native_restore_size == (64, 36)
    result = s.call('record')
    assert result['file'].endswith('.mp4') and recording.closed
    assert s.size == s.renderer.size == (64, 36) and s.native_restore_size is None
    assert {frame.shape for frame in recording.frames} == {(54, 96, 3)}


def test_the_output_keeps_drawing_while_a_stopped_recording_finishes_encoding(recording_output):
    s, window, recordings = recording_output
    s.call('native_output', enabled=True, monitor='window')
    s.call('tracker_transport', enabled=True, playing=True, beat=0)
    s.call('record')
    recording = recordings[-1]
    wait_for(s, lambda: len(recording.frames) >= 2)
    recording.finished = threading.Event()
    stopped = []
    caller = threading.Thread(target=lambda: stopped.append(s.call('record')))
    caller.start()
    assert recording.finishing.wait(3)
    shown = window.presented
    wait_for(s, lambda: window.presented >= shown + 3)   # still flipping while FFmpeg drains
    assert s.call('tracker_transport', playing=True)['playing'] and s.recorder is None and not stopped
    recording.finished.set()
    caller.join(3)
    assert stopped[0]['file'].endswith('.mp4') and recording.closed and s.notice == 'Video recording saved.'


def test_reopening_output_preserves_the_size_pending_from_before_recording(recording_output):
    s, window, recordings = recording_output
    s.call('native_output', enabled=True, monitor='window')
    s.call('record')
    s.call('native_output', enabled=False)
    s.call('native_output', enabled=True, monitor='window')
    s.call('record')  # stopping while the output shows must keep the output's size
    assert window.active and s.size == (96, 54) and s.native_restore_size == (64, 36)
    s.call('native_output', enabled=True, monitor='other')
    assert s.size == (128, 72) and s.native_restore_size == (64, 36)
    s.call('native_output', enabled=False)
    assert s.size == (64, 36) and s.native_restore_size is None


@pytest.mark.parametrize('failure', ['write', 'close'])
def test_failed_recording_also_restores_size_after_output_closes(recording_output, failure):
    s, window, recordings = recording_output
    s.call('native_output', enabled=True, monitor='window')
    s.call('record')
    recording = recordings[-1]
    s.call('native_output', enabled=False)
    if failure == 'write':
        recording.fail.set()
        wait_for(s, lambda: s.size == (64, 36) and s.native_restore_size is None)
        s.call('source', path='p.py')  # finish the render loop's automatic recording cleanup
    else:
        recording.close_error = True
        with pytest.raises(ValueError, match='test encoder close failed'):
            s.call('record')
    assert s.thread.is_alive() and s.recorder is None and recording.closed
    assert s.size == s.renderer.size == (64, 36) and s.native_restore_size is None
    assert 'test encoder' in s.error


def test_rejected_resize_during_recording_does_not_schedule_a_restore(recording_output):
    s, window, recordings = recording_output
    s.call('record')
    with pytest.raises(ValueError, match='Stop recording'):
        s.call('native_output', enabled=True, monitor='window')
    assert not window.active and s.size == (64, 36) and s.native_restore_size is None
    renderer = s.renderer
    s.call('record')
    assert s.renderer is renderer  # stopping this recording must not reload the patch


def test_failed_restore_keeps_rendering_and_retains_the_original_size(recording_output):
    s, window, recordings = recording_output
    s.call('native_output', enabled=True, monitor='window')
    s.call('record')
    s.call('native_output', enabled=False)
    (s.root / 'p.py').write_text('syntax !')
    s.call('record')
    assert s.thread.is_alive() and s.size == s.renderer.size == (96, 54)
    assert s.native_restore_size == (64, 36)
    assert 'size was not restored' in s.notice


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


def test_recording_keeps_one_frame_per_tick_when_frames_arrive_near_slot_edges(monkeypatch):
    """Frames at exactly the recording rate (the native output's flips), with a millisecond of jitter,
    each fill one slot. Counted from when Record was pressed, frames that happened to sit near slot
    edges dropped one frame and repeated the next (a simulation of that case did so for 128 of 600)."""
    import random
    import hydrapy.export
    from hydrapy.studio import Recorder
    written = []

    def save_mp4(frames, path, fps, encoder):
        written.extend(frames)
    monkeypatch.setattr(hydrapy.export, 'save_mp4', save_mp4)
    recorder = Recorder(None, 60.)
    random.seed(4)
    first = recorder.start + .98 / 60  # the first frame lands just before a slot edge
    for k in range(170):  # fewer than the encoder queue holds: this loop never yields to its thread
        recorder.push(k, first + k / 60 + random.uniform(-.001, .001))
    recorder.close()
    assert written == list(range(170))


def test_editor_preview_stays_small_while_recording_with_the_output_showing(recording_output):
    s, window, recordings = recording_output
    sizes, read = [], s._read_frame

    def spy(size=None):
        sizes.append(size)
        return read(size)
    s._read_frame = spy
    s.call('native_output', enabled=True, monitor='window')
    s.call('tracker_transport', enabled=True, playing=True, beat=0)
    s.call('record')
    wait_for(s, lambda: len(recordings[-1].frames) >= 2)
    sizes.clear()
    frame = s.preview.frame(s.preview.output, -1, timeout=3, thumbnails=False)
    s.preview.frame(s.preview.output, frame['sequence'] if frame else -1, timeout=3, thumbnails=False)
    assert s._confidence_size() in sizes  # a scaled read for the editor, not the recording's full frame
    assert None in sizes                  # the recording still reads full frames
    s.call('record')


@pytest.mark.parametrize('backend', ['cpu', 'gl'])
def test_feedback_carries_on_when_the_output_changes_the_render_size(tmp_path, backend):
    """Opening the fullscreen output reloads the patch at the monitor's size; its feedback history
    is scaled across instead of starting again from black."""
    import numpy as np
    from hydrapy.studio import Session
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsrc(o0).blend(solid(1, .5, 0), .2).out()\n')
    try:
        s = Session(tmp_path, 'p.py', backend=backend, size=(64, 36))
    except Exception as exc:
        pytest.skip(f'no {backend} renderer here: {exc}')
    try:
        if backend == 'gl' and not hasattr(s.renderer, 'ctx'):
            pytest.skip('no GL context here')
        s.call('tracker_transport', enabled=True, playing=True, beat=0)
        time.sleep(.5)                                  # the feedback converges towards the solid colour
        s.call('pause', paused=True)
        handle = s._handle

        def resize(action, data):
            if action == 'resize_for_test':
                s._resize((96, 54))
                return {}
            return handle(action, data)
        s._handle = resize
        s.call('resize_for_test')
        frame = s.call('snapshot') and s.last_frame     # the first frame at the new size, still paused
        assert frame.shape == (54, 96, 3)
        assert frame[..., 0].mean() > 100               # not black: the history came across, scaled
        if backend == 'cpu':
            assert {b.shape[:2] for b in s.renderer.env.buffers.values()} == {(54, 96)}
    finally:
        s.close()
