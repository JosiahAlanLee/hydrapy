"""Bounded stress diagnostics: overload, failed storage, crashes and real Studio lifecycle."""
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hydrapy import studio_diagnostics as diagnostics
from hydrapy.studio import Session


def wait(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), 'condition timed out'


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audio(**changes):
    values = dict(xruns=0, underruns=0, late_packets=0, suspensions=0, jack_frame_at=lambda t: 123)
    values.update(changes)
    return SimpleNamespace(**values)


def frame(recorder, at=None, **changes):
    values = dict(now=time.monotonic() if at is None else at, simulation=1, beat=2,
                  render_ms=1, readback_ms=2, work_ms=3, fps=30, depth=0, repeats=0, audio=audio())
    values.update(changes)
    recorder.frame(**values)


@pytest.fixture(autouse=True)
def cheap_environment(monkeypatch):
    monkeypatch.setattr(diagnostics, 'environment', lambda: {'test': True})


def test_checkpoint_incident_history_and_clean_close(tmp_path):
    path = tmp_path / 'take.jsonl'
    recorder = diagnostics.Diagnostics(path, interval=.05)
    now = time.monotonic()
    for i in range(10):
        frame(recorder, now + i / 30)
    frame(recorder, now + 1, work_ms=180, depth=150, audio=audio(underruns=3))
    recorder.event('scene', active=['fracture'])
    wait(lambda: any(r['kind'] == 'summary' for r in records(path)))
    assert recorder.close()
    rows = records(path)
    assert rows[0]['format'] == diagnostics.FORMAT
    incident = next(r for r in rows if r['kind'] == 'incident')
    assert {'underruns', 'frame_stall', 'encoder_backlog'} <= set(incident['reasons'])
    assert len(incident['frames']) == 11
    assert incident['frames'][-1][incident['columns'].index('jack_frame_estimate')] == 123
    summary = next(r for r in rows if r['kind'] == 'summary' and r['frames'])
    assert summary['timing_ms']['work_ms']['maximum'] == 180
    assert summary['counters']['underruns'] == 3
    assert rows[-1]['kind'] == 'closed' and rows[-1]['complete']
    assert path.stat().st_mode & 0o777 == 0o600


def test_stalled_writer_drops_records_and_shutdown_is_bounded(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow_environment():
        entered.set()
        release.wait(5)
        return {}
    monkeypatch.setattr(diagnostics, 'environment', slow_environment)
    recorder = diagnostics.Diagnostics(tmp_path / 'slow.jsonl', capacity=2, event_capacity=2)
    try:
        assert entered.wait(2)
        for _ in range(100):
            frame(recorder)
            recorder.event('scene', active=['a'])
        assert recorder.write_index == len(recorder.rows) == 2
        assert len(recorder.events) == 2
        assert recorder.dropped_frames == recorder.dropped_events == 98
        assert not recorder.close(timeout=.01)
        assert recorder.thread.is_alive()
    finally:
        release.set()
        recorder.thread.join(5)
    assert not recorder.thread.is_alive()
    assert records(recorder.path)[-1]['dropped_frames'] == 98


def test_write_failure_disables_diagnostics_without_raising_in_producer(tmp_path, monkeypatch):
    def fail(self, record):
        raise OSError('disk full')
    monkeypatch.setattr(diagnostics.Diagnostics, '_emit', fail)
    recorder = diagnostics.Diagnostics(tmp_path / 'failed.jsonl')
    wait(lambda: recorder.finished)
    frame(recorder)
    recorder.event('shutdown')
    assert recorder.disabled and 'disk full' in recorder.error
    assert not recorder.close()


def test_file_limit_is_explicit_and_bounded(tmp_path):
    recorder = diagnostics.Diagnostics(tmp_path / 'limit.jsonl', max_bytes=4096)
    recorder.event('error', detail='x' * 5000)
    wait(lambda: recorder.finished)
    assert recorder.disabled and 'size limit' in recorder.error
    rows = records(recorder.path)
    assert rows[-1]['kind'] == 'limit' and not rows[-1]['complete']
    assert recorder.path.stat().st_size <= 4096
    assert not any(r['kind'] == 'closed' for r in rows)


def test_existing_file_and_symlink_are_not_overwritten(tmp_path):
    target = tmp_path / 'take.jsonl'
    target.write_text('keep')
    link = tmp_path / 'link.jsonl'
    link.symlink_to(target)
    for path in (target, link):
        with pytest.raises(FileExistsError):
            diagnostics.Diagnostics(path)
    assert target.read_text() == 'keep'
    missing = tmp_path / 'missing.jsonl'
    dangling = tmp_path / 'dangling.jsonl'
    dangling.symlink_to(missing)
    with pytest.raises(FileExistsError):
        diagnostics.Diagnostics(dangling)
    assert not missing.exists()


def test_single_producer_ring_never_reuses_an_unread_row(tmp_path, monkeypatch):
    observed = []
    # Delay the consumer sometimes so both the full and the concurrent-drain cases occur.
    def observe(self, row):
        observed.append(row)
        if len(observed) % 8 == 0:
            time.sleep(.001)
    monkeypatch.setattr(diagnostics.Diagnostics, '_observe', observe)
    recorder = diagnostics.Diagnostics(tmp_path / 'ring.jsonl', capacity=16)
    for i in range(1, 1001):
        frame(recorder, at=float(i), simulation=float(i))
        if i % 32 == 0:
            time.sleep(.002)
    assert recorder.close()
    assert len(observed) + recorder.dropped_frames == 1000
    assert all(row[0] == row[1] == row[2] for row in observed)
    assert all(a[0] < b[0] for a, b in zip(observed, observed[1:]))


def test_counter_reset_is_not_a_negative_delta(tmp_path):
    recorder = diagnostics.Diagnostics(tmp_path / 'reset.jsonl', interval=.05)
    frame(recorder, repeats=12)
    wait(lambda: any(r['kind'] == 'summary' and r['frames'] for r in records(recorder.path)))
    frame(recorder, repeats=1)
    assert recorder.close()
    summaries = [r for r in records(recorder.path) if r['kind'] == 'summary' and r['frames']]
    assert summaries[-1]['counter_deltas']['repeated_frames'] is None
    assert summaries[-1]['counter_deltas']['xruns'] == 0


def test_audio_counters_are_checkpointed_without_render_progress(tmp_path):
    source = audio(error='')
    recorder = diagnostics.Diagnostics(tmp_path / 'audio.jsonl', audio=source, interval=.05)
    wait(lambda: any(r['kind'] == 'summary' for r in records(recorder.path)))
    source.underruns = 19
    source.error = 'producer failed'
    wait(lambda: any(r.get('audio_error') == source.error for r in records(recorder.path)))
    assert recorder.close()
    summary = [r for r in records(recorder.path) if r['kind'] == 'summary'][-1]
    assert summary['counters']['underruns'] == 19 and summary['frames'] == 0


def test_studio_tracks_scene_load_record_and_upstream_errors(tmp_path, monkeypatch):
    # Use the actual render thread and recording queue; don't need a video codec or JACK.
    monkeypatch.setattr('hydrapy.export.save_mp4', lambda frames, path, **kw: list(frames))
    patch = tmp_path / 'scene.py'
    patch.write_text("from hydrapy import *\nscene('red')(lambda: solid(1,0,0))\n"
                     "scene('blue')(lambda: solid(0,0,1))\nrender(o0)\n")
    path = tmp_path / 'diagnostics.jsonl'
    session = Session(tmp_path, patch.name, backend='cpu', size=(32, 18), fps=60, diagnostics=path)
    try:
        import copy
        score = copy.deepcopy(session.tracker.document)
        score['patterns'][0]['scenes'] = [dict(r=0, name='red', fade=0), dict(r=1, name='blue', fade=1)]
        session.call('tracker_score', document=score, revision=session.tracker.revision)
        session.call('tracker_transport', enabled=True, playing=True, beat=0)
        session.call('record')
        wait(lambda: session.diagnostics.frame_count >= 4)
        session.audio.underruns = 7  # upstream loss without any recorder xrun
        session.call('tracker_transport', beat=.4)
        time.sleep(.2)
        session.call('record')
        assert not session.error
        session.call('load', path='scene.py')
    finally:
        session.close()
    rows = records(path)
    events = [r for r in rows if r['kind'] == 'event']
    assert any(r['event'] == 'patch' and r['backend'] == 'CPU' and len(r['sha256']) == 64 for r in events)
    assert any(r['event'] == 'score' and len(r['sha256']) == 64 for r in events)
    assert any(r['event'] == 'scene' and 'blue' in r['active'] for r in events)
    assert any(r['event'] == 'state' and r['recording'] for r in events)
    assert any(r['event'] == 'command_end' and r['action'] == '_recording_finished' for r in events)
    summary = [r for r in rows if r['kind'] == 'summary' and r['counters']][-1]
    assert summary['counters']['underruns'] == 7 and summary['counters']['xruns'] == 0
    assert rows[-1]['kind'] == 'closed'


def test_patch_without_scenes_and_failed_load_do_not_break_diagnostics(tmp_path):
    (tmp_path / 'simple.py').write_text('from hydrapy import *\nosc(3).out()\n')
    (tmp_path / 'bad.py').write_text('invalid ! python')
    session = Session(tmp_path, 'simple.py', backend='cpu', size=(32, 18), diagnostics='take.jsonl')
    try:
        wait(lambda: session.diagnostics.frame_count >= 3)
        assert not session.error and not session.paused
        with pytest.raises(ValueError, match='SyntaxError'):
            session.call('load', path='bad.py')
    finally:
        session.close()
    rows = records(tmp_path / 'take.jsonl')
    assert any(r.get('event') == 'load' and r.get('state') == 'error' for r in rows)


def test_actual_gl_renderer_is_identified(tmp_path):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(f'GL unavailable: {exc}')
    renderer = ctx.info['GL_RENDERER']
    ctx.release()
    (tmp_path / 'simple.py').write_text('from hydrapy import *\nosc(3).out()\n')
    session = Session(tmp_path, 'simple.py', backend='gl', size=(32, 18), diagnostics='gl.jsonl')
    try:
        wait(lambda: session.diagnostics.frame_count >= 2)
        assert not session.error
    finally:
        session.close()
    patch = next(r for r in records(tmp_path / 'gl.jsonl') if r.get('event') == 'patch')
    assert patch['backend'] == 'GL' and patch['graphics']['GL_RENDERER'] == renderer


def test_disabled_mode_does_not_create_a_diagnostic_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'Diagnostics', lambda *a, **kw: pytest.fail('unexpected recorder'))
    (tmp_path / 'simple.py').write_text('from hydrapy import *\nosc(3).out()\n')
    session = Session(tmp_path, 'simple.py', backend='cpu', size=(32, 18))
    try:
        assert session.diagnostics is None
    finally:
        session.close()
    assert not list(tmp_path.rglob('*.jsonl'))


def test_checkpoints_survive_process_kill(tmp_path):
    path = tmp_path / 'killed.jsonl'
    code = ('import time; from hydrapy.studio_diagnostics import Diagnostics; '
            f'd = Diagnostics({str(path)!r}, interval=.05); '
            'd.event("load", state="loading"); time.sleep(30)')
    child = subprocess.Popen([sys.executable, '-c', code])
    try:
        wait(lambda: path.exists() and '"kind":"summary"' in path.read_text(), timeout=10)
        child.kill()
        child.wait(timeout=5)
        rows = records(path)
        assert rows[0]['kind'] == 'run'
        assert any(r.get('event') == 'load' for r in rows)
        assert not any(r['kind'] == 'closed' for r in rows)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_cli_passes_diagnostics_to_studio(tmp_path, monkeypatch):
    from hydrapy import cli
    calls = []
    monkeypatch.setattr('hydrapy.studio.serve', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(sys, 'argv', ['hydrapy', 'studio', '--root', str(tmp_path),
                                    '--no-native-output', '--diagnostics', 'take.jsonl'])
    cli.main()
    assert calls[0]['diagnostics'] == 'take.jsonl'
