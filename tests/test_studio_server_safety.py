"""Device-free regressions for Studio's filesystem and HTTP boundaries."""
import http.client
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy import Ctx
from hydrapy.live import CvOut, ResourceStage
from hydrapy.studio import Session, atomic_text, make_server


def test_atomic_text_does_not_follow_temporary_symlink(tmp_path):
    victim = tmp_path / 'victim'
    victim.write_text('keep')
    target = tmp_path / 'patch.py'
    target.with_name('patch.py.tmp').symlink_to(victim)
    atomic_text(target, 'new source')
    assert victim.read_text() == 'keep'
    assert target.read_text() == 'new source'


def test_atomic_text_failure_removes_temporary(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr('hydrapy.studio.os.replace', fail)
    with pytest.raises(OSError):
        atomic_text(tmp_path / 'patch.py', 'new source')
    assert list(tmp_path.iterdir()) == []


def test_generated_export_refuses_symlink_outside_workspace(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / '.hydrapy').symlink_to(outside, target_is_directory=True)
    session = Session.__new__(Session)
    session.root, session.path = root, root / 'patch.py'
    with pytest.raises(ValueError, match='workspace root'):
        session._new_output('png')
    assert list(outside.iterdir()) == []


def test_non_ascii_token_is_rejected_with_http_reply():
    server, _ = make_server(SimpleNamespace(), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        connection.request('GET', '/api/state', headers={'X-Hydrapy-Token': '\xe9'})
        assert connection.getresponse().status == 403
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_cv_finite_extreme_cannot_poison_callback():
    with ResourceStage():
        output = CvOut(channels=[0]).send(0, 1e300)
        with np.errstate(all='ignore'):
            output.tick(Ctx())
            data = np.empty((8, 1), np.float32)
            output._callback(data, 8, None, None)
        assert np.isfinite(data).all()
        assert (np.abs(data) <= 1).all()


@pytest.mark.parametrize('index', [-1, 1, True, .5])
def test_cv_route_requires_valid_logical_channel(index):
    with ResourceStage():
        output = CvOut(channels=[0])
        with pytest.raises(ValueError):
            output.send(index, 1.)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_cv_requires_positive_finite_voltage_scale(scale):
    with ResourceStage():
        with pytest.raises(ValueError):
            CvOut(channels=[0], volts_per_fs=scale)


@pytest.mark.parametrize('failure', [False, True])
def test_recording_completion_publishes_only_on_render_thread(tmp_path, monkeypatch, failure):
    mutations = []
    class CheckedSession(Session):
        def __setattr__(self, name, value):
            if name in ('error', 'notice') and getattr(self, 'check_mutations', False):
                mutations.append(threading.get_ident())
            super().__setattr__(name, value)
    class Recorder:
        def __init__(self, path, fps):
            self.path = path
        def push(self, *args):
            pass
        def close(self):
            if failure:
                raise RuntimeError('encoder failed')
    monkeypatch.setattr('hydrapy.studio.Recorder', Recorder)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(0).out()\n')
    session = CheckedSession(tmp_path, 'p.py', backend='cpu', size=(32, 18))
    try:
        session.call('record')
        session.check_mutations = True
        if failure:
            with pytest.raises(ValueError, match='encoder failed'):
                session.call('record')
            assert 'encoder failed' in session.error
        else:
            session.call('record')
            assert session.notice == 'Video recording saved.'
        assert mutations
        assert set(mutations) == {session.thread.ident}
    finally:
        session.check_mutations = False
        session.close()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_es40_direct_positive_saturation_preserves_polarity(dtype):
    from hydrapy.es40 import ES40CV
    cv = ES40CV([dict(channels=[2, 3], header=1, map={'1': 'pitch1'})],
                direct='0=pitch1', open_stream=False)
    volts = np.full((2, len(cv.mapping)), 100., dtype=dtype)
    output = np.zeros((2, 4), np.int32)
    with np.errstate(all='ignore'):
        cv._write_volts(output, volts)
    assert np.all(output[:, 0] == np.iinfo(np.int32).max)


def test_saved_tracker_endpoint_refuses_external_symlink(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'tracker.json').write_text('{"private": true}')
    (root / '.hydrapy').symlink_to(outside, target_is_directory=True)
    server, token = make_server(SimpleNamespace(root=root), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        connection.request('GET', '/api/tracker_saved', headers={'X-Hydrapy-Token': token})
        response = connection.getresponse()
        assert response.status == 400
        assert b'private' not in response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('status', [None, 'not json'])
def test_dead_export_with_unreadable_status_is_not_running(tmp_path, monkeypatch, status):
    import json
    import time
    from pathlib import Path
    def launch(args, **kwargs):
        job = Path(args[-1])
        assert json.loads(job.read_text())['format'] == 'png'
        state = job.with_name('status.json')
        if status is None:
            state.unlink()
        else:
            state.write_text(status)
        return SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr('hydrapy.studio.subprocess.Popen', launch)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(0).out()\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18))
    try:
        session.call('export', format='png', backend='cpu', size=[32, 18])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with session.lock:
                jobs = session.state.get('jobs', [])
            if jobs and jobs[0]['state'] == 'error':
                break
            time.sleep(.01)
        assert jobs and jobs[0]['state'] == 'error'
        assert 'exited (1)' in jobs[0]['error']
    finally:
        session.close()


@pytest.mark.parametrize('channels', [[], [-1], [256], [True], [0, 0], [.5]])
def test_cv_invalid_physical_channels_are_refused_before_activation(channels):
    with ResourceStage():
        with pytest.raises(ValueError):
            CvOut(channels=channels)
