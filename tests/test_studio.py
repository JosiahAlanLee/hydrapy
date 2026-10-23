"""Workspace behavior: controls, rollback, exports, and the loopback boundary."""
import json
import sys
import threading
import time
import types
import urllib.request
import urllib.error

import numpy as np
import pytest

from hydrapy import Param, Patch, osc, reset_patch, solid, Ctx, Fn
from hydrapy.controls import apply_controls
from hydrapy.inspect import capabilities, catalog
from hydrapy.live import ResourceStage, close_resources
from hydrapy.studio import Session, make_server, project_path, digest


@pytest.fixture(autouse=True)
def clean():
    close_resources()
    reset_patch()
    yield
    close_resources()
    reset_patch()


def test_controls_are_signals_with_soft_ranges_and_atomic_presets():
    p = Patch()
    x = Param('frequency', 3, min=1, max=5, patch=p)
    y = Param('colour', .2, patch=p)
    x.set(12)
    assert x(Ctx()) == 12
    x.set_modulation('lfo', depth=2, rate=1)
    assert x(Ctx(time=.25)) == pytest.approx(14)
    with pytest.raises(ValueError):
        apply_controls(p, {'frequency': 8, 'colour': float('nan')})
    assert x.base == 12 and y.base == .2
    apply_controls(p, {'frequency': {'value': 8, 'modulation': 'none'}})
    assert x(Ctx(time=.25)) == 8
    with pytest.raises(ValueError):
        Param('frequency', patch=p)


def test_export_check_finds_nested_live_signals_and_baked_functions():
    from hydrapy.live import LiveValue
    from hydrapy import FrameSource, src
    p = Patch()
    osc(LiveValue('cv', .5) * 20 + Fn(lambda c: c.time)).out(p.o0, patch=p)
    src(FrameSource(lambda t: np.zeros((2, 2, 3), np.uint8))).out(p.o1, patch=p)
    result = capabilities(p, 'html')
    statuses = {x['status'] for x in result['findings']}
    assert not result['ready']
    assert {'fixed value', 'baked loop', 'requires baking'} <= statuses
    assert capabilities(p, 'gl')['ready']


def test_catalog_does_not_execute_and_paths_cannot_escape(tmp_path):
    (tmp_path / 'example.py').write_text('"""Example texture."""\nraise RuntimeError("must not execute")\n')
    assert catalog(tmp_path)[0]['description'] == 'Example texture.'
    with pytest.raises(ValueError):
        project_path(tmp_path, '../outside.py')
    (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        project_path(tmp_path, 'escape/outside.py')


def fake_sounddevice(monkeypatch):
    active = {}
    class Stream:
        def __init__(self, device=None, **kw):
            if device == 'bad':
                raise RuntimeError('device refused')
            if device in active:
                raise RuntimeError('device busy')
            self.device, self.callback = device, kw['callback']
        def start(self):
            active[self.device] = self
        def stop(self):
            pass
        def close(self):
            active.pop(self.device, None)
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(
        OutputStream=Stream, InputStream=Stream, query_devices=lambda *a: {'default_samplerate':48000}))
    return active


def test_staged_cv_keeps_old_stream_until_commit_and_restores_after_failure(monkeypatch):
    from hydrapy.live import CvOut
    active = fake_sounddevice(monkeypatch)
    old = CvOut(device='good').send(0, 3.)
    stream = old.stream
    with pytest.raises(SyntaxError):
        with ResourceStage():
            CvOut(device='good').send(0, 5.)
            assert active['good'] is stream
            raise SyntaxError('edited file')
    assert active['good'] is stream and old.routes[0](Ctx()) == 3.
    with pytest.raises(RuntimeError, match='device refused'):
        with ResourceStage() as stage:
            CvOut(device='bad')
            stage.commit()
    assert active['good'] is old.stream
    assert old.routes[0](Ctx()) == 3.
    with ResourceStage() as stage:
        new = CvOut(device='good').send(0, 7.)
        assert active['good'] is old.stream
        stage.commit()
    assert active['good'] is new.stream
    assert new.routes[0](Ctx()) == 7.


def test_checking_hardware_patch_does_not_import_or_open_devices(monkeypatch):
    from hydrapy.live import CvIn, CvOut, Midi, Link, OscIn, OscOut
    for name in ('sounddevice', 'mido', 'aalink', 'pythonosc'):
        monkeypatch.setitem(sys.modules, name, None)
    with ResourceStage():
        cv = CvIn()
        CvOut().send(0, cv.cv[0])
        midi = Midi()
        midi.cc(74)
        assert Link().bpm == 120
        OscIn().value('/x')
        OscOut().send('/x', 1)


def test_load_patch_rejection_does_not_clear_running_outputs(tmp_path):
    from hydrapy.cli import load_patch
    good = tmp_path / 'good.py'
    good.write_text('from hydrapy import *\nsolid(.6,0,0).out(o0)\n')
    with ResourceStage() as stage:
        p = load_patch(str(good))
        stage.commit(False)
    chain = p.o0.chain
    bad = tmp_path / 'bad.py'
    bad.write_text('from hydrapy import *\nsolid(0,1,0).out(o0)\nraise ValueError("bad edit")\n')
    with pytest.raises(ValueError, match='bad edit'):
        with ResourceStage():
            load_patch(str(bad))
    assert p.o0.chain is chain
    assert p.o0.chain.render(size=(8,8))[...,0].mean() == pytest.approx(.6)


def test_explicit_reset_in_a_patch_is_isolated_during_reload(tmp_path):
    from hydrapy.cli import load_patch
    path = tmp_path / 'reset.py'
    source = 'from hydrapy import *\nreset_patch(bpm=90)\nsolid(.6,0,0).out(o0)\n'
    path.write_text(source)
    with ResourceStage() as stage:
        old = load_patch(str(path))
        stage.commit(False)
    chain = old.o0.chain
    assert old.bpm == 90
    path.write_text(source.replace('.6,0,0', '0,1,0') + 'raise ValueError("bad edit")\n')
    with pytest.raises(ValueError, match='bad edit'):
        with ResourceStage():
            load_patch(str(path))
    assert old.o0.chain is chain


def test_staging_file_outputs_does_not_overwrite_a_recording(tmp_path):
    from hydrapy.live import CvOut, run_hooks
    path = tmp_path / 'take.wav'
    path.write_bytes(b'previous take')
    with ResourceStage():
        out = CvOut(file=str(path)).send(0, 5)
        run_hooks(Ctx())
        assert out._frames == []
    assert path.read_bytes() == b'previous take'


def test_bad_catalog_metadata_does_not_hide_other_patches(tmp_path):
    (tmp_path / 'bad.py').write_text('STUDIO = []\n')
    (tmp_path / 'good.py').write_text('"""Working patch."""\n')
    found = {row['path']: row for row in catalog(tmp_path)}
    assert found['bad.py']['tags'] == ['Needs attention']
    assert found['good.py']['description'] == 'Working patch.'


def sample_patch(tmp_path):
    p = tmp_path / 'patch.py'
    source = "from hydrapy import *\nlevel=Param('Level', .2)\nsolid(level,0,0).out(o0)\nsolid(0,.5,0).out(o1)\n"
    p.write_text(source)
    return p, source


def test_workspace_controls_presets_failed_save_and_real_export(tmp_path):
    path, source = sample_patch(tmp_path)
    session = Session(tmp_path, path.name, backend='cpu', size=(32,18), fps=12)
    try:
        session.call('controls', values={'Level':.7})
        session.call('preset_save', name='Red')
        session.call('controls', values={'Level':.1})
        session.call('preset_load', name='Red')
        assert session.renderer.patch.controls['Level'].base == .7
        old = session.renderer
        with pytest.raises(ValueError, match='SyntaxError'):
            session.call('save', path='patch.py', source=source+'syntax !', version=digest(source))
        assert session.renderer is old
        assert session.renderer.patch.controls['Level'].base == .7
        with pytest.raises(ValueError, match='changed outside'):
            session.call('save', path='patch.py', source=source, version='old')
        # Fix the saved file while retaining the last good renderer's values.
        session.call('save', path='patch.py', source=source, version=digest(source+'syntax !'))
        assert session.renderer.patch.controls['Level'].base == .7
        session.call('output', index=1)
        assert session.last_frame[0,0,1] == 128
        job = session.call('export', format='png', size=[32,18], fps=12, seconds=1, backend='cpu')
        proc = session.job_processes[job['id']]
        assert proc.wait(timeout=30) == 0
        status = json.loads((tmp_path/'.hydrapy/jobs'/job['id']/'status.json').read_text())
        assert status['state'] == 'done'
        from PIL import Image
        im = np.asarray(Image.open(tmp_path/job['file']))
        assert im[0,0,1] == 128
    finally:
        session.close()


def test_workspace_http_requires_session_token_and_rejects_cross_origin(tmp_path):
    path, source = sample_patch(tmp_path)
    session = Session(tmp_path, path.name, backend='cpu', size=(32,18), fps=12)
    server, token = make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base+'/api/state')
        assert error.value.code == 403
        good = urllib.request.Request(base+'/api/state', headers={'X-Hydrapy-Token':token})
        assert json.load(urllib.request.urlopen(good))['path'] == 'patch.py'
        bad = urllib.request.Request(base+'/api/pause', b'{}', headers={
            'X-Hydrapy-Token':token, 'Content-Type':'application/json', 'Origin':'https://elsewhere.invalid'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        session.close()


def test_reload_preserves_all_feedback_history_and_exposes_error(tmp_path):
    from hydrapy.cli import load_patch
    from hydrapy.backend_gl import GLRenderer, _reload, _create_context
    try:
        ctx = _create_context()
    except Exception:
        pytest.skip('GL unavailable')
    path = tmp_path/'patch.py'
    source = 'from hydrapy import *\nsrc(o0).add(solid(.1,0,0)).out(o0)\ndelay(o0,3).out(o1)\n'
    path.write_text(source)
    p = load_patch(str(path))
    r = GLRenderer(p,size=(16,8),ctx=ctx)
    for i in range(5):
        r.step()
    r.patch.render(r.patch.o1)
    before = [fb['fbo'].read(components=4,dtype='f4') for fb in r.outs['o0']['ring']]
    new = _reload(r,lambda:load_patch(str(path)),ctx,60)
    assert new is not r
    assert new.patch.display.index == 1
    assert before == [fb['fbo'].read(components=4,dtype='f4') for fb in new.outs['o0']['ring']]
    path.write_text(source+'raise ValueError("bad")\n')
    same = _reload(new,lambda:load_patch(str(path)),ctx,60)
    assert same is new and 'bad' in same.reload_error
    same.release()
    ctx.release()


def test_deferred_audio_routes_use_the_actual_device_rate(monkeypatch):
    from hydrapy.live import CvOut, Soundtrack
    fake_sounddevice(monkeypatch)
    sys.modules['sounddevice'].query_devices = lambda *a: {'default_samplerate': 44100}
    p = Patch()
    solid(.5).out(p.o0, patch=p)
    with ResourceStage() as stage:
        out = CvOut(device='good')
        track = Soundtrack(p.o0, patch=p)
        out.send(0, track)
        assert out.audio_routes[0].samplerate == 48000
        stage.commit()
    assert out.samplerate == out.audio_routes[0].samplerate == 44100


def test_recording_repeats_held_frames_at_wall_clock_rate(tmp_path, monkeypatch):
    from hydrapy.studio import Recorder
    clock = [100.]
    monkeypatch.setattr('hydrapy.studio.time.monotonic', lambda: clock[0])
    received = []
    def save(frames, path, **kw):
        received.extend(frames)
    monkeypatch.setattr('hydrapy.export.save_mp4', save)
    recorder = Recorder(tmp_path/'take.mp4', 10)
    red = np.full((2,2,3), 10, np.uint8)
    recorder.push(red)
    clock[0] += .3
    recorder.push(red)
    recorder.close()
    assert len(received) in (3,4)
    assert all(np.array_equal(f,red) for f in received)
