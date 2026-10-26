"""Preview loss/backpressure must not change synthesis or durable output."""
import io
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import pytest

from hydrapy import Patch, Soundtrack, delay, noise, osc, solid, src, voronoi
from hydrapy.studio import Session, make_server
from hydrapy.studio_preview import Preview


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('condition timed out')


def test_slow_encoder_has_one_latest_slot_and_invalidates_old_patch(monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    calls = []
    def encode(frame, size=None):
        calls.append(int(frame[0, 0, 0]))
        entered.set()
        assert resume.wait(5)
        return bytes([calls[-1]])
    monkeypatch.setattr('hydrapy.studio_preview.jpeg', encode)
    p = Preview()
    def submit(n):
        p.submit('o0', np.full((4, 4, 3), n, np.uint8), {'captured': time.monotonic()})
    try:
        submit(1)
        assert entered.wait(2)
        for n in range(2, 100):
            submit(n)
        assert p.snapshot()['pending'] == 1 and p.snapshot()['replaced'] == 97
        resume.set()
        until(lambda: p.snapshot()['encoded'] == 2)
        assert calls == [1, 99]
        assert p.frame('o0', timeout=0)['data'] == bytes([99])
        resume.clear(); entered.clear()
        submit(100)
        assert entered.wait(2)
        p.reset('o1')
        resume.set()
        until(lambda: p.pending is None)
        assert p.frame('o0', timeout=0) is None
        assert p.frame('o1', timeout=.05) is None  # old in-flight encode cannot publish
    finally:
        resume.set(); p.close()
    assert not p.thread.is_alive()


def test_slow_consumer_gets_latest_and_shutdown_wakes_waiters():
    p = Preview()
    try:
        for n in range(1, 6):
            p.submit('o0', np.full((2, 2, 3), n * 30, np.uint8), {'captured': time.monotonic()})
            until(lambda: p.snapshot()['encoded'] == n)
        frame = p.frame('o0', after=1, timeout=0)
        assert frame['sequence'] == 5
        assert np.asarray(Image.open(io.BytesIO(frame['data']))).mean() == pytest.approx(150, abs=1)
        got = []
        waiter = threading.Thread(target=lambda: got.append(p.frame('o0', after=5, timeout=10)))
        waiter.start()
        p.close(); waiter.join(timeout=1)
        assert got == [None] and not waiter.is_alive()
    finally:
        p.close()


def test_thumbnail_reads_are_staggered_and_stop_without_viewers():
    p = Preview()
    try:
        now = time.monotonic()
        p.frame('o0', timeout=0)
        main, thumbnail = p.plan(now)
        assert main and thumbnail == 'o0'
        assert p.plan(now + .01) == (False, None)
        thumbs = [p.plan(now + x)[1] for x in (.13, .26, .39, .52)]
        assert thumbs == ['o1', 'o2', 'o3', 'o0']
        assert p.plan(now + 3) == (False, None)
    finally:
        p.close()


def test_frame_only_viewer_does_not_request_thumbnails_or_cancel_studio_viewer():
    p = Preview()
    try:
        p.frame('o0', timeout=0, thumbnails=False)
        now = time.monotonic()
        assert p.plan(now) == (True, None)
        # A simultaneous Studio window still gets its output picker thumbnails.
        p.frame('o0', timeout=0)
        p.frame('o0', timeout=0, thumbnails=False)
        assert p.plan(now + .01) == (True, 'o0')
        # Gorgon keeps requesting selected frames after Studio goes away.
        p.frame('o0', timeout=0, thumbnails=False)
        assert p.plan(now + 3) == (True, None)
    finally:
        p.close()


def renderer(backend, patch, size=(32, 18)):
    if backend == 'cpu':
        from hydrapy.backend_np import Renderer
        return Renderer(patch, size=size, fps=60)
    from hydrapy.backend_gl import GLRenderer, _create_context
    try:
        ctx = _create_context()
    except Exception:
        pytest.skip('GL unavailable')
    return GLRenderer(patch, size=size, fps=60, ctx=ctx)


def release(r):
    if hasattr(r, 'release'):
        r.release(); r.ctx.release()


@pytest.mark.parametrize('backend', ['cpu', 'gl'])
@pytest.mark.parametrize('multi', [False, True])
def test_preview_drops_preserve_feedback_timing_and_audio(backend, multi):
    from hydrapy.live import register_hook, clear_hooks
    def run(preview):
        p = Patch(size=(32, 18)); p.frame_rate = 12
        src(p.o0).scale(1.01).blend(osc(10, .2), .1).out(p.o0, patch=p)
        delay(p.o0, 3).out(p.o1, patch=p, scale=.5)
        if multi:
            noise(4, .2).modulate(noise(7), .2).blur(3, 3).out(p.o2, patch=p)
            src(p.o2).blend(src(p.o1), .4).add(voronoi(7), .2).blur(2, 3).out(p.o3, patch=p)
        sound = Soundtrack(p.o0, patch=p, fps=24)
        r = renderer(backend, p)
        session = Session.__new__(Session)
        session.renderer, session.preview = r, Preview()
        session.aux_preview = Preview()
        session.aux_preview.reset('o1')
        session.path, session.paused, session.preview_dirty = None, False, True
        session.clock_time, session.render_metrics = 0., {}
        hooks = []
        register_hook(lambda ctx, renderer: hooks.append(ctx.time))
        try:
            for i in range(60):
                r.step(i / 60)
                if preview:
                    # A slow consumer: ask for only every seventh frame.
                    if i % 7 == 0:
                        session.preview.frame('o0', timeout=0)
                    if i % 5 == 0:
                        session.aux_preview.frame('o1', timeout=0)
                    session._pictures()
            buffers = []
            for o in p.active_outputs():
                p.display = o
                buffers.append(r.read_rgba() if hasattr(r, 'read_rgba') else r.displayed().copy())
            return buffers, r.frame, r.clock.updates, hooks, sound.samples.copy()
        finally:
            clear_hooks(); session.preview.close(); session.aux_preview.close(); release(r)
    baseline, preview = run(False), run(True)
    assert baseline[1:4] == preview[1:4]
    assert baseline[1:3] == (60, 12)
    assert all(np.array_equal(a, b) for a, b in zip(baseline[0], preview[0]))
    assert np.array_equal(baseline[4], preview[4])


def test_thumbnail_failure_restores_output():
    p = Patch(); solid(.5).out(p.o0, patch=p)
    r = renderer('cpu', p)
    s = Session.__new__(Session)
    s.renderer, s.preview = r, Preview()
    s.path, s.paused, s.preview_dirty = None, False, True
    s.clock_time, s.render_metrics = 0., {}
    try:
        s.preview.frame('o0', timeout=0)
        s.preview.thumbnail_index = 1
        def read():
            if p.display.index == 1:
                raise RuntimeError('readback failed')
            return np.zeros((18, 32, 3), np.uint8)
        s._read_frame = read
        with pytest.raises(RuntimeError, match='readback failed'):
            s._pictures()
        assert p.display is p.o0
    finally:
        s.preview.close()


def test_blocked_jpeg_does_not_block_render_record_snapshot_or_pause(tmp_path, monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    from hydrapy.studio_preview import jpeg
    def blocked(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return jpeg(*args, **kwargs)
    monkeypatch.setattr('hydrapy.studio_preview.jpeg', blocked)
    class Recorder:
        def __init__(self, path, fps):
            self.path, self.frames = path, []
        def push(self, frame, at=None):
            self.frames.append(frame.copy())
        def close(self):
            pass
    monkeypatch.setattr('hydrapy.studio.Recorder', Recorder)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(.2,0,0).out(o0)\nsolid(0,.5,0).out(o1)\n')
    s = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        assert entered.wait(2)
        initial = s.renderer.frame
        s.call('record'); recorder = s.recorder
        until(lambda: len(recorder.frames) >= 5)
        assert s.renderer.frame >= initial + 5 and not s.paused
        s.call('output', index=1)
        s.call('pause', paused=True)
        snapshot = s.call('snapshot')
        image = np.asarray(Image.open(tmp_path / snapshot['file']))
        assert np.all(image[..., 1] == 128) and np.all(image[..., 0] == 0)
        assert s.renderer.patch.display.index == 1
        s.call('record')
        assert all(frame.shape == (18, 32, 3) for frame in recorder.frames)
        assert any(np.all(frame[..., 1] == 128) for frame in recorder.frames)
    finally:
        resume.set(); s.close()


@pytest.mark.parametrize('backend', ['cpu', 'gl'])
def test_next_frame_http_auth_freshness_metrics_and_pause(tmp_path, backend):
    if backend == 'gl':
        p = Patch(); solid(.5).out(p.o0, patch=p)
        r = renderer('gl', p); release(r)
    (tmp_path / 'p.py').write_text('from hydrapy import *\nosc(12).out(o0)\n')
    s = Session(tmp_path, 'p.py', backend=backend, size=(64, 36), fps=30)
    server, token = make_server(s, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}/api/frame'
    def fetch(after=-1, **headers):
        return urllib.request.urlopen(urllib.request.Request(base + f'?after={after}',
            headers={'X-Hydrapy-Token': token, **headers}), timeout=3)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            fetch(Origin='https://untrusted.invalid')
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base)
        s.call('preview_profile', enabled=True)
        seq = -1
        for _ in range(5):
            with fetch(seq) as response:
                assert response.status == 200
                new = int(response.headers['X-Preview-Sequence'])
                assert new > seq
                seq = new
                assert Image.open(io.BytesIO(response.read())).size == (64, 36)
                timing = response.headers['Server-Timing']
                assert 'readback_ms' in timing and 'jpeg_ms' in timing and 'render_cpu_ms' in timing
        if backend == 'gl':
            assert 'gpu_ms' in timing
        s.call('pause', paused=True)
        with fetch(seq) as response:
            if response.status == 200:  # final frame already in flight at pause
                seq = int(response.headers['X-Preview-Sequence'])
        with fetch(seq) as response:
            assert response.status == 204
        assert not s.preview_error and not s.error
    finally:
        server.shutdown(); server.server_close(); s.close()
