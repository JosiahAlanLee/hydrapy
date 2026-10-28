"""Tracker ownership, deterministic automation, exports and output isolation."""
import copy
import json
import threading
import time
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

from hydrapy import Ctx, Param, Patch, solid, src
from hydrapy.backend_np import Renderer, iter_frames
from hydrapy.studio import Session, make_server
from hydrapy.studio_tracker import Tracker, validate, value_at


def score():
    return copy.deepcopy(Tracker().document)


def test_interpolation_including_exact_hold_boundaries():
    p = score()['patterns'][0]
    p['keys']['z'] = [{'r': 0, 'v': 0.}, {'r': 16, 'v': 1.}]
    for mode, expected in [('hold', 0.), ('linear', .25), ('smooth', .15625)]:
        p['easing']['z'] = mode
        assert value_at(p, 'z', 4) == expected
        assert value_at(p, 'z', 16) == 1.
        assert value_at(p, 'z', 63) == 1.


@pytest.mark.parametrize('change', [
    lambda d: d.update(bpm=float('nan')),
    lambda d: d.update(key=False),
    lambda d: d.update(key=True),
    lambda d: d.update(key='0'),
    lambda d: d.update(key=None),
    lambda d: d.update(pattern=4),
    lambda d: d['bindings'].update(z='Roll'),
    lambda d: d['modes'].update(aux='o7'),
    lambda d: d['patterns'][0]['keys']['z'].append({'r': 0, 'v': .5}),
    lambda d: d['patterns'][0]['audio'][0].__setitem__(0, {'note': '<script>', 'vel': 90}),
])
def test_rejected_score_does_not_change_the_running_document(change):
    tracker = Tracker()
    old = tracker.document
    d = score(); change(d)
    with pytest.raises(ValueError):
        tracker.replace(d, 1., 0)
    assert tracker.document is old
    assert tracker.revision == 0


@pytest.mark.parametrize('fmt', ['hydrapy-tracker-1', 'hydrapy-tracker-2'])
@pytest.mark.parametrize('where', [
    lambda d, x: d.update(x_note=x),                                          # an unknown key at the top
    lambda d, x: d.update(x_editor={'zoom': [1, x]}),                         # nested in one
    lambda d, x: d['patterns'][1].update(x_pattern=x),                        # in a pattern
    lambda d, x: d['patterns'][1]['audio'][2].__setitem__(0, {'note': 'C-4', 'vel': 90, 'x_hit': x}),   # in a hit
    lambda d, x: d['patterns'][0]['keys']['z'].insert(1, {'r': 3, 'v': .5, 'x_key': [x]}),
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_versions_1_and_2_reject_non_finite_numbers_in_unknown_keys(fmt, where, value):
    """As version 3 does (score-v3.md): NaN and Infinity aren't JSON for the browser or Rack."""
    d = score()
    d['format'] = fmt
    if fmt == 'hydrapy-tracker-2':
        d['sounds'] = [{'machine': None, 'params': {}} for _ in range(4)]
    validate(d)
    where(d, value)
    with pytest.raises(ValueError, match=r'A score holds only finite numbers \(no NaN or Infinity\)'):
        validate(d)
    with pytest.raises(ValueError, match='finite'):                           # what Studio parses from a body
        validate(json.loads(json.dumps(d)))


@pytest.mark.parametrize('key', [0, 2, 5, 7, 0., 2., 5., 7.])
def test_numeric_keys_keep_the_existing_transposition(key):
    from hydrapy.tracker_cv import TrackerCV
    document = score()
    document['key'] = key
    document['patterns'][0]['audio'][0][0] = {'note': 'C-4', 'vel': 100}
    tracker = Tracker(document)
    tracker.enabled = True
    cv = TrackerCV({0: 'pitch1'}, open_stream=False)
    cv.publish(tracker, 0., 100., True)
    assert cv.render(1, 100.)[0, 0] == pytest.approx((key - 2) / 12)


def test_tempo_loop_seek_revision_and_copy_ownership():
    tracker = Tracker(); tracker.enabled = True
    tracker.seek(3, 10)
    beat = tracker.beat(11)
    d = score(); d['bpm'] = 90
    tracker.replace(d, 11, 0)
    assert tracker.beat(11) == beat
    assert tracker.beat(12) == pytest.approx(beat + 1.5)
    d['bpm'] = 200
    assert tracker.document['bpm'] == 90
    with pytest.raises(ValueError, match='another window'):
        tracker.replace(d, 12, 0)
    d['loop'] = True; d['pattern'] = 2
    tracker.replace(d, 12, 1)
    tracker.seek(47, 12)
    assert tracker.beat(12.6) == pytest.approx(33)


@pytest.mark.parametrize('backend', ['cpu', 'gl'])
def test_offline_automation_matches_live_steps_with_feedback_and_frame_rate(backend):
    def patch():
        p = Patch()
        z = Param('Dolly Z', .35, patch=p)
        solid(z, 0, 0).out(p.o0, patch=p)
        src(p.o1).blend(src(p.o0), .3).out(p.o1, patch=p)
        p.render(p.o1); p.frame_rate = 12
        return p
    factory, render = Renderer, iter_frames
    if backend == 'gl':
        pytest.importorskip('moderngl')
        from hydrapy.backend_gl import GLRenderer, iter_frames_gl
        factory, render = GLRenderer, iter_frames_gl
    p = patch()
    try:
        r = factory(p, size=(32, 18), fps=30)
    except Exception as error:
        if backend == 'gl': pytest.skip(str(error))
        raise
    try:
        tracker = Tracker(); tracker.enabled = True
        expected = []
        for i in range(30):
            tracker.apply(p, i / 30)
            r.step(i / 30)
            if backend == 'gl': expected.append(r.read())
            else:
                from hydrapy.backend_np import to_uint8
                expected.append(to_uint8(r.displayed(), p.composite_alpha))
        q = patch(); tracker = Tracker(); tracker.enabled = True
        actual = list(tracker.frames(q, seconds=1, fps=30, size=(32,18), backend=backend))
        assert np.array_equal(actual, expected)
    finally:
        if hasattr(r, 'release'): r.release()


def test_session_seek_preserves_renderer_and_feedback_and_exports_score(tmp_path):
    path = tmp_path/'patch.py'
    path.write_text("from hydrapy import *\nz=Param('Dolly Z', .35)\nsolid(z,0,0).out(o0)\nsrc(o1).blend(src(o0),.2).out(o1)\n")
    session = Session(tmp_path, path.name, backend='cpu', size=(32,18), fps=30)
    try:
        session.call('tracker_transport', enabled=True, playing=True)
        time.sleep(.1)
        session.call('pause', paused=True)
        renderer, clock, frame = session.renderer, session.clock_time, session.renderer.frame
        history = renderer.env.buffers[1].copy()
        session.call('tracker_transport', beat=40, playing=False)
        assert session.renderer is renderer and session.clock_time == clock and renderer.frame == frame
        assert np.array_equal(history, renderer.env.buffers[1])
        assert session.tracker.beat(clock) == 40
        session.call('tracker_save', revision=session.tracker.revision)
        assert validate(json.loads((tmp_path/'.hydrapy/tracker.json').read_text())) == session.tracker.document
        job = session.call('export', format='png', size=[32,18], backend='cpu', seconds=1, fps=30)
        assert session.job_processes[job['id']].wait(30) == 0
        spec = json.loads((tmp_path/'.hydrapy/jobs'/job['id']/'job.json').read_text())
        assert spec['tracker']['beat'] == 40
        from PIL import Image
        pixels = np.asarray(Image.open(tmp_path/job['file']))
        assert pixels[0,0,0] == pytest.approx(value_at(session.tracker.document['patterns'][2], 'z', 32)*255, abs=1)
        with pytest.raises(ValueError, match='static HTML'):
            session.call('export', format='html')
    finally:
        session.close()


def test_http_assets_auth_independent_aux_and_output_restore(tmp_path):
    (tmp_path/'p.py').write_text('from hydrapy import *\nsolid(.8,0,0).out(o0)\nsolid(0,.8,0).out(o1)\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32,18), fps=30)
    server, token = make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    def get(path, auth=True):
        return urllib.request.urlopen(urllib.request.Request(base+path, headers={'X-Hydrapy-Token':token} if auth else {}), timeout=5)
    try:
        for path in ('/tracker/', '/tracker/app.js', '/tracker/backend.js', '/tracker/style.css'):
            with get(path, False) as response: assert response.status == 200
        with pytest.raises(urllib.error.HTTPError) as e: get('/api/tracker', False)
        assert e.value.code == 403
        with get('/api/tracker') as response: assert json.load(response)['path'] == 'p.py'
        from PIL import Image
        import io
        for bus, output, channel in [('program', 'o0', 0), ('aux', 'o1', 1)]:
            with get(f'/api/frame?bus={bus}&output={output}') as response:
                assert response.status == 200
                data = np.asarray(Image.open(io.BytesIO(response.read())))
                assert data[0,0,channel] > 195
        assert session.renderer.patch.display.name == 'o0'
        session.call('output', index=1)
        assert session.tracker.document['modes']['program'] == 'o1'
        session.call('pause', paused=True)
        d=score();d['modes']['aux']='o0'
        session.call('tracker_score', document=d, revision=session.tracker.revision)
        with get('/api/frame?bus=aux&output=o0') as response: assert response.status == 200
        assert session.renderer.patch.display.name == 'o0'
    finally:
        server.shutdown(); server.server_close(); session.close()


def test_seek_and_blackout_do_not_discard_elapsed_simulation_time():
    session = Session.__new__(Session)
    session.renderer = types.SimpleNamespace(patch=Patch())
    session.tracker, session.tracker_blackout = Tracker(), False
    session.clock_time, session.last_wall = 5., 10.
    session.paused, session.armed = False, False
    session._handle('tracker_transport', {'enabled': True, 'beat': 16})
    session._handle('tracker_transport', {'blackout': True})
    assert session.last_wall == 10. and session.clock_time == 5.
    assert session.tracker.beat(5.) == 16


def test_external_clock_exclusion_reload_and_rejected_edits(tmp_path):
    source = "from hydrapy import *\nz=Param('Dolly Z', .35)\nsolid(z,0,0).out(o0)\n"
    (tmp_path/'p.py').write_text(source)
    (tmp_path/'clock.py').write_text(source + 'patch().transport=Transport(120)\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32,18), fps=30)
    try:
        session.call('tracker_transport', enabled=True, playing=False, beat=20)
        session.call('load', path='p.py')
        assert session.tracker.enabled and session.tracker.beat(session.clock_time) == 20
        old = session.renderer
        with pytest.raises(ValueError):
            session.call('save', path='p.py', source=source+'broken !', version=session.state['version'])
        assert session.renderer is old and session.tracker.enabled
        session.call('load', path='clock.py')
        assert not session.tracker.enabled
        session.call('tracker_transport', enabled=True)
        with pytest.raises(ValueError, match='external clock'):
            session.call('connections', armed=True)
        assert not session.armed
        session.call('tracker_transport', enabled=False)
        session.call('connections', armed=True)
        with pytest.raises(ValueError, match='external clock'):
            session.call('tracker_transport', enabled=True)
        assert not session.tracker.enabled
    finally:
        session.close()


def test_studio_output_sync_is_not_an_edit_conflict():
    tracker = Tracker()
    base = tracker.revision
    tracker.sync_program('o2', 1.)
    assert tracker.document['modes']['program'] == 'o2'
    assert tracker.revision == base + 1
    # An editor still holding the pre-sync revision is accepted, and the
    # server's program output survives its stale copy of the routing.
    d = score(); d['bpm'] = 120
    tracker.replace(d, 1., base)
    assert tracker.document['bpm'] == 120
    assert tracker.document['modes']['program'] == 'o2'
    # Repeating a sync to the same output changes nothing.
    revision = tracker.revision
    tracker.sync_program('o2', 1.)
    assert tracker.revision == revision


def test_a_real_concurrent_edit_is_still_rejected_after_a_sync():
    tracker = Tracker()
    tracker.replace(score(), 1., 0)        # window A edits
    tracker.sync_program('o3', 1.)         # Studio switches output
    with pytest.raises(ValueError, match='another window'):
        tracker.replace(score(), 1., 0)    # window B never saw A's edit
    with pytest.raises(ValueError, match='another window'):
        tracker.replace(score(), 1., tracker.revision + 1)
    with pytest.raises(ValueError, match='another window'):
        tracker.replace(score(), 1., True)


def test_switching_studio_output_does_not_reject_the_open_tracker(tmp_path):
    (tmp_path/'p.py').write_text("from hydrapy import *\nsolid(1,0,0).out(o0)\nsolid(0,1,0).out(o1)\n")
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32,18), fps=30)
    try:
        base = session.tracker.revision
        session.call('output', index=1)
        assert session.tracker.document['modes']['program'] == 'o1'
        d = copy.deepcopy(session.tracker.document); d['modes']['program'] = 'o0'; d['bpm'] = 100
        result = session.call('tracker_score', document=d, revision=base)
        assert result['document']['bpm'] == 100
        assert result['document']['modes']['program'] == 'o1'
        assert session.renderer.patch.display.name == 'o1'
        session.call('output', index=0)    # another sync after the edit
        session.call('tracker_save', revision=result['revision'])
        assert json.loads((tmp_path/'.hydrapy'/'tracker.json').read_text())['bpm'] == 100
        with pytest.raises(ValueError, match='another window'):
            session.call('tracker_save', revision=base)   # predates the edit
    finally:
        session.close()
