"""The bridge for local clients such as the Gorgon VCV module."""
import json
import os
import stat
import threading
import time
import types
import urllib.error
import urllib.request

import pytest

from hydrapy.studio import Session, advertise, bridge_file, make_server, withdraw

PATCH = ("from hydrapy import *\n"
         "z = Param('Dolly Z', .35)\nroll = Param('Roll', 0, min=-45, max=45)\nwarp = Param('Warp', .5)\n"
         "solid(z, 0, 0).out(o0)\n")


@pytest.fixture
def studio(tmp_path):
    (tmp_path / 'p.py').write_text(PATCH)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    server, token = make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'

    def request(path, body=None, auth=True):
        headers = {'X-Hydrapy-Token': token} if auth else {}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        data = None if body is None else json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=5) as r:
            return json.load(r)

    def bridge(**query):
        deadline = time.monotonic() + 5
        while True:                                # the first state lands after one render iteration
            state = request('/api/bridge' + ('?' + '&'.join(f'{k}={v}' for k, v in query.items()) if query else ''))
            if 'revision' in state or time.monotonic() > deadline:
                return state
            time.sleep(.01)

    try:
        yield types.SimpleNamespace(session=session, request=request, bridge=bridge, token=token, port=server.server_port)
    finally:
        server.shutdown(); server.server_close(); session.close()


def test_bridge_state_needs_the_token_and_resends_the_score_only_on_change(studio):
    with pytest.raises(urllib.error.HTTPError) as e:
        studio.request('/api/bridge', auth=False)
    assert e.value.code == 403
    state = studio.bridge()
    for key in ('revision', 'enabled', 'playing', 'transport_revision', 'anchor_beat', 'anchor_time', 'clock_time', 'wall', 'server_now'):
        assert key in state
    assert state['document']['format'] == 'hydrapy-tracker-1'
    assert [c['name'] for c in state['controls']] == ['Dolly Z', 'Roll', 'Warp']
    assert state['server_now'] >= state['wall']
    assert 'document' not in studio.bridge(revision=state['revision'])
    assert 'document' in studio.bridge(revision=state['revision'] - 1)


def test_bridge_playhead_extrapolates_to_the_tracker_beat(studio):
    studio.request('/api/tracker_transport', dict(enabled=True, playing=True, beat=4))
    time.sleep(.2)
    s = studio.bridge()
    assert s['enabled'] and s['playing']
    clock = s['clock_time'] + (s['server_now'] - s['wall'])
    beat = s['anchor_beat'] + (clock - s['anchor_time']) * s['document']['bpm'] / 60
    assert beat == pytest.approx(studio.session.tracker.beat(studio.session.clock_time), abs=.2)


def test_bridge_params_set_controls_but_not_ones_the_tracker_drives(studio):
    result = studio.request('/api/bridge_params', dict(values={'Warp': .8, 'Roll': 400, 'Nope': 1, 'Dolly Z': True}))
    assert sorted(result['applied']) == ['Roll', 'Warp']
    assert result['ignored'] == {'Nope': 'unknown control', 'Dolly Z': 'not a finite number'}
    controls = studio.session.renderer.patch.controls
    assert controls['Warp'].base == .8 and controls['Roll'].base == 45      # clamped to the slider range
    studio.request('/api/tracker_transport', dict(enabled=True))
    result = studio.request('/api/bridge_params', dict(values={'Roll': 10, 'Warp': .1}))
    assert result['applied'] == ['Warp'] and result['ignored'] == {'Roll': 'driven by tracker automation'}
    with pytest.raises(urllib.error.HTTPError):
        studio.request('/api/bridge_params', dict(values=[1, 2]))


def test_bridge_file_is_private_and_only_withdrawn_by_its_owner(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDRAPY_STUDIO_FILE', str(tmp_path / 'home' / 'studio.json'))
    path = advertise(8765, 'secret', tmp_path)
    assert path == bridge_file()
    data = json.loads(path.read_text())
    assert data['url'] == 'http://127.0.0.1:8765' and data['token'] == 'secret' and data['pid'] == os.getpid()
    if os.name == 'posix':
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    withdraw(path, 'someone else')
    assert path.exists()
    withdraw(path, 'secret')
    assert not path.exists()
    withdraw(path, 'secret')                        # already gone: no error


def test_all_followers_share_the_pre_render_anchor(tmp_path):
    """An expensive frame must neither delay publication nor retimestamp its beat."""
    import numpy as np
    from hydrapy.tracker_cv import TrackerCV
    (tmp_path/'p.py').write_text(PATCH)
    cv = TrackerCV({0: 'clock'}, open_stream=False)
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30, cv=cv)
    entered, release = threading.Event(), threading.Event()
    try:
        session.call('tracker_transport', enabled=True, playing=False, beat=4)
        step = session.renderer.step
        def slow_step(t):
            entered.set()
            assert release.wait(5)
            return step(t)
        session.renderer.step = slow_step
        session.call('tracker_transport', playing=True)
        assert entered.wait(5)
        with session.lock:
            browser, bridge = session.tracker_state.copy(), session.bridge_state.copy()
        direct = cv.snapshot
        assert browser['playing'] and bridge['playing'] and direct['playing']
        assert browser['sampled'] == bridge['wall'] == direct['wall']
        assert browser['transport_revision'] == bridge['transport_revision'] == direct['transport_revision']
        # Compare the browser, bridge and direct CV at a later timestamp.
        at = browser['sampled'] + .080
        browser_beat = browser['beat'] + .080 * browser['document']['bpm'] / 60
        clock = bridge['clock_time'] + at - bridge['wall']
        bridge_beat = bridge['anchor_beat'] + (clock - bridge['anchor_time']) * bridge['document']['bpm'] / 60
        cv_beat = cv.beats(direct, np.array([at]))[0][0]
        assert browser_beat == pytest.approx(bridge_beat)
        assert browser_beat == pytest.approx(cv_beat)
    finally:
        release.set()
        session.close()
    assert not session.bridge_state['playing'] and not session.tracker_state['playing']


def test_transport_revision_changes_on_seek_but_not_score_edit(studio):
    studio.request('/api/tracker_transport', dict(enabled=True, playing=False, beat=4))
    # Transport replies are synchronous and carry the same revision as snapshots.
    first = studio.request('/api/tracker_transport', dict(beat=4.25))
    second = studio.request('/api/tracker_transport', dict(beat=4.25))
    assert second['transport_revision'] == first['transport_revision'] + 1
    document = second['document']
    document['bpm'] = 140
    edited = studio.request('/api/tracker_score', dict(document=document, revision=second['revision']))
    assert edited['transport_revision'] == second['transport_revision']
