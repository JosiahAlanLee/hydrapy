"""Studio for a live set: recordings on the audio clock, JACK input for Audio, load status, the Program
route check, and JACK audio kept alive through a long patch load. No JACK server: a fake client."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy import Audio, Patch, reset_patch
from hydrapy.live import close_resources
from hydrapy.signals import LiveRing
from hydrapy.studio import Recorder, Session, make_server, timestamps_path
from hydrapy.studio_audio import StudioAudio, check_inputs
from hydrapy import recording
from hydrapy.studio_tracker import Tracker


@pytest.fixture(autouse=True)
def clean():
    close_resources()
    reset_patch()
    yield
    close_resources()
    reset_patch()


class FakeJack:
    """A JACK client: ports with buffers, activation and connections, all recorded in ``events``."""
    samplerate, blocksize, last_frame_time, frames_since_cycle_start = 48000, 128, 0, 0
    name = 'HydraPy'

    def __init__(self, name, **kw):
        self.events, self.active, self.closed = [], False, False
        self.connections = set()                         # (source, destination)
        self.outports = SimpleNamespace(register=lambda n: self._register(n))
        self.inports = SimpleNamespace(register=lambda n: self._register(n))
        self.registered = {}

    def _register(self, name):
        buffer = np.zeros(self.blocksize, np.float32)
        port = SimpleNamespace(name='HydraPy:' + name, buffer=buffer, get_array=lambda: buffer)
        def unregister():
            self.registered.pop(port.name)
            self.connections = {c for c in self.connections if port.name not in c}
        port.unregister = unregister
        self.registered[port.name] = port
        self.events.append(('register', name))
        return port

    def __getattr__(self, name):
        if name.startswith('set_') and name.endswith('_callback'):
            return lambda fn: setattr(self, 'on_' + name[4:-9], fn)
        raise AttributeError(name)

    def activate(self):
        self.active = True
        self.events.append('activate')

    def deactivate(self):
        # JACK drops a deactivated client's connections
        self.active = False
        self.connections = {c for c in self.connections if not any(p in self.registered for p in c)}
        self.events.append('deactivate')

    def get_all_connections(self, port):
        return [SimpleNamespace(name=b if a == port.name else a) for a, b in self.connections if port.name in (a, b)]

    def connect(self, source, destination):
        assert self.active, 'JACK connects only active clients'
        self.connections.add((source, destination))

    def close(self):
        self.closed = True


def jack_audio(monkeypatch=None):
    audio = StudioAudio(client_factory=FakeJack)
    audio._produce = lambda: None
    return audio


# --- 1. recordings stamped with the audio clock -----------------------------------------------------

def test_recorder_sidecar_stamps_each_frame_with_present_render_and_jack_time(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr('hydrapy.export.save_mp4', lambda frames, path, **kw: received.extend(frames))
    jack = SimpleNamespace(samplerate=48000, client=SimpleNamespace(name='HydraPy'),
                           jack_frame_at=lambda t: int(round(t * 48000)) % 2**32)
    recorder = Recorder(tmp_path / 'take.mp4', 10)
    recorder.jack = jack
    frame = np.zeros((2, 2, 3), np.uint8)
    recorder.push(frame, 100.00, 99.99)
    recorder.push(frame, 100.10, 100.09)
    recorder.push(frame, 100.30, 100.29)               # late: fills two slots with one picture
    recorder.close()
    stamps = json.loads(timestamps_path(tmp_path / 'take.mp4').read_text())
    assert timestamps_path(tmp_path / 'take.mp4').name == 'take.timestamps.json'
    assert stamps['format'] == 'hydrapy-recording-timestamps-1' and stamps['video'] == 'take.mp4'
    assert stamps['frames'] == len(received) == len(stamps['present']) == 4
    assert stamps['present'] == [100.0, 100.1, 100.3, 100.3] and stamps['origin'] == 100.0
    assert stamps['rendered'] == [99.99, 100.09, 100.29, 100.29]
    assert stamps['jack']['samplerate'] == 48000
    assert stamps['jack']['frame_time'] == [4800000, 4804800, 4814400, 4814400]


def test_recorder_without_jack_still_writes_monotonic_stamps(tmp_path, monkeypatch):
    monkeypatch.setattr('hydrapy.export.save_mp4', lambda frames, path, **kw: list(frames))
    recorder = Recorder(tmp_path / 'take.mp4', 30)
    recorder.jack = StudioAudio()                       # JACK audio off
    recorder.push(np.zeros((2, 2, 3), np.uint8), 5.)
    recorder.close()
    stamps = recording.load(tmp_path / 'take.mp4')
    assert stamps['jack'] is None and stamps['present'] == [5.]


def test_jack_frame_at_follows_the_last_process_cycle():
    audio = jack_audio()
    assert audio.jack_frame_at(1.) is None            # not running
    audio.configure(dict(enabled=True))
    try:
        audio.client.last_frame_time = 2**32 - 64
        audio._process(128)
        at, wall, _, _ = audio.anchor
        assert audio.jack_frame_at(wall) == 2**32 - 64
        assert audio.jack_frame_at(wall + .5) == (2**32 - 64 + 24000) % 2**32   # wraps like frame_time
    finally:
        audio.close()


def test_session_recording_returns_its_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr('hydrapy.export.save_mp4', lambda frames, path, **kw: list(frames))
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(.5).out()\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        session.call('record')
        time.sleep(.3)
        result = session.call('record')
        stamps = json.loads((tmp_path / result['timestamps']).read_text())
        assert result['timestamps'].endswith('.timestamps.json') and stamps['frames'] >= 3
        assert stamps['present'] == sorted(stamps['present'])
        assert all(r <= p for r, p in zip(stamps['rendered'], stamps['present']))
    finally:
        session.close()


def synthetic(fps=30., sr=48000, start=2**32 - 30000, frames=300, drift_ppm=0., repeat=(10, 11)):
    """Timestamps like Studio's: frame k presented at t0 + k / fps (with a little jitter), JACK running."""
    rng = np.random.default_rng(1)
    t0 = 1000.
    present, jack = [], []
    for k in range(frames):
        if k in repeat[1:]:
            present.append(present[-1]); jack.append(jack[-1]); continue
        t = t0 + k / fps + rng.uniform(-.0005, .0005)
        present.append(t)
        jack.append(int(round(start + (t - t0) * sr * (1 + drift_ppm * 1e-6))) % 2**32)
    return dict(format='hydrapy-recording-timestamps-1', video='take.mp4', fps=fps, frames=frames, origin=t0,
                present=present, rendered=present, jack=dict(samplerate=sr, frame_time=jack, client='HydraPy'))


def test_fit_and_offset_from_the_jack_clock_across_a_wrap():
    stamps = synthetic(drift_ppm=40)
    line = recording.fit(stamps)
    assert line['start'] % 2**32 == pytest.approx(2**32 - 30000, abs=20)
    assert line['drift_ppm'] == pytest.approx(40, abs=2)
    assert line['residual_ms'] < .01
    # the capture began 0.25 s (12000 frames) before video time 0, on the other side of the wrap
    assert recording.offset(stamps, audio_start_frame=2**32 - 42000) == pytest.approx(.25, abs=5e-4)
    # ... or 0.1 s after it (wrapped): the audio must be delayed
    assert recording.offset(stamps, audio_start_frame=(2**32 - 30000 + 4800) % 2**32) == pytest.approx(-.1, abs=5e-4)
    assert recording.offset(stamps, audio_start_monotonic=999.5) == pytest.approx(.5, abs=5e-4)
    with pytest.raises(ValueError):
        recording.offset(stamps)


def test_mux_places_the_audio_by_the_timestamps(tmp_path):
    import wave
    from hydrapy.export import save_mp4
    fps, sr, seconds = 25, 48000, 2.
    save_mp4((np.full((32, 32, 3), 40, np.uint8) for _ in range(int(fps * seconds))), str(tmp_path / 'take.mp4'), fps=fps)
    stamps = synthetic(fps=fps, sr=sr, start=1_000_000, frames=int(fps * seconds), repeat=())
    timestamps_path(tmp_path / 'take.mp4').write_text(json.dumps(stamps))
    # the capture starts 0.3 s before video time 0 and clicks at video time 1.0 s (capture time 1.3 s)
    audio = np.zeros(int(sr * 3), np.int16)
    audio[int(1.3 * sr):int(1.3 * sr) + 240] = 20000
    with wave.open(str(tmp_path / 'cap.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
    shift = recording.mux(tmp_path / 'take.mp4', tmp_path / 'cap.wav', tmp_path / 'out.mp4',
                          audio_start_frame=1_000_000 - int(.3 * sr))
    assert shift == pytest.approx(.3, abs=1e-3)
    from hydrapy.encoding import _ffmpeg_candidates
    pcm = subprocess.run([_ffmpeg_candidates()[0], '-v', 'error', '-i', str(tmp_path / 'out.mp4'), '-map', '0:a',
                          '-f', 's16le', '-ac', '1', '-ar', str(sr), '-'], capture_output=True, check=True).stdout
    decoded = np.abs(np.frombuffer(pcm, np.int16).astype(float))
    onset = np.flatnonzero(decoded > 5000)[0] / sr
    assert onset == pytest.approx(1.0, abs=.03)       # AAC priming aside, the click lands on video time 1.0


# --- 2. live JACK input for Audio ------------------------------------------------------------------

def test_live_ring_copies_and_returns_the_newest_samples_across_its_wrap():
    ring = LiveRing(8)
    assert not ring.latest(4).any()
    ring.write(np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(ring.latest(3), [3, 4, 5])
    np.testing.assert_array_equal(ring.latest(7), [0, 0, 1, 2, 3, 4, 5])
    ring.write(np.arange(6, 12, dtype=np.float32))     # wraps
    np.testing.assert_array_equal(ring.latest(8), np.arange(4, 12))
    ring.write(np.arange(100, 120, dtype=np.float32))   # a block larger than the ring
    np.testing.assert_array_equal(ring.latest(4), [116, 117, 118, 119])
    assert ring.written == 31


def test_audio_jack_declares_a_studio_input():
    p = Patch()
    a = Audio(jack='mix', patch=p)
    b = Audio.live('mix', patch=p)
    assert p.audio_inputs == [a, b] and b.jack == 'mix' and b.sr == 48000
    assert not any(f.value(SimpleNamespace(time=0.)) for f in a.fft)   # silent until fed
    with pytest.raises(ValueError, match='port'):
        Audio(jack='my mix', patch=p)
    with pytest.raises(ValueError, match='live input'):
        Audio(np.zeros(4), jack='mix', patch=p)
    clash = Patch()
    Audio(jack='optical_L', patch=clash)
    with pytest.raises(ValueError, match='clashes'):
        check_inputs(clash)


def test_input_port_feeds_the_analysers_from_a_copy_only_callback():
    p = Patch()
    bands, also = Audio(jack='mix', patch=p, sr=44100), Audio(jack='mix', patch=p)
    other = Audio(jack='kick', patch=p)
    audio = jack_audio()
    audio.attach(p)
    audio.configure(dict(enabled=True))
    try:
        client = audio.client
        assert ('register', 'mix') in client.events and ('register', 'kick') in client.events
        assert bands.sr == also.sr == 48000 and bands._ring is also._ring and other._ring is not bands._ring
        mix = client.registered['HydraPy:mix']
        tone = np.sin(2 * np.pi * 220 * np.arange(1024) / 48000).astype(np.float32)
        for start in range(0, 1024, 128):
            mix.buffer[:] = tone[start:start + 128]
            audio._process(128)
        np.testing.assert_array_equal(bands._ring.latest(1024), tone)
        assert bands.vol.value(SimpleNamespace(time=1.)) > 0
        assert not other._ring.latest(16).any()
        described = {i['name']: i for i in audio.describe()['inputs']}
        assert described['mix'] == dict(port='HydraPy:mix', name='mix', frames=1024, analysers=2)
        # a reload that keeps `mix` keeps its port (and connections); `kick` goes
        client.activate()
        client.connect('Rack:out1', 'HydraPy:mix')
        q = Patch()
        again = Audio(jack='mix', patch=q)
        audio.attach(q)
        assert client.registered.keys() >= {'HydraPy:mix'} and 'HydraPy:kick' not in client.registered
        assert ('Rack:out1', 'HydraPy:mix') in client.connections and again._ring is bands._ring
        assert [port.name for port, _ in audio.inputs] == ['HydraPy:mix']
    finally:
        audio.close()
    assert audio.inputs == () and not audio.inports
    assert not bands._ring.latest(1024).any()          # silent once JACK audio stops, not frozen


def test_session_patch_with_live_audio_is_fed_by_studio(tmp_path):
    (tmp_path / 'p.py').write_text('from hydrapy import *\nbands = Audio.live("mix")\n'
                                   'solid(bands.fft[0], 0, 0).out()\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        assert 'enable JACK audio' in session.notice
        session.audio.client_factory = FakeJack
        session.audio._produce = lambda: None
        session.call('pause', paused=True)
        session.audio.configure(dict(enabled=True))
        analyser = session.renderer.patch.audio_inputs[0]
        port = session.audio.inports['mix']
        assert analyser._ring is session.audio.rings['mix'] and port.name == 'HydraPy:mix'
        port.buffer[:] = 1.
        session.audio._process(128)
        assert analyser._ring.written == 128
    finally:
        session.close()


# --- 3. patch status in the API --------------------------------------------------------------------

SLOW = 'import time\nfrom hydrapy import *\ntime.sleep({delay})\nsolid(.5).out()\n'


def wait(condition, timeout=10.):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, 'condition timed out'
        time.sleep(.01)


def test_status_follows_a_load_through_to_ready_and_error(tmp_path):
    (tmp_path / 'slow.py').write_text(SLOW.format(delay=.6))
    (tmp_path / 'bad.py').write_text('from hydrapy import *\nraise RuntimeError("no such scene")\n')
    session = Session(tmp_path, backend='cpu', size=(32, 18), fps=30)
    server, token = make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def status():
        request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/status',
                                         headers={'X-Hydrapy-Token': token})
        return json.load(urllib.request.urlopen(request, timeout=5))
    try:
        assert status()['state'] == 'idle' and not status()['ready']
        loader = threading.Thread(target=session.call, args=('load',), kwargs=dict(path='slow.py'))
        loader.start()
        wait(lambda: status()['state'] == 'loading')
        seen = status()
        assert seen['path'] == 'slow.py' and not seen['ready'] and seen['seconds'] >= 0
        loader.join()
        done = status()
        assert done['state'] == 'ready' and done['ready'] and done['error'] == '' and done['patch'] == 'slow.py'
        assert done['seconds'] >= .5 and {'compiled_variants', 'banks', 'prewarmed_banks'} <= done.keys()
        with pytest.raises(ValueError, match='no such scene'):
            session.call('load', path='bad.py')
        failed = status()
        assert failed['state'] == 'error' and 'no such scene' in failed['error'] and not failed['ready']
    finally:
        server.shutdown(); server.server_close(); session.close()


def test_load_counts_compiled_variants_and_banks():
    session = Session.__new__(Session)
    bank = SimpleNamespace(name='bank1', count=3, array_at=lambda t: None)
    renderer = SimpleNamespace(_variants={1: 0, 2: 0}, media={'bank1': {'texs': {0: 0, 1: 0, 2: 0}}},
                               programs=[SimpleNamespace(sampler_uniforms=[SimpleNamespace(source=bank)] * 2)])
    assert session._load_counts(renderer) == dict(compiled_variants=2, banks=1, prewarmed_banks=1, bank_frames=3)


def test_served_status_is_up_before_the_startup_patch_finishes(tmp_path):
    (tmp_path / 'slow.py').write_text(SLOW.format(delay=2))
    bridge = tmp_path / 'studio.json'
    env = dict(os.environ, HYDRAPY_STUDIO_FILE=str(bridge), PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    code = ('from hydrapy.studio import serve; '
            f'serve({str(tmp_path)!r}, "slow.py", backend="cpu", size=(32, 18), port=0, open_browser=False)')
    proc = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        wait(lambda: bridge.exists() and bridge.stat().st_size > 0, 30)
        info = json.loads(bridge.read_text())

        def status():
            request = urllib.request.Request(info['url'] + '/api/status', headers={'X-Hydrapy-Token': info['token']})
            return json.load(urllib.request.urlopen(request, timeout=10))
        states = [status()['state']]
        wait(lambda: states.append(status()['state']) or states[-1] == 'ready', 30)
        assert states[0] in ('idle', 'loading') and 'loading' in states
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    assert not bridge.exists()


# --- 4. the score's Program route against the patch ------------------------------------------------

def score(program):
    d = Tracker().document
    d = json.loads(json.dumps(d))
    d['modes']['program'] = program
    return d


def test_score_routing_program_away_from_the_patch_display_warns(tmp_path):
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(1,0,0).out(o0)\nsrc(o0).invert().out(o1)\nrender(o1)\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        assert session.patch_display == 'o1' and session.route_notice == ''
        result = session.call('tracker_score', document=score('o0'), revision=session.tracker.revision)
        assert 'routes Program to o0' in result['notice'] and 'render(o1)' in result['notice']
        assert session.notice == session.route_notice == result['notice']
        # a later edit that keeps the route doesn't warn again, but the notice stays up
        again = session.call('tracker_score', document=score('o0'), revision=session.tracker.revision)
        assert 'notice' not in again and session.route_notice
        wait(lambda: session.tracker_state.get('route_notice') == session.route_notice)
        assert session.status()['route_notice'] == session.route_notice
        fixed = session.call('tracker_score', document=score('o1'), revision=session.tracker.revision)
        assert 'notice' not in fixed and session.route_notice == ''
        # loading a patch over a score routed elsewhere: the patch wins, and says so
        session.call('tracker_score', document=score('o2'), revision=session.tracker.revision)
        warning = session.route_notice
        assert 'notice' not in session.call('load', path='p.py')   # a reload keeps the route shown
        assert session.route_notice == warning and session.tracker.document['modes']['program'] == 'o2'
        (tmp_path / 'q.py').write_text((tmp_path / 'p.py').read_text())
        loaded = session.call('load', path='q.py')
        assert 'routed Program to o2' in loaded['notice'] and session.tracker.document['modes']['program'] == 'o1'
        session.call('output', index=0)                # chosen in Studio: deliberate
        assert session.route_notice == ''
    finally:
        session.close()


def test_agreeing_routes_never_warn(tmp_path):
    (tmp_path / 'p.py').write_text('from hydrapy import *\nsolid(1,0,0).out(o0)\n')
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        result = session.call('tracker_score', document=score('o0'), revision=session.tracker.revision)
        assert 'notice' not in result and session.route_notice == ''
        assert 'notice' not in session.call('load', path='p.py')
    finally:
        session.close()


# --- 5. JACK audio kept alive through a long load --------------------------------------------------

def test_slow_load_runs_with_the_client_deactivated_and_reconnects_it(tmp_path):
    (tmp_path / 'slow.py').write_text(SLOW.format(delay=.4))
    session = Session(tmp_path, backend='cpu', size=(32, 18), fps=30)
    audio = session.audio
    audio.client_factory, audio._produce = FakeJack, (lambda: None)
    try:
        audio.configure(dict(enabled=True))
        client = audio.client
        client.connect('HydraPy:optical_L', 'Rack:in1')
        client.connect('HydraPy:breaks_R', 'Rack:in4')
        during = []
        loader = threading.Thread(target=session.call, args=('load',), kwargs=dict(path='slow.py'))
        loader.start()
        wait(lambda: session.status()['state'] == 'loading')
        during.append((client.active, audio.suspended, audio.describe()['active']))
        loader.join()
        assert during == [(False, True, True)]         # deactivated (still enabled) while the patch ran
        assert client.events[-2:] == ['deactivate', 'activate'] and client.active and not audio.suspended
        assert client.connections == {('HydraPy:optical_L', 'Rack:in1'), ('HydraPy:breaks_R', 'Rack:in4')}
        assert audio.suspensions == 1 and audio.client is client and audio.error == ''
        # a score edit with no new crossfade to compile leaves the client alone
        session.call('tracker_score', document=score('o0'), revision=session.tracker.revision)
        assert client.events[-1] == 'activate' and audio.suspensions == 1
    finally:
        session.close()


def test_held_is_a_no_op_without_a_running_client_and_reports_a_failed_resume():
    audio = jack_audio()
    with audio.held() as suspended:
        assert not suspended
    audio.configure(dict(enabled=True))
    try:
        epoch = audio.epoch
        audio.client.activate = lambda: (_ for _ in ()).throw(RuntimeError('server gone'))
        with audio.held('a test') as suspended:
            assert suspended and audio.epoch == epoch + 1 and audio.anchor is None
        assert 'did not resume after a test: server gone' in audio.error
    finally:
        audio.close()


def test_new_crossfades_compile_with_the_client_deactivated_and_known_ones_do_not(tmp_path):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import _create_context
    try:
        _create_context().release()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')
    (tmp_path / 'p.py').write_text("from hydrapy import *\nscene('drift')(lambda: osc(8, .1, 1.2))\n"
                                   "scene('third')(lambda: noise(3, .2))\nscene('fracture')(lambda: voronoi(5, .3, .3))\n")
    session = Session(tmp_path, 'p.py', backend='gl', size=(32, 18), fps=30)
    audio = session.audio
    audio.client_factory, audio._produce = FakeJack, (lambda: None)
    try:
        audio.configure(dict(enabled=True))
        client = audio.client
        d = score('o0')
        d['patterns'][0]['scenes'] = [{'r': 0, 'name': 'drift', 'fade': 0}, {'r': 16, 'name': 'third', 'fade': 8}]
        session.call('tracker_score', document=d, revision=session.tracker.revision)
        assert client.events[-2:] == ['deactivate', 'activate'] and audio.suspensions == 1
        assert session.renderer.warmed_fades == {('drift', 'third')}
        d['patterns'][0]['scenes'][1]['fade'] = 4                  # the same crossfade, shorter: nothing new
        session.call('tracker_score', document=d, revision=session.tracker.revision)
        assert audio.suspensions == 1
        assert session.status()['compiled_variants'] == len(session.renderer._variants) > 0
    finally:
        session.close()
