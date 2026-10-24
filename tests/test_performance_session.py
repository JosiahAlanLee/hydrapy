"""Control-layer integration; no GL context, audio device, model, or GPU."""
from fractions import Fraction
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hydrapy.performance import Binding, Event, Message, Process, Route, Session, capture_vortex
from hydrapy.session import Children, OscSender, main, preview_html, run_session


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class Sender:
    def __init__(self, clock, cost=0):
        self.clock, self.cost, self.messages = clock, cost, []

    def send(self, route, message):
        self.messages.append((self.clock(), route, message.address, message.args))
        self.clock.time += self.cost


def score(events=(), **kwargs):
    values = dict(bpm=120, beats=2, routes={"voice": Route(57130, reset=(Message("/gate", (0,)),))},
                  events=events)
    values.update(kwargs)
    return Session(**values)


def test_roundtrip_and_stable_order_with_multiple_routes():
    original = Session.from_json((ROOT / "examples/performance/session.json").read_text())
    decoded = Session.from_json(original.to_json())
    assert decoded.to_dict() == original.to_dict()
    assert [e.address for e in decoded.events if e.beat == 0] == [
        "/voice/level", "/voice/pitch", "/voice/grit", "/voice/gate"]
    assert decoded.routes["picture"].reset == (Message("/voice/level", (0,)),)


@pytest.mark.parametrize("update", [
    {"bpm": 0}, {"bpm": True}, {"bpm": float("nan")}, {"bpm": 1e-320},
    {"beats": float("inf")}, {"beats": -2}, {"routez": {}},
    {"events": [{"beat": 2, "route": "voice", "address": "/freq"}]},
    {"events": [{"beat": 1, "duration": 2, "route": "voice", "address": "/freq"}]},
    {"events": [{"beat": 0, "route": "typo", "address": "/freq"}]},
    {"events": [{"beat": 0, "route": "voice", "address": "freq"}]},
    {"events": [{"beat": 0, "route": "voice", "address": "/freq", "args": [float("nan")]}]},
    {"events": [{"beat": 0, "route": "voice", "address": "/freq", "args": [2 ** 40]}]},
    {"events": [{"beat": 0, "route": "voice", "address": "/freq", "args": [[1, 2]]}]},
    {"processes": [{"name": "bad", "argv": "pd -nogui"}]},
    {"routes": {"voice": {"port": 0}}},
    {"routes": {"voice": {"port": True}}},
])
def test_invalid_score_fails_before_playback(update):
    data = score().to_dict()
    data.update(update)
    with pytest.raises(ValueError):
        Session.from_json(json.dumps(data))


def test_deadlines_do_not_accumulate_send_cost_across_repeats():
    clock = Clock()
    sender = Sender(clock, cost=.004)
    session = score([Event(i / 4, "voice", "/pitch", (i,)) for i in range(8)])
    rows = []
    stats = run_session(session, sender, repeats=10, lead_in=.1, clock=clock,
                        sleep=clock.sleep, record=rows.append)
    expected = [.1 + repeat + i / 8 for repeat in range(10) for i in range(8)]
    assert [m[0] for m in sender.messages[:-1]] == pytest.approx(expected)
    assert sender.messages[-1][0] == pytest.approx(10.1)
    assert sender.messages[-1][1:] == ("voice", "/gate", (0,))
    assert stats["sent"] == 80 and stats["dropped_late"] == 0
    assert max(row["lateness_seconds"] for row in rows) < 1e-9
    assert all(row["dispatched_seconds"] - row["observed_seconds"] == pytest.approx(.004) for row in rows)


def test_stall_drops_trigger_but_applies_gate_release():
    clock = Clock()
    sender, rows = Sender(clock), []
    def stalled_sleep(seconds):
        clock.time += seconds + 2
    session = score([Event(0, "voice", "/gate", (1,), kind="trigger"),
                     Event(.5, "voice", "/gate", (0,))])
    stats = run_session(session, sender, clock=clock, sleep=stalled_sleep, record=rows.append)
    assert stats["sent"] == 1 and stats["dropped_late"] == 1
    assert [m[3] for m in sender.messages] == [(0,), (0,)]
    assert [r["status"] for r in rows] == ["dropped_late", "sent"]
    assert rows[0]["dispatched_seconds"] is None


def test_interrupt_sends_reset_and_does_not_continue_score():
    clock, sender = Clock(), None
    sender = Sender(clock)
    def interrupt(_):
        raise KeyboardInterrupt
    session = score([Event(0, "voice", "/gate", (1,)), Event(1, "voice", "/gate", (0,))])
    with pytest.raises(KeyboardInterrupt):
        run_session(session, sender, lead_in=0, clock=clock, sleep=interrupt)
    assert [m[3] for m in sender.messages] == [(1,), (0,)]


def test_reset_attempts_all_routes_without_masking_playback_error(capsys):
    class FailingSender:
        def __init__(self):
            self.calls = []
        def send(self, route, message):
            self.calls.append((route, message.address))
            if message.address == "/event":
                raise RuntimeError("playback failure")
            if route == "a":
                raise OSError("reset failure")
    session = score([Event(0, "a", "/event")], routes={
        name: Route(57130, reset=(Message("/reset"),)) for name in ("a", "b")})
    sender = FailingSender()
    with pytest.raises(RuntimeError, match="playback failure"):
        run_session(session, sender, lead_in=0)
    assert sender.calls == [("a", "/event"), ("a", "/reset"), ("b", "/reset")]
    assert "reset failure" in capsys.readouterr().err


def vortex_event(start, end, value, part_start=None):
    return SimpleNamespace(whole=SimpleNamespace(begin=start, end=end), value=value,
                           has_onset=lambda: part_start is None or part_start == start)


def test_capture_fractional_onsets_binding_order_and_window_boundaries():
    events = [vortex_event(0, 2, {"n": 0}, part_start=1),  # earlier note, no new onset
              vortex_event(1, Fraction(3, 2), {"n": Fraction(7, 12), "gain": .4}),
              vortex_event(Fraction(7, 4), 3, {"n": 1, "gain": .5}),
              vortex_event(2, 3, {"n": 2, "gain": .6})]  # end boundary excluded
    result = capture_vortex(events, [Binding("voice", "/note", ("n", "gain"), "trigger")],
                            start_cycle=1, cycles=1, beats_per_cycle=4)
    assert len(result) == 2
    assert [(e.beat, e.duration) for e in result] == [(0, 2), (3, 1)]
    assert result[0].args == pytest.approx((7 / 12, .4))
    assert result[0].kind == "trigger"


def test_capture_rejects_missing_controls_and_continuous_patterns():
    bindings = [Binding("voice", "/pitch", ("n",))]
    with pytest.raises(ValueError, match="missing fields"):
        capture_vortex([vortex_event(0, 1, {"note": 60})], bindings)
    with pytest.raises(ValueError, match="continuous"):
        capture_vortex([SimpleNamespace(whole=None)], bindings)


def test_actual_sardine_vortex_capture():
    vortex = pytest.importorskip("sardine_core.sequences.tidal_parser")
    pattern = vortex.n("0 ~ [1 2] 3")
    result = capture_vortex(pattern.query(vortex.TimeSpan(0, 1)),
                            [Binding("voice", "/pitch", ("n",))])
    assert [e.beat for e in result] == [0, 2, 2.5, 3]
    assert [e.duration for e in result] == [1, .5, .5, 1]
    assert [e.args for e in result] == [(0,), (1,), (2,), (3,)]


def test_real_udp_session_updates_hydrapy_signals_and_resets():
    pytest.importorskip("pythonosc")
    from hydrapy.live import OscIn, close_resources
    control = OscIn(host="127.0.0.1", port=0)
    pitch, gate = control.value("/pitch"), control.value("/gate")
    session = score([Event(0, "voice", "/pitch", (.75,)), Event(.1, "voice", "/gate", (1,))],
                    bpm=600, beats=.5,
                    routes={"voice": Route(control.port, reset=(Message("/gate", (0,)),))})
    sender = OscSender(session.routes)
    try:
        run_session(session, sender, lead_in=0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and control.latest.get("/gate") != [0]:
            time.sleep(.005)
        ctx = SimpleNamespace(time=0)
        assert pitch.value(ctx) == pytest.approx(.75)
        assert gate.value(ctx) == 0
    finally:
        sender.close()
        close_resources()


def test_example_patch_changes_cpu_pixels_without_audio_or_model(monkeypatch):
    from hydrapy import Renderer, default_patch, reset_patch
    from hydrapy.live import close_resources
    monkeypatch.setenv("HYDRAPY_SESSION_PORT", "0")
    existing_modules = set(sys.modules)
    reset_patch()
    try:
        ns = runpy.run_path(str(ROOT / "patches/session_oscillator.py"))
        patch = default_patch()
        renderer = Renderer(patch, size=(48, 48))
        before = renderer.step(0).copy()
        ns["control"]._handle("/voice/pitch", 1.5)
        after = renderer.step(.1)
        assert not np.allclose(before, after)
        assert len(patch.probes[0].samples) > 0
        assert not ({"torch", "sounddevice"} & (set(sys.modules) - existing_modules))
    finally:
        close_resources()
        reset_patch()


def test_inspection_modes_never_open_sender_or_launch(tmp_path, monkeypatch, capsys):
    import hydrapy.session as runner
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection tried to activate I/O")
    monkeypatch.setattr(runner, "OscSender", forbidden)
    monkeypatch.setattr(runner, "Children", forbidden)
    path = ROOT / "examples/performance/session.json"
    main([str(path), "--dry-run", "--launch"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["seconds_per_play"] == 4
    assert plan["schedule"][-1]["seconds"] == 3.5
    out = tmp_path / "score.html"
    main([str(path), "--preview", str(out), "--launch"])
    page = out.read_text()
    assert "/voice/pitch" in page and "127.0.0.1:57130" in page


def test_preview_escapes_score_text():
    session = score([Event(0, "voice", "/label", ('<script>alert(1)</script>',))], name="<b>clip</b>")
    page = preview_html(session)
    assert "<script>" not in page and "<b>clip</b>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;clip&lt;/b&gt;" in page


def test_child_failure_is_reported_and_owned_children_are_reaped(tmp_path):
    specs = [Process("worker", ("{python}", "-c", "import time; time.sleep(60)")),
             Process("failed", ("{python}", "-c", "raise SystemExit(7)"))]
    children = Children(specs, tmp_path)
    with pytest.raises(RuntimeError, match="failed.*7"):
        with children:
            children.children[1][1].wait(timeout=5)
            children.check()
    assert all(child.poll() is not None for _, child in children.children)


def test_failed_launch_cleans_up_earlier_child(tmp_path):
    specs = [Process("worker", ("{python}", "-c", "import time; time.sleep(60)")),
             Process("missing", (str(tmp_path / "missing-executable"),))]
    children = Children(specs, tmp_path)
    with pytest.raises(FileNotFoundError):
        with children:
            pass
    assert all(child.poll() is not None for _, child in children.children)


POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process-group shutdown")


def _running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _inherited_ignore(name):
    """SIGINT/SIGHUP ignored by whatever started pytest stay ignored in the runner, by design."""
    return signal.getsignal(getattr(signal, name)) == signal.SIG_IGN


@pytest.fixture
def launched(tmp_path):
    """Start ``python -m hydrapy.session --launch`` on a long score with one owned child.

    Returns (runner, recv, child_pid) after the first gate message has arrived and the
    child is up; ``recv(timeout)`` reads the next OSC message's args from the route.
    """
    from pythonosc.osc_message import OscMessage
    started = []

    def start(child_setup="", prefix=()):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        pid_path = tmp_path / "child.pid"
        code = ("import os,signal,time; from pathlib import Path; " + child_setup +
                f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)")
        session = score([Event(0, "voice", "/gate", (1,))], beats=128,
                        routes={"voice": Route(receiver.getsockname()[1], reset=(Message("/gate", (0,)),))},
                        processes=(Process("owned", ("{python}", "-c", code)),))
        path = tmp_path / "session.json"
        path.write_text(session.to_json())
        runner = subprocess.Popen([*prefix, sys.executable, "-m", "hydrapy.session", str(path),
                                   "--launch", "--lead-in", ".2"], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started.append((runner, receiver, pid_path))

        def recv(timeout=5):
            receiver.settimeout(timeout)
            return OscMessage(receiver.recv(4096)).params

        assert recv() == [1]
        deadline = time.monotonic() + 5
        while not (pid_path.exists() and pid_path.read_text()) and time.monotonic() < deadline:
            time.sleep(.01)
        return runner, recv, int(pid_path.read_text())

    yield start
    for runner, receiver, pid_path in started:
        # Child first: a leaked child holds the runner's pipes open.
        text = pid_path.read_text() if pid_path.exists() else ""
        if text:
            try:
                os.kill(int(text), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if runner.poll() is None:
            runner.kill()
            runner.communicate(timeout=5)
        receiver.close()


@POSIX
@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_cli_signal_resets_route_and_reaps_its_child(launched, name):
    if name != "SIGTERM" and _inherited_ignore(name):
        pytest.skip(f"{name} is ignored by whatever started pytest")
    sig = getattr(signal, name)
    runner, recv, child = launched()
    runner.send_signal(sig)
    assert recv() == [0]
    runner.communicate(timeout=5)
    assert runner.returncode == 128 + sig
    assert not _running(child)


@POSIX
@pytest.mark.parametrize("second", ["SIGTERM", "SIGINT"])
def test_repeated_signal_does_not_cancel_kill_escalation(launched, second):
    if second == "SIGINT" and _inherited_ignore("SIGINT"):
        pytest.skip("SIGINT is ignored by whatever started pytest")
    # This child ignores SIGTERM, so only the runner's SIGKILL fallback can stop it.
    runner, recv, child = launched("signal.signal(signal.SIGTERM, signal.SIG_IGN); ")
    runner.send_signal(signal.SIGTERM)
    assert recv() == [0]              # resets are out; the runner now waits 2 s on the child
    time.sleep(.3)
    runner.send_signal(getattr(signal, second))
    runner.communicate(timeout=10)
    assert runner.returncode == 128 + signal.SIGTERM
    assert not _running(child)


@POSIX
@pytest.mark.skipif(shutil.which("nohup") is None, reason="needs nohup")
def test_nohup_session_plays_through_hangup(launched):
    runner, recv, child = launched(prefix=("nohup",))
    runner.send_signal(signal.SIGHUP)
    with pytest.raises(socket.timeout):
        recv(.5)                      # no reset: the hangup was ignored
    assert runner.poll() is None and _running(child)
    runner.send_signal(signal.SIGTERM)
    assert recv() == [0]
    runner.communicate(timeout=5)
    assert runner.returncode == 128 + signal.SIGTERM
    assert not _running(child)
