"""Control-layer integration; no GL context, audio device, model, or GPU."""
from fractions import Fraction
import json
import os
from pathlib import Path
import runpy
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group shutdown")
def test_cli_sigterm_resets_route_and_reaps_its_child(tmp_path):
    from pythonosc.osc_message import OscMessage
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(5)
    pid_path = tmp_path / "child.pid"
    code = ("import os,time; from pathlib import Path; "
            f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)")
    session = score([Event(0, "voice", "/gate", (1,))], beats=128,
                    routes={"voice": Route(receiver.getsockname()[1], reset=(Message("/gate", (0,)),))},
                    processes=(Process("owned", ("{python}", "-c", code)),))
    path = tmp_path / "session.json"
    path.write_text(session.to_json())
    process = subprocess.Popen([sys.executable, "-m", "hydrapy.session", str(path),
                                "--launch", "--lead-in", ".2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert OscMessage(receiver.recv(4096)).params == [1]
        process.send_signal(signal.SIGTERM)
        assert OscMessage(receiver.recv(4096)).params == [0]
        process.communicate(timeout=5)
        assert process.returncode == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        receiver.close()
