"""Finite OSC session runner. Control-rate scheduling, never an audio callback."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import html
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from .performance import Session, _number


class OscSender:
    """One UDP socket; python-osc is imported only for an actual performance."""
    def __init__(self, routes):
        try:
            from pythonosc.osc_message_builder import OscMessageBuilder
        except ImportError as exc:
            raise RuntimeError('OSC sessions need python-osc: pip install -e ".[osc]"') from exc
        self.builder = OscMessageBuilder
        # Resolve endpoints once, outside the scheduler's deadlines.
        self.endpoints = {name: socket.getaddrinfo(r.host, r.port, socket.AF_INET,
                                                  socket.SOCK_DGRAM)[0][4]
                          for name, r in routes.items()}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, route, message):
        builder = self.builder(address=message.address)
        for arg in message.args:
            builder.add_arg(arg)
        self.socket.sendto(builder.build().dgram, self.endpoints[route])

    def close(self):
        self.socket.close()


class Children:
    """Own only the processes explicitly started by this session."""
    def __init__(self, specs, root):
        self.specs, self.root, self.children = specs, Path(root), []

    def __enter__(self):
        try:
            for spec in self.specs:
                argv = [sys.executable if arg == "{python}" else arg for arg in spec.argv]
                child = subprocess.Popen(argv, cwd=self.root / spec.cwd,
                                         start_new_session=os.name == "posix")
                self.children.append((spec.name, child))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def check(self):
        for name, child in self.children:
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"session process {name!r} exited with status {code}")

    def __exit__(self, *_):
        for _, child in reversed(self.children):
            try:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGTERM)
                elif child.poll() is None:
                    child.terminate()
            except ProcessLookupError:
                pass
        for _, child in reversed(self.children):
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    child.kill()
                child.wait()


def run_session(session, sender, *, repeats=1, lead_in=1, max_trigger_lateness=.05,
                clock=time.monotonic, sleep=time.sleep, record=None, check=None):
    """Play against absolute monotonic deadlines and reset routes on every exit.

    Late state changes still apply (including gate-off). Triggers that miss
    their deadline by more than max_trigger_lateness are dropped and recorded.
    OSC messages are sent immediately when due, without OSC bundle timetags.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    lead_in = _number(lead_in, "lead-in")
    limit = _number(max_trigger_lateness, "max trigger lateness")
    seconds_per_beat = 60 / session.bpm
    origin = clock() + lead_in
    stats = dict(sent=0, dropped_late=0, max_lateness_seconds=0.0)

    def wait_until(deadline):
        while True:
            if check:
                check()
            remaining = deadline - clock()
            if remaining <= 0:
                return
            sleep(min(remaining, .05))

    error = None
    try:
        for repeat in range(repeats):
            for index, event in enumerate(session.events):
                due = (repeat * session.beats + event.beat) * seconds_per_beat
                wait_until(origin + due)
                actual = clock() - origin
                lateness = max(0, actual - due)
                status = "dropped_late" if event.kind == "trigger" and lateness > limit else "sent"
                dispatched = None
                if status == "sent":
                    sender.send(event.route, event)
                    dispatched = clock() - origin
                stats[status] += 1
                stats["max_lateness_seconds"] = max(stats["max_lateness_seconds"], lateness)
                if record:
                    record(dict(repeat=repeat, index=index, event=event.to_dict(),
                                scheduled_seconds=due, observed_seconds=actual,
                                dispatched_seconds=dispatched,
                                lateness_seconds=lateness, status=status))
            # A clip's final held value lasts to its declared end, even with no events.
            wait_until(origin + (repeat + 1) * session.beats * seconds_per_beat)
    except BaseException as exc:
        error = exc
        raise
    finally:
        reset_errors = []
        for name, route in session.routes.items():
            for message in route.reset:
                try:
                    sender.send(name, message)
                except Exception as exc:
                    reset_errors.append(f"{name}: {exc}")
        if reset_errors:
            detail = "route reset failed: " + "; ".join(reset_errors)
            if error is None:
                raise RuntimeError(detail)
            print(detail, file=sys.stderr)
    return stats


def preview_html(session):
    """A static arrangement and routing view, with exact event values below it."""
    esc = lambda value: html.escape(str(value), quote=True)
    lanes = {}
    for event in session.events:
        lanes.setdefault((event.route, event.address), []).append(event)
    rows = []
    for (route, address), events in lanes.items():
        markers = []
        for event in events:
            title = f"beat {event.beat:g}: {event.args} ({event.kind})"
            width = max(.35, event.duration / session.beats * 100)
            markers.append(f'<span class="event" style="left:{event.beat/session.beats*100:.5f}%;'
                           f'width:{min(width,100-event.beat/session.beats*100):.5f}%" '
                           f'title="{esc(title)}"><span>{esc(", ".join(map(str,event.args)))}</span></span>')
        endpoint = session.routes[route]
        rows.append(f'<div class="label"><b>{esc(route)}</b><code>{esc(address)}</code>'
                    f'<small>{esc(endpoint.host)}:{endpoint.port}</small></div>'
                    f'<div class="lane">{"".join(markers)}</div>')
    table = "".join(f'<tr><td>{e.beat:g}</td><td>{esc(e.route)}</td><td>{esc(e.address)}</td>'
                    f'<td>{esc(json.dumps(e.args))}</td><td>{e.duration:g}</td><td>{e.kind}</td></tr>'
                    for e in session.events)
    # At most 32 labels even for long clips; positions are still exact.
    step = max(1, math.ceil(session.beats / 32))
    ticks = "".join(f'<span style="left:{beat/session.beats*100:.5f}%">{beat}</span>'
                    for beat in range(0, math.ceil(session.beats), step))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(session.name)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#111820;color:#e6edf3;font:15px system-ui}}
main{{max-width:1200px;margin:36px auto;padding:0 24px}}h1{{font-size:28px;margin-bottom:8px}}
p,small{{color:#a9bdc9}}code{{font:13px monospace}}.scroll{{overflow-x:auto;padding:18px 0}}
.score{{display:grid;grid-template-columns:240px minmax(440px,1fr);min-width:700px;gap:1px;background:#34424e}}
.label,.lane,.ruler{{background:#18232e;min-height:72px;padding:10px}}.label>*{{display:block;margin:3px 0}}
.lane,.ruler{{position:relative;overflow:hidden}}.ruler{{min-height:34px}}.ruler>span{{position:absolute}}
.event{{position:absolute;top:14px;height:42px;min-width:3px;background:#d49a65;border-left:2px solid #ffd7a9}}
.event>span{{display:block;color:#111820;font:12px monospace;padding:4px;overflow:hidden;white-space:nowrap}}
table{{border-collapse:collapse;width:100%;font:13px monospace}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #34424e}}
</style><main><h1>{esc(session.name)}</h1><p>{session.bpm:g} BPM · {session.beats:g} beats ·
{len(session.events)} events. Static score preview; this page sends no events and plays no audio.</p>
<div class="scroll"><div class="score"><div class="label">Beat</div><div class="ruler">{ticks}</div>
{"".join(rows)}</div></div><p>Bars show captured durations. State changes hold until the next message;
durations do not generate gate releases. Route resets run when playback ends.</p>
<div class="scroll"><table><thead><tr><th>Beat</th><th>Route</th><th>Address</th><th>Values</th>
<th>Duration</th><th>Kind</th></tr></thead><tbody>{table}</tbody></table></div></main></html>'''


def add_arguments(parser):
    parser.add_argument("session", help="hydrapy-session-1 JSON file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print the schedule; no sockets or children")
    mode.add_argument("--preview", metavar="HTML", help="write a static arrangement preview; no playback")
    parser.add_argument("--launch", action="store_true", help="also launch the session's explicit child commands")
    parser.add_argument("--repeat", type=int, default=1, help="number of complete clip plays")
    parser.add_argument("--lead-in", type=float, default=1, help="startup allowance in seconds (not a readiness handshake)")
    parser.add_argument("--max-trigger-lateness", type=float, default=.05)
    parser.add_argument("--record", help="new JSONL dispatch log (refuses to overwrite)")


def execute(args):
    path = Path(args.session).resolve()
    session = Session.from_json(path.read_text(encoding="utf-8"))
    if args.repeat < 1:
        raise ValueError("repeat must be a positive integer")
    _number(args.lead_in, "lead-in")
    _number(args.max_trigger_lateness, "max trigger lateness")
    if args.dry_run:
        plan = session.to_dict()
        plan["repeats"] = args.repeat
        plan["seconds_per_play"] = session.beats * 60 / session.bpm
        plan["schedule"] = [{**e.to_dict(), "seconds": e.beat * 60 / session.bpm,
                             "endpoint": f"{session.routes[e.route].host}:{session.routes[e.route].port}"}
                            for e in session.events]
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    if args.preview:
        Path(args.preview).write_text(preview_html(session), encoding="utf-8")
        print(f"wrote {args.preview}")
        return
    with ExitStack() as stack:
        if threading.current_thread() is threading.main_thread():
            # Kill, Ctrl-C and a closed terminal (SIGHUP) share one teardown; repeats are
            # ignored once it starts so they cannot cut off resets or SIGKILL escalation.
            # SIGINT/SIGHUP already ignored by the launcher (a background job, nohup) stay so.
            caught = [signal.SIGTERM] + [s for s in (signal.SIGINT, getattr(signal, "SIGHUP", None))
                                         if s is not None and signal.getsignal(s) != signal.SIG_IGN]
            def terminate(signum, frame):
                for s in caught:
                    signal.signal(s, signal.SIG_IGN)
                raise SystemExit(128 + signum)
            for s in caught:
                stack.callback(signal.signal, s, signal.signal(s, terminate))
        log = stack.enter_context(open(args.record, "x", encoding="utf-8")) if args.record else None
        if log:
            log.write(json.dumps(dict(type="session", session=session.to_dict(), repeats=args.repeat)) + "\n")
            log.flush()
        sender = OscSender(session.routes)
        stack.callback(sender.close)
        children = stack.enter_context(Children(session.processes if args.launch else (), path.parent))
        def record(row):
            if log:
                log.write(json.dumps(dict(type="dispatch", **row), allow_nan=False) + "\n")
                log.flush()
        result = run_session(session, sender, repeats=args.repeat, lead_in=args.lead_in,
                             max_trigger_lateness=args.max_trigger_lateness,
                             check=children.check, record=record)
        print(json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        execute(parser.parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"session: {exc}\n")


if __name__ == "__main__":
    main()
