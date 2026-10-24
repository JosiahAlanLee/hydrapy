"""Portable, finite event clips for local OSC performances.

This module opens no devices or connections and imports no pattern engine.
Times are quarter-note beats relative to the start of the clip. Durations are
descriptive: note-offs/gate releases must be explicit events or route resets.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Mapping


SCHEMA = "hydrapy-session-1"


def _number(value, name, minimum=0, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}")
    return value


def _name(value, name):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a nonempty string without control characters")
    return value


def _fields(data, allowed, where):
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    unknown = data.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {where} fields: {', '.join(sorted(unknown))}")


def _args(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("OSC args must be a list or tuple of numbers and strings")
    result = []
    for value in values:
        if isinstance(value, bool):
            result.append(int(value))
        elif isinstance(value, int):
            if not -(2 ** 31) <= value < 2 ** 31:
                raise ValueError("OSC integers must fit in 32 bits")
            result.append(value)
        elif isinstance(value, Real):
            value = float(value)
            if not math.isfinite(value) or abs(value) > 3.402823466e38:
                raise ValueError("OSC floats must be finite and fit in 32 bits")
            result.append(value)
        elif isinstance(value, str) and "\0" not in value:
            result.append(value)
        else:
            raise ValueError("OSC args must contain only numbers and strings")
    return tuple(result)


@dataclass(frozen=True)
class Message:
    address: str
    args: tuple = ()

    def __post_init__(self):
        _name(self.address, "OSC address")
        if not self.address.startswith("/") or any(c in self.address for c in " #*,?[]{}"):
            raise ValueError("use a concrete OSC address beginning with / (no wildcards)")
        object.__setattr__(self, "args", _args(self.args))

    def to_dict(self):
        return dict(address=self.address, args=list(self.args))


@dataclass(frozen=True)
class Event:
    beat: float
    route: str
    address: str
    args: tuple = ()
    duration: float = 0
    kind: str = "set"

    def __post_init__(self):
        object.__setattr__(self, "beat", _number(self.beat, "event beat"))
        object.__setattr__(self, "duration", _number(self.duration, "event duration"))
        _name(self.route, "route name")
        object.__setattr__(self, "args", Message(self.address, self.args).args)
        if self.kind not in ("set", "trigger"):
            raise ValueError("event kind must be set or trigger")

    def to_dict(self):
        return dict(beat=self.beat, route=self.route, address=self.address,
                    args=list(self.args), duration=self.duration, kind=self.kind)


@dataclass(frozen=True)
class Route:
    port: int
    host: str = "127.0.0.1"
    reset: tuple = ()

    def __post_init__(self):
        _name(self.host, "OSC host")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("OSC port must be an integer in 1..65535")
        if not all(isinstance(message, Message) for message in self.reset):
            raise ValueError("route reset must contain Message objects")
        object.__setattr__(self, "reset", tuple(self.reset))

    def to_dict(self):
        return dict(host=self.host, port=self.port, reset=[m.to_dict() for m in self.reset])


@dataclass(frozen=True)
class Process:
    name: str
    argv: tuple
    cwd: str = "."

    def __post_init__(self):
        _name(self.name, "process name")
        if not isinstance(self.argv, (list, tuple)) or not self.argv:
            raise ValueError("process argv must be a nonempty list (not a shell command)")
        for arg in self.argv:
            if not isinstance(arg, str) or "\0" in arg:
                raise ValueError("process argv must contain strings without NUL")
        if not self.argv[0]:
            raise ValueError("process executable must not be empty")
        _name(self.cwd, "process cwd")
        object.__setattr__(self, "argv", tuple(self.argv))

    def to_dict(self):
        return dict(name=self.name, argv=list(self.argv), cwd=self.cwd)


@dataclass
class Session:
    bpm: float
    beats: float
    routes: dict
    events: tuple
    name: str = "Performance"
    processes: tuple = ()

    def __post_init__(self):
        self.bpm = _number(self.bpm, "bpm", positive=True)
        self.beats = _number(self.beats, "session beats", positive=True)
        if not math.isfinite(self.beats * 60 / self.bpm):
            raise ValueError("session duration in seconds must be finite")
        _name(self.name, "session name")
        if not isinstance(self.routes, dict) or not self.routes:
            raise ValueError("a session needs at least one named route")
        for name, route in self.routes.items():
            _name(name, "route name")
            if not isinstance(route, Route):
                raise ValueError("session routes must contain Route objects")
        self.events = tuple(self.events)
        self.routes = dict(self.routes)
        for event in self.events:
            if not isinstance(event, Event):
                raise ValueError("session events must contain Event objects")
            if event.route not in self.routes:
                raise ValueError(f"unknown route: {event.route}")
            if event.beat >= self.beats or event.beat + event.duration > self.beats + 1e-9:
                raise ValueError("event starts and durations must fit inside the session")
        self.events = tuple(sorted(self.events, key=lambda e: e.beat))
        if not all(isinstance(p, Process) for p in self.processes):
            raise ValueError("session processes must contain Process objects")
        self.processes = tuple(self.processes)
        if len({p.name for p in self.processes}) != len(self.processes):
            raise ValueError("process names must be unique")

    def to_dict(self):
        return dict(schema=SCHEMA, name=self.name, bpm=self.bpm, beats=self.beats,
                    routes={name: route.to_dict() for name, route in self.routes.items()},
                    events=[e.to_dict() for e in self.events],
                    processes=[p.to_dict() for p in self.processes])

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        _fields(data, ("schema", "name", "bpm", "beats", "routes", "events", "processes"), "session")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"expected {SCHEMA}")
        try:
            routes = {}
            if not isinstance(data["routes"], dict):
                raise ValueError("routes must be an object")
            for name, route in data["routes"].items():
                _fields(route, ("host", "port", "reset"), "route")
                resets = []
                for message in route.get("reset", []):
                    _fields(message, ("address", "args"), "reset message")
                    resets.append(Message(**message))
                routes[name] = Route(**{**route, "reset": tuple(resets)})
            events = []
            for event in data["events"]:
                _fields(event, ("beat", "route", "address", "args", "duration", "kind"), "event")
                events.append(Event(**event))
            processes = []
            for process in data.get("processes", []):
                _fields(process, ("name", "argv", "cwd"), "process")
                processes.append(Process(**process))
            return cls(bpm=data["bpm"], beats=data["beats"], routes=routes, events=events,
                       name=data.get("name", "Performance"), processes=processes)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid session: {exc}") from exc


@dataclass(frozen=True)
class Binding:
    """Map named Vortex control fields to one OSC message, in argument order."""
    route: str
    address: str
    fields: tuple
    kind: str = "set"

    def __post_init__(self):
        Event(0, self.route, self.address, kind=self.kind)
        if not isinstance(self.fields, (list, tuple)) or not self.fields:
            raise ValueError("a binding needs at least one control field")
        for name in self.fields:
            _name(name, "control field")
        object.__setattr__(self, "fields", tuple(self.fields))


def capture_vortex(events, bindings, *, start_cycle=0, cycles=1, beats_per_cycle=4):
    """Capture queried Vortex events without importing/starting Sardine.

    Pass ``pattern.query(TimeSpan(start, start + cycles))`` from Sardine.
    Only discrete onsets in that half-open window are captured. Values must
    be control dictionaries; missing bound fields are errors, not silence.
    Continuous patterns must first be discretized in the pattern engine.
    """
    start = _number(start_cycle, "start cycle")
    length = _number(cycles, "cycles", positive=True)
    scale = _number(beats_per_cycle, "beats per cycle", positive=True)
    bindings = tuple(bindings)
    if not bindings or not all(isinstance(b, Binding) for b in bindings):
        raise ValueError("capture needs at least one Binding")
    result = []
    for event in events:
        if event.whole is None:
            raise ValueError("continuous Vortex events need explicit discretization before capture")
        if not event.has_onset():
            continue
        onset, end = float(event.whole.begin), float(event.whole.end)
        if not math.isfinite(onset) or not math.isfinite(end) or end < onset:
            raise ValueError("invalid Vortex event span")
        if not start <= onset < start + length:
            continue
        if not isinstance(event.value, Mapping):
            raise ValueError("capture a Vortex control pattern, e.g. n('0 2 4 7')")
        for binding in bindings:
            missing = set(binding.fields) - event.value.keys()
            if missing:
                raise ValueError(f"Vortex event missing fields: {', '.join(sorted(missing))}")
            result.append(Event((onset - start) * scale, binding.route, binding.address,
                                tuple(event.value[name] for name in binding.fields),
                                (min(end, start + length) - onset) * scale, binding.kind))
    return tuple(sorted(result, key=lambda e: e.beat))
