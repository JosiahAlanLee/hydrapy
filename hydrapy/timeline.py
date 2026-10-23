"""Sparse musical keyframes, evaluated identically in Python and HTML.

Rocket-style interpolation; this module does not implement Rocket's editor
protocol or rewind stateful simulations.
"""
from __future__ import annotations

import bisect
import json
import math

from .signals import Signal


class Track(Signal):
    """Non-looping (time, value[, mode]) keyframes; mode belongs to the left key.

    Modes: step, linear, smooth (smoothstep), ramp (quadratic ease-in).
    Times must be strictly increasing; outside the range the endpoint holds.
    units='beats' uses ctx.bpm unless an explicit bpm is supplied.
    """
    MODES = ("step", "linear", "smooth", "ramp")

    def __init__(self, keys, units="seconds", bpm=None):
        if units not in ("seconds", "beats"):
            raise ValueError("units must be 'seconds' or 'beats'")
        if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
            raise ValueError("bpm must be positive and finite")
        rows = []
        for key in keys:
            if len(key) not in (2, 3):
                raise ValueError("keyframes must be (time, value[, mode])")
            t, v = map(float, key[:2])
            mode = key[2] if len(key) == 3 else "linear"
            if not math.isfinite(t) or not math.isfinite(v) or mode not in self.MODES:
                raise ValueError("keyframes need finite values and a supported interpolation mode")
            if rows and t <= rows[-1][0]:
                raise ValueError("keyframe times must be strictly increasing")
            rows.append((t, v, mode))
        if not rows:
            raise ValueError("a track needs at least one keyframe")
        self.keys = tuple(rows)
        self.units, self.bpm = units, bpm
        self._times = tuple(k[0] for k in rows)

    def value(self, ctx):
        t = ctx.time * ((ctx.bpm if self.bpm is None else self.bpm) / 60 if self.units == "beats" else 1)
        if not math.isfinite(t):
            raise ValueError("track time must be finite")
        i = bisect.bisect_right(self._times, t) - 1
        if i < 0:
            return self.keys[0][1]
        if i >= len(self.keys) - 1:
            return self.keys[-1][1]
        a, b = self.keys[i:i + 2]
        u = (t - a[0]) / (b[0] - a[0])
        if a[2] == "step": u = 0
        elif a[2] == "smooth": u = u * u * (3 - 2 * u)
        elif a[2] == "ramp": u *= u
        return a[1] + (b[1] - a[1]) * u

    def js(self):
        clock = "time"
        if self.units == "beats":
            clock += "*(" + ("bpm" if self.bpm is None else repr(float(self.bpm))) + ")/60"
        rows = json.dumps(self.keys, separators=(",", ":"))
        return ("(function(t,k){if(t<=k[0][0])return k[0][1];var i=0;"
                "while(i+1<k.length&&k[i+1][0]<=t)i++;if(i===k.length-1)return k[i][1];"
                "var a=k[i],b=k[i+1],u=(t-a[0])/(b[0]-a[0]);"
                "if(a[2]==='step')u=0;else if(a[2]==='smooth')u=u*u*(3-2*u);"
                "else if(a[2]==='ramp')u=u*u;return a[1]+(b[1]-a[1])*u;"
                "})(" + clock + "," + rows + ")")

    def __repr__(self):
        return f"Track({self.keys!r}, units={self.units!r}, bpm={self.bpm!r})"


class Timeline(dict):
    """Named Track signals with a portable JSON cue format (not Rocket binary)."""
    def __init__(self, tracks):
        super().__init__(tracks)
        if not all(isinstance(k, str) and isinstance(v, Track) for k, v in self.items()):
            raise ValueError("timeline maps string names to Track instances")

    def to_json(self):
        return json.dumps({"schema": "hydrapy-timeline-1", "tracks": {
            name: dict(keys=track.keys, units=track.units, bpm=track.bpm)
            for name, track in self.items()}}, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if data.get("schema") != "hydrapy-timeline-1":
            raise ValueError("expected hydrapy-timeline-1")
        return cls({name: Track(**track) for name, track in data["tracks"].items()})
