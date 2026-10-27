"""Named scenes, and the Gorgon tracker's lanes as patch signals.

A scene is a named chain for one output, chosen by the tracker's SCENE lane::

    @scene('drift')
    def _():
        return osc(8).rotate(roll)

    @scene('fracture')
    def _():
        return voronoi(5).kaleid(6)

The output becomes ``base.blend(drift, w1).blend(fracture, w2)...``, where each
weight is 1 for the scene playing and 0 for the rest, and crossfades in between.
``base`` is whatever the patch sent to that output before its first scene
(opaque black if nothing). A blend at amount 0 is compiled out of the GL shader,
so a scene costs nothing while it is not playing; Studio compiles every scene's
shader when the patch loads, so changing scene never compiles mid-performance.

Without the tracker driving them (``hydrapy live``, ``render``, or before the
tracker first plays), the first scene declared for each output plays, as it
does for an output whose scenes the score never names. A change holds until the
next one, wrapping round the cycle; a fade never runs past the next change.

``tracker`` (the ``hydrapy.tracker`` package) reads the audio lanes and
playhead, per frame, as signals::

    osc(tracker.note(1) / 12).scale(1 + tracker.hit(2, decay=.3))

``gate(n)``  1 while lane n's gate is open, else 0
``hit(n, decay=.25)``  1 at each hit on lane n, falling to 0 over ``decay`` seconds
``note(n)``  the last hit's MIDI note on lane n, key-transposed as the CV is (held)
``vel(n)``   the last hit's velocity on lane n, 0..1 (held)
``beat()``   the tracker's beat, 0..64;  ``row()`` the sixteenth row, 0..63

Lanes are 1-6; 5 and 6 play only in a version-3 score, where ``beat()`` is the
beat within the cycle and ``row()`` the playing pattern's row.

Without the tracker these are 0 (``note`` is 60). Muted lanes and a stopped
tracker send no gates or hits; held values keep their last value.
"""
from __future__ import annotations

import re

from .core import Chain, Output, default_patch
from .signals import Signal, _num

NAME = re.compile(r'[A-Za-z0-9_-]{1,24}')


class SceneWeight(Signal):
    """How much of scene ``name`` is showing (0..1). Set by the tracker each frame."""

    def __init__(self, name, patch):
        self.name, self.patch = name, patch

    def value(self, ctx):
        weights = self.patch.scene_weights
        if weights is None:
            return 1.0 if self.name in _first_scenes(self.patch) else 0.0
        return float(weights.get(self.name, 0.0))

    def js(self):
        # A static export shows the scene that is playing as it is exported.
        return _num(self(None))

    def __repr__(self):
        return f"SceneWeight({self.name!r})"


def _first_scenes(patch):
    """The first scene declared for each output: what plays when nothing chooses."""
    first = {}
    for name, (index, _, _) in patch.scenes.items():
        first.setdefault(index, name)
    return set(first.values())


def scene(name, output: Output | None = None, *, patch=None):
    """Declare a named scene: the decorated function returns the chain for ``output`` (default o0)."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError('Scene names are 1-24 letters, digits, - or _')
    p = patch or default_patch()
    scenes = p.scenes
    if name in scenes:
        raise ValueError(f'Scene {name!r} is declared twice')
    index = (output or p.o0).index

    def decorate(fn):
        chain = fn()
        if not isinstance(chain, Chain):
            raise TypeError(f'Scene {name!r} must return a chain, not {type(chain).__name__}')
        if index not in p.scene_bases:
            from . import solid
            p.scene_bases[index] = p.outputs[index].chain or solid(0, 0, 0, 1)
        composed = p.outputs[index].chain if any(i == index for i, _, _ in scenes.values()) else p.scene_bases[index]
        composed = composed.blend(chain, SceneWeight(name, p))
        scenes[name] = (index, chain, composed.transforms[-1])
        p.outputs[index].chain = composed
        return fn

    return decorate


def scene_names(patch):
    return list(patch.scenes)


def check_scenes(patch):
    """Raise if an output's scenes were replaced by a later ``.out()`` to the same output."""
    for name, (index, _, transform) in patch.scenes.items():
        chain = patch.outputs[index].chain
        if chain is None or not any(t is transform for t in chain.transforms):
            raise ValueError(f'o{index} was reassigned after its scenes (scene {name!r} is no longer drawn); '
                             f'send to o{index} before declaring its scenes, or use another output')


def scene_transforms(patch):
    """name -> the blend transform that draws it (its gate in the GL renderer)."""
    return {name: transform for name, (_, _, transform) in patch.scenes.items()}


class TrackerSignal(Signal):
    """One of the tracker's per-frame values, read from the patch the tracker drives."""

    DEFAULTS = {'gate': 0., 'hit': 0., 'note': 60., 'vel': 0., 'beat': 0., 'row': 0.}

    def __init__(self, kind, lane=None, decay=.25, patch=None):
        self.kind, self.lane, self.decay = kind, lane, float(decay)
        self.patch = patch or default_patch()

    def value(self, ctx):
        frame = self.patch.tracker_frame
        if frame is None:
            return self.DEFAULTS[self.kind]
        if self.kind in ('beat', 'row'):
            return float(frame[self.kind])
        if self.kind == 'hit':
            since = frame['since'][self.lane]
            return 0.0 if since is None or since >= self.decay else 1.0 - since / self.decay
        return float(frame[self.kind][self.lane])

    def js(self):
        return _num(self(None))

    def __repr__(self):
        return f"tracker.{self.kind}({'' if self.lane is None else self.lane + 1})"


class _Tracker:
    """``tracker.gate(1)`` etc.: lanes are numbered 1-6, as in the tracker; 5 and 6 need a version-3 score."""

    @staticmethod
    def _lane(n):
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 6:
            raise ValueError('Tracker lanes are 1-6')
        return n - 1

    def gate(self, lane):
        return TrackerSignal('gate', self._lane(lane))

    def hit(self, lane, decay=.25):
        if not 0 < float(decay) <= 30:
            raise ValueError('decay must be between 0 and 30 seconds')
        return TrackerSignal('hit', self._lane(lane), decay)

    def note(self, lane):
        return TrackerSignal('note', self._lane(lane))

    def vel(self, lane):
        return TrackerSignal('vel', self._lane(lane))

    def beat(self):
        return TrackerSignal('beat')

    def row(self):
        return TrackerSignal('row')

