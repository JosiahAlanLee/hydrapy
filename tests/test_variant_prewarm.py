"""Planned takes: GLRenderer.prewarm_take compiles what a rehearsal of the score reaches, and ``hold``
keeps a take from compiling at all; variant keys leave out the gates inside a skipped transform."""
import copy

import numpy as np
import pytest

import hydrapy.core as core
from hydrapy import Param, noise, osc, scene, tracker, voronoi
from hydrapy.signals import Fn, Signal
from hydrapy.studio_tracker import Tracker, validate


@pytest.fixture
def patch():
    previous = core._default_patch
    p = core.Patch()
    core._default_patch = p
    yield p
    core._default_patch = previous


@pytest.fixture
def gl():
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import _create_context
    try:
        return _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')


class Outside(Signal):
    """A value no rehearsal can predict (an audio band, a live input): set by the test."""
    v = .5

    def value(self, ctx):
        return Outside.v


def score_with(*changes):
    d = copy.deepcopy(Tracker().document)
    d['loop'], d['pattern'] = False, 0
    for p, row, name, fade in changes:
        d['patterns'][p].setdefault('scenes', []).append({'r': row, 'name': name, 'fade': fade})
    return validate(d)


def build(patch):
    warp = Param('Warp', .5)
    pulse = Fn(lambda c: float(int(c.time * 2) % 3 == 0))               # time-driven: on and off for 0.5 s
    scene('a')(lambda: osc(8, .1, 1.2).modulate(noise(3), warp).add(voronoi(4), pulse))
    scene('b')(lambda: voronoi(5, .3, .3).kaleid(6).modulate(osc(3), Outside() * 1))
    scene('c')(lambda: noise(3, .2).add(osc(2), tracker.gate(1)).swirl(pulse * .3))
    return warp


def test_gates_inside_a_skipped_transform_are_not_part_of_the_key(patch, gl):
    from hydrapy.backend_gl import GLRenderer
    warp = build(patch)
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=gl)
    try:
        patch.scene_weights = {'a': 0, 'b': 1, 'c': 0}
        r.step(0.)
        before = len(r._variants)
        for i, w in enumerate([0, .5, 0, .5, 0]):        # a is hidden: its own gates change nothing drawn
            warp.set(w)
            r.step(.75 + i / 30)                           # pulse on and off too
        assert len(r._variants) == before
        (key,) = {k for n, k in r._variants if k is not None}
        a_blend = patch.scenes['a'][2]
        assert id(a_blend) in key and len(key) == 2        # a's and c's blends, nothing inside them
    finally:
        r.release()


def frames(r, t, patch, n, outside_at):
    out = []
    for i in range(n):
        Outside.v = 0. if i in outside_at else .5
        t.apply(patch, i / 30)
        r.step(i / 30)
        out.append(r.read())
    return out


def test_prewarm_take_compiles_nothing_mid_take_and_draws_the_same_frames(patch, gl):
    from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer
    build(patch)
    doc = score_with((0, 0, 'a', 0), (0, 12, 'b', 6), (0, 28, 'c', 8), (0, 44, 'a', 0))
    seconds = 64 * 60 / doc['bpm']
    n = int(seconds * 30)
    outside_at = set(range(n // 2, n // 2 + 3 * LIVE_SETTLE))      # an unplanned idle set, held long enough
    runs = {}
    for planned in (True, False):
        r = GLRenderer(patch, size=(32, 18), fps=30, ctx=gl)
        r.settle = LIVE_SETTLE
        try:
            if planned:
                patch.scene_weights, patch.tracker_frame = {'b': 1}, None
                assert r.prewarm_take(doc, seconds) > 0 and r.hold
                assert patch.scene_weights == {'b': 1} and patch.tracker_frame is None   # put back
            else:
                r.prewarm_scenes()
            t = Tracker(copy.deepcopy(doc))
            t.seek(0., 0.)
            t.enabled = True
            built = len(r._variants)
            runs[planned] = frames(r, t, patch, n, outside_at), len(r._variants) - built
        finally:
            r.release()
    (warm, compiled), (cold, cold_compiled) = runs[True], runs[False]
    assert compiled == 0 and cold_compiled > 0
    assert all(np.array_equal(a, b) for a, b in zip(warm, cold))


def test_releasing_hold_compiles_after_settle_again(patch, gl):
    from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer
    build(patch)
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=gl)
    r.settle = LIVE_SETTLE
    try:
        r.prewarm_take(score_with((0, 0, 'b', 0)), 2.)
        patch.scene_weights = {'a': 0, 'b': 1, 'c': 0}
        Outside.v = 0.
        before = len(r._variants)
        for i in range(LIVE_SETTLE + 2):
            r.step(i / 30)
        assert len(r._variants) == before                     # held: the valid compiled variant draws
        r.hold = False
        for i in range(LIVE_SETTLE + 2):
            r.step(1 + i / 30)
        assert len(r._variants) == before + 1
    finally:
        Outside.v = .5
        r.release()
