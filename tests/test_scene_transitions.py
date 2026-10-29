"""Scene transitions: the transition op, and per-output keyed crossfades chosen before scenes."""
import copy

import numpy as np
import pytest

import hydrapy.core as core
from hydrapy import Patch, gradient, o1, osc, scene, scene_transition, shape, solid, voronoi
from hydrapy.backend_np import Renderer, render_chain
from hydrapy.glsl import gates
from hydrapy.signals import Fn


@pytest.fixture
def patch():
    previous = core._default_patch
    p = core.Patch()
    core._default_patch = p
    yield p
    core._default_patch = previous


def build_scenes(p, scenes=(('day', lambda: osc(8, .1, 1.2)), ('night', lambda: voronoi(5, .3, .3)))):
    for name, fn in scenes:
        scene(name, patch=p)(fn)
    return p


# --- the scene_transition API -------------------------------------------------

def test_scene_transition_composes_transitions_and_blend_stays_the_default(patch):
    scene_transition('luma', softness=.2, patch=patch)
    build_scenes(patch)
    assert [t.op.name for t in patch.o0.chain.transforms] == ['solid', 'transition', 'transition']
    assert all(t.args[2] == 1 and t.args[3] == .2 and t.args[4] == 6
               for t in patch.o0.chain.transforms if t.op.name == 'transition')

    q = Patch()
    build_scenes(q)
    assert all(t.op.name == 'blend' for t in q.o0.chain.transforms[1:])   # unchanged default


def test_scene_transition_takes_numbers_and_is_per_output(patch):
    scene_transition(2, scale=9, patch=patch)                              # shadow, as a number
    build_scenes(patch)
    scene_transition('grain', output=o1, patch=patch)
    scene('a', o1, patch=patch)(lambda: osc(3))
    assert [t.args[2] for t in patch.o0.chain.transforms if t.op.name == 'transition'] == [2, 2]
    assert patch.o0.chain.transforms[-1].args[4] == 9
    assert patch.o1.chain.transforms[-1].op.name == 'transition' and patch.o1.chain.transforms[-1].args[2] == 3


def test_scene_transition_validation(patch):
    build_scenes(patch)
    for bad, message in [(lambda: scene_transition('luma'), 'before o0'),
                         (lambda: scene_transition('bogus'), 'style is'),
                         (lambda: scene_transition(7), 'style is')]:
        with pytest.raises(ValueError, match=message):
            bad()
    q = Patch()
    for bad, message in [(lambda: scene_transition('luma', softness=-1, patch=q), 'softness'),
                         (lambda: scene_transition('luma', scale=0, patch=q), 'scale')]:
        with pytest.raises(ValueError, match=message):
            bad()
    q = Patch()                                                             # a later call replaces the choice
    scene_transition('shadow', patch=q)
    scene_transition('grain', scale=4, patch=q)
    build_scenes(q)
    assert q.o0.chain.transforms[-1].args[2] == 3 and q.o0.chain.transforms[-1].args[4] == 4


# --- the transition op --------------------------------------------------------

def test_every_style_is_exact_at_amount_0_and_1():
    outgoing = render_chain(osc(8, .1, 1.2), t=.3, size=(32, 18))
    incoming = render_chain(voronoi(5, .3, .3).kaleid(6), t=.3, size=(32, 18))
    for style in (0, 1, 2, 3):
        a = render_chain(osc(8, .1, 1.2).transition(voronoi(5, .3, .3).kaleid(6), 0, style=style, seed=2),
                         t=.3, size=(32, 18))
        b = render_chain(osc(8, .1, 1.2).transition(voronoi(5, .3, .3).kaleid(6), 1, style=style, seed=2),
                         t=.3, size=(32, 18))
        assert np.array_equal(a, outgoing)
        assert np.array_equal(b, incoming)


def test_style_0_is_bit_identical_to_blend():
    incoming = voronoi(5, .3, .3).kaleid(6)
    for amount in (0.1, .5, .9):
        a = render_chain(osc(8, .1, 1.2).transition(incoming, amount, style=0), t=.3, size=(32, 18))
        b = render_chain(osc(8, .1, 1.2).blend(incoming, amount), t=.3, size=(32, 18))
        assert np.array_equal(a, b)


def test_luma_appears_brightest_first_and_is_monotonic_in_amount():
    # incoming = gradient, whose luminance rises with x; outgoing black, so out = m * incoming.
    c0, c1 = solid(0, 0, 0), gradient(0)
    incoming = c1.render(0., size=(64, 18))

    def m(amount):
        got = render_chain(c0.transition(c1, amount, style=1, softness=.2), t=0., size=(64, 18))
        return got[..., 0] / np.maximum(incoming[..., 0], 1e-6)
    mid = m(.5)
    assert np.corrcoef(mid.mean(0), incoming[..., 0].mean(0))[0, 1] > .9   # bright side (right) leads
    frames = [m(a) for a in np.linspace(0, 1, 41)]
    assert (np.diff(np.stack(frames), axis=0) >= -1e-6).all()              # every pixel only grows


def test_shadow_floods_in_through_the_outgoing_dark_and_is_monotonic_in_amount():
    # outgoing = gradient (dark on the left), incoming red: the incoming floods in from the dark side.
    c0, c1 = gradient(0), solid(1, 0, 0)
    outgoing = c0.render(0., size=(64, 18))

    def m(amount):
        got = render_chain(c0.transition(c1, amount, style=2, softness=.2), t=0., size=(64, 18))
        return (got[..., 0] - outgoing[..., 0]) / (1 - outgoing[..., 0])
    mid = m(.5)
    assert np.corrcoef(mid.mean(0), outgoing[..., 0].mean(0))[0, 1] < -.9  # dark side (left) leads
    frames = [m(a) for a in np.linspace(0, 1, 41)]
    assert (np.diff(np.stack(frames), axis=0) >= -1e-6).all()


def test_grain_dissolves_through_a_soft_textured_front():
    c0, c1 = solid(0, 0, 0), solid(1, 1, 1)
    a = render_chain(c0.transition(c1, .5, style=3, softness=.1, scale=8, seed=1), t=0., size=(64, 36))
    b = render_chain(c0.transition(c1, .5, style=3, softness=.1, scale=8, seed=2), t=0., size=(64, 36))
    front = a[..., 0]
    assert (front == 0).any() and (front == 1).any() and ((front > 0) & (front < 1)).any()  # soft, not a cut
    assert np.abs(front - front[0, 0]).max() > .5                                # textured
    assert not np.array_equal(a, b)                                              # the seed fixes the pattern


def test_transition_amount_gates_like_blend():
    from hydrapy import noise
    w = Fn(lambda c: 0.0)
    ch = osc(10).transition(noise(3), w, style=1).transition(shape(4), 0, style=3)
    found = [(tr.op.name, v) for tr, v, ident in gates(ch)]
    assert ("transition", w) in found and ("transition", 0) in found


# --- gating on the CPU renderer ----------------------------------------------

def test_cpu_skips_transition_scenes_that_are_not_showing(patch):
    from hydrapy.signals import Signal

    class CountedSignal(Signal):
        def __init__(self, v):
            self.v, self.calls = v, 0

        def value(self, ctx):
            self.calls += 1
            return self.v

    scene_transition('luma', patch=patch)
    freq = {name: CountedSignal(4 + k) for k, name in enumerate(('drift', 'fracture', 'third'))}
    for name, f in freq.items():
        scene(name, patch=patch)(lambda f=f: osc(f, .1))
    r = Renderer(patch, size=(16, 9), fps=10)
    for w, drawn in [({'drift': 1}, {'drift'}), ({'drift': 1, 'fracture': .5}, {'drift', 'fracture'}),
                     ({'third': 1}, {'third'})]:
        patch.scene_weights = w
        for f in freq.values():
            f.calls = 0
        r.step(.5)
        assert {name for name, f in freq.items() if f.calls} == drawn


# --- the GL renderer ---------------------------------------------------------

def _gl_ctx():
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import _create_context
    try:
        return _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')


def test_gl_transition_scene_matches_a_blend_scene_at_extreme_weights(patch):
    ctx = _gl_ctx()
    from hydrapy.backend_gl import GLRenderer
    scene_transition('luma', patch=patch)
    build_scenes(patch)
    plain = Patch()
    build_scenes(plain)
    try:
        for w in ({'day': 1, 'night': 0}, {'day': 0, 'night': 1}):
            patch.scene_weights, plain.scene_weights = w, w
            a = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
            b = GLRenderer(plain, size=(32, 18), fps=30, ctx=ctx)
            try:
                a.prewarm_scenes(); a.step(.7)
                b.prewarm_scenes(); b.step(.7)
                assert np.array_equal(a.read_rgba(), b.read_rgba())
            finally:
                a.release(); b.release()
    finally:
        ctx.release()


def test_gl_transition_scenes_prewarm_and_never_compile_while_playing(patch):
    ctx = _gl_ctx()
    from hydrapy.backend_gl import GLRenderer, LIVE_SETTLE
    from hydrapy.studio_tracker import Tracker, scene_fades, validate
    scene_transition('luma', patch=patch)
    build_scenes(patch, scenes=(('drift', lambda: osc(8, .1, 1.2)), ('fracture', lambda: voronoi(5, .3, .3)),
                                ('third', lambda: gradient(0))))
    d = copy.deepcopy(Tracker().document)
    d['patterns'][0]['scenes'] = [{'r': 0, 'name': 'drift', 'fade': 0},
                                  {'r': 16, 'name': 'third', 'fade': 8},
                                  {'r': 40, 'name': 'fracture', 'fade': 4}]
    d = validate(d)
    t = Tracker(d)
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
    r.settle = LIVE_SETTLE
    try:
        assert r.prewarm_scenes(scene_fades(d, patch.scenes)) == 5
        t.enabled = True
        compiles = 0
        for i in range(int(64 * 60 / d['bpm'] * 30)):
            before = len(r._variants)
            t.apply(patch, i / 30)
            r.step(i / 30)
            compiles += len(r._variants) - before
        assert compiles == 0
    finally:
        r.release()
        ctx.release()


def test_gl_transition_scene_at_zero_weight_costs_no_shader():
    ctx = _gl_ctx()
    from hydrapy.backend_gl import GLRenderer, LIVE_SETTLE
    p = Patch(size=(48, 27))
    scene_transition('luma', patch=p)
    build_scenes(p, scenes=(('day', lambda: osc(8, .1, 1.2)),       # the non-playing scene has a texture op
                            ('night', lambda: voronoi(5, .3, .3).blur(2, 1))))
    p.scene_weights = {'day': 1, 'night': 0}
    r = GLRenderer(p, size=(48, 27), fps=30, ctx=ctx)
    r.settle = LIVE_SETTLE
    try:
        r.prewarm_scenes()
        r.step(0.)
        key = r._active['o0']
        assert key is not None and len(key) == 1          # the non-playing scene's transition is compiled out
        skipped = next(v for (name, k), v in r._variants.items() if name == 'o0' and k)
        assert len(skipped) < len(r._variants[('o0', None)])   # fewer passes than the full program
    finally:
        r.release()
        ctx.release()


def test_gl_transition_matches_numpy():
    ctx = _gl_ctx()
    from hydrapy.backend_gl import GLRenderer
    for style in ('luma', 'shadow'):
        q = Patch(size=(48, 27))
        scene_transition(style, patch=q)
        build_scenes(q, scenes=(('day', lambda: gradient(0)), ('night', lambda: osc(8, .1, 1.2))))
        q.scene_weights = {'day': .4, 'night': .6}
        g = GLRenderer(q, size=(48, 27), fps=30, ctx=ctx)
        try:
            g.prewarm_scenes(); g.step(.7)
            n = Renderer(q, size=(48, 27), fps=30).step(.7)
            assert np.abs(g.read_rgba() - n).max() < 0.002, style
        finally:
            g.release()
    # grain's hash differs on the GPU only around fract() boundaries (FMA contraction): the same
    # front, but a few cells flip. Compare statistically.
    q = Patch(size=(48, 27))
    scene_transition('grain', scale=6, patch=q)
    build_scenes(q, scenes=(('day', lambda: gradient(0)), ('night', lambda: osc(8, .1, 1.2))))
    q.scene_weights = {'day': .4, 'night': .6}
    g = GLRenderer(q, size=(48, 27), fps=30, ctx=ctx)
    try:
        g.prewarm_scenes(); g.step(.7)
        n = Renderer(q, size=(48, 27), fps=30).step(.7)
        d = np.abs(g.read_rgba() - n)
        assert d.max() < 0.05 and (d > 1e-3).mean() < 0.1
    finally:
        g.release()
        ctx.release()
