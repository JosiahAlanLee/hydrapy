"""Optimized rendering paths must retain sampling, state and branch semantics."""
import numpy as np
import pytest

import hydrapy as H
from hydrapy.backend_np import Env
from hydrapy.noise import h31, segd
from hydrapy.nca import CLAMP, FILTERS, perceive, step_np
from hydrapy.ops_stroke import _stroke_alpha
from hydrapy.sdf3d import _bound, _distance, _flatten, _np_march


@pytest.fixture
def glctx():
    pytest.importorskip("moderngl")
    from hydrapy.backend_gl import _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(str(exc))
    yield ctx
    ctx.release()


@pytest.mark.parametrize("filter", ["linear", "nearest"])
@pytest.mark.parametrize("offset", [(0, 0), (1, -1), (37, -23), (.5, 1.5), (-1.25, .3)])
@pytest.mark.parametrize("warped,small", [(False, False), (True, False), (False, True)])
def test_offset_sampler_matches_general_sampler(filter, offset, warped, small):
    env = Env(31, 19, filter=filter)
    tex = np.random.default_rng(7).random((7 if small else 19, 11 if small else 31, 4), dtype=np.float32)
    before = tex.copy()
    st = env.base_st * .81 - .2 if warped else env.base_st
    expected = env.sample(tex, st.copy() + np.asarray(offset, np.float32) / env.res)
    actual = env.sample_offset(tex, st, offset)
    np.testing.assert_allclose(actual, expected, atol=7e-6)
    np.testing.assert_array_equal(tex, before)


def test_identity_sample_keeps_texture_storage_private():
    env = Env(16, 8)
    tex = np.ones((8, 16, 4), np.float32)
    env.sample(tex, env.base_st)[...] = 0
    assert np.all(tex == 1)


@pytest.mark.parametrize("warp", ["none", "rotate", "fold", "pixelate"])
@pytest.mark.parametrize("width,taper", [(.012, 0), (-.03, .8), (.001, 1.2), (.5, 0)])
def test_bounded_strokes_match_full_frame_distance(warp, width, taper):
    q = (Env(91, 57).base_st - .5) * [1.6, 1]
    if warp == "rotate": q = q @ np.array([[.8, -.6], [.6, .8]])
    if warp == "fold": q = np.sin(q * 7) * .4
    if warp == "pixelate": q = np.floor(q * 20) / 20
    pts = np.array([[-.6, .1], [.2, -.25], [.43, .3], [.43, .3]], np.float32)
    expected = np.zeros(q.shape[:2], np.float32)
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        tail = 1 - (i + .5) / (len(pts) - 1)
        w = width * (1 - taper * tail * .85)
        expected = np.maximum(expected, np.exp(-2 * (segd(q, a, b) / w)**2) * (1 - taper * tail * .7))
    np.testing.assert_allclose(_stroke_alpha(q, pts, width, taper), expected, atol=1e-7)


FIELDS = [
    H.SDF3.sphere(-.6), H.SDF3.box(.2, -.8, .4), H.SDF3.torus(-.6, .2),
    H.SDF3.torus().rotate(.7, -.3, 1).scale(-.8).translate(.3, -.6, .2),
    H.SDF3.box(.2, .3, .4).fold(4, scale=1.3, x=-.3, y=.5, z=.2, angle=.7),
    H.SDF3.sphere().union(H.SDF3.box().translate(.8), .7),
    H.SDF3.sphere().intersect(H.SDF3.box(.2, .8, .4)),
    H.SDF3.sphere().subtract(H.SDF3.box(.2, .8, .4)),
]


@pytest.mark.parametrize("field", FIELDS)
def test_sdf_sphere_bound_encloses_distance_shell(field):
    p = np.random.default_rng(3).normal(size=(2000, 3)).astype(np.float32) * 2
    plan, args = _flatten(field)
    assert np.min(_distance(plan, p, args) - (np.linalg.norm(p, axis=-1) - _bound(plan, args))) > -1e-5


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("distance,far,steps", [(3, 10, 96), (.1, .4, 12), (-2, 8, 64)])
def test_culled_march_matches_full_grid_reference(field, distance, far, steps):
    from hydrapy.ops_3d import _rot3
    env = Env(47, 31)
    plan, args = _flatten(field)
    rot = _rot3(.25, .3, 0)
    q = (env.base_st - .5) * 2 * np.array([47 / 31, 1], np.float32)
    rd = np.concatenate([q * np.tan(np.radians(45) * .5), -np.ones((31, 47, 1), np.float32)], axis=-1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    rd = rd @ rot
    ro = np.array([0, 0, distance], np.float32) @ rot
    t, hit = np.zeros((31, 47), np.float32), np.zeros((31, 47), bool)
    for _ in range(steps):
        d = _distance(plan, ro + rd * t[..., None], args)
        active = (~hit) & (t <= far)
        hit |= active & (np.abs(d) <= .001)
        t += np.where(active & ~hit, np.maximum(np.abs(d) * .85, .001 * .25), 0)
    actual = _np_march(plan, len(args), env.base_st, args + [.25, .3, 0, distance, 45, steps, .001, far, 2], env)
    np.testing.assert_array_equal(actual[..., 3], hit)
    np.testing.assert_allclose(actual[..., 0], np.where(hit, np.clip(t / far, 0, 1), 0), atol=1e-6)


def reference_perception(state):
    h, w, c = state.shape
    features = np.zeros((h, w, c, 4), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            features += np.roll(state, (dy, -dx), axis=(0, 1))[..., None] * FILTERS[:, dy + 1, dx + 1]
    return features.reshape(h, w, 4 * c)


@pytest.mark.parametrize("rate", [0, .5, 1])
def test_sparse_neural_update_matches_dense_reference(rate):
    rng = np.random.default_rng(9)
    model = H.NCAModel.random(12, 32, seed=3, scale=.3)
    state = rng.normal(0, .4, (17, 23, 12)).astype(np.float32)
    before = state.copy()
    features = reference_perception(state)
    np.testing.assert_allclose(perceive(state), features, atol=2e-6)
    y, x = np.mgrid[:17, :23].astype(np.float32)
    fire = h31(np.stack([x, y, np.full_like(x, 4)], axis=-1)) < rate
    delta = np.maximum(features @ model.w1.T + model.b1, 0) @ model.w2.T
    expected = np.clip(state + delta * fire[..., None], -CLAMP, CLAMP)
    np.testing.assert_allclose(step_np(model, state, 4, rate), expected, atol=1e-6)
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("channels", [4, 12, 16])
@pytest.mark.parametrize("rate", [0, .5, 1])
def test_nca_multi_target_matches_single_target(glctx, monkeypatch, channels, rate):
    from hydrapy.backend_gl import GLRenderer
    if glctx.info.get("GL_MAX_DRAW_BUFFERS", 1) < channels // 4:
        pytest.skip("insufficient draw buffers")
    states = []
    for single in (False, True):
        if single:
            monkeypatch.setitem(glctx.info, "GL_MAX_DRAW_BUFFERS", 1)
        p = H.Patch(size=(32, 24))
        H.shape(4, .15).out(p.o1, patch=p)
        H.osc(4, .1).out(p.o2, patch=p)
        nca = H.NCA(H.NCAModel.random(channels, 16, seed=5), size=(16, 12), speed=30,
                    rate=rate, perturb=.003, damage=p.o1, paint=p.o2, paint_amount=.2)
        H.src(nca).out(p.o0, patch=p)
        r = GLRenderer(p, fps=30, ctx=glctx)
        try:
            for i in range(5): r.step(i / 30)
            assert len(r._ncas[nca.name]["steps"]) == (channels // 4 if single else 1)
            states.append(r.nca_state(nca))
        finally:
            r.release()
    np.testing.assert_allclose(states[0], states[1], atol=1e-6)


@pytest.mark.parametrize("iterations", [1, 2, 3, 4, 5])
def test_shared_texture_prefix_does_not_alias_iteration_targets(glctx, monkeypatch, iterations):
    from hydrapy.backend_gl import GLRenderer
    from hydrapy.glsl import _Compiler
    base = H.osc(6, .03, .5).bake()
    # The first branch is evaluated in the final pass, after the second branch's
    # intermediate iterations. Its input must survive that ping-pong unchanged.
    chain = base.blur(.6, iterations=1).blend(base.blur(1.3, iterations=iterations), .4)
    chain = chain.mult(base.normalMap(.03).relight())
    p = H.Patch(size=(64, 32)); chain.out(p.o0, patch=p)
    frames, passes = [], []
    original = _Compiler.__init__
    class NoCache(dict):
        def get(self, key, default=None): return default
    def uncached(self, *a, **kw):
        original(self, *a, **kw)
        self._prefix_textures = NoCache()
    for disable in (False, True):
        if disable: monkeypatch.setattr(_Compiler, "__init__", uncached)
        r = GLRenderer(p, fps=30, ctx=glctx)
        try:
            frames.append([])
            passes.append(len(r._passes))
            for i in range(3):
                r.step(i / 30)
                frames[-1].append(r.read_rgba())
        finally:
            r.release()
    assert passes[0] < passes[1]
    np.testing.assert_allclose(frames[0], frames[1], atol=.001)


@pytest.mark.parametrize("composite", [False, True])
def test_demo_gpu_mix_matches_display_byte_blending(glctx, composite):
    from docs.bullfrog_demo import FrameMixer
    from hydrapy.backend_gl import GLRenderer
    size = (31, 19)  # RGB rows must not acquire alignment padding
    renderers = []
    mixer = FrameMixer(glctx, size)
    try:
        for chain in (H.osc(5, .1, .8).color(1.5, .3, -.4, .4), H.gradient().color(.4, 1.7, 1, .7)):
            p = H.Patch(size=size); p.composite_alpha = composite
            chain.out(p.o0, patch=p)
            r = GLRenderer(p, ctx=glctx)
            renderers.append(r); r.step(.7)
        a, b = [r.read().astype(np.float32) for r in renderers]
        for amount, fade in ((0, 1), (1, 1), (.37, 1), (.73, .41), (0, 0)):
            expected = np.clip(a * (1 - amount) + b * amount, 0, 255).astype(np.uint8)
            expected = (expected.astype(np.float32) * fade).astype(np.uint8)
            actual = mixer.read(*renderers, amount, fade)
            # GPU multiply-add and byte rounding can differ by one level.
            assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1
    finally:
        mixer.release()
        for r in renderers: r.release()
