"""Animated engine semantics, time reproducibility, geometry and renderer contracts."""
import json

import numpy as np
import pytest

import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.export import chain_to_python


def test_cellular_bit_order_history_and_age_match_upstream_example():
    # The upstream README/source example encodes this neighborhood as 27.
    kernel = ((-2, 1), (-1, 0), (0, 0), (1, 0), (2, 1))
    ca = H.MultiCA1D([H.CARule(1 << 27, kernel, states=8)], 5, wrap=False)
    ca.history[:] = [[0, 1, 0, 3, 0], [2, 0, 1, 2, 1]]
    assert ca.update()[2] == 1
    assert np.array_equal(ca.history[1], [0, 1, 0, 3, 0])
    # A live right neighbor sets bit 2, selecting rule bit 4 (first entry = LSB).
    ca = H.MultiCA1D([H.CARule(1 << 4)], 3, wrap=False)
    ca.current[:] = [0, 0, 1]
    assert np.array_equal(ca.update(), [0, 1, 0])


def test_cellular_masks_reset_saturation_and_totalistic_rules():
    ca = H.MultiCA1D([H.CARule(3, ((0, 0),), 3, True),
                      H.CARule(3, ((0, 0),), 3, False)], 4)
    ca.mask[:] = [0, 0, 1, 1]
    ca.current[:] = [1, 2, 1, 2]
    assert np.array_equal(ca.update(), [2, 0, 2, 2])
    # Two occupied cells select bit 2, regardless of positions or ages.
    ca = H.MultiCA1D([H.CARule(1 << 2, states=8, positional=False)], 3)
    ca.current[:] = [1, 0, 6]
    assert np.array_equal(ca.update(), [2, 1, 7])


def test_cellular_wrap_wide_offsets_negative_lags_and_large_rules():
    rule = H.CARule(2, ((-1, 0), (0, -1)))
    wrapped, bounded = [H.MultiCA1D([rule], 3, wrap=x) for x in (True, False)]
    for c in (wrapped, bounded):
        c.current[-1] = 1
    assert np.array_equal(wrapped.update(), [1, 0, 0])
    assert np.array_equal(bounded.update(), [0, 0, 0])
    ca = H.MultiCA1D([H.CARule(2, ((-7, 0),))], 3)
    ca.current[-1] = 1
    assert np.array_equal(ca.update(), [1, 0, 0])
    ca = H.MultiCA1D([H.CARule(1 << 127, tuple((x, 0) for x in range(-3, 4)), states=256)], 8)
    ca.current[:] = 255
    assert np.all(ca.update() == 255)


def test_cellular_animation_is_fps_independent_and_replayable():
    sources = [H.Cellular(width=32, height=16, rate=25, seed=42, perturb=.02) for _ in range(3)]
    for fps, source in zip((24, 60), sources):
        for i in range(fps * 2):
            source.frame_at(i / fps)
    expected = sources[2].frame_at(2).copy()
    assert np.ptp(expected[..., 0]) > 50
    for source in sources[:2]:
        assert np.array_equal(source.frame_at(2), expected)
        source.frame_at(.1)
        assert np.array_equal(source.frame_at(2), expected)
        assert source._rows.shape == (16, 32)
    # Sub-generation times hold exactly, while a new generation moves the window.
    assert np.array_equal(sources[2].frame_at(2), sources[2].frame_at(2.01))
    assert not np.array_equal(sources[2].frame_at(2.04), expected)


def test_hidden_cellular_keeps_up_so_its_scene_returns_without_a_stall():
    import time
    p = H.Patch(size=(64, 36))
    cells = H.Cellular(width=320, height=180, rate=120, seed=3)
    H.solid(0, 0, 0).out(p.o0, patch=p)
    H.scene('plain', patch=p)(lambda: H.solid(.2, .2, .2))
    H.scene('cells', patch=p)(lambda: H.src(cells))
    p.scene_weights = {'plain': 1.0, 'cells': 0.0}
    r = GLRenderer(p, size=(64, 36), ctx=_create_context(), fps=30)
    for k in range(30 * 20):                     # twenty seconds with the automaton's scene hidden
        r.step(k / 30)
    assert cells._generation == 599 * 4           # kept up while hidden: 120 generations a second
    p.scene_weights = {'plain': 0.0, 'cells': 1.0}
    t0 = time.perf_counter()
    r.step(600 / 30)
    assert time.perf_counter() - t0 < .05        # no replay of 2400 generations on the way back
    fresh = H.Cellular(width=320, height=180, rate=120, seed=3)
    np.testing.assert_array_equal(fresh.frame_at(600 / 30), cells.frame_at(600 / 30))   # still a function of time


def test_cellular_glyphs_and_python_roundtrip():
    c = H.Cellular(width=12, height=8, chars=" .:+#", cell=(8, 12), seed=3)
    assert c.frame_at(0).shape == (96, 96, 4)
    assert np.ptp(c.frame_at(0)[..., 0]) > 0
    encoded = chain_to_python(H.src(c))
    decoded = eval(encoded, vars(H))
    assert np.array_equal(H.src(c).render(1, (96, 96)), decoded.render(1, (96, 96)))


@pytest.mark.parametrize("factory", [
    lambda: H.CARule(-1), lambda: H.CARule(1, states=257),
    lambda: H.CARule(1, ((0, 65),)), lambda: H.CARule(1, ()),
    lambda: H.MultiCA1D([], 3), lambda: H.Cellular(rate=0),
    lambda: H.Cellular(width=2.5), lambda: H.Cellular(chars=""),
    lambda: H.Cellular(perturb=2), lambda: H.Cellular(cell=(0, 8)),
])
def test_invalid_cellular_configuration(factory):
    with pytest.raises(ValueError):
        factory()


def test_frame_source_validation_single_draw_cache_and_export_errors():
    calls = []
    def draw(t):
        calls.append(t)
        return np.full((3, 5, 3), int(t), np.uint8)
    source = H.FrameSource(draw)
    assert source.frame_at(1).shape == (3, 5, 4)
    source.array_at(1); source.frame_at(1)
    assert calls == [1]
    assert np.all(source.frame_at(2)[..., 3] == 255)
    assert calls == [1, 2]
    with pytest.raises(ValueError, match="finite"):
        source.frame_at(float("nan"))
    with pytest.raises(ValueError, match="uint8"):
        H.FrameSource(lambda t: np.zeros((2, 3, 3))).frame_at(0)
    p = H.Patch(); H.src(source).out(p.o0, patch=p)
    with pytest.raises(ValueError, match="FrameSource"):
        H.to_html(p)
    with pytest.raises(ValueError, match="FrameSource"):
        H.patch_to_hydra(p)
    with pytest.raises(ValueError, match="original draw callable"):
        chain_to_python(H.src(source))


@pytest.fixture
def glctx():
    pytest.importorskip("moderngl")
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(str(exc))
    yield ctx
    ctx.release()


def test_frame_source_cpu_gl_orientation_alpha_resize_and_bounded_texture(glctx):
    def draw(t):
        h = 4 if t < 3 else 6
        a = np.zeros((h, 8, 4), np.uint8)
        a[:h // 2] = [255, 0, 0, 128]
        a[h // 2:] = [0, int(t * 15) % 256, 255, 255]
        return a
    source = H.FrameSource(draw)
    p = H.Patch(size=(8, 4)); H.src(source).out(p.o0, patch=p)
    r = GLRenderer(p, ctx=glctx)
    try:
        for i in range(100):
            t = i / 30
            r.step(t)
            gpu = r.read_rgba()
            cpu = p.o0.chain.render(t, p.size)
            assert np.max(np.abs(cpu - gpu)) < .005
            if i == 0:
                first_tex = r.media[source.name]["tex"]
            if t < 3:
                assert r.media[source.name]["tex"] is first_tex
        assert len(r.media) == 1 and "texs" not in r.media[source.name]
        assert r.media[source.name]["tex"].size == (8, 6)
    finally:
        r.release()


def test_sdf_geometry_boolean_and_morph_semantics():
    circle = H.sdfCircle(.25).render(size=(101, 101))[..., 0]
    assert circle[50, 50] == pytest.approx(-.25)
    assert circle[50, 75] == pytest.approx(25 / 101 - .25, abs=1e-7)
    a, b = H.sdfCircle(.25, .4), H.sdfBox(.3, .4, .02, .6)
    av, bv = a.render(size=(80, 60))[..., 0], b.render(size=(80, 60))[..., 0]
    assert np.allclose(a.sdfUnion(b).render(size=(80, 60))[..., 0], np.minimum(av, bv))
    assert np.allclose(a.sdfIntersect(b).render(size=(80, 60))[..., 0], np.maximum(av, bv))
    assert np.allclose(a.sdfSubtract(b).render(size=(80, 60))[..., 0], np.maximum(av, -bv))
    assert np.allclose(a.blend(b, .3).render(size=(80, 60))[..., 0], .7 * av + .3 * bv)
    mask = a.sdfSubtract(b).sdfMask().render(size=(80, 60))
    assert mask[..., :3].min() == 0 and mask[..., :3].max() == 1
    assert np.all(mask[..., 3] == 1)


@pytest.mark.parametrize("chain", [
    H.orbitVoronoi(13, 3, .38, -.4, .2, .003, mode) for mode in (0, 1, 2)
] + [
    H.sdfCircle(H.Lfo(.1, lo=.2, hi=.3), .4).sdfUnion(H.sdfBox(.2, .3, .05, .6), .12).sdfBands(14, .4, H.Time() * .2),
    H.sdfCircle(.35).sdfSubtract(H.sdfBox(.35, .4, H.noise(2) * .05)).sdfMask(.01, .005),
])
def test_generative_cpu_gpu_parity_and_native_html(chain, glctx):
    p = H.Patch(size=(96, 64)); chain.out(p.o0, patch=p)
    r = GLRenderer(p, ctx=glctx)
    try:
        for t in (0.2, 2.7):
            r.step(t)
            assert np.abs(chain.render(t, p.size) - r.read_rgba()).mean() < .002
        assert "<canvas" in H.to_html(p)
    finally:
        r.release()


def test_curve_endpoints_closure_replay_and_input_ownership():
    points = np.array([[0., 0.], [.3, 1.], [1., 0.]])
    saved = points.copy()
    smooth = H.subdivide(points, 2)
    assert np.array_equal(smooth[[0, -1]], points[[0, -1]])
    closed = H.subdivide(points, 2, closed=True)
    assert np.array_equal(closed[0], closed[-1])
    fuzzy = H.fuzzy_paths([smooth, closed], seed=12, time=.5)
    again = H.fuzzy_paths([smooth, closed], seed=12, time=.5)
    nearby = H.fuzzy_paths([smooth, closed], seed=12, time=.501)
    assert len(fuzzy) == 8
    for a, b, c in zip(fuzzy, again, nearby):
        assert np.array_equal(a, b)
        assert np.max(np.abs(a - c)) < .0001
    assert np.array_equal(fuzzy[0][[0, -1]], smooth[[0, -1]])
    assert np.array_equal(fuzzy[-1][0], fuzzy[-1][-1])
    assert np.array_equal(saved, points)


def test_flora_schema_projection_and_detail(tmp_path):
    model = dict(schema="wisconsin-flora-1", units="metres", up="Z", strokes=[
        dict(points=[[0, 0, 0], [1, 2, 3]], detail=0),
        dict(points=[[0, 0, 1], [2, 0, 3]], detail=2)])
    path = tmp_path / "plant.json"; path.write_text(json.dumps(model))
    paths = H.flora_lines(path, azimuth=0, elevation=0, detail=0)
    assert len(paths) == 1 and np.allclose(paths[0], [[0, 0], [1, 3]])
    assert np.allclose(H.flora_lines(model, azimuth=90, elevation=0)[0], [[0, 0], [-2, 3]])
    with pytest.raises(ValueError, match="wisconsin-flora-1"):
        H.flora_lines({})


def test_prewarm_banks_uploads_every_frame_before_it_shows():
    frames = [np.full((8, 12, 3), k * 20, np.uint8) for k in range(6)]
    bank = H.Bank(frames, rate=6)
    p = H.Patch(size=(12, 8))
    H.src(bank).out(p.o0, patch=p)
    r = GLRenderer(p, size=(12, 8), ctx=_create_context(), fps=6)
    assert r.prewarm_banks() == 6 and r.prewarm_banks() == 0          # all six now, none twice
    texs = r.media[bank.name]['texs']
    for k in range(6):                                                  # playing them uploads nothing new
        r.step(k / 6)
        assert len(texs) == 6
    assert r.prewarm_scenes() == 0 and len(texs) == 6                   # prewarm_scenes warms banks too (none left)
