"""The opt-in compute path must preserve the fragment renderer's semantics."""
import numpy as np
import pytest

import hydrapy as H
from hydrapy.glsl import compile_chain


@pytest.fixture
def glctx():
    pytest.importorskip("moderngl")
    from hydrapy.backend_gl import _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(str(exc))
    if ctx.version_code < 430:
        ctx.release()
        pytest.skip("OpenGL 4.3 required for compute equivalence")
    yield ctx
    ctx.release()


def image(size):
    im = H.Image("compute-test-memory")
    im._array = np.random.default_rng(4).random((size[1], size[0], 4), dtype=np.float32)
    # Black RD seeds and transparent paint, including wrapped boundaries.
    im._array[::3, ::4] = 0
    return im


@pytest.mark.parametrize("tail,count", [
    (lambda c: c.rd(iterations=4), 4),
    (lambda c: c.rd(iterations=4).rotate(.3), 3),
    (lambda c: c.bleed(iterations=3).color(.5), 2),
    (lambda c: c.bleed(iterations=1).modulate(H.noise(), .1), 0),
    (lambda c: c.rd(iterations=1).blend(H.osc()), 0),
    (lambda c: c.bleed(iterations=3).blur(), 3),
    (lambda c: c.rd(iterations=2).bake().rotate(.2), 2),
])
def test_only_whole_unwarped_calls_are_eligible(tail, count):
    prog = compile_chain(tail(H.osc()))
    assert sum(p.compute is not None for p in prog.passes) == count
    # Portable shaders are always populated, even when compute is eligible.
    assert all("gl_FragColor" in p.source and "imageStore" not in p.source for p in prog.passes)


def test_custom_operator_named_rd_does_not_select_builtin_kernel():
    from hydrapy.core import Chain, Op, Transform
    custom = Op("rd", "texture", [], "return vec4(0.2);")
    chain = Chain(H.osc().transforms + [Transform(custom, [])])
    assert not any(p.compute for p in compile_chain(chain).passes)


def test_context_requests_compute_then_falls_back_to_33(monkeypatch):
    moderngl = pytest.importorskip("moderngl")
    from hydrapy.backend_gl import _create_context
    calls, result = [], object()
    def create(**kwargs):
        calls.append(kwargs)
        if kwargs["require"] == 330:
            return result
        raise RuntimeError("compute context unavailable")
    monkeypatch.setattr(moderngl, "create_standalone_context", create)
    assert _create_context(compute=True) is result
    assert calls == [{"require": 430}, {"require": 430, "backend": "egl"}, {"require": 330}]


def compare(glctx, patch, count, frames=1, resize=None, atol=.002):
    from hydrapy.backend_gl import GLRenderer
    rendered = []
    for compute in (False, True):
        r = GLRenderer(patch, ctx=glctx, fps=30, compute=compute)
        try:
            assert r.compute_passes == (count if compute else 0)
            assert not r.compute_fallbacks
            rendered.append([])
            for i in range(frames):
                r.step(i / 30)
                rendered[-1].append(r.read_rgba())
            if resize:
                r.resize(resize)
                r.step(frames / 30)
                rendered[-1].append(r.read_rgba())
        finally:
            r.release()
    for fragment, compute in zip(*rendered):
        assert np.isfinite(compute).all()
        np.testing.assert_allclose(compute, fragment, rtol=0, atol=atol)


@pytest.mark.parametrize("name", ["rd", "bleed"])
@pytest.mark.parametrize("size", [(1, 1), (1, 19), (19, 1), (47, 31), (64, 32)])
@pytest.mark.parametrize("iterations", [1, 4, 16])
def test_tiled_kernels_match_fragment_at_edges_and_partial_groups(glctx, name, size, iterations):
    p = H.Patch(size=size)
    getattr(H.src(image(size)), name)(iterations=iterations).out(p.o0, patch=p)
    compare(glctx, p, iterations)


@pytest.mark.parametrize("name", ["rd", "bleed"])
def test_signals_feedback_outputs_and_resize(glctx, name):
    p = H.Patch(size=(47, 31))
    source = H.src(p.o1).blend(H.src(image(p.size)), .3)
    args = {"feed": .055 + H.Time() * .001} if name == "rd" else {"grow": .08 + H.Time() * .01}
    getattr(source, name)(iterations=4, **args).out(p.o1, patch=p)
    # Current-frame output dependency, delay history, and mixed fragment/compute.
    H.src(p.o1).blend(H.delay(p.o1, 1), .2).out(p.o0, patch=p)
    compare(glctx, p, 4, frames=6, resize=(53, 29), atol=.004)


@pytest.mark.parametrize("name", ["rd", "bleed"])
@pytest.mark.parametrize("tail", [
    lambda c: c.rotate(.31).scale(.83),
    lambda c: c.modulate(H.osc(7), .13).color(.7, .9, .2, .6),
    lambda c: c.blur(.8).layer(H.shape(3, .2).color(1, .1, .4, .3)),
])
def test_transformed_and_composited_final_iteration(glctx, name, tail):
    p = H.Patch(size=(47, 31))
    chain = getattr(H.src(image(p.size)), name)(iterations=4)
    tail(chain).out(p.o0, patch=p)
    count = sum(p.compute is not None for p in compile_chain(tail(chain)).passes)
    assert count in (3, 4)
    compare(glctx, p, count)


@pytest.mark.parametrize("iterations", [1, 2, 4])
def test_shared_prefix_survives_ping_pong(glctx, iterations):
    p = H.Patch(size=(47, 31))
    base = H.src(image(p.size)).bake()
    base.bleed(iterations=1).blend(base.rd(iterations=iterations), .4).out(p.o0, patch=p)
    compare(glctx, p, iterations - 1)


@pytest.mark.parametrize("reason", ["version", "compile"])
def test_fallback_renders_the_reference_and_reports_reason(glctx, monkeypatch, reason):
    from hydrapy.backend_gl import GLRenderer
    p = H.Patch(size=(47, 31))
    H.src(image(p.size)).rd().bleed().out(p.o0, patch=p)
    reference = GLRenderer(p, ctx=glctx)
    try:
        reference.step(0)
        expected = reference.read_rgba()
    finally:
        reference.release()
    if reason == "version":
        monkeypatch.setattr(glctx, "version_code", 330)
    else:
        def reject(source):
            raise RuntimeError("test driver rejection")
        monkeypatch.setattr(glctx, "compute_shader", reject)
    with pytest.warns(RuntimeWarning, match="using fragment shaders"):
        r = GLRenderer(p, ctx=glctx, compute=True)
    try:
        assert r.compute_passes == 0 and r.compute_fallbacks
        r.step(0)
        np.testing.assert_array_equal(r.read_rgba(), expected)
    finally:
        r.release()


def test_cli_compute_selects_gpu_without_gpu_flag(monkeypatch, tmp_path):
    from hydrapy import backend_gl, cli, export
    called = {}
    patch = tmp_path / "compute.py"
    patch.write_text("from hydrapy import *\nosc().rd().out(o0)\n")
    def frames(patch, **kwargs):
        called.update(kwargs)
        return iter([np.zeros((2, 2, 3), np.uint8)])
    monkeypatch.setattr(backend_gl, "iter_frames_gl", frames)
    monkeypatch.setattr(export, "save_png", lambda *args, **kwargs: None)
    cli.main(["render", str(patch), "--compute", "--out", str(tmp_path / "frame.png")])
    assert called["compute"] is True
