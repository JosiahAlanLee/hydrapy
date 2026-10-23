import json
import math
import shutil
import subprocess

import numpy as np
import pytest

import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.export import chain_to_python


@pytest.fixture(scope="module")
def glctx():
    pytest.importorskip("moderngl")
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(str(exc))
    yield ctx
    ctx.release()


def test_normals_use_height_units_and_preserve_alpha():
    for size in ((80, 40), (160, 80), (80, 80)):
        c = H.gradient(0).color(1, 1, 1, .3).normalMap(.1).render(size=size)
        expected = np.array([-.1 * size[1]/size[0], 0, 1])
        expected /= np.linalg.norm(expected)
        assert np.allclose(c[2:-2, 2:-2, :3], .5 + .5*expected, atol=1e-6)
        assert np.allclose(c[..., 3], .3)
    flat = H.solid(.4, .4, .4, .7).normalMap().render(size=(13, 9))
    assert np.allclose(flat, [.5, .5, 1, .7])


def test_material_lighting_and_zero_refraction():
    n = H.solid(.5, .5, 1, .4)
    front = n.relight(elevation=math.pi/2, ambient=.1, diffuse=.7, specular=.2).render()
    back = n.relight(elevation=-math.pi/2, ambient=.1, diffuse=.7, specular=.2).render()
    assert np.allclose(front, [1, 1, 1, .4])
    assert np.allclose(back, [.1, .1, .1, .4], atol=1e-6)
    source = H.osc(7, .1, .5)
    assert np.array_equal(source.render(), source.refractMap(n, 0).render())
    uv = H.gradient(0).envMap(n).render(size=(33, 25))
    assert np.allclose(uv[..., 0], .75)


MATERIAL_CHAINS = [
    H.rings(4).normalMap(.02),
    H.fbm(2).normalMap(H.osc(3)*.03).relight(H.Time(), .6, .1, .6, .3, 24),
    H.osc(4).envMap(H.rings(2).normalMap(.04), H.Time()*.1),
    H.osc(7).refractMap(H.rings(3).normalMap(.03), .04),
    H.sdfMetaballs(6, .12, .3, .15, .7, 4).sdfBands(18, .2),
]


def scene():
    return H.SDF3.torus(.6, .2).rotate(.3, .2, 0).union(
        H.SDF3.sphere(.36).translate(H.Lfo(.2, lo=-.45, hi=.45), .2, 0), .15).subtract(
        H.SDF3.box(.12, .8, .12))


@pytest.mark.parametrize("chain", MATERIAL_CHAINS + [
    H.raymarch(scene(), mode=mode) for mode in (0, 1, 2)
] + [
    H.raymarch(H.SDF3.box(.3, .3, .3).fold(3, angle=.13).scale(.8), steps=128),
    H.raymarch(H.SDF3.box(.5, .5, .5).intersect(H.SDF3.sphere(.6))),
])
def test_demoscene_cpu_gpu_python_and_html(chain, glctx):
    p = H.Patch(size=(80, 56)); chain.out(p.o0, patch=p)
    r = GLRenderer(p, ctx=glctx)
    try:
        for t in (.2, 1.7):
            r.step(t)
            cpu, gpu = chain.render(t, p.size), r.read_rgba()
            assert np.isfinite(cpu).all() and np.isfinite(gpu).all()
            assert np.abs(cpu-gpu).mean() < .004
            assert np.quantile(np.abs(cpu-gpu), .98) < .015
        encoded = chain_to_python(chain)
        decoded = eval(encoded, vars(H))
        assert np.allclose(chain.render(.4, p.size), decoded.render(.4, p.size), atol=1e-5)
        html = H.to_html(p)
        assert "<canvas" in html and "void main" in html
    finally:
        r.release()


def test_metaball_single_circle_and_replay():
    a = H.sdfMetaballs(count=1, radius=.2, spread=0)
    assert np.allclose(a.render(size=(100, 50)), H.sdfCircle(.2).render(size=(100, 50)))
    field = H.sdfMetaballs()
    first = field.render(.5, (64, 40))
    field.render(10, (64, 40))
    assert np.array_equal(first, field.render(.5, (64, 40)))
    assert not np.array_equal(first, field.render(.6, (64, 40)))


def test_sdf_distance_geometry_and_scale():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]])
    assert np.allclose(H.SDF3.sphere(1).distance(points), [-1, 0, 1])
    assert H.SDF3.box(.5, 1, 2).distance([0, 0, 0]) == pytest.approx(-.5)
    torus = H.SDF3.torus(1, .25)
    assert np.allclose(torus.distance([[1, 0, 0], [1, .25, 0], [0, 0, 0]]), [-.25, 0, .75])
    a = H.SDF3.box(.4, .2, .3).rotate(.1, .3, .6)
    assert np.allclose(a.scale(2).distance(points), 2*a.distance(points/2))
    assert np.allclose(a.translate(1, 2, 3).distance(points+[1, 2, 3]), a.distance(points))
    b = H.SDF3.sphere(.3)
    av, bv = a.distance(points), b.distance(points)
    assert np.allclose(a.union(b).distance(points), np.minimum(av, bv))
    assert np.allclose(a.subtract(b).distance(points), np.maximum(av, -bv))
    assert np.allclose(a.intersect(b).distance(points), np.maximum(av, bv))


def test_raymarch_hit_miss_depth_and_bounded_steps():
    sphere = H.SDF3.sphere(.5)
    camera = dict(rotX=0, rotY=0, distance=3, far=5)
    depth = H.raymarch(sphere, mode=2, **camera).render(size=(51, 51))
    assert depth[25, 25, 0] == pytest.approx(2.5/5, abs=.001)
    assert depth[25, 25, 3] == 1 and depth[0, 0, 3] == 0
    normal = H.raymarch(sphere, mode=1, **camera).render(size=(51, 51))
    assert np.allclose(normal[25, 25], [.5, .5, 1, 1], atol=.001)
    assert not H.raymarch(sphere, steps=1, **camera).render(size=(20, 20))[..., 3].any()
    assert not H.raymarch(sphere, far=.1).render(size=(20, 20))[..., 3].any()
    assert np.isfinite(H.raymarch(sphere, distance=0).render(size=(20, 20))).all()


def test_same_sdf_topology_shares_helpers_and_separate_scenes_compile(glctx):
    a, b = H.raymarch(H.SDF3.sphere(.3)), H.raymarch(H.SDF3.sphere(.5))
    chain = a.add(b).add(H.raymarch(H.SDF3.box(.2, .3, .4)))
    source = H.compile_chain(chain).source
    assert source.count("float _sdfMap_") == 2
    p = H.Patch(size=(32, 24)); chain.out(p.o0, patch=p)
    r = GLRenderer(p, ctx=glctx)
    try: r.step(0)
    finally: r.release()


def test_spline_sweep_projection_and_svg():
    p = np.array([[-.5, -.6, 0], [.2, -.2, .3], [-.1, .3, -.2], [.5, .7, 0]])
    saved = p.copy()
    curve = H.catmull_rom(p, samples=5)
    assert np.array_equal(curve[::5], p)
    wire = H.sweep_wire(curve, .1, sides=8, twist=1.2, rings=3)
    for line in wire[:8]:
        assert np.allclose(np.linalg.norm(line-curve, axis=1), .1)
    for ring in wire[8:]: assert np.array_equal(ring[0], ring[-1])
    projected = H.project_wire(wire, aspect=1.5)
    assert all(np.isfinite(line).all() for line in projected)
    assert "<svg" in H.to_svg(projected)
    assert np.array_equal(p, saved)
    closed = H.catmull_rom(p, samples=5, closed=True)
    assert np.array_equal(closed[0], closed[-1])
    lathe = H.lathe_wire([[.2, -.5], [.4, 0], [.2, .5]], sides=8)
    assert len(lathe) == 11


def test_near_plane_clips_and_splits_wire_paths():
    lines = H.project_wire([[[0, 0, 0], [.1, 0, 4], [.2, 0, 0]]], distance=3, near=.1)
    assert len(lines) == 2 and all(len(p) == 2 for p in lines)
    assert H.project_wire([[[0, 0, 4], [.1, 0, 5]]], distance=3) == []


def test_projection_preserves_square_in_svg_and_wide_canvas():
    square = [[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0], [-.5, -.5, 0]]
    svg_path = H.project_wire([square], aspect=1)[0]
    screen_path = H.project_wire([square], aspect=16/9)[0]
    assert np.ptp(svg_path[:, 0]) == pytest.approx(np.ptp(svg_path[:, 1]))
    pixel_span = np.ptp(screen_path, axis=0) * [640, 360]
    assert pixel_span[0] == pytest.approx(pixel_span[1])


@pytest.mark.parametrize("mode,quarter", [("step", 0), ("linear", .25), ("smooth", .15625), ("ramp", .0625)])
def test_track_interpolation_boundaries_and_units(mode, quarter):
    track = H.Track([(1, 0, mode), (5, 1), (8, 2)])
    assert track(H.Ctx(time=2)) == pytest.approx(quarter)
    assert track(H.Ctx(time=-3)) == 0
    assert track(H.Ctx(time=5)) == 1
    assert track(H.Ctx(time=1000)) == 2
    beats = H.Track([(0, 0), (4, 1)], units="beats")
    assert beats(H.Ctx(time=1, bpm=120)) == .5
    assert H.Track([(0, 0), (4, 1)], units="beats", bpm=60)(H.Ctx(time=1, bpm=120)) == .25


def test_track_json_python_html_and_js_parity():
    node = shutil.which("node")
    if not node: pytest.skip("Node not installed")
    track = H.Track([(0, .1, "smooth"), (2, .7, "step"), (5, .5, "ramp"), (8, .9)], units="beats")
    timeline = H.Timeline({"light": track})
    assert H.Timeline.from_json(timeline.to_json()).to_json() == timeline.to_json()
    times = [-1, 0, .3, .999, 1, 2.4999, 2.5, 3, 5]
    script = "var bpm=120; console.log(JSON.stringify(" + json.dumps(times) + ".map(function(time){return " + track.js() + ";})));"
    js = json.loads(subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout)
    assert np.allclose(js, [track(H.Ctx(time=t, bpm=120)) for t in times], atol=1e-12)
    chain = H.osc(track*10+2)
    decoded = eval(chain_to_python(chain), vars(H))
    assert np.allclose(decoded.render(1.2), chain.render(1.2))
    p = H.Patch(); chain.out(p.o0, patch=p)
    assert "k.length" in H.to_html(p)


@pytest.mark.parametrize("factory", [
    lambda: H.Track([]), lambda: H.Track([(1, 0), (1, 1)]),
    lambda: H.Track([(0, float("nan"))]), lambda: H.Track([(0, 1, "bad")]),
    lambda: H.Track([(0, 1)], bpm=0), lambda: H.SDF3.sphere(float("inf")),
    lambda: H.SDF3.box().fold(11), lambda: H.raymarch(H.SDF3.sphere(), unknown=1),
    lambda: H.sweep_wire([[0, 0, 0], [0, 0, 0]]),
    lambda: H.project_wire([[[0, 0, 0], [1, 1, 1]]], near=0),
])
def test_invalid_demoscene_inputs(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()
