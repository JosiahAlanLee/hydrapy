"""hyperkaleid / mobius / hyperfade: the Poincaré disk and band, on both backends."""
import math

import numpy as np
import pytest

import hydrapy as H
from hydrapy import Patch, hyperfade, osc, solid
from hydrapy.backend_np import Env
from hydrapy.ops_hyperbolic import _np_hyperkaleid, _to_disk, _asp, triangle


def _gl():
    try:
        from hydrapy.backend_gl import _create_context
        return _create_context()
    except Exception:
        return None


@pytest.fixture
def gl_context():
    ctx = _gl()
    try:
        yield ctx
    finally:
        if ctx is not None:
            ctx.release()


def _fold(x, y, p, q, env):
    """Fold disk points through the op (shift 0, zoom 1) and return them as disk points again."""
    asp = _asp(env)
    st = np.stack([x / (2 * asp[0]) + 0.5, y / 2 + 0.5], axis=-1).astype(np.float32)
    out = _np_hyperkaleid(st, p, q, 0, 0, 0, 0, 1, env)
    _, _, _, _, rho_b = triangle(p, q)
    return (out[..., 0] - 0.5) * 2 * asp[0] * rho_b, (out[..., 1] - 0.5) * 2 * rho_b


def test_triangle_matches_the_textbook_right_triangle():
    alpha, rho_c, x0, r, rho_b = triangle(3, 100)          # the modular group's (2, 3, inf) triangle
    assert alpha == pytest.approx(math.pi / 3)
    assert rho_c == pytest.approx(math.tan(math.pi / 12))   # C at hyperbolic distance acosh(2 / sqrt 3)
    assert (x0, r) == pytest.approx((2.0, math.sqrt(3)))    # the geodesic |z - 2| = sqrt 3
    assert rho_b == 1.0                                     # B on the boundary
    assert x0 * x0 - r * r == pytest.approx(1.0)            # orthogonal to the unit circle
    alpha, rho_c, x0, r, rho_b = triangle(4, 5)
    assert 0 < rho_c < rho_b < 1 and x0 * x0 - r * r == pytest.approx(1.0)


@pytest.mark.parametrize("p,q", [(4, 5), (3, 7), (7, 3), (3, 100), (5, 4)])
def test_every_point_folds_into_the_fundamental_triangle(p, q):
    env = Env(64, 36)
    rng = np.random.default_rng(1)
    n = 4000
    rad = np.sqrt(rng.random(n)) * 0.999
    ang = rng.random(n) * 2 * math.pi
    x, y = (rad * np.cos(ang)).reshape(1, -1), (rad * np.sin(ang)).reshape(1, -1)
    fx, fy = _fold(x, y, p, q, env)
    alpha, rho_c, x0, r, rho_b = triangle(p, q)
    a = np.arctan2(fy, fx)
    assert (a >= -1e-5).all() and (a <= alpha + 1e-5).all()                 # inside the wedge, always
    done = (np.hypot(fx - x0, fy) >= r - 1e-5) & (np.hypot(fx, fy) <= rho_b + 1e-5)
    assert done.mean() > 0.995                   # the fold cap leaves only sub-pixel points at the boundary unconverged
    near = np.hypot(x, y) < 0.97
    assert done[near].all()


def test_the_tiling_is_invariant_under_the_group_s_reflections():
    env = Env(64, 36)
    rng = np.random.default_rng(2)
    n = 2000
    rad = np.sqrt(rng.random(n)) * 0.98
    ang = rng.random(n) * 2 * math.pi
    x, y = (rad * np.cos(ang)).reshape(1, -1), (rad * np.sin(ang)).reshape(1, -1)
    p, q = 4, 5
    alpha, rho_c, x0, r, rho_b = triangle(p, q)
    fx, fy = _fold(x, y, p, q, env)
    # reflection in the real axis
    gx, gy = _fold(x, -y, p, q, env)
    assert np.allclose(fx, gx, atol=1e-4) and np.allclose(fy, gy, atol=1e-4)
    # reflection in the ray at angle pi/p
    c, s = math.cos(2 * alpha), math.sin(2 * alpha)
    gx, gy = _fold(x * c + y * s, x * s - y * c, p, q, env)
    assert np.allclose(fx, gx, atol=1e-4) and np.allclose(fy, gy, atol=1e-4)
    # inversion in the circle that carries the side BC
    d2 = (x - x0) ** 2 + y ** 2
    ix, iy = x0 + (x - x0) * r * r / d2, y * r * r / d2
    keep = np.hypot(ix, iy) < 0.999
    gx, gy = _fold(ix, iy, p, q, env)
    assert np.allclose(fx[keep], gx[keep], atol=1e-4) and np.allclose(fy[keep], gy[keep], atol=1e-4)


def test_mobius_translations_along_one_geodesic_undo_each_other():
    pic = osc(5, 0, 0.3).add(osc(3, 0, 0.1).rotate(1.1))
    a = pic.mobius(0.7, 0.4).mobius(-0.7, 0.4).render(0, size=(96, 54))
    b = pic.render(0, size=(96, 54))
    inside = np.hypot((np.arange(96)[None, :] + 0.5) / 96 - 0.5, (np.arange(54)[:, None] + 0.5) / 54 - 0.5) < 0.42
    assert np.abs(a - b)[inside].max() < 0.03                     # bilinear resampling twice, inside the disk
    band = pic.mobius(0.7, 0, 0, 1).mobius(-0.7, 0, 0, 1).render(0, size=(96, 54))
    assert np.abs(band - b)[:, 16:-16].max() < 0.03


def test_hyperfade_is_one_in_the_middle_and_zero_past_the_boundary():
    a = hyperfade(0.2).render(0, size=(96, 54))
    assert a[27, 48, 3] == pytest.approx(1.0) and a[27, 48, :3].min() == 1.0
    assert np.allclose(a[..., :3], a[..., 3:4])                    # grey: the fade is in RGB as well as alpha
    assert a[0, 0, 3] == 0.0 and a[27, 0, 3] == 0.0                 # the corners and the sides are outside the disk
    assert 0.0 < a[0, 48, 3] < 0.5 < a[2, 48, 3]                    # the top edge fades, over about .2 of 1 - |z|^2
    band = hyperfade(0.2, 1).render(0, size=(96, 54))
    assert band[27, 2, 3] == pytest.approx(1.0) and band[0, 48, 3] < 0.3 < band[3, 48, 3]   # the band's sides stay, its edges fade


def test_mult_hyperfade_fades_a_hyperkaleid_picture_to_black_at_the_rim(gl_context):
    pic = osc(20, 0.1, 1).hyperkaleid(4, 5)
    h = 160
    size = (h, h)
    plain = pic.render(0.0, size=size)
    faded = pic.mult(hyperfade(0.15)).render(0.0, size=size)
    rad = np.hypot((np.arange(h)[None, :] + 0.5) / h - 0.5, (np.arange(h)[:, None] + 0.5) / h - 0.5) * 2
    middle, rim, outside = rad < 0.4, (rad > 0.95) & (rad < 1.0), rad > 1.0
    assert np.allclose(plain[middle, :3], faded[middle, :3])        # the middle is untouched
    assert plain[rim, :3].mean() > 0.1 and faded[rim, :3].mean() < 0.4 * plain[rim, :3].mean()   # the rim fades
    assert faded[outside, :3].max() == 0.0                          # and outside the disk is black
    if gl_context is None:
        return
    from hydrapy.backend_gl import GLRenderer
    p = Patch(size=size)
    pic.mult(hyperfade(0.15)).out(p.o0, patch=p)
    r = GLRenderer(p, size=size, fps=10, ctx=gl_context)
    try:
        r.step(0.0)
        b = r.read_rgba().astype(np.float32)
        if b.max() > 1.0:
            b = b / 255.0
    finally:
        r.release()
    assert b[rim, :3].mean() < 0.4 * plain[rim, :3].mean()          # the GPU fades the rim too
    assert b[outside, :3].max() < 1e-3


def test_a_hyperkaleid_picture_has_the_symmetry_of_its_group():
    pic = solid(0.9, 0.85, 0.7).layer(H.disc(0.6, 0.6, 0.05, 0.003).color(0.1, 0.2, 0.6)).hyperkaleid(4, 5) \
        .mult(hyperfade(0.1))
    a = pic.render(0, size=(128, 72))
    inside = np.hypot((np.arange(128)[None, :] + 0.5) / 128 * 16 / 9 - 8 / 9, (np.arange(72)[:, None] + 0.5) / 72 - 0.5) < 0.4
    # four-fold about the centre: the frame mirrored in y gives the same picture
    assert np.abs(a - a[::-1])[inside].mean() < 0.02
    assert np.abs(a - a[:, ::-1])[inside].mean() < 0.02


def test_gl_matches_numpy(gl_context):
    if gl_context is None:
        pytest.skip("no GL context")
    from hydrapy.backend_gl import GLRenderer
    pic = osc(4, 0, 0.3).blend(osc(2.5, 0, 0.1).rotate(0.9), 0.5)
    for chain in (pic.hyperkaleid(4, 5, 0.3, 0.5, 0.2).mult(hyperfade(0.15)),
                  pic.hyperkaleid(3, 100, 0, 0, 0, 1).mult(hyperfade(0.15, 1)),
                  pic.mobius(0.6, 0.3, 0.4)):
        w, h = 160, 90
        a = np.clip(chain.render(0.0, size=(w, h)), 0, 1)
        p = Patch(size=(w, h))
        chain.out(p.o0, patch=p)
        r = GLRenderer(p, size=(w, h), fps=10, ctx=gl_context)
        try:
            r.step(0.0)
            b = r.read_rgba().astype(np.float32)
            if b.max() > 1.0:
                b = b / 255.0
        finally:
            r.release()
        assert b.shape[:2] == (h, w)
        diffs = []
        for cand in (a, a[::-1]):
            keep = (cand[..., 3] > 0.5) & (b[..., 3] > 0.5)
            diffs.append(np.abs(cand[..., :3] - b[..., :3])[keep].mean())
        assert min(diffs) < 0.03, diffs
