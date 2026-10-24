"""ca(): Life-like rules per pixel, asynchronous updates, spontaneous flips — on both backends."""
import os

import numpy as np
import pytest

import hydrapy as H
from hydrapy import CA_MODES, CA_RULES, Patch, osc, solid, src
from hydrapy.backend_np import Renderer
from hydrapy.ops_automata import rule_masks


def _gl():
    try:
        from hydrapy.backend_gl import _create_context
        return _create_context()
    except Exception:
        return None


CTX = _gl()
SIZE = (16, 12)


def _renderers(build):
    out = [("numpy", Renderer(build(), size=SIZE, fps=10))]
    if CTX is not None:
        from hydrapy.backend_gl import GLRenderer
        out.append(("gl", GLRenderer(build(), size=SIZE, fps=10, ctx=CTX)))
    return out


def _frame(r, t):
    a = r.step(t)
    return a if a is not None and not hasattr(r, "read_rgba") else r.read_rgba()


def _start(cells):
    """A frame with these (x, y) cells alive at t=0, then ``ca`` feeding back on o0."""
    img = np.zeros((SIZE[1], SIZE[0], 4), np.uint8)
    img[..., 3] = 255
    for x, y in cells:
        img[y, x, :3] = 255
    return H.FrameSource(lambda t: img if t < 0.05 else None)


def test_rule_strings_become_masks():
    assert rule_masks("B3/S23") == (1 << 3, (1 << 2) | (1 << 3))
    assert rule_masks("b2/s") == (4, 0)
    assert len(CA_RULES) == 18 and CA_MODES["life"] == 0 and CA_MODES["brians-brain"] == 4
    with pytest.raises(ValueError):
        rule_masks("23/3")


def test_a_blinker_blinks_on_both_backends():
    def build(mode=0):
        p = Patch(size=SIZE)
        start = _start([(7, 6), (8, 6), (9, 6)])
        once = ((0.05 - H.Time()) * 1000).clip(0, 1)
        src(p.o0).ca(mode).blend(src(start), once).out(p.o0, patch=p)
        return p
    for name, r in _renderers(build):
        frames = [(_frame(r, i / 10)[..., 0] > 0.99) for i in range(4)]
        assert frames[0][6, 7:10].all() and frames[0].sum() == 3, name
        assert frames[1][5:8, 8].all() and frames[1].sum() == 3, name           # vertical
        assert frames[2][6, 7:10].all() and frames[2].sum() == 3, name           # horizontal again
        if hasattr(r, "release"):
            r.release()


def test_asynchrony_and_flips():
    def build(p_=1.0, flip=0.0, mode=0):
        def make():
            p = Patch(size=SIZE)
            start = _start([(7, 6), (8, 6), (9, 6)])
            once = ((0.05 - H.Time()) * 1000).clip(0, 1)
            src(p.o0).ca(mode, p_, flip).blend(src(start), once).out(p.o0, patch=p)
            return p
        return make
    for name, r in _renderers(build(p_=0.0)):                  # nobody updates: the blinker is frozen
        frames = [(_frame(r, i / 10)[..., 0] > 0.99) for i in range(4)]
        assert all((f == frames[0]).all() for f in frames), name
    for name, r in _renderers(build(flip=1.0)):                # everyone flips after updating: the inverse
        _frame(r, 0)
        f = _frame(r, 0.1)[..., 0] > 0.99
        assert (~f).sum() == 3 and not f[5:8, 8].any(), name
    for name, r in _renderers(build(p_=0.5, flip=0.0, mode=CA_MODES["seeds"])):
        alive = [int((_frame(r, i / 10)[..., 0] > 0.99).sum()) for i in range(6)]
        assert alive[-1] > 3, name                             # seeds still grows when half the cells wait


def test_generations_rules_leave_a_dying_tail():
    def build():
        p = Patch(size=SIZE)
        start = _start([(8, 6)])
        once = ((0.05 - H.Time()) * 1000).clip(0, 1)
        src(p.o0).ca(CA_MODES["star-wars"]).blend(src(start), once).out(p.o0, patch=p)   # 4 states, S345
        return p
    for name, r in _renderers(build):
        _frame(r, 0)
        a = _frame(r, 0.1)
        assert abs(a[6, 8, 0] - 2 / 3) < 0.01, name             # a lonely cell dies through 2/3 ...
        b = _frame(r, 0.2)
        assert abs(b[6, 8, 0] - 1 / 3) < 0.01, name             # ... and 1/3
        assert _frame(r, 0.3)[6, 8, 0] < 0.01, name               # then it is dead
        if hasattr(r, "release"):
            r.release()


def test_mode_can_be_a_field():
    prog = H.compile_chain(src(H.o0).ca(osc(4).mult(solid(16, 16, 16)), osc(2)))
    assert "_carule(" in prog.source and "osc(" in prog.source


def test_parity_copies_a_pattern_and_symmetric_draws_keep_symmetry():
    size = (32, 24)
    def build(mode, p_=1.0, flip=0.0, sym=1):
        def make():
            img = np.zeros((size[1], size[0], 4), np.uint8); img[..., 3] = 255
            for x, y in ((15, 11), (16, 11), (15, 12), (16, 12), (14, 11), (17, 12)):   # point-symmetric seed
                img[y, x, :3] = 255
            img[11, 14, :3] = 255; img[11, 17, :3] = 255; img[12, 14, :3] = 255; img[12, 17, :3] = 255
            start = H.FrameSource(lambda t: img if t < 0.05 else None)
            p = Patch(size=size)
            once = ((0.05 - H.Time()) * 1000).clip(0, 1)
            src(p.o0).ca(mode, p_, flip, 0.9, 0, sym).blend(src(start), once).out(p.o0, patch=p)
            return p
        return make
    rs = [("numpy", Renderer(build(CA_MODES["parity"])(), size=size, fps=10))]
    for name, r in rs:
        frames = [r.step(i / 10)[..., 0] > 0.99 for i in range(5)]
        assert frames[4].sum() == 9 * frames[0].sum(), name      # after 4 steps: the seed and eight copies
    for mode in (CA_MODES["parity-4"], CA_MODES["brians-brain"], CA_MODES["anneal"]):
        r = Renderer(build(mode, p_=0.6, flip=0.02)(), size=size, fps=10)
        for i in range(12):
            a = r.step(i / 10)[..., 0]
            assert np.array_equal(a, a[::-1, :]) and np.array_equal(a, a[:, ::-1]), (mode, i)


def test_mirrorgrid_gives_mirrored_cells_identical_coordinates():
    n, m = 20, 12
    field = H.noise(7, 0.3).add(H.shape(64, 0.3, 0.2)).mirrorgrid(n, m)
    a = field.render(0.7, size=(n, m))
    assert np.array_equal(a, a[::-1]) and np.array_equal(a, a[:, ::-1])
    if CTX is not None:
        from hydrapy.backend_gl import GLRenderer
        p = Patch(size=(n * 4, m * 4))
        field.out(p.o0, patch=p, scale=0.25)
        r = GLRenderer(p, size=(n * 4, m * 4), fps=10, ctx=CTX)
        r.step(0.7)
        b = r.read_rgba()
        assert b.shape[:2] == (m, n) and np.array_equal(b, b[::-1]) and np.array_equal(b, b[:, ::-1])
        r.release()


def test_patch_41_stays_symmetric(monkeypatch):
    import runpy
    monkeypatch.setenv("UNFOLDING_SIZE", "640x368")           # 160x92 cells: small, and both halves even
    H.reset_patch()
    runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "patches", "41_unfolding.py"))
    p = H.default_patch()
    p.o0.chain = None                                      # the field only; the display isn't under test
    p.render(p.o1)
    r = Renderer(p, size=(640, 368), fps=30)
    for i in range(70):                                    # past the first folds (32, 64)
        r.step(i / 30)
    x = np.round(r.env.texture(p.o1)[..., 0] * 5)          # the state levels of the field itself
    assert x.shape[:2] == (92, 160) and x.sum() > 0
    assert np.array_equal(x, x[::-1]) and np.array_equal(x, x[:, ::-1])
    H.reset_patch()
