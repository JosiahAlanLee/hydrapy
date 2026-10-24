"""ca(): Life-like rules per pixel, asynchronous updates, spontaneous flips — on both backends."""
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
    assert len(CA_RULES) == 16 and CA_MODES["life"] == 0 and CA_MODES["brians-brain"] == 4
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
