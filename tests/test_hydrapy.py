"""Run with:  python -m pytest tests -q
GPU parity tests are skipped unless moderngl can create a context."""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydrapy as H  # noqa: E402
from hydrapy import *  # noqa: E402,F401,F403
from hydrapy.backend_np import Renderer  # noqa: E402
from hydrapy.glsl import compile_chain, compile_patch  # noqa: E402

SIZE = (96, 54)


def _gl_available():
    try:
        from hydrapy.backend_gl import _create_context
        _create_context()
        return True
    except Exception:
        return False


def every_op_chain():
    base = osc(8, 0.1, 0.6)
    other = noise(3, 0.2)
    for name, op in REGISTRY.items():
        if op.kind == "src":
            if name in ("src", "prev", "delay"):
                continue
            yield name, getattr(H, name)()
        elif op.kind in ("coord", "color", "texture"):
            yield name, getattr(base, name)()
        else:
            yield name, getattr(base, name)(other)


@pytest.mark.parametrize("name,chain", list(every_op_chain()))
def test_every_op_renders_on_cpu(name, chain):
    frame = chain.render(1.3, size=SIZE)
    assert frame.shape == (SIZE[1], SIZE[0], 4)
    assert np.isfinite(frame).all()


@pytest.mark.parametrize("name,chain", [(n, c) for n, c in every_op_chain() if not REGISTRY[n].cpu_only])
def test_every_op_compiles_to_glsl(name, chain):
    prog = compile_chain(chain)
    assert "void main" in prog.source
    assert name + "(" in prog.source


def test_cpu_only_op_refuses_glsl():
    with pytest.raises(NotImplementedError):
        compile_chain(osc().pixelsort())


def test_chain_must_start_with_source():
    with pytest.raises(ValueError):
        Chain().rotate(1).render()


def test_hydra_nesting_order_matches_hydra():
    src_ = osc(10).rotate(0.3).scale(2).color(1, 0, 0).glsl()
    assert "color(osc(rotate(scale(st, 2.0, 1.0, 1.0, 0.5, 0.5), 0.3, 0.0), 10.0, 0.1, 0.0), 1.0, 0.0, 0.0, 1.0)" in src_


def test_texture_ops_split_passes():
    prog = compile_chain(osc().blur(2, iterations=3).edge())
    targets = [p.target for p in prog.passes]
    assert targets[-1] == "o0"
    assert len(targets) == 5  # prefix, 2 extra blur iterations, edge prefix, final
    assert all(t.startswith("tmp") for t in targets[:-1])


def test_feedback_accumulates():
    p = Patch(size=SIZE)
    src(o0).blend(solid(1, 1, 1), 0.1).out(o0, patch=p)
    r = Renderer(p, size=SIZE)
    means = [r.step(i / 30).mean() for i in range(5)]
    assert all(b > a for a, b in zip(means, means[1:]))
    assert abs(means[0] - 0.1) < 1e-3  # 0.9 * black + 0.1 * white


def test_per_pixel_parameter():
    a = osc(noise(3).r() * 20 + 5).render(0.5, size=SIZE)
    b = osc(15).render(0.5, size=SIZE)
    assert not np.allclose(a, b)
    assert "(noise(" in osc(noise(3)).glsl()


def test_python_operators():
    c = osc() * noise() + shape() | gradient() ^ rings()   # ^ binds tighter than | in Python
    names = [t.op.name for t in c.transforms]
    assert names == ["osc", "mult", "add", "layer"]
    assert [t.op.name for t in c.transforms[-1].args[0].transforms] == ["gradient", "diff"]


def test_signals_vs_js_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    from hydrapy.signals import JS_HELPERS
    sigs = {
        "lfo": Lfo(0.7, "sine", 2, 5, 0.1), "seq": Seq([1, 2, 3, 5]).fast(3).smooth().ease("easeInOutCubic"),
        "env": Env(Clock(1), 0.1, 0.4, 2), "sh": SH(5, 1, 3, 2), "drift": Drift(1.5, 0, 2, 1),
        "expr": (Lfo(1) * 2 + SH(3)) ** 2 % 1.5 - Drift(2) / 3, "fn1": Lfo(0.3).sin().abs().clip(0.1, 0.9).quantize(4),
    }
    times = [0, 0.37, 1.0, 2.71, 5.5, 13.3]
    js = JS_HELPERS + "\nvar bpm=30, __mx=0, __my=0, __fft=[0,0,0,0], __vol=0, __flux=0;\nvar out={};\n"
    for name, s in sigs.items():
        js += f"out[{name!r}] = [{','.join(str(t) for t in times)}].map(function(time){{ return {s.js()}; }});\n"
    js += "console.log(JSON.stringify(out));"
    res = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout)
    for name, s in sigs.items():
        py = [s(Ctx(time=t, bpm=30)) for t in times]
        assert max(abs(a - b) for a, b in zip(py, res[name])) < 1e-9, name


def test_seq_matches_hydra_array_semantics():
    s = Seq([1, 2, 3])
    # bpm 30 -> 0.5 steps per second: 2 s per value
    assert s(Ctx(time=0.0)) == 1 and s(Ctx(time=1.99)) == 1 and s(Ctx(time=2.0)) == 2 and s(Ctx(time=5.9)) == 3


def test_hydra_import_roundtrip():
    from hydrapy.hydra_import import from_hydra
    sketch = """
    osc(10, 0.1, 1.2).rotate(() => time * 0.1).kaleid([3,4,5].fast(2).smooth())
      .modulate(noise(3).scale(({time}) => 1 + Math.sin(time)*0.5), 0.2).blend(src(o0).scale(1.01), 0.5).out(o0)
    render(o0)
    """
    p = from_hydra(sketch, Patch())
    chain = p.o0.chain
    assert [t.op.name for t in chain.transforms] == ["osc", "rotate", "kaleid", "modulate", "blend"]
    fn = chain.transforms[1].args[0]
    assert abs(fn(Ctx(time=2.0)) - 0.2) < 1e-9
    js = H.patch_to_hydra(p)
    assert "osc(10, 0.1, 1.2).rotate(" in js and "[3,4,5].fast(2).smooth(1)" in js and "render(o0)" in js
    assert "time * 0.1" in js


def test_hydra_export_rejects_new_ops_unless_loose():
    with pytest.raises(ValueError):
        osc().fold().hydra()
    assert "dropped .fold()" in H.chain_to_hydra(osc().fold(), strict=False)


def test_html_export_contains_passes():
    p = Patch(size=SIZE)
    osc(Lfo(1)).blur().out(o0, patch=p)
    html = H.to_html(p)
    assert "<canvas" in html and "tmp0" in html and "__fract" in html and "u_0" in html


def test_audio_signals(tmp_path):
    import wave
    sr = 22050
    t = np.arange(sr) / sr
    tone = (np.sin(2 * np.pi * 80 * t) * (t < 0.5)).astype(np.float32)
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((tone * 32767).astype("<i2").tobytes())
    a = Audio.from_file(str(tmp_path / "a.wav")).set_cutoff(0).set_scale(5)
    assert a.band_at(0.25, 0) > a.band_at(0.75, 0)
    assert a.fft[0](Ctx(time=0.25)) == a.band_at(0.25, 0)


@pytest.mark.skipif(not _gl_available(), reason="no GL context")
@pytest.mark.parametrize("name,chain", [(n, c) for n, c in every_op_chain()
                                         if not REGISTRY[n].cpu_only and n not in ("white", "grain", "glitch")])
def test_cpu_and_gpu_agree(name, chain):
    from hydrapy.backend_gl import GLRenderer
    p = Patch(size=SIZE)
    chain.out(p.o0, patch=p)
    rn, rg = Renderer(p, size=SIZE, fps=30), GLRenderer(p, size=SIZE, fps=30)
    a = np.clip(np.nan_to_num(rn.step(1.3)[..., :3]), 0, 1)
    rg.step(1.3)
    b = rg.read().astype(np.float32) / 255.0
    d = np.abs(a - b)
    assert d.mean() < 0.02 and (d < 0.05).mean() > 0.97, name


# --- paint-on-film, quilting, banks ---------------------------------------------------

def test_stain_is_multiplicative_dye():
    red = solid(1, 0.1, 0.05, 1.0)               # a red dye, thickness 1
    out = solid(1, 1, 1).stain(red).render(0, size=(8, 4))[0, 0]
    assert out[0] > 0.99 and out[1] < 0.1 and out[2] < 0.1
    clear = solid(1, 1, 1).stain(solid(1, 0, 0, 0.0)).render(0, size=(8, 4))[0, 0]
    assert clear[:3].min() > 0.99                # thickness 0 leaves the leader clear
    doubled = solid(1, 1, 1).stain(red).stain(solid(0.1, 0.1, 1, 1.0)).render(0, size=(8, 4))[0, 0]
    assert doubled[:3].max() < 0.15              # red over blue goes black, like ink


def test_alpha_is_straight_not_premultiplied():
    a = solid(1, 0.3, 0.1).alpha(solid(0.5, 0.5, 0.5)).render(0, size=(8, 4))[0, 0]
    assert abs(a[0] - 1.0) < 1e-5 and abs(a[3] - 0.5) < 1e-3


def test_drops_seed_changes_painting_and_pan_moves_it():
    a = drops(seed=1).render(0, size=(64, 36))
    b = drops(seed=2).render(0, size=(64, 36))
    assert not np.allclose(a, b)
    c = drops(seed=1).pan(0, 0, 0, 24).render(1 / 24, size=(64, 36))   # one frame-height later
    assert not np.allclose(a, c)


def test_quilt_is_deterministic_and_sized():
    from hydrapy.synth import quilt
    src_img = veins(3, 4, 2, 0, 1).render(0, size=(96, 64))[..., :3]
    a = quilt(src_img, size=(80, 50), block=16, overlap=4, seed=5, candidates=200)
    b = quilt(src_img, size=(80, 50), block=16, overlap=4, seed=5, candidates=200)
    c = quilt(src_img, size=(80, 50), block=16, overlap=4, seed=6, candidates=200)
    assert a.shape == (50, 80, 3) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert 0.0 <= a.min() and a.max() <= 1.0


def test_bank_index_matches_js_runtime():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    from hydrapy.signals import JS_HELPERS
    from hydrapy.synth import bank_index
    js = JS_HELPERS + """
    function bankIndex(cfg, n, time){ var k = Math.floor(time * cfg.rate / Math.max(cfg.hold, 1));
      if (cfg.mode === 'random') return ((Math.floor(__hash11(k + cfg.seed * 131.0) * n)) % n + n) % n; return ((k % n) + n) % n; }
    var out = []; [0, 0.1, 0.37, 1.0, 2.71, 9.9].forEach(function(t){ out.push(bankIndex({rate: 24, hold: 2, mode: 'random', seed: 3}, 7, t)); });
    console.log(JSON.stringify(out));"""
    res = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout)
    py = [bank_index(t, 7, 24, 2, "random", 3) for t in (0, 0.1, 0.37, 1.0, 2.71, 9.9)]
    assert py == res


def test_recipes_compile_and_render():
    from hydrapy.recipes import film, handpainted, mothlight
    for chain in (film(handpainted(seed=SH(8, 0, 99), bleed=1)), film(handpainted(seed=2, strip=24, leader="black")),
                  film(mothlight(seed=3))):
        frame = chain.render(0.3, size=(64, 36))
        assert np.isfinite(frame).all()
        assert "void main" in compile_chain(chain).source


def test_delay_reads_exact_history_on_both_backends():
    size = (48, 27)
    p = Patch(size=size)
    shape(32, 0.1, 0.01).scroll(Time() * 0.1, 0).out(p.o0, patch=p)
    src(o0).color(1, 0, 0).add(delay(o0, 3).color(0, 1, 0)).out(p.o1, patch=p)
    p.render(p.o1)
    rn = Renderer(p, size=size, fps=30)
    hist = []
    for i in range(8):
        a = rn.step(i / 30)
        hist.append(rn.env.texture(p.o0)[::-1][..., 0].copy())
        if i >= 3:
            assert np.abs(a[..., 1] - hist[i - 3]).max() < 1e-6
    if _gl_available():
        from hydrapy.backend_gl import GLRenderer
        rg = GLRenderer(p, size=size, fps=30)
        rn2 = Renderer(p, size=size, fps=30)
        for i in range(8):
            a = np.clip(rn2.step(i / 30)[..., :3], 0, 1)
            rg.step(i / 30)
            assert np.abs(a - rg.read() / 255.0).mean() < 0.01
        assert len(rg.outs["o0"]["ring"]) == 5


def test_lenient_import_of_a_p5_sketch():
    from hydrapy.hydra_import import from_hydra
    from hydrapy.export import patch_to_python
    sketch = """
    new p5();
    function rnd_btw(min, max) {return fxrand() * (max - min) + min;}
    let hydra = new Hydra ({detectAudio: false, canvas:c})
    //hydra
    src(o0)
    .layer(src(s0))
    .blend(osc())
    .pixelate(150,150)
    .out(o0)
    render(o0)
    function setup() {
        createCanvas(windowWidth, windowHeight, WEBGL);
        s0.init({ src:drawingContext.canvas })
    }
    """
    p = from_hydra(sketch, Patch())
    assert [t.op.name for t in p.o0.chain.transforms] == ["src", "layer", "blend", "pixelate"]
    assert p.import_notes == []
    code = patch_to_python(p)
    ns = {}
    exec(code, ns)                       # the export is valid Python
    assert "External('s0')" in code
    assert np.isfinite(p.o0.chain.render(0.2, size=(32, 18))).all()


def test_fx_is_deterministic():
    a, b = Fx("ooABC"), Fx("ooABC")
    assert [a.rnd_btw(0, 1) for _ in range(5)] == [b.rnd_btw(0, 1) for _ in range(5)]
    assert Fx("ooABC").rnd_int(3, 8) in range(3, 9) and Fx(1).rand() != Fx(2).rand()
