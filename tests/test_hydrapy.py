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
    b = np.clip(rg.read_rgba()[..., :3], 0, 1)
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


# --- live I/O (no hardware needed: loopback OSC, synthetic CV blocks, files, probes) -----------------

def test_osc_loopback_values_and_triggers():
    pytest.importorskip("pythonosc")
    import time
    from hydrapy.live import OscIn, OscOut, clear_hooks
    oi = OscIn(port=0)
    try:
        v = oi.value("/synth/freq", default=10, lo=1, hi=100)
        tr = oi.trig("/kick", length=0.5)
        oo = OscOut("127.0.0.1", oi.port)
        oo.send("/synth/freq", Lfo(1, lo=20, hi=40)).send("/kick", 1.0)
        oo.tick(Ctx(time=0.25))
        for _ in range(50):
            if tr.count:
                break
            time.sleep(0.01)
        assert abs(v(Ctx(time=0)) - 40.0) < 1e-6 and tr.count == 1 and 0.0 <= tr(Ctx()) < 1.0
    finally:
        oi.close(); clear_hooks()


def test_link_transport_runs_on_the_beat():
    pytest.importorskip("aalink")
    import time
    from hydrapy.live import Link
    lk = Link(bpm=120)
    try:
        t1 = lk.now(); time.sleep(0.2); t2 = lk.now()
        assert 0.1 < t2 - t1 < 0.5 and abs(lk.bpm - 120) < 1e-6
        p = Patch(bpm=lk.bpm)
        assert 0.0 <= Clock(1)(Ctx(time=lk.now(), bpm=lk.bpm)) < 1.0
    finally:
        lk.close()


def test_cvin_levels_gates_and_audio_from_blocks():
    import time
    from hydrapy.live import CvIn
    cv = CvIn(channels=[0, 1, 2], samplerate=48000, _file="fake")
    trig = cv.trig(1, threshold=1.0, length=0.5)
    aud = cv.audio(2)
    t = np.arange(1024) / 48000
    block = np.zeros((1024, 3), np.float32)
    block[:, 0] = 0.25; block[:, 2] = 0.5 * np.sin(2 * np.pi * 440 * t)
    cv.process(block, now=time.monotonic())
    assert abs(cv.cv[0](Ctx(time=0.1)) - 2.5) < 1e-5 and abs(cv.uni(0)(Ctx(time=0.1)) - 0.25) < 1e-5
    assert cv.gate(1)(Ctx(time=0.1)) == 0.0 and trig(Ctx(time=0.1)) == 1.0
    block[:, 1] = 0.5
    cv.process(block, now=time.monotonic())
    assert cv.gate(1)(Ctx(time=0.2)) == 1.0 and trig(Ctx(time=0.2)) < 0.2
    assert aud.band_at(0.2, 0) > aud.band_at(0.2, 3)        # a 440 Hz tone sits in the lowest band


def test_cvfile_and_cvout_roundtrip(tmp_path):
    import wave
    from hydrapy.live import CvIn, CvOut, Probe, clear_hooks
    sr, n = 48000, 48000
    tt = np.arange(n) / sr
    d = np.zeros((n, 2), np.float32)
    d[:, 0] = np.linspace(-0.5, 0.5, n)
    d[:, 1] = (np.mod(tt, 0.5) < 0.1) * 0.5
    with wave.open(str(tmp_path / "cv.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((d * 32767).astype("<i2").tobytes())
    cf = CvIn.from_file(str(tmp_path / "cv.wav"), volts_per_fs=10)
    assert abs(cf.cv[0](Ctx(time=0.25)) + 2.5) < 0.02 and abs(cf.cv[0](Ctx(time=0.75)) - 2.5) < 0.02
    assert cf.gate(1)(Ctx(time=0.05)) == 1.0 and cf.gate(1)(Ctx(time=0.3)) == 0.0
    assert abs(cf.trig(1, 1.0, 0.25)(Ctx(time=0.6)) - 0.4) < 0.01
    # probe -> CvOut file
    p = Patch(size=(48, 27))
    solid(Lfo(0.5), 0.2, 0.2).out(p.o0, patch=p)
    pr = Probe(p.o0, 0.5, 0.5, 0.2, "r", lo=0, hi=5, patch=p)
    out = CvOut(file=str(tmp_path / "out.wav"), channels=[0], samplerate=sr, fps=30)
    out.send(0, pr)
    r = Renderer(p, size=(48, 27), fps=30)
    for i in range(30):
        r.step(i / 30)
    out.close(); clear_hooks()
    back = CvIn.from_file(str(tmp_path / "out.wav"))
    assert abs(back.cv[0](Ctx(time=0.5)) - 5.0 * Lfo(0.5)(Ctx(time=14 / 30))) < 0.15


# --- entropy locking -------------------------------------------------------------------------

def test_fx_matches_fxhash_sfc32_bit_for_bit():
    # expected values come from the 2021 fxrand() snippet run in node for this hash
    fx = Fx("ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567", legacy=True)
    expected = [0.66925048828125, 0.01876902603544295, 0.2789015178568661, 0.4374880369286984,
                0.4301606831140816, 0.6958099333569407, 0.3829852268099785, 0.7686102406587452]
    assert [fx.rand() for _ in range(8)] == expected


def test_fx_matches_the_fxhash_project_sdk():
    # $fx.rand() from @fxhash/project-sdk 0.0.17 (dist/fxhash.min.js) with ?fxhash=<hash>, run in node
    expected = {
        "ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567":
            [0.048488462110981345, 0.7305173503700644, 0.6664658468216658, 0.490762609988451],
        "0x3f1a5c0e0d6b2a99887766554433221100ffeeddccbbaa998877665544332211":
            [0.8978666930925101, 0.8322722127195448, 0.7598719340749085, 0.7777691385708749],
        "3f1a5c0e0d6b2a99887766554433221100ffeeddccbbaa998877665544332211":
            [0.5148867010138929, 0.4338844979647547, 0.5572817819193006, 0.6722511269617826],
        "tz1VSUr8wwNhLAzempoch5d6hLRiTh8Cjcjb":
            [0.7786013532895595, 0.962840897962451, 0.017749151913449168, 0.9952561378013343],
    }
    for h, vals in expected.items():
        fx = Fx(h)
        assert [fx.rand() for _ in range(4)] == vals, h


def test_entropy_lock_recurs():
    fx = Fx("ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567", lock=0.02)
    vals = [fx.rand() for _ in range(400)]
    assert fx.locks > 0 and vals.count(vals[0]) == fx.locks + 1      # every lock restarts the stream exactly
    el = EntropyLock("ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567", rate=12, lock=0.003, steps=512)
    seq = [el(Ctx(time=i / 12)) for i in range(512)]
    assert len(set(seq)) < 400                                           # a free stream would give ~512 distinct values
    assert "__steptable(" in el.js()


def test_tear_and_shred_remap_coordinates():
    base = osc(10).render(0.0, size=(64, 36))
    torn = osc(10).tear(3.0, 3, 0.2, 0.3).render(0.0, size=(64, 36))
    shredded = voronoi(5).shred(2.0, 4, 0.3, 1.5).render(0.0, size=(64, 36))
    assert not np.allclose(base, torn) and np.isfinite(shredded).all()
    assert "tear(" in osc().tear(EntropyLock("x", 12, 0.01, steps=16)).glsl()


# --- video synthesis, tape, film -----------------------------------------------------------------

def test_psort_fully_sorts_rows_given_enough_passes():
    p = Patch(size=(48, 3))
    white(60, 1).psort(0.0, 0, iterations=50).out(p.o0, patch=p)
    rows = Renderer(p, size=(48, 3), fps=30).step(1.0)[..., 0]
    assert all(np.all(np.diff(r) >= -1e-6) for r in rows)
    # the GLSL gets the 0-based iteration index as a trailing argument, one pass per iteration
    prog = compile_chain(osc(6).psort(0.3, 0, iterations=3))
    assert [ps.expr.rsplit(", ", 1)[-1] for ps in prog.passes[1:]] == ["0.0)", "1.0)", "2.0)"]
    assert "float _pass" in prog.source and "uniform float _pass" not in prog.source


def test_psort_leaves_dark_runs_alone():
    p = Patch(size=(32, 2))
    # a bright run on the left, darkness on the right: only the run may change
    left = gradient(0).r().thresh(0.5, 0.0).invert().mult(white(60, 2).brightness(0.3))
    left.psort(0.2, 0, iterations=40).out(p.o0, patch=p)
    before = left.render(1.0, size=(32, 2))[..., 0]
    after = Renderer(p, size=(32, 2), fps=30).step(1.0)[..., 0]
    assert before[:, 16:].max() == 0.0 and before[:, :16].min() > 0.2
    assert np.allclose(before[:, 16:], after[:, 16:])
    assert np.allclose(np.sort(before[:, :16], axis=1), after[:, :16], atol=1e-5)


def test_bits_is_bitwise_logic_on_quantised_channels():
    a, b = solid(0.6, 0.6, 0.6), solid(0.3, 0.3, 0.3)          # 4 bits: 9 and 5 -> xor 12, and 1, or 13
    x = a.bits(b, 0, 4).render(0, size=(2, 2))[0, 0, 0]
    n = a.bits(b, 1, 4).render(0, size=(2, 2))[0, 0, 0]
    o = a.bits(b, 2, 4).render(0, size=(2, 2))[0, 0, 0]
    assert (round(x * 15), round(n * 15), round(o * 15)) == (12, 1, 13)


def test_ruttetra_draws_lines_with_alpha_and_lifts_them():
    flat = solid(0, 0, 0).ruttetra(20, 0.3, 1.0, 0.0).render(0, size=(64, 64))
    assert 0.2 < flat[..., 3].mean() < 0.6 and flat[..., :3].max() < 1e-6      # lines but no colour on black
    lit = solid(1, 1, 1).ruttetra(20, 0.3, 1.0, 0.0).render(0, size=(64, 64))
    rows_flat = np.where(flat[..., 3].mean(1) > 0.3)[0]
    rows_lit = np.where(lit[..., 3].mean(1) > 0.3)[0]
    assert rows_lit.mean() < rows_flat.mean() - 5                            # brightness lifts the raster (row 0 = top)


def test_contours_sinefold_filmstock_gate_shapes():
    ramp = gradient(0).r()                                                   # a grey ramp 0..1 left to right
    c = ramp.contours(4, 0.1, 0.0).render(0, size=(128, 4))[0, :, 0]
    assert 3 <= (np.diff((c > 0.5).astype(int)) > 0).sum() <= 5                # isolines where luminance crosses levels
    s = ramp.sinefold(2, 0).render(0, size=(128, 4))[0, :, 0]
    assert s.min() < 0.05 and s.max() > 0.95 and (np.diff(np.sign(np.diff(s))) != 0).sum() >= 3
    trix = osc(5).filmstock(3).render(0, size=(16, 4))
    assert np.allclose(trix[..., 0], trix[..., 1]) and np.allclose(trix[..., 1], trix[..., 2])
    g = gate(0.5, 0.5, 0.05, 0.0).render(0, size=(64, 36))[..., 3]
    assert g[18, 32] == 1.0 and g[0, 0] == 0.0 and 0.2 < g.mean() < 0.3


def test_tape_ops_keep_the_picture_and_add_defects():
    pic = osc(6, 0.1, 1.2).rotate(0.4)
    base = pic.render(1.0, size=(96, 54))[..., :3]
    bled = pic.yiq(0.05, 2.0, 0.01).render(1.0, size=(96, 54))[..., :3]
    assert abs(bled.mean() - base.mean()) < 0.03 and not np.allclose(bled, base)
    jit = pic.linejitter(0.1, 30, 0.0, 1).render(1.0, size=(96, 54))[..., :3]
    assert np.abs(jit - base).mean() > 0.01 and np.abs(jit.mean(1) - base.mean(1)).max() < 0.08  # rows keep their content
    hs = pic.headswitch(0.1, 0.3, 30, 1).render(1.0, size=(96, 54))[..., :3]
    assert np.allclose(hs[:-8], base[:-8]) and not np.allclose(hs[-4:], base[-4:])                 # only the bottom band tears (row 0 = top)
    d = dropouts(2.0, 0.5, 30, 1).render(1.0, size=(96, 54))
    assert 0 < (d[..., 3] > 0).mean() < 0.2 and np.allclose(d[..., 0], d[..., 3])


def test_frame_rate_holds_time_on_both_backends_and_in_html():
    from hydrapy.export import build_spec
    p = Patch(size=(16, 8))
    p.frame_rate = 4
    solid(Time(), 0, 0).out(p.o0, patch=p)
    rn = Renderer(p, size=(16, 8), fps=30)
    assert abs(rn.step(0.3)[0, 0, 0] - 0.25) < 1e-6 and abs(rn.step(0.49)[0, 0, 0] - 0.25) < 1e-6
    assert build_spec(p)["frameRate"] == 4
    if _gl_available():
        from hydrapy.backend_gl import GLRenderer
        rg = GLRenderer(p, size=(16, 8), fps=30)
        rg.step(0.3)
        assert abs(rg.read()[0, 0, 0] / 255.0 - 0.25) < 0.01
        rg.release()


def test_vhs_and_super8_recipes_render_on_both_backends():
    from hydrapy.recipes import super8, vhs
    pic = osc(5, 0.1, 1.1).kaleid(4)
    for chain in (vhs(pic), vhs(pic, generation=3), super8(pic, hold=False), super8(pic, stock=3, hold=False)):
        frame = chain.render(1.0, size=SIZE)
        assert np.isfinite(frame).all() and 0.05 < frame[..., :3].mean() < 0.95
        assert "void main" in compile_chain(chain).source
    assert default_patch().frame_rate is None                                   # hold=False left the patch alone


def test_python_export_round_trips_native_signals_and_frame_rate():
    from hydrapy.export import patch_to_python
    from hydrapy.recipes import super8
    reset_patch()
    super8(osc(5).kaleid(4), stock=2).out(o0)
    osc(20).rotate((Lfo(0.5) * 2 + SH(3)).sin() * 3).kaleid(Lfo(0.2, lo=2, hi=8).quantize(1)).color(Env(Clock(2), 0.02), 1, Mouse("x")).out(o1)
    py = patch_to_python()
    before = [Renderer(default_patch(), size=(32, 18), fps=30).step(t) for t in (0.3, 1.9)]
    reset_patch()
    exec(py, {})
    assert default_patch().frame_rate == 18 and "Const(0)" not in py
    after = [Renderer(default_patch(), size=(32, 18), fps=30).step(t) for t in (0.3, 1.9)]
    assert all(np.array_equal(np.nan_to_num(a), np.nan_to_num(b)) for a, b in zip(before, after))
    reset_patch()


# --- strokes, paper, print, drawn sound ---------------------------------------------------------

def test_strokes_are_white_with_alpha_and_follow_their_curves():
    circle = arc(0.3, 0, 6.2832, 0.02, 0, 0).render(0, size=(128, 128))
    r = np.hypot(*np.meshgrid((np.arange(128) + 0.5) / 128 - 0.5, (np.arange(128) + 0.5) / 128 - 0.5))
    on = circle[..., 3] > 0.5
    assert np.allclose(circle[..., :3], 1) and 0.26 < r[on].min() and r[on].max() < 0.34
    lis = harmonograph(1, 1, 0, 0, 1.5708, 0, 1, 0.3, 0.02).render(0, size=(128, 128))[..., 3]   # phase pi/2: a circle
    assert np.abs(lis - circle[..., 3]).mean() < 0.02
    w = wave(2, 0.2, 0.02, 0, 0, 1, 0).render(0, size=(128, 72))[..., 3]
    cols = np.where(w.max(0) > 0.5)[0]
    assert cols.min() <= 1 and cols.max() >= 126 and 0.01 < (w > 0.5).mean() < 0.3    # spans the width, is a line
    tapered = arc(0.3, 0, 3.0, 0.03, 1.0, 0).render(0, size=(128, 128))[..., 3]
    head, tail = tapered[:, :64].sum(), tapered[:, 64:].sum()                         # the arc ends on the left (3 rad)
    assert head > tail * 1.5                                                            # the head (end of the arc) is heavier


def test_dab_charcoal_smear_halftone_boil_polar_repeat():
    d = dab(0.3, 0.5, 0.0, 0.0, 0.3).render(0, size=(96, 96))[..., 3]
    assert d[48, 48] > 0.95 and d[48, 5] == 0 and d.mean() < 0.4
    stroke = arc(0.3, 0, 6.2832, 0.03, 0, 0)
    plain, paper = stroke.render(0, size=(96, 96))[..., 3], stroke.charcoal(1.0, 2.0, 60).render(0, size=(96, 96))[..., 3]
    assert paper.mean() < plain.mean() and (paper > 0.01).sum() > 0                    # tooth takes charcoal away
    bar = shape(4, 0.3, 0.002).scale(1, 0.1, 1)                                            # a thin vertical bar at the centre
    plain, sm = bar.render(0, size=(96, 54))[..., 0], bar.smear(0.0, 0.3).render(0, size=(96, 54))[..., 0]
    cols_plain, cols_sm = np.where(plain.sum(0) > 0.01)[0], np.where(sm.sum(0) > 0.01)[0]
    assert abs(sm.sum() - plain.sum()) < 0.5 and len(cols_sm) > len(cols_plain) * 3        # a normalised drag: same ink, spread out
    assert cols_sm.min() >= cols_plain.min() - 1 and cols_sm.max() > cols_plain.max() + 10  # to the right only
    pic = osc(6, 0.1, 1.2).rotate(0.4)
    base = pic.render(1.0, size=(96, 54))[..., :3]
    ht = solid(0.5, 0.5, 0.5).halftone(8, 0.0, 1.0).render(0, size=(96, 96))[..., 0]
    assert ((ht < 0.05) | (ht > 0.95)).mean() > 0.7 and 0.25 < ht.mean() < 0.75             # dots: ink or paper, soft edges
    assert solid(1, 1, 1).halftone(8).render(0, size=(32, 32))[..., :3].min() > 0.99      # white stays paper
    b = pic.boil(0.05, 12, 4).render(1.0, size=(96, 54))[..., :3]
    assert np.abs(b - base).mean() > 0.01 and abs(b.mean() - base.mean()) < 0.03
    sq = shape(4, 0.1, 0.002).scroll(-0.3, 0).polarRepeat(5).render(0, size=(96, 96))[..., 0]
    ang = np.arctan2(*np.meshgrid((np.arange(96) + 0.5) / 96 - 0.5, (np.arange(96) + 0.5) / 96 - 0.5)[::-1])
    hist = np.histogram(ang[sq > 0.5], bins=10, range=(-np.pi, np.pi))[0]
    assert (hist > 0).sum() >= 5                                                          # five copies around the centre


def test_soundtrack_reads_the_strip_as_audio():
    import math
    p = Patch(size=(160, 90))
    osc(10 * 2 * math.pi, 0, 0).rotate(1.5708).out(p.o0, patch=p)                        # 10 stripes per frame -> 240 Hz at 24 fps
    tr = Soundtrack(p.o0, x=0.5, width=0.02, fps=24, patch=p)
    r = Renderer(p, size=(160, 90), fps=24)
    for i in range(24):
        r.step(i / 24)
    s = tr.samples
    assert len(s) == 44100 and tr.duration == 1.0
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), 1 / tr.sr)
    assert abs(freqs[spec.argmax()] - 240) < 2 and abs(s.mean()) < 0.02                  # the drawn pitch, DC removed
    assert 0.2 < tr(Ctx(time=1.0)) < 0.8                                                  # still a probe: the strip's mean
    from hydrapy.export import build_spec
    assert build_spec(p)["soundtrack"]["output"] == "o0"


def test_save_mp4_muxes_a_soundtrack(tmp_path):
    import math
    pytest.importorskip("imageio_ffmpeg")
    p = Patch(size=(64, 36))
    osc(4 * 2 * math.pi, 0, 0).rotate(1.5708).out(p.o0, patch=p)
    tr = Soundtrack(p.o0, fps=12, patch=p)
    r = Renderer(p, size=(64, 36), fps=12)
    frames = list(r.frames(12))
    out = save_mp4(frames, str(tmp_path / "t.mp4"), fps=12, audio=tr)
    import imageio_ffmpeg, subprocess
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", out], capture_output=True, text=True).stderr
    assert "Audio: aac" in info and "Video: h264" in info
    tr.wav(str(tmp_path / "t.wav"))
    assert (tmp_path / "t.wav").stat().st_size > 44


def test_studie_and_cutout_recipes_render():
    from hydrapy.recipes import cutout, studie
    reset_patch()
    studie(o0, comets=5, waves=1).out(o0)
    cutout(osc(5).kaleid(3), screen=4, hold=True).out(o1)
    assert default_patch().frame_rate == 12
    r = Renderer(default_patch(), size=SIZE, fps=24)
    for t in (0.0, 0.5, 1.0):
        r.step(t)
    a = r.env.texture(o0); b = r.env.texture(o1)
    assert np.isfinite(a).all() and a[..., :3].max() > 0.3 and a[..., :3].mean() < 0.3    # strokes on black, with trails
    assert np.isfinite(b).all() and 0.1 < b[..., :3].mean() < 0.95
    assert len(compile_patch()) == 2
    reset_patch()


def test_soundtrack_keeps_its_own_clock_and_follows_a_frame_hold():
    import math
    p = Patch(size=(64, 36))
    osc(10 * 2 * math.pi, 0, 0).rotate(1.5708).out(p.o0, patch=p)
    tr = Soundtrack(p.o0, fps=24, patch=p)
    r = Renderer(p, size=(64, 36), fps=48)                     # rendered twice as fast as the track reads
    for i in range(96):
        r.step(i / 48)
    assert len(tr.chunks) == 48 and abs(tr.duration - 2.0) < 0.01
    p2 = Patch(size=(64, 36)); p2.frame_rate = 18                # Super 8 sound ran at 18 frames a second
    osc(10 * 2 * math.pi, 0, 0).rotate(1.5708).out(p2.o0, patch=p2)
    tr2 = Soundtrack(p2.o0, fps=24, patch=p2)
    r2 = Renderer(p2, size=(64, 36), fps=24)
    for i in range(48):
        r2.step(i / 24)
    assert tr2.rate == 18 and len(tr2.chunks) == 36 and abs(tr2.duration - 2.0) < 0.01


def test_cvout_carries_a_soundtrack_at_audio_rate(tmp_path):
    import math
    import wave as wavemod
    from hydrapy.live import AudioQueue, CvOut, clear_hooks
    p = Patch(size=(64, 36))
    osc(10 * 2 * math.pi, 0, 0).rotate(1.5708).out(p.o0, patch=p)
    tr = Soundtrack(p.o0, fps=24, gain=0.5, patch=p)             # ±0.5 FS: ±5 V out of an ES-3
    out = CvOut(file=str(tmp_path / "cv.wav"), channels=(0, 1), fps=24, samplerate=48000)
    out.send(0, tr).send(1, Lfo(0.5, lo=0, hi=5))
    r = Renderer(p, size=(64, 36), fps=24)
    for i in range(48):
        r.step(i / 24)
    out.close(); out.close()                                     # idempotent
    clear_hooks()
    with wavemod.open(str(tmp_path / "cv.wav")) as w:
        sr, data = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()) / 32767.0
    assert sr == 48000 and data.shape == (96000, 2)
    spec = np.abs(np.fft.rfft(data[:, 0] * np.hanning(len(data)))); fr = np.fft.rfftfreq(len(data), 1 / sr); spec[fr < 30] = 0
    assert abs(fr[spec.argmax()] - 240) < 2 and 0.45 < np.abs(data[:, 0]).max() < 0.56      # the drawn tone, resampled, at ±5 V
    assert data[:, 1].min() > -0.01 and abs(data[:, 1].max() - 0.5) < 0.02                  # the CV beside it, 0..5 V
    # the live queue: resamples, pulls across chunk boundaries, holds the last chunk when it runs dry
    tr.listeners.clear()
    q = AudioQueue(tr, 48000)
    q.push(np.linspace(-1, 1, 1837, dtype=np.float32)); q.push(np.full(1837, 0.5, np.float32))
    queued = [c.copy() for c in q._q]
    a, b = q.pull(3000), q.pull(3000)
    assert a[0] == -1 and a[1998] > 0.99 and 0.5 < a[1999] < 1
    assert (a[2001:] == 0.5).all()                           # interpolate through the seam
    expected = np.concatenate([*queued, np.tile(queued[-1], 3)])[:6000]
    np.testing.assert_array_equal(np.concatenate([a, b]), expected)  # hold the whole converted chunk
    q.close()


# --- specialised GL variants: transforms at their identity value are compiled out ----------------

def test_gates_find_identity_parameters_but_not_per_pixel_ones():
    from hydrapy.glsl import gates
    w = Fn(lambda c: 0.0)
    ch = osc(10).blend(noise(3).add(voronoi(4), w), w).modulate(noise(2), 0).mult(shape(4), osc(2))
    found = [(tr.op.name, v) for tr, v, ident in gates(ch)]
    assert ("blend", w) in found and ("add", w) in found and ("modulate", 0) in found
    assert all(name != "mult" for name, _ in found)          # a per-pixel amount is never gated


@pytest.mark.skipif(not _gl_available(), reason="no GL")
def test_gl_variants_skip_faded_branches_and_match_the_full_program():
    from hydrapy.backend_gl import GLRenderer
    fade = Fn(lambda c: 0.0 if c.time < 0.5 or c.time > 1.0 else (c.time - 0.5) * 2)
    def build():
        p = Patch(size=(48, 27))
        heavy = noise(4, 0.3).blur(2, 2).color(0.2, 0.9, 0.4)               # a texture op: extra passes
        src(p.o0).scale(1.01).blend(osc(8, 0.1, 1.2).add(heavy, fade), 0.5).out(p.o0, patch=p)
        return p
    fast, full = GLRenderer(build(), size=(48, 27), fps=10), GLRenderer(build(), size=(48, 27), fps=10)
    full.specialise = False
    for i in range(16):
        t = i / 10
        fast.step(t), full.step(t)
        assert np.array_equal(fast.read_rgba(), full.read_rgba()), t
    keys = [k for (name, k) in fast._variants if name == "o0"]
    assert None in keys and any(k for k in keys)               # the full program and a variant without the blur
    skipped = next(v for (name, k), v in fast._variants.items() if name == "o0" and k)
    assert len(skipped) < len(fast._variants[("o0", None)])    # fewer passes while the branch is faded out
    fast.release(); full.release()


@pytest.mark.skipif(not _gl_available(), reason="no OpenGL context")
def test_prewarm_compiles_the_score_before_the_first_frame():
    from hydrapy.backend_gl import GLRenderer
    fade = Fn(lambda c: 0.0 if c.time < 0.5 or c.time > 1.0 else (c.time - 0.5) * 2)
    p = Patch(size=(32, 18))
    osc(8, 0.1, 1.2).add(noise(4, 0.3).blur(2, 2), fade).out(p.o0, patch=p)
    r = GLRenderer(p, size=(32, 18), fps=10)
    assert r.prewarm([i / 10 for i in range(16)]) == 1          # one variant: the blur faded out
    def no_compiles(*a, **k):
        raise AssertionError("compiled mid-render")
    r._build = no_compiles
    for i in range(16):
        r.step(i / 10)
    r.release()
    # a gate on audio analysis can't be predicted, so it's left alone
    q = Patch(size=(32, 18))
    osc(8).add(noise(4), Fft(2)).out(q.o0, patch=q)
    r = GLRenderer(q, size=(32, 18), fps=10)
    assert r.prewarm([0.0, 1.0]) == 0
    r.release()


@pytest.mark.skipif(not _gl_available(), reason="no OpenGL context")
def test_live_settle_skips_compiles_for_values_that_only_touch_zero():
    from hydrapy.backend_gl import GLRenderer
    # zero for one frame at t=0.3, then for good from t=1.0
    amount = Fn(lambda c: 0.0 if abs(c.time - 0.3) < 0.01 or c.time >= 0.95 else 0.5)
    def build():
        p = Patch(size=(32, 18))
        osc(8, 0.1, 1.2).add(noise(4, 0.3).blur(2, 2), amount).out(p.o0, patch=p)
        return p
    live, full = GLRenderer(build(), size=(32, 18), fps=10), GLRenderer(build(), size=(32, 18), fps=10)
    live.settle, full.specialise = 4, False
    for i in range(8):
        live.step(i / 10), full.step(i / 10)
        assert np.array_equal(live.read_rgba(), full.read_rgba()), i
    assert [k for (name, k) in live._variants if name == "o0"] == [None]     # the blip compiled nothing
    for i in range(8, 16):
        live.step(i / 10), full.step(i / 10)
        assert np.array_equal(live.read_rgba(), full.read_rgba()), i
    assert len(live._variants) == 2 and live._active["o0"]                 # the lasting zero did, after 4 frames
    live.release(); full.release()


def test_mipblur_is_smooth_cheap_and_the_same_on_both_backends():
    from hydrapy.glsl import gates
    def base():
        return shape(4, 0.3, 0.01).repeat(3, 3).add(osc(30, 0.1, 1.2).mask(shape(40, 0.2, 0.01)))
    def chain(r):
        return base().mipblur(r)
    prog = compile_chain(chain(6))
    assert [p.mipmap for p in prog.passes] == [True, False]
    assert any(name == "mipblur" and v == 6 for name, v in [(tr.op.name, v) for tr, v, _ in gates(chain(6))])
    sharp = chain(0).render(0.5, size=(64, 36))
    assert np.allclose(sharp, base().render(0.5, size=(64, 36)))           # radius 0: no blur
    soft = chain(12).render(0.5, size=(64, 36))
    assert soft[..., :3].std() < sharp[..., :3].std() * 0.7
    assert "_texLod(_tex" in prog.source and "#define _texLod(s, uv, l) texture2D(s, uv)" in prog.source
    if _gl_available():
        from hydrapy.backend_gl import GLRenderer
        for r in (0.5, 3, 12, 40):
            p = Patch(size=(64, 36)); chain(r).out(p.o0, patch=p)
            g = GLRenderer(p, size=(64, 36), fps=10); g.step(0.5)
            a = np.clip(g.read_rgba(), 0, 1); g.release()
            b = np.clip(Renderer(p, size=(64, 36), fps=10).step(0.5), 0, 1)
            assert np.abs(a - b).max() < 0.01, r


def test_html_export_builds_mip_levels_for_mipblur():
    p = Patch(size=(32, 18))
    osc(10).mipblur(Lfo(0.2, lo=0, hi=20)).out(p.o0, patch=p)
    html = H.to_html(p)
    assert '"mipmap": true' in html and "generateMipmap" in html and "#version 300 es" in html


def test_fast_noise_texture_is_the_same_in_python_and_javascript():
    from hydrapy.noise import NOISE_TEXTURE_JS, noise_texture
    tex = noise_texture()
    assert tex.shape == (256, 256, 4) and tex.dtype == np.uint8
    assert tex[0, 0, 1] == tex[0, 1, 0] and tex[0, 0, 2] == tex[1, 0, 0] and tex[255, 255, 3] == tex[0, 0, 0]
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = NOISE_TEXTURE_JS + "\nprocess.stdout.write(Buffer.from(noiseTexture()).toString('hex'));"
    out = subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout
    assert bytes.fromhex(out) == tex.tobytes()


@pytest.mark.skipif(not _gl_available(), reason="no OpenGL context")
def test_fast_noise_looks_like_noise_and_every_noise_op_compiles():
    from hydrapy.backend_gl import GLRenderer
    shots = {}
    for fast in (False, True):
        p = Patch(size=(96, 54)); p.fast_noise = fast
        noise(4, 0.2).out(p.o0, patch=p)
        g = GLRenderer(p, size=(96, 54), fps=10); g.step(1.0)
        shots[fast] = g.read_rgba()[..., 0]; g.release()
    assert not np.allclose(shots[True], shots[False])
    assert 0.6 < shots[True].std() / shots[False].std() < 1.4
    assert 0.6 < shots[True].mean() / shots[False].mean() < 1.4
    users = [n for n, op in REGISTRY.items() if not op.cpu_only and "_noise(" in (op.glsl + op.glsl_helpers)]
    assert len(users) >= 8
    p = Patch(size=(32, 18)); p.fast_noise = True
    chains = dict(every_op_chain())
    for i, n in enumerate(users[:4]):
        chains[n].out(p.outputs[i], patch=p)
    g = GLRenderer(p, size=(32, 18), fps=10); g.step(0.5)
    assert np.isfinite(g.read_rgba()).all(); g.release()
    for n in users[4:]:
        q = Patch(size=(32, 18)); q.fast_noise = True
        chains[n].out(q.o0, patch=q)
        g = GLRenderer(q, size=(32, 18), fps=10); g.step(0.5); g.release()


def test_fast_noise_travels_with_the_patch():
    p = Patch(size=(32, 18)); p.fast_noise = True
    noise(3).out(p.o0, patch=p)
    html = H.to_html(p)
    assert '"fastNoise": true' in html and "_NOISE_TEX" in html and "function noiseTexture" in html
    from hydrapy.export import patch_to_python
    assert "set_fast_noise(True)" in patch_to_python(p)


def test_repeated_subchains_are_computed_once_per_pass():
    import hydrapy.glsl as G
    def build():
        n = noise(3, 0.2).color(0.3, 0.6, 0.9)
        # the same chain object twice, and an equal chain built separately: one noise call each way
        return osc(10).blend(n, 0.5).add(n.invert(), 0.3).mult(noise(3, 0.2).color(0.3, 0.6, 0.9), 0.4)
    prog = compile_chain(build())
    p = prog.passes[-1]
    assert p.lets and p.source.count("noise(st") + p.source.count("noise(_st") <= 2
    assert p.source.count("vec4 _s0 =") == 1
    G.SHARE_SUBCHAINS = False
    try:
        plain = compile_chain(build())
    finally:
        G.SHARE_SUBCHAINS = True
    assert not plain.passes[-1].lets
    a = build().render(0.4, size=(32, 18))
    if _gl_available():
        from hydrapy.backend_gl import GLRenderer
        outs = []
        for share in (True, False):
            G.SHARE_SUBCHAINS = share
            try:
                q = Patch(size=(32, 18)); build().out(q.o0, patch=q)
                r = GLRenderer(q, size=(32, 18), fps=10); r.step(0.4); outs.append(r.read_rgba()); r.release()
            finally:
                G.SHARE_SUBCHAINS = True
        assert np.array_equal(outs[0], outs[1])
        assert np.abs(outs[0] - a).max() < 0.01


def test_structurally_equal_signals_share_one_uniform():
    l = Lfo(0.3)
    ch = osc(10, l * 0.5 + 0.1).rotate(l * 0.5 + 0.1).kaleid(Lfo(0.3) * 2)
    prog = compile_chain(ch)
    assert len(prog.float_uniforms) == 2        # l*0.5+0.1 once; the second Lfo is another object


# --- per-output resolution -------------------------------------------------------------------

def _scaled_patch():
    p = Patch(size=(64, 36))
    noise(3, 0.2).color(0.9, 0.5, 0.2).blur(1).out(p.o1, patch=p, scale=0.25)
    src(p.o2).scale(1.02).blend(osc(6, 0.1, 1.2), 0.2).out(p.o2, patch=p, scale=0.5)
    src(p.o1).add(src(p.o2), 0.5).out(p.o0, patch=p)
    return p


def test_scaled_outputs_render_smaller_and_upscale_on_numpy():
    p = _scaled_patch()
    assert p.output_size(p.o1, (64, 36)) == (16, 9) and p.output_size("o2", (64, 36)) == (32, 18)
    r = Renderer(p, size=(64, 36), fps=10)
    for i in range(3):
        img = r.step(i / 10)
    assert img.shape == (36, 64, 4) and np.isfinite(img).all()
    assert r.env.buffers[1].shape == (9, 16, 4) and r.env.buffers[2].shape == (18, 32, 4)
    with pytest.raises(ValueError):
        osc().out(p.o3, patch=p, scale=0)
    from hydrapy.export import patch_to_python
    assert "out(o1, scale=0.25)" in patch_to_python(p)


@pytest.mark.skipif(not _gl_available(), reason="no GL")
def test_scaled_outputs_on_gl_match_numpy_and_read_at_their_size():
    from hydrapy.backend_gl import GLRenderer
    rn, rg = Renderer(_scaled_patch(), size=(64, 36), fps=10), GLRenderer(_scaled_patch(), size=(64, 36), fps=10)
    for i in range(4):
        a = rn.step(i / 10)
        rg.step(i / 10)
    assert rg.outs["o1"]["ring"][0]["tex"].size == (16, 9) and rg.outs["o2"]["ring"][0]["tex"].size == (32, 18)
    assert np.abs(np.clip(a[..., :3], 0, 1) - rg.read() / 255.0).mean() < 0.02
    rg.patch.render(rg.patch.o1)
    assert rg.read_rgba().shape == (9, 16, 4) and rg.read().shape == (36, 64, 3)
    rg.release()


def test_html_export_carries_output_scales():
    from hydrapy.export import build_spec
    spec = build_spec(_scaled_patch(), size=(64, 36))
    assert spec["scales"] == {"o1": 0.25, "o2": 0.5}
    from hydrapy.export import RUNTIME_JS
    assert "function osz(n)" in RUNTIME_JS and "bindRef(s.ref, s.delay, o.name)" in RUNTIME_JS


@pytest.mark.skipif(not _gl_available(), reason="no GL")
def test_pipelined_readback_yields_the_same_frames():
    from hydrapy.backend_gl import GLRenderer
    def frames(pipelined):
        p = Patch(size=(40, 24))
        src(p.o0).scale(1.03).blend(osc(7, 0.2, 1.1).rotate(0, 0.3), 0.3).out(p.o0, patch=p)
        r = GLRenderer(p, size=(40, 24), fps=12)
        out = list(r.frames(9, pipelined=pipelined))
        r.release()
        return out
    a, b = frames(False), frames(True)
    assert len(a) == len(b) == 9 and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_frame_source_may_report_an_unchanged_frame():
    calls = []
    def draw(t):
        calls.append(t)
        if t >= 1.0:
            return None
        return np.full((4, 6, 4), int(t * 100), np.uint8)
    fs = FrameSource(draw)
    a = fs.frame_at(0.5)
    v = fs.version
    b = fs.frame_at(1.5)
    assert b is a and fs.version == v and calls == [0.5, 1.5]
    if _gl_available():
        from hydrapy.backend_gl import GLRenderer
        p = Patch(size=(6, 4)); src(fs).out(p.o0, patch=p)
        r = GLRenderer(p, size=(6, 4), fps=10)
        r.step(0.5); first = r.read_rgba().copy()
        r.step(1.5)
        assert np.array_equal(r.read_rgba(), first) and r.media[fs.name]["version"] == fs.version
        r.release()
