"""Neural cellular automata, vector out (XY), plotter output."""
import math
import os
import sys
import wave
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydrapy import (NCA, XY, Contours, Ctx, Harmonograph, Lissajous, NCAModel, Patch, Paths, Renderer, Rose,  # noqa: E402
                     chain_contours, noise, shape, solid, src, streamlines, to_plotter_svg)
from hydrapy.live import CvOut, LiveValue, clear_hooks  # noqa: E402
from hydrapy.nca import CLAMP, step_np, visible  # noqa: E402
from hydrapy.vector import XYVoice, order_paths, resample, simplify, travel  # noqa: E402

MODELS = os.path.join(os.path.dirname(__file__), "..", "patches", "models")


def _gl():
    try:
        from hydrapy.backend_gl import _create_context
        _create_context().release()
        return True
    except Exception:
        return False


GL = _gl()


# --- NCA -------------------------------------------------------------------------------------

def test_nca_model_roundtrip_and_validation(tmp_path):
    m = NCAModel.random(8, 32, seed=2)
    p = m.save(str(tmp_path / "m.npz"))
    m2 = NCAModel.load(p)
    assert np.array_equal(m.w1, m2.w1) and m2.channels == 8 and m2.hidden == 32 and m2.meta["seed"] == 2
    with pytest.raises(ValueError):
        NCAModel(np.zeros((4, 24)), np.zeros(4), np.zeros((6, 4)))            # 6 channels: not RGBA-packable


def test_nca_step_is_local_stochastic_and_clamped():
    m = NCAModel.random(12, 48, seed=0, scale=50.0)
    st = np.zeros((20, 30, 12), np.float32)
    st[10, 15] = 1.0
    a = step_np(m, st, 0, rate=1.0)
    changed = np.abs(a - st).max(-1) > 0
    ys, xs = np.nonzero(changed & (np.abs(st).max(-1) == 0) & (np.abs(a - step_np(m, np.zeros_like(st), 0, rate=1.0)).max(-1) > 1e-6))
    assert ys.min() >= 9 and ys.max() <= 11 and xs.min() >= 14 and xs.max() <= 16     # only the 3x3 neighbourhood sees the cell
    half = step_np(m, st, 0, rate=0.5)
    frac = (np.abs(half - st).max(-1) > 0).mean() / max((np.abs(a - st).max(-1) > 0).mean(), 1e-9)
    assert 0.35 < frac < 0.65                                                           # about half the cells fire
    assert np.abs(step_np(m, st * 100, 0, rate=1.0)).max() <= CLAMP
    assert np.array_equal(step_np(m, st, 5, 0.5), step_np(m, st, 5, 0.5))             # deterministic per step index


def test_nca_damage_paint_and_clock():
    m = NCAModel.random(12, 32, seed=1)
    st = np.full((10, 10, 12), 0.3, np.float32)
    dmg = np.zeros((10, 10, 4), np.float32); dmg[:5] = 1.0
    out = step_np(m, st, 0, rate=0.0, damage=dmg)                                       # rate 0: only the damage acts
    assert np.all(out[:5] == 0) and np.allclose(out[5:], 0.3)
    paint = np.zeros((10, 10, 4), np.float32); paint[..., 0] = 1.0
    out = step_np(m, st, 0, rate=0.0, paint=paint, paint_amount=1.0)
    assert np.allclose(out[..., 0], 0.5) and np.allclose(out[..., 1], -0.5) and np.allclose(out[..., 3:], 0.3)
    n = NCA(m, size=(8, 8), speed=10, max_steps=4)
    assert [n.steps_due(t) for t in (0.0, 0.05, 0.1, 0.35, 1.0, 1.05)] == [1, 0, 1, 2, 4, 0]
    n.step_index = 7
    assert n.steps_due(0.5) == 1 and n.step_index == 0                                  # time went back: start over


def test_shipped_gravel_rule_grows_and_heals():
    m = NCAModel.load(os.path.join(MODELS, "gravel.npz"))
    st = np.zeros((40, 40, m.channels), np.float32)
    for i in range(160):
        st = step_np(m, st, i)
    lum = visible(st)[..., :3].mean(-1)
    assert lum.std() > 0.08 and np.abs(st).max() < CLAMP                                # texture, not grey, not blown
    st[10:30, 10:30] = 0
    for i in range(160, 260):
        st = step_np(m, st, i)
    healed = visible(st)[10:30, 10:30, :3].mean(-1)
    assert healed.std() > 0.6 * lum.std()


@pytest.mark.skipif(not GL, reason="no GL context")
def test_nca_gpu_matches_numpy_step_for_step():
    from hydrapy.backend_gl import GLRenderer
    m = NCAModel.random(12, 64, seed=4, scale=0.3)
    states = []
    for backend in ("np", "gl"):
        p = Patch(size=(48, 27))
        shape(4, 0.3).out(p.o1, patch=p)
        osc_paint = solid(0.9, 0.2, 0.1).mask(shape(3, 0.2))
        osc_paint.out(p.o2, patch=p)
        n = NCA(m, size=(32, 18), speed=24, damage=p.o1, paint=p.o2, paint_amount=0.3)
        src(n).out(p.o0, patch=p)
        r = Renderer(p, size=(48, 27), fps=24) if backend == "np" else GLRenderer(p, size=(48, 27), fps=24)
        for i in range(4):
            r.step(i / 24)
        states.append(n.state if backend == "np" else r.nca_state(n))
        if backend == "gl":
            r.release()
    assert np.abs(states[0] - states[1]).max() < 1e-4


def test_nca_html_spec_carries_the_rule():
    from hydrapy.export import build_spec
    p = Patch(size=(32, 18))
    src(NCA(NCAModel.random(12, 16, seed=0), size=(16, 9), damage=p.o1)).out(p.o0, patch=p)
    spec = build_spec(p)
    (n,) = spec["ncas"]
    assert n["w"] == 16 and n["textures"] == 3 and len(n["steps"]) == 3 and n["damage"] == "o1"
    assert "_h31" in n["steps"][0] and "gl_FragColor" in n["view"]


def test_nca_trainer_step_matches_runtime():
    pytest.importorskip("torch")
    from hydrapy.nca_train import torch_step
    m = NCAModel.random(12, 32, seed=3, scale=1.0)
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.3, (12, 16, 12)).astype(np.float32)
    s[..., 0] += np.linspace(-0.5, 0.5, 12)[:, None]                                     # orientation matters
    a = torch_step(m, s, rate=1.0)[::-1]                                                 # image rows down; runtime rows up
    b = step_np(m, s[::-1].copy(), 0, rate=1.0)
    assert np.abs(a - b).max() < 1e-4


def test_nca_trainer_runs_a_few_iterations():
    pytest.importorskip("torch")
    from hydrapy.nca_train import train
    img = (np.random.default_rng(0).random((32, 32, 3)) > 0.5).astype(np.float32)
    m = train(img, scale=32, size=16, steps=(4, 6), iters=3, hidden=16, log_every=0)
    assert m.channels == 12 and m.hidden == 16 and np.isfinite(m.w1).all()
    tuned = train(img, scale=32, size=16, steps=(4, 6), iters=1, lr=1e-6, log_every=0, init=m)   # fine-tune from it
    assert tuned.hidden == 16 and np.abs(tuned.w1 - m.w1).max() < 1e-3 and not np.array_equal(tuned.w1, m.w1)


# --- XY --------------------------------------------------------------------------------------

def _run_xy(xy, seconds=2.0, fps=24):
    p = Patch(size=(16, 9))
    solid(0, 0, 0).out(p.o0, patch=p)
    r = Renderer(p, size=(16, 9), fps=fps)
    for i in range(int(seconds * fps)):
        r.step(i / fps)


def _peak(a, sr):
    s = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    return np.fft.rfftfreq(len(a), 1 / sr)[s.argmax()]


def test_xy_beam_draws_the_curve_at_its_rate():
    clear_hooks()
    xy = XY(Lissajous(3, 2, speed=0), rate=80, sr=48000, fps=24, gain=0.5)
    _run_xy(xy)
    x, y, z = xy.x.samples, xy.y.samples, xy.z.samples
    assert len(x) == 96000 and len(y) == 96000                                        # exact sample count
    assert abs(_peak(x, 48000) - 240) < 2 and abs(_peak(y, 48000) - 160) < 2           # 3 x 80 Hz and 2 x 80 Hz
    assert np.abs(x).max() <= 0.5 + 1e-6 and 0.01 < 1 - z.mean() < 0.06               # gain, blanked travel
    drawn = np.abs(np.diff(x))[z[1:] > 0.5]
    assert drawn.max() < 0.05                                                           # continuous across frame joins
    assert xy.stereo().shape == (96000, 2)
    assert xy.dynamic_array()[..., 1].max() > 0.5                                      # the phosphor preview lit up
    xy.close(); clear_hooks()


@pytest.mark.parametrize('hz', [440.0, 659.255, 880.0, 1174.66])
def test_xy_pitch_is_exact_between_whole_sample_periods(hz):
    clear_hooks()
    xy = XY(Lissajous(1, 1, speed=0), rate=hz, sr=48000, fps=60, gain=0.5, jump=0)
    _run_xy(xy, 2.0, fps=60)
    x = xy.x.samples
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))                            # one rising crossing a cycle
    measured = (len(up) - 1) / ((up[-1] - up[0]) / 48000)
    assert abs(1200 * np.log2(measured / hz)) < 1                                # cents; whole-sample periods were ~16 at 880
    xy.close(); clear_hooks()


def test_xy_gain_ramps_across_each_frame():
    clear_hooks()
    level = LiveValue('env', 0.0)
    xy = XY(Lissajous(1, 1, speed=0), rate=100, sr=48000, fps=24, gain=level, jump=0)
    p = Patch(size=(16, 9))
    solid(0, 0, 0).out(p.o0, patch=p)
    r = Renderer(p, size=(16, 9), fps=24)
    for i in range(24):
        level.set(0.5 if i >= 12 else 0.0)
        r.step(i / 24)
    x = xy.x.samples
    assert np.abs(np.diff(x)).max() < 0.02                                       # no step where the envelope opens
    assert np.abs(x[-2000:]).max() > 0.44                                         # size .9 x gain .5
    xy.close(); clear_hooks()


def test_xy_voice_stays_continuous_however_frames_arrive():
    clear_hooks()
    hz = 440.0
    xy = XY(Lissajous(1, 2, phase=1.5708, speed=0), rate=hz, sr=48000, fps=60, gain=0.5, jump=0)
    p = Patch(size=(16, 9))
    solid(0, 0, 0).out(p.o0, patch=p)
    r = Renderer(p, size=(16, 9), fps=60)
    left, right = xy.x.live_source(48000), xy.y.live_source(48000)
    assert left.voice is right.voice
    frame, xs, ys = 0, [], []
    for frames, blocks in [(1, 1), (3, 1), (0, 4), (5, 2), (1, 6), (2, 1)] * 4:   # bursts and stalls
        for _ in range(frames):
            r.step(frame / 60); frame += 1
        for _ in range(blocks):
            xs.append(left.pull(1024)); ys.append(right.pull(1024))
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert len(x) == len(y) == 1024 * 60
    whole = XYVoice(xy, 48000).channels[0].pull(len(x))                              # the same beam, never interrupted
    np.testing.assert_allclose(x, whole, atol=1e-4)                                   # so never a phase jump
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    assert abs(1200 * np.log2((len(up) - 1) / ((up[-1] - up[0]) / 48000) / hz)) < 1
    assert abs(_peak(y, 48000) - 2 * hz) < 2                                         # y an octave up, in step
    xy.close(); clear_hooks()


def test_xy_joins_and_leaves_the_patch_audio_streams():
    clear_hooks()
    p = Patch(size=(16, 9))
    xy = XY(Lissajous(3, 2), patch=p)
    assert p.audio_streams == [xy.x, xy.y] and xy.x.label.endswith(' x')
    xy.close(); clear_hooks()
    assert p.audio_streams == []


def test_xy_routes_to_cvout_file(tmp_path):
    clear_hooks()
    xy = XY(Rose(4, speed=0), rate=50, sr=48000, fps=24, gain=0.4)
    out = CvOut(file=str(tmp_path / "xy.wav"), channels=(0, 1, 2), fps=24, samplerate=48000)
    out.send(0, xy.x).send(1, xy.y).send(2, xy.z)
    _run_xy(xy, 1.0)
    out.close(); xy.close(); clear_hooks()
    with wave.open(str(tmp_path / "xy.wav")) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 3) / 32767
    assert data.shape[0] == 48000 and np.abs(data[:, :2]).max() <= 0.41
    assert set(np.unique(np.round(data[:, 2]))) <= {0.0, 1.0}


def test_traversal_blanks_jumps_between_paths():
    clear_hooks()
    xy = XY([[(-1, -1), (1, -1)], [(-1, 1), (1, 1)]], rate=50, sr=48000)
    tr = xy.traversal(xy.source(Ctx()), 960)
    assert tr.shape == (960, 3)
    dark = tr[tr[:, 2] < 0.5]
    assert len(dark) > 0 and np.all(np.abs(tr[tr[:, 2] > 0.5, 1]) > 0.9)                # lines drawn lit, travel dark
    xy.close(); clear_hooks()


def test_contours_trace_an_output():
    clear_hooks()
    p = Patch(size=(96, 96))
    shape(64, 0.5, 0.01).out(p.o0, patch=p)
    c = Contours(p.o0, levels=[0.5], patch=p)
    Renderer(p, size=(96, 96), fps=24).step(0)
    assert len(c.paths) >= 1
    r = np.hypot(*c.paths[0].T)
    assert 0.4 < r.mean() < 0.6 and r.std() < 0.03                                     # a circle of radius ~0.5


# --- plotter -------------------------------------------------------------------------------

def test_path_utilities():
    p = np.array([[0, 0], [1, 0], [1, 1]], float)
    q = resample(p, 21)
    assert len(q) == 21 and np.allclose(q[10], [1, 0])
    line = np.stack([np.linspace(0, 1, 50), np.zeros(50)], -1)
    assert len(simplify(line, 1e-3)) == 2
    paths = [np.array([[x, 0.0], [x + 0.1, 0.0]]) for x in np.random.default_rng(0).permutation(np.linspace(-1, 1, 20))]
    assert travel(order_paths(paths))[1] < 0.5 * travel(paths)[1]


def test_plotter_svg_and_fields(tmp_path):
    field = noise(1.5, 0.0)
    cs = chain_contours(field, levels=5, size=(120, 80))
    fl = streamlines(field, n=60, steps=30, size=(60, 40))
    assert len(cs) > 3 and len(fl) > 10 and all(np.abs(p).max() <= 1.0 + 1e-9 for p in fl)
    svg = to_plotter_svg(cs, str(tmp_path / "c.svg"), size_mm=(297, 210), margin_mm=10, layers=[cs, fl])
    root = ET.fromstring(svg)
    assert root.get("width") == "297mm" and root.get("viewBox") == "0 0 297 210"
    pts = [tuple(map(float, xy.split(","))) for pl in root.iter("{http://www.w3.org/2000/svg}polyline")
           for xy in pl.get("points").split()]
    xs, ys = zip(*pts)
    assert min(xs) >= 10 - 1e-6 and max(xs) <= 287 + 1e-6 and min(ys) >= 10 - 1e-6 and max(ys) <= 200 + 1e-6
    assert len(list(root)) == 2                                                          # one group per pen
    h = Harmonograph(3, 2, 3.01, 2.005, 0, 0.6, 12, 0.95, 0, points=4000).at(0)
    assert len(h) == 1 and h[0].shape == (4000, 2)
    assert Paths.svg('<svg xmlns="http://www.w3.org/2000/svg"><line x1="0" y1="0" x2="10" y2="10"/></svg>').at(0)[0].shape == (2, 2)


# --- marbling ------------------------------------------------------------------------------

def test_marbling_moves_are_exactly_invertible_and_compose():
    from hydrapy import osc
    base = osc(7, 0, 1.1).rotate(0.6)
    size = (96, 54)
    ref = base.render(0, size=size)
    there_and_back = base.tine(0.4, 0.05, 0.2, 0.03, 0.1).tine(0.4, 0.05, -0.2, 0.03, 0.1).render(0, size=size)
    assert np.abs(there_and_back - ref).mean() < 0.01                     # the comb drawn back un-marbles
    wav = base.ripple(0.3, 0.05, 0.2, 1.0).ripple(0.3, -0.05, 0.2, 1.0).render(0, size=size)
    assert np.abs(wav - ref).mean() < 0.01
    sw = base.swirl(0.5, 0.5, 0.3, 0.2, 0.05).swirl(0.5, 0.5, -0.3, 0.2, 0.05).render(0, size=size)
    assert np.abs(sw - ref).mean() < 0.01
    two = base.inkdrop(0.4, 0.5, 0.1).inkdrop(0.4, 0.5, 0.15).render(0, size=size)
    one = base.inkdrop(0.4, 0.5, math.hypot(0.1, 0.15)).render(0, size=size)
    assert np.abs(two - one).mean() < 0.01                                 # drops add area: r^2 = a^2 + b^2


def test_floating_drops_fill_holes_with_ink_or_water():
    from hydrapy.recipes import floating_drops
    water = solid(0.9, 0.9, 0.9)
    img = floating_drops(water, [(0.5, 0.5, 0.2, (0.0, 0.0, 0.5)), (0.5, 0.5, 0.1, None)]).render(0, size=(90, 90))
    centre, ring, outside = img[45, 45, :3], img[45, 45 + 15, :3], img[2, 2, :3]     # ink pushed to 0.1..0.224
    assert np.allclose(centre, 0.9, atol=0.02) and ring[2] > 0.4 and ring[0] < 0.1 and np.allclose(outside, 0.9)


# --- physarum ------------------------------------------------------------------------------

def test_physarum_agents_follow_trails_and_networks_form():
    from hydrapy import Physarum
    m = Physarum(size=(96, 64), agents=(64, 32), seed=1)
    p = m.values(Ctx())
    for _ in range(150):
        m.step_np(p)
    t = m.trail
    assert np.isfinite(t).all() and t.max() > 5 * t.mean()                 # trail concentrated into veins
    assert (t > t.mean()).mean() < 0.4
    s = m.initial_state()
    assert s.shape == (2048, 4) and 0 <= s[:, 0].min() and s[:, 0].max() < 96 and s[:, 1].max() < 64


def test_physarum_food_attracts():
    from hydrapy import Physarum
    food = np.zeros((64, 96), np.float32); food[:, :16] = 1.0
    m = Physarum(size=(96, 64), agents=(64, 32), seed=2)
    p = m.values(Ctx()); p["food_weight"] = 5.0
    for _ in range(120):
        m.step_np(p, food)
    x = m.state[:, 0]
    assert (x < 16).mean() > 0.4                                              # well over the 1/6 a uniform crowd would put there


@pytest.mark.skipif(not GL, reason="no GL context")
def test_physarum_gpu_matches_numpy_statistics():
    from hydrapy import Physarum
    from hydrapy.backend_gl import GLRenderer
    out = {}
    for backend in ("np", "gl"):
        p = Patch(size=(96, 64))
        m = Physarum(size=(96, 64), agents=(64, 32), speed=24, seed=3)
        src(m).out(p.o0, patch=p)
        r = Renderer(p, size=(96, 64), fps=24) if backend == "np" else GLRenderer(p, size=(96, 64), fps=24)
        for i in range(2):
            r.step(i / 24)
        out[backend] = m.gl_state(r)[1] if backend == "gl" else m.trail
        if backend == "gl":
            r.release()
    a, b = out["np"], out["gl"]
    assert abs(a.sum() - b.sum()) < 1e-3 * a.sum()                     # the same mass of trail
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9               # in the same places (hash rounding moves a few agents a cell)


def test_physarum_html_spec():
    from hydrapy import Physarum
    from hydrapy.export import build_spec
    p = Patch(size=(32, 18))
    src(Physarum(size=(32, 18), agents=(8, 8), food=p.o1)).out(p.o0, patch=p)
    (s,) = build_spec(p)["physarum"]
    assert s["aw"] == 8 and s["food"] == "o1" and "gl_VertexID" in s["depositVert"] and s["depositVert"].startswith("#version 300 es")
    assert set(s["params"]) >= {"sensor_angle", "turn", "deposit", "decay"}


# --- beats and flow ------------------------------------------------------------------------

def _drums(bpm, sr=22050, dur=12.0, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.normal(size=int(dur * sr)); period = 60 / bpm
    bt = np.arange(0.7, dur - 0.1, period); n = int(0.08 * sr); env = np.exp(-np.arange(n) / (0.01 * sr))
    for i, b in enumerate(bt):
        k = int(b * sr)
        x[k:k + n] += (1.0 if i % 4 == 0 else 0.7) * env * (rng.normal(size=n) * 0.6 + np.sin(2 * np.pi * 60 * np.arange(n) / sr))
    return x, sr, bt


def test_beats_found_within_milliseconds():
    from hydrapy import Beats
    for bpm in (92, 128):
        x, sr, truth = _drums(bpm)
        b = Beats.from_samples(x, sr, method="native")
        assert abs(b.tempo - bpm) < 1.5
        err = [np.min(np.abs(b.times - t)) for t in truth[2:-2]]
        assert max(err) < 0.03
    x, sr, truth = _drums(150, sr=44100, seed=1)
    b = Beats.from_samples(x, sr, method="native", bpm=150)                      # a given tempo: only the phase is found
    assert max(np.min(np.abs(b.times - t)) for t in truth[2:-2]) < 0.03
    assert len(b.double().times) == 2 * len(b.times) - 1 and len(b.half().times) == (len(b.times) + 1) // 2


def test_beat_signals_and_their_javascript():
    from hydrapy import Beats
    from hydrapy.signals import JS_HELPERS
    b = Beats([1.0, 1.5, 2.0, 2.5, 3.0], onset=np.linspace(0, 1, 100).astype(np.float32), onset_fps=25)
    assert b.phase(Ctx(time=1.25)) == pytest.approx(0.5) and b.phase(Ctx(time=3.25)) == pytest.approx(0.5)   # extrapolates past the end
    assert b.bar(4)(Ctx(time=2.0)) == pytest.approx(0.5) and 0.9 < b.pulse(0.02, 0.4)(Ctx(time=1.01)) <= 1.0
    import shutil, subprocess, json
    node = shutil.which("node")
    if node:
        js = JS_HELPERS + "\nconsole.log(JSON.stringify([" + ",".join(
            f"(function(time){{return {s.js()};}})({t})" for s, t in ((b.phase, 1.25), (b.bar(4), 2.0), (b.phase, 0.5), (b.onset, 2.0))) + "]))"
        vals = json.loads(subprocess.check_output([node, "-e", js], text=True))
        assert vals[0] == pytest.approx(0.5) and vals[1] == pytest.approx(0.5) and vals[2] == pytest.approx(0.0)
        assert vals[3] == pytest.approx(b.onset(Ctx(time=2.0)), abs=0.02)


def test_optical_flow_measures_motion(tmp_path):
    pytest.importorskip("cv2")
    import imageio.v2 as iio
    from hydrapy import Flow
    from hydrapy.core import Video
    path = str(tmp_path / "m.mp4")
    w = iio.get_writer(path, fps=24, codec="libx264", quality=8, macro_block_size=None, pixelformat="yuv420p")
    tex = (np.random.default_rng(0).random((40, 40, 3)) * 255).astype(np.uint8)
    for i in range(12):
        f = np.full((120, 160, 3), 40, np.uint8)
        f[40 - i:80 - i, 20 + 3 * i:60 + 3 * i] = tex                              # right 3 px, up 1 px per frame
        w.append_data(f)
    w.close()
    a = Flow(Video(path), width=160).dynamic_array(0.3)
    moving = a[a[..., 2] > 0.05]
    assert len(moving) > 500
    assert abs((moving[:, 0] - 0.5).mean() * 160 - 3) < 0.6 and abs((moving[:, 1] - 0.5).mean() * 120 - 1) < 0.4


# --- live plumbing -------------------------------------------------------------------------

@pytest.mark.skipif(not GL, reason="no GL context")
def test_live_reload_carries_feedback_over_and_survives_a_broken_edit():
    from hydrapy import Patch as P
    from hydrapy.backend_gl import GLRenderer, _create_context, _reload
    ctx = _create_context()
    a = P(size=(32, 18))
    src(a.o0).add(solid(0.1, 0, 0)).out(a.o0, patch=a)
    r = GLRenderer(a, size=(32, 18), ctx=ctx, fps=24)
    for i in range(5):
        r.step()
    red = r.read_rgba()[..., 0].mean()

    def good():
        b = P(size=(32, 18))
        src(b.o0).add(solid(0, 0.1, 0)).out(b.o0, patch=b)
        return b

    def bad():
        raise SyntaxError("half-typed patch")

    same = _reload(r, bad, ctx, 24)
    assert same is r                                                   # the old patch keeps running
    r2 = _reload(r, good, ctx, 24)
    assert r2 is not r and r2.clock.calls == 5
    for i in range(3):
        r2.step()
    img = r2.read_rgba()
    assert abs(img[..., 0].mean() - red) < 1e-3 and abs(img[..., 1].mean() - 0.3) < 0.01   # red carried over, green added
    r2.release()


def test_video_out_streams_to_ffmpeg(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as iio
    from hydrapy import osc
    from hydrapy.live import VideoOut, close_resources
    p = Patch(size=(64, 36))
    osc(6).out(p.o0, patch=p)
    out = VideoOut(str(tmp_path / "live.mkv"), fps=24)
    r = Renderer(p, size=(64, 36), fps=24)
    for i in range(24):
        r.step(i / 24)
    close_resources()                                                 # closes the stream (and anything else the patch opened)
    rd = iio.get_reader(str(tmp_path / "live.mkv"))
    n = sum(1 for _ in rd)
    assert n == 24 and out.frames == 24
