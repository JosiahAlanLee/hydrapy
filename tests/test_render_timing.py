"""Timing, alpha and I/O regressions exercised through public render paths."""
import json
import math
import shutil
import subprocess
import wave

import numpy as np
import pytest

from hydrapy import (Ctx, Decay, Fn, Patch, Probe, Renderer, Soundtrack, arc,
                     gradient, iter_frames, osc, solid, src)
from hydrapy.backend_gl import GLRenderer, _create_context, iter_frames_gl
from hydrapy.export import AUDIO_HELPERS_JS, RUNTIME_JS, build_spec, patch_to_python, to_image
from hydrapy.live import AudioQueue, CvOut, clear_hooks, register_hook


def gl_context():
    pytest.importorskip("moderngl")
    try:
        return _create_context()
    except Exception as exc:
        pytest.skip(str(exc))


def renderer(backend, patch, fps=60):
    if backend == "numpy":
        return Renderer(patch, fps=fps)
    return GLRenderer(patch, fps=fps, ctx=gl_context())


def read(r, t):
    f = r.step(t)
    return r.read_rgba() if isinstance(r, GLRenderer) else f.copy()


def release(r):
    if isinstance(r, GLRenderer):
        r.release()


@pytest.mark.parametrize("backend", ["numpy", "gl"])
def test_hold_freezes_feedback_but_keeps_presentation_hooks(backend):
    p = Patch(size=(32, 18)); p.frame_rate = 18
    src(p.o0).add(solid(.1, .1, .1)).out(p.o0, patch=p)
    r = renderer(backend, p)
    hooks = []
    register_hook(lambda ctx, r: hooks.append(ctx.time))
    try:
        first = read(r, 0)
        for i in (1, 2, 3):
            assert np.array_equal(read(r, i / 60), first)
        for i in range(4, 60):
            read(r, i / 60)
        assert r.clock.updates == 18
        assert r.frame == len(hooks) == 60
    finally:
        clear_hooks(); release(r)


@pytest.mark.parametrize("backend", ["numpy", "gl"])
@pytest.mark.parametrize("fps,hold", [(24, None), (60, None), (60, 12)])
def test_feedback_half_life_is_in_seconds(backend, fps, hold):
    p = Patch(size=(8, 8)); p.frame_rate = hold
    decay = Decay(.5)
    impulse = Fn(lambda c: 1.0 if c.frame == 0 else 0.0)
    src(p.o0).mult(solid(decay, decay, decay)).add(solid(impulse, impulse, impulse)).out(p.o0, patch=p)
    r = renderer(backend, p, fps)
    try:
        for i in range(2 * fps + 1):
            result = read(r, i / fps)
        assert result[..., 0].mean() == pytest.approx(1 / 16, abs=.003)
    finally:
        release(r)


@pytest.mark.parametrize("backend", ["numpy", "gl"])
def test_stroke_coverage_applied_once_in_display_and_sound(backend):
    p = Patch(size=(96, 64))
    c = arc(.3, 0, 2 * math.pi, .04, 0, 0).charcoal(.3, 0, 60)
    c.out(p.o0, patch=p)
    sound = Soundtrack(p.o0, x=.5, fps=24, ac=False, patch=p)
    r = renderer(backend, p, 24)
    try:
        rgba = read(r, 0)
        assert np.allclose(rgba[..., :3], 1)
        image = np.asarray(to_image(rgba))
        assert np.max(np.abs(image[..., 0] / 255 - rgba[..., 3])) < .005
        if backend == "gl":
            assert np.abs(r.read().astype(float) - image).max() <= 1
        # The hidden white RGB outside the stroke must not become a constant tone.
        assert np.ptp(sound.samples) > .5
        black = solid(0, 0, 0).layer(c).render(size=p.size)
        assert np.allclose(black[..., 0], c.render(size=p.size)[..., 3])
    finally:
        release(r)


def test_source_over_preserves_transparent_colour():
    a = solid(0, 0, 1, .5).over(solid(1, 0, 0, .5)).render(size=(4, 4))[0, 0]
    assert a == pytest.approx([2 / 3, 0, 1 / 3, .75])


@pytest.mark.parametrize("size", [(160, 90), (90, 160)])
def test_polar_repeat_keeps_a_circular_orbit_and_offers_legacy_mode(size):
    w, h = size
    source = arc(.25, .01, .7, .016, .3, 0)
    corrected = source.polarRepeat(7).render(size=size)[..., 3]
    legacy = source.polarRepeat(7, aspect=0).render(size=size)[..., 3]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot((xx + .5 - w / 2) / h, (yy + .5 - h / 2) / h)
    assert np.abs(rad[corrected > .7] - .25).max() < .02
    assert np.abs(corrected - legacy).mean() > .01


@pytest.mark.parametrize("fps,hold", [(12, None), (24, None), (60, None), (24, 18)])
@pytest.mark.parametrize("backend", ["numpy", "gl"])
def test_audio_sample_count_matches_duration_at_different_frame_rates(backend, fps, hold):
    p = Patch(size=(40, 40)); p.frame_rate = hold
    osc(10 * 2 * math.pi, 0, 0).rotate(math.pi / 2).out(p.o0, patch=p)
    sound = Soundtrack(p.o0, fps=24, patch=p)
    r = renderer(backend, p, fps)
    try:
        for i in range(2 * fps):
            r.step(i / fps)
        assert len(sound.samples) == 88200
        assert sound.duration == 2
    finally:
        release(r)


def test_fractional_resampling_is_continuous_across_chunks():
    p = Patch(); sound = Soundtrack(p.o0, fps=29, sr=44100, ac=False, patch=p)
    q = AudioQueue(sound, 48000)
    lengths = []
    original = q.push
    def receive(chunk):
        original(chunk)
        lengths.append(len(q._q[-1]))
    sound.listeners = [receive]
    for i in range(29 * 10):
        sound.update(np.ones((32, 1, 4)), i / 29, 1 / 29)
    assert len(sound.samples) == 441000
    assert sum(lengths) == 480000
    q.close()


@pytest.mark.parametrize("channel", ["r", "g", "b", "a", "luma"])
def test_gpu_reduced_probe_matches_region_mean(channel):
    p = Patch(size=(37, 61))
    gradient().alpha(osc(4, 0, 0)).out(p.o0, patch=p)
    probe = Probe(p.o0, x=.72, y=.12, size=.78, channel=channel, patch=p)
    original = probe.update_mean
    shapes = []
    def capture(rgba, t):
        shapes.append(rgba.shape)
        original(rgba, t)
    probe.update_mean = capture
    cpu = Renderer(p)
    cpu.step(0)
    expected = probe(Ctx())
    shapes.clear()
    gpu = renderer("gl", p)
    try:
        gpu.step(0)
        assert shapes == [(1, 1, 4)]
        assert probe(Ctx()) == pytest.approx(expected, abs=.001)
    finally:
        release(gpu)


def test_browser_audio_holds_whole_buffers_and_exact_sample_counts():
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    script = AUDIO_HELPERS_JS + """
const q = hydrapyAudioQueue(4); q.push(new Float32Array([0,1,2,3]));
const out = [];
for (let i=0;i<3;i++){const b=new Float32Array(6);q.pull(b);out.push(Array.from(b));}
const next = hydrapySampleClock(44100,24);let n=0;
for(let i=0;i<24*3600;i++)n+=next();
console.log(JSON.stringify({out,n}));
"""
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result == {"out": [[0,1,2,3,0,1], [2,3,0,1,2,3], [0,1,2,3,0,1]], "n": 44100 * 3600}
    subprocess.run([node, "--check"], input=RUNTIME_JS, text=True, check=True, capture_output=True)


def test_iterator_export_and_native_signal_round_trip():
    p = Patch(size=(16, 16)); p.composite_alpha = False
    solid(Decay(.7), 0, 0).out(p.o0, patch=p)
    frames = iter_frames(p, seconds=.5, fps=12)
    assert iter(frames) is frames
    assert len(list(frames)) == 6
    code = patch_to_python(p)
    assert "Decay(0.7)" in code and "composite_alpha = False" in code
    assert build_spec(p)["compositeAlpha"] is False


@pytest.mark.parametrize("backend,fps", [("numpy", 12), ("gl", 60)])
def test_cv_file_uses_presentation_rate_and_excludes_warmup(tmp_path, backend, fps):
    if backend == 'gl':
        gl_context().release()
    p = Patch(size=(24, 24)); p.frame_rate = 18
    osc(4 * 2 * math.pi, 0, 0).rotate(math.pi / 2).out(p.o0, patch=p)
    tr = Soundtrack(p.o0, fps=24, patch=p)
    path = tmp_path / 'routes.wav'
    out = CvOut(file=str(path), channels=(0, 1), fps=30, samplerate=48000)
    out.send(0, tr).send(1, 5)
    frames = (iter_frames if backend == 'numpy' else iter_frames_gl)(p, seconds=2, fps=fps, warmup=3)
    try:
        for _ in frames:
            pass
        out.close()
        with wave.open(str(path)) as w:
            assert w.getnframes() == 96000
            data = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, 2) / 32767
        assert np.allclose(data[:, 1], .5, atol=1e-4)
        assert data[:, 0].std() > .1
    finally:
        out.close(); clear_hooks()


def test_browser_holds_feedback_but_keeps_display_refreshing():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    body = 'function frame(now){' + RUNTIME_JS.split('function frame(now){', 1)[1].split('// the optical soundtrack:', 1)[0]
    harness = """
var SPEC={speed:1,frameRate:18,display:'o0',bpm:120}, t0=null, paused=false;
var lastRenderTime=null,curTime=0,audio=null,track=null,W=8,H=8;
var mx=0,my=0,fft=[],vol=0,flux=0,copyProg={},copyLocs={};
var framebuffer=null, draws=0, displays=0, times=[];
var gl={FRAMEBUFFER:1,TRIANGLES:2,TEXTURE0:3,TEXTURE_2D:4,
bindFramebuffer:function(_,fb){framebuffer=fb;},viewport:function(){},useProgram:function(){},
uniform1f:function(){},uniform2f:function(){},uniform1i:function(){},activeTexture:function(){},bindTexture:function(){},
drawArrays:function(){if(framebuffer===null)displays++;else draws++;}};
var requestAnimationFrame=function(){},bindRef=function(){};
var outs={o0:{ring:[{fb:{},tex:{}},{fb:{},tex:{}}]}};
var outputs=[{name:'o0',samplers:[],floats:[{name:'test',fn:function(t){times.push(t);return 1;}}],passes:[{prog:{},target:'o0',locs:{}}]}];
"""
    script = harness + body + '\nfor(var i=0;i<60;i++)frame(i*1000/60);console.log(JSON.stringify({draws,displays,times}));'
    result = json.loads(subprocess.check_output([node, '-e', script], text=True))
    assert result['draws'] == len(result['times']) == 18
    assert result['displays'] == 60
    assert result['times'] == pytest.approx([i / 18 for i in range(18)])


def test_material_recipe_separates_motion_surface_and_paper():
    from hydrapy.recipes import on_paper, studie
    p = Patch(size=(64, 40))
    still = studie(p.o0, motion=0, surface=0, trails=0)
    assert np.array_equal(still.render(0, p.size), still.render(.5, p.size))
    jitter = studie(p.o0, motion=0, surface=.01, trails=0)
    assert np.abs(jitter.render(0, p.size) - jitter.render(.5, p.size)).mean() > .001
    pigment = arc(.15, 0, 4, .02, 0, .3).color(.6, .1, .1)
    a, b = [on_paper(pigment).render(t, p.size) for t in (0, 2)]
    assert np.array_equal(a[:5], b[:5])  # untouched paper stays fixed while pigment moves
    assert np.abs(a-b).mean() > .001


@pytest.mark.parametrize("backend", ["numpy", "gl"])
def test_warmup_clear_can_start_inside_an_already_held_frame(backend):
    if backend == 'gl':
        gl_context().release()
    p = Patch(size=(16, 16)); p.frame_rate = 18
    osc(20).out(p.o0, patch=p)
    track = Soundtrack(p.o0, patch=p)
    frames = (iter_frames if backend == "numpy" else iter_frames_gl)(p, start=1.02, seconds=1, fps=60, warmup=1)
    for _ in frames:
        pass
    assert track.duration == 1.0
