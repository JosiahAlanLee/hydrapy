"""Sample-clock regressions; simulated time, with no audio device or wall-clock waits."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import wave

import numpy as np
import pytest

from hydrapy import Fn, Patch, Renderer, Soundtrack, save_mp4, solid
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.live import AudioQueue, CvOut, clear_hooks


def make_renderer(backend, patch, fps):
    if backend == "numpy":
        return Renderer(patch, fps=fps)
    pytest.importorskip("moderngl")
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(str(exc))
    return GLRenderer(patch, fps=fps, ctx=ctx)


def release(renderer):
    if isinstance(renderer, GLRenderer):
        renderer.release()
        renderer.ctx.release()


@pytest.mark.parametrize("sr,device_sr", [(44100, 48000), (48000, 44100), (44100, 44100)])
def test_resampling_does_not_depend_on_chunk_boundaries(sr, device_sr):
    # A continuous tone split into awkward sizes, including single-sample chunks.
    samples = np.sin(2 * np.pi * 997 * np.arange(sr) / sr).astype(np.float32)

    def convert(sizes):
        track = Soundtrack(sr=sr, patch=Patch())
        queue = AudioQueue(track, device_sr)
        result = []
        start = 0
        try:
            for size in sizes:
                queue.push(samples[start:start + size])
                start += size
                # Drain every push: the live queue deliberately drops excess backlog.
                n = sum(map(len, queue._q))
                if n:
                    result.append(queue.pull(n))
            return np.concatenate(result)
        finally:
            queue.close()

    sizes = [1, 1, 1, 1837, 1838, 17, 256, 1999, 2]
    sizes += [sr - sum(sizes)]
    whole, split = convert([sr]), convert(sizes)
    assert len(whole) == len(split) == device_sr
    np.testing.assert_allclose(split, whole, atol=2e-7, rtol=0)
    # Linear interpolation of a 997 Hz tone remains close to the same tone at
    # the destination clock (a fixed sub-sample causal delay is permitted).
    spectrum = np.abs(np.fft.rfft(split * np.hanning(len(split))))
    assert np.argmax(spectrum) == 997


@pytest.mark.parametrize("fps", [18, 24, 30])
@pytest.mark.parametrize("sr", [44100, 48000])
def test_optical_waveform_uses_uniform_sample_times(fps, sr):
    # Successive pictures are portions of ONE 317 Hz sine, not repeated frames.
    # A tall strip limits pixel interpolation error so fractional-sample phase
    # resets cannot hide behind low image resolution or a broad FFT tolerance.
    track = Soundtrack(fps=fps, sr=sr, ac=False, patch=Patch())
    height = 8192
    for frame in range(2 * fps):
        t = frame / fps + (np.arange(height) + .5) / (height * fps)
        column = .5 + .4 * np.sin(2 * np.pi * 317 * t)
        track._emit(column)
    samples = track.samples
    expected = .8 * np.sin(2 * np.pi * 317 * (np.arange(len(samples)) + .5) / sr)
    assert len(samples) == 2 * sr
    assert np.sqrt(np.mean((samples - expected) ** 2)) < 1e-4
    assert np.max(np.abs(samples - expected)) < .006


@pytest.mark.parametrize("sr,device_sr", [(44100, 48000), (48000, 44100)])
def test_clear_resets_queue_and_resampling_phase(sr, device_sr):
    track = Soundtrack(sr=sr, ac=False, patch=Patch())
    queue = AudioQueue(track, device_sr)
    try:
        track._emit(np.ones(32))
        queue.pull(11)                         # partially consumed pre-clear audio
        track.clear()
        silence = round(device_sr / track.rate)
        assert np.array_equal(queue.pull(silence), np.zeros(silence))
        track._emit(np.zeros(32))
        n = round(len(track.samples) * device_sr / sr)
        assert np.array_equal(queue.pull(n), -np.ones(n))
    finally:
        queue.close()


@pytest.mark.parametrize("sr,device_sr", [(44100, 48000), (48000, 44100)])
def test_cv_file_matches_live_callback_samples(tmp_path, sr, device_sr):
    track = Soundtrack(sr=sr, fps=24, ac=False, patch=Patch())
    path = tmp_path / "cv.wav"
    out = CvOut(file=str(path), channels=(0, 1), samplerate=device_sr)
    out.send(1, track)
    live = []
    try:
        for i in range(48):
            t = i / 24 + (np.arange(1024) + .5) / (24 * 1024)
            track._emit(.5 + .4 * np.sin(2 * np.pi * 317 * t))
            queue = out.audio_routes[1]
            remaining = sum(map(len, queue._q))
            while remaining:
                count = min(remaining, 256)           # real callback block boundaries
                block = np.empty((count, 2), np.float32)
                out._callback(block, count, None, None)
                assert not block[:, 0].any()
                live.append(block[:, 1].copy())
                remaining -= count
        out.close()
        with wave.open(str(path)) as wav:
            assert wav.getnframes() == 2 * device_sr
            samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").reshape(-1, 2)
        assert not samples[:, 0].any()
        np.testing.assert_allclose(samples[:, 1] / 32767, np.concatenate(live), atol=1 / 32767)
    finally:
        out.close()
        clear_hooks()


def test_high_pass_state_survives_optical_boundaries():
    # A constant strip is a single step through a stateful one-pole filter, not
    # a new attack at each optical frame. Exercise variable 1837/1838 lengths.
    track = Soundtrack(fps=24, gain=.5, ac=True, patch=Patch())
    for _ in range(24):
        track._emit(np.ones(32))
    expected = .5 * track._hp_a ** np.arange(track.sr)
    np.testing.assert_allclose(track.samples, expected, atol=1e-7)


def test_live_queue_holds_whole_waveforms_and_bounds_backlog():
    track = Soundtrack(sr=48000, patch=Patch())
    queue = AudioQueue(track, 48000)
    try:
        queue.push(np.array([.1, .2, .3, .4], np.float32))
        assert queue.pull(3) == pytest.approx([.1, .2, .3])
        assert queue.pull(7) == pytest.approx([.4, .1, .2, .3, .4, .1, .2])
        for i in range(20):
            queue.push(np.full(4, i / 20, np.float32))
        assert len(queue._q) == 6
        assert queue.pull(4) == pytest.approx([.7] * 4)
    finally:
        queue.close()
    assert queue.push not in track.listeners


def test_late_renderer_holds_previous_picture_and_rewind_resets_audio():
    track = Soundtrack(fps=24, sr=44100, ac=False, patch=Patch())
    queue = AudioQueue(track, 48000)
    try:
        track.update(np.ones((32, 1, 4)), 0, 1 / 48)
        track.update(np.zeros((32, 1, 4)), .5, 1 / 48)
        assert (track.samples[:22050] == 1).all()  # missing pictures hold white
        assert (track.samples[22050:] == -1).all()
        track.update(np.ones((32, 1, 4)), 0, 1 / 48)
        assert len(track.samples) == 1838          # rewind starts a new recording
        assert (queue.pull(2001) == 1).all()       # no stale black frames
    finally:
        queue.close()


@pytest.mark.parametrize("backend,sr,render_fps,hold", [
    ("numpy", 44100, 24, None), ("gl", 48000, 30, 18),
])
def test_ten_minute_video_audio_sync(tmp_path, backend, sr, render_fps, hold, av_ffmpeg):
    ffmpeg, ffprobe = av_ffmpeg, shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe required for encoded A/V synchronization")
    seconds = 600
    p = Patch(size=(16, 16)); p.frame_rate = hold
    # A simultaneous visual/audio step halfway through the performance verifies
    # timeline alignment as well as container duration. No real-time waiting.
    level = Fn(lambda ctx: .5 if ctx.time < seconds / 2 else .9)
    solid(level, level, level).out(p.o0, patch=p)
    track = Soundtrack(fps=24, sr=sr, ac=False, patch=p)
    r = make_renderer(backend, p, render_fps)
    path = tmp_path / "sync.mp4"
    try:
        save_mp4(r.frames(seconds * render_fps), path, fps=render_fps,
                 audio=track, encoder="software", ffmpeg=ffmpeg)
        assert len(track.samples) == seconds * sr
        # The DC step reaches the PCM exactly at the midpoint, not progressively late.
        assert np.flatnonzero(track.samples > .4)[0] == seconds * sr // 2
        info = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)], text=True))
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        assert int(video["nb_frames"]) == seconds * render_fps
        assert int(audio["sample_rate"]) == sr
        assert float(video["start_time"]) == float(audio["start_time"]) == 0
        assert float(video["duration"]) == seconds
        # MP4/AAC duration metadata may be quantized to milliseconds; decoded AAC
        # has padding, so it must not be used as the source PCM sample count.
        assert abs(float(audio["duration"]) - seconds) <= .001 + 1 / sr
        raw = subprocess.check_output([
            ffmpeg, "-v", "error", "-i", str(path), "-ss", str(seconds / 2 - .1),
            "-t", "0.2", "-vn", "-f", "f32le", "-ac", "1", "pipe:1"])
        decoded = np.frombuffer(raw, "<f4")
        assert abs(np.flatnonzero(decoded > .4)[0] / sr - .1) < .002
        raw_video = subprocess.check_output([
            ffmpeg, "-v", "error", "-ss", str(seconds / 2 - 1 / render_fps),
            "-i", str(path), "-frames:v", "2", "-an", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "pipe:1"])
        pictures = np.frombuffer(raw_video, np.uint8).reshape(2, 16, 16, 3)
        assert pictures[0].mean() == pytest.approx(128, abs=3)
        assert pictures[1].mean() == pytest.approx(230, abs=3)
    finally:
        release(r)


@pytest.mark.parametrize("backend", ["numpy", "gl"])
def test_painted_oscillator_patch_still_records_both_voices(tmp_path, monkeypatch, backend):
    import hydrapy

    def no_input(*args, **kwargs):
        raise RuntimeError("test without an audio interface")

    def output(*args, **kwargs):
        if "file" not in kwargs:
            raise RuntimeError("test without an audio interface")
        return CvOut(*args, **kwargs)

    path = Path(__file__).resolve().parents[1] / "patches/26_painted_oscillator.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hydrapy, "CvIn", no_input)
    monkeypatch.setattr(hydrapy, "CvOut", output)
    p = hydrapy.reset_patch()
    namespace = runpy.run_path(str(path))
    p.size = (64, 128)
    r = make_renderer(backend, p, 30)
    out = namespace["out"]
    try:
        for i in range(60):
            r.step(i / 30)
        for name in ("trackA", "trackB"):
            assert len(namespace[name].samples) == 88200
            assert np.std(namespace[name].samples) > .01
        out.close()
        with wave.open(str(tmp_path / "painted_oscillator.wav")) as wav:
            assert wav.getframerate() == 48000
            assert wav.getnchannels() == 3
            assert wav.getnframes() == 96000
    finally:
        out.close()
        clear_hooks()
        release(r)
        hydrapy.reset_patch()


@pytest.mark.parametrize("fps", [18, 24, 30])
@pytest.mark.parametrize("sr,device_sr,render_fps,seconds", [
    (44100, 48000, 60, 600),
    (48000, 44100, 12, 1800),
    (44100, 44100, 24, 1800),
    (48000, 48000, 30, 600),
])
def test_long_performance_counts_and_pitch(fps, sr, device_sr, render_fps, seconds):
    track = Soundtrack(fps=fps, sr=sr, ac=False, record=False, patch=Patch())
    queue = AudioQueue(track, device_sr)
    source_count = output_count = ticks = 0
    windows = []

    def inspect(chunk):
        nonlocal source_count, output_count, ticks
        source_count += len(chunk)
        ticks += 1
        n = sum(map(len, queue._q))
        audio = queue.pull(n)
        output_count += n
        # Independent time bounds: quantization may move an endpoint by one
        # sample; it must never accumulate across optical frames.
        assert abs(source_count - ticks * sr / fps) < .500001
        assert abs(output_count - ticks * device_sr / fps) < .500001 * (1 + device_sr / sr)
        if ticks <= fps or ticks > (seconds - 1) * fps:
            windows.append(audio)

    track.listeners.append(inspect)             # queue.push runs before inspection
    # Same strip as a 10-cycle drawn oscillator. update/advance exercise the
    # optical scheduler at slower and faster presentation rates, holding pixels.
    column = .5 + .4 * np.sin(2 * np.pi * 10 * (np.arange(1024) + .5) / 1024)
    rgba = np.repeat(column[::-1, None, None], 4, axis=2)
    try:
        track.update(rgba, 0, 1 / render_fps)
        for frame in range(1, seconds * render_fps):
            track.advance(frame / render_fps, 1 / render_fps)
        assert ticks == seconds * fps
        assert source_count == seconds * sr
        assert output_count == seconds * device_sr
        assert not track.chunks                    # bounded memory for long live sets
        # Compare first/last second separately: correct counts alone can conceal
        # a pitch shift or phase restart in the resampler.
        audio = np.concatenate(windows)
        assert len(audio) == 2 * device_sr
        first, last = audio[:device_sr], audio[device_sr:]
        for window in (first, last):
            spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
            assert np.argmax(spectrum) == 10 * fps
        np.testing.assert_allclose(first[4:], last[4:], atol=2e-6, rtol=0)
    finally:
        track.listeners.remove(inspect)
        queue.close()
