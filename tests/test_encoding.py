"""Encoder selection, single-pass export, and failure recovery regressions."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from hydrapy import Patch, Soundtrack, iter_frames, osc, save_mp4, video_encoders
from hydrapy import encoding
from hydrapy.cli import main


@pytest.fixture(scope="module")
def ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable is None:
        executable = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    try:
        encoding._select_encoder("software", (64, 36), 12, 8, None, executable)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return executable


def inspect_video(path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe required for stream metadata assertions")
    result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"]


def fake_inventory(monkeypatch, probe):
    monkeypatch.setattr(encoding, "_ffmpeg_candidates", lambda exe: ["system", "bundled"])
    monkeypatch.setattr(encoding, "_encoder_names", lambda exe: ({"h264_nvenc", "libx264"}, ""))
    monkeypatch.setattr(encoding, "_probe", probe)


def test_auto_checks_hardware_in_other_binary_before_software(monkeypatch):
    calls = []

    def probe(exe, name, *args):
        calls.append((exe, name))
        available = exe == "bundled" or name == "libx264"
        return encoding.VideoEncoder(name, exe, available, "missing driver" if not available else "")

    fake_inventory(monkeypatch, probe)
    selected = encoding._select_encoder("auto", (64, 36), 12, 8, None)
    assert selected.hardware and selected.ffmpeg == "bundled"
    assert calls == [("system", "h264_nvenc"), ("bundled", "h264_nvenc")]


def test_auto_falls_back_but_explicit_hardware_does_not(monkeypatch):
    def probe(exe, name, *args):
        return encoding.VideoEncoder(name, exe, name == "libx264", "missing driver")

    fake_inventory(monkeypatch, probe)
    selected = encoding._select_encoder("auto", (64, 36), 12, 8, None)
    assert selected.name == "libx264" and not selected.hardware
    with pytest.raises(RuntimeError, match="missing driver"):
        encoding._select_encoder("nvenc", (64, 36), 12, 8, None)
    results = video_encoders(size=(64, 36), fps=12)
    assert not any(result.available and result.hardware for result in results)
    assert any(result.available for result in results)


def test_explicit_binary_and_environment_override_are_authoritative(monkeypatch):
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "/custom/ffmpeg")
    assert encoding._ffmpeg_candidates() == ["/custom/ffmpeg"]
    assert encoding._ffmpeg_candidates("/explicit/ffmpeg") == ["/explicit/ffmpeg"]


def test_probe_runs_at_requested_size_and_reports_runtime_failure(monkeypatch):
    def run(cmd, **kwargs):
        assert cmd[cmd.index("-s:v") + 1] == "1920x1080"
        assert cmd[cmd.index("-r") + 1] == "60.0"
        assert len(kwargs["input"]) == 1920 * 1080 * 3
        assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
        assert cmd[cmd.index("-b:v") + 1] == "16M"
        return subprocess.CompletedProcess(cmd, 1, stderr=b"Cannot load libcuda.so.1")

    monkeypatch.setattr(encoding.subprocess, "run", run)
    result = encoding._probe("ffmpeg", "h264_nvenc", (1920, 1080), 60, 8, "16M")
    assert not result.available and "libcuda" in result.reason


def test_videotoolbox_probe_forbids_silent_software_encoding(monkeypatch):
    def run(cmd, **kwargs):
        assert cmd[cmd.index("-allow_sw") + 1] == "0"
        return subprocess.CompletedProcess(cmd, 0, stderr=b"")

    monkeypatch.setattr(encoding.subprocess, "run", run)
    result = encoding._probe("ffmpeg", "h264_videotoolbox", (64, 36), 12, 8, None)
    assert result.available and result.hardware


@pytest.mark.parametrize("scale,channels,expected", [(1, 3, (64, 36)), (2, 4, (130, 74))])
@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv"])
def test_software_export_streams_once_scales_crops_and_preserves_rgb(tmp_path, ffmpeg, scale, channels, expected, suffix):
    events = []

    def frames():
        try:
            for n in range(6):
                events.append(n)
                frame = np.zeros((37, 65, channels), dtype=np.uint8)
                frame[..., :3] = (40 + n * 20, 70, 180)
                yield frame
        finally:
            events.append("closed")

    def selected(encoder):
        assert encoder.name == "libx264" and events == [0]
        events.append("selected")

    path = tmp_path / ("video" + suffix)
    assert save_mp4(frames(), path, fps=12, encoder="software", ffmpeg=ffmpeg,
                    scale=scale, on_encoder=selected) == path
    assert events == [0, "selected", 1, 2, 3, 4, 5, "closed"]
    assert list(tmp_path.iterdir()) == [path]
    decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True).stdout
    pictures = np.frombuffer(decoded, np.uint8).reshape(6, expected[1], expected[0], 3)
    for n, picture in enumerate(pictures):
        assert np.max(np.abs(picture.mean(axis=(0, 1)) - [40 + n * 20, 70, 180])) < 6


def test_lazy_soundtrack_is_muxed_after_rendering(tmp_path, ffmpeg):
    patch = Patch(size=(64, 36))
    osc(8, .1, 1).out(patch.o0, patch=patch)
    track = Soundtrack(patch.o0, fps=12, record=True, keep=2, patch=patch)
    path = tmp_path / "sound.mp4"
    assert len(track.samples) == 0
    save_mp4(iter_frames(patch, seconds=1, fps=12), path, fps=12, audio=track,
             encoder="software", ffmpeg=ffmpeg)
    assert len(track.samples) == 44100
    streams = inspect_video(path)
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert video["r_frame_rate"] == "12/1" and int(video["nb_frames"]) == 12
    assert audio["codec_name"] == "aac" and float(audio["duration"]) == pytest.approx(1, abs=.03)


def test_explicit_encoder_failure_consumes_only_first_frame_and_preserves_output(tmp_path, monkeypatch):
    def probe(exe, name, *args):
        return encoding.VideoEncoder(name, exe, False, "missing driver")

    fake_inventory(monkeypatch, probe)
    events = []

    def frames():
        try:
            events.append("first")
            yield np.zeros((36, 64, 3), np.uint8)
            pytest.fail("failed hardware selection must not advance the renderer")
        finally:
            events.append("closed")

    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="missing driver"):
        save_mp4(frames(), path, encoder="nvenc")
    assert events == ["first", "closed"]
    assert path.read_bytes() == b"original" and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["encoder", "renderer", "dimensions", "audio"])
def test_midstream_failures_preserve_destination_and_close_frames(tmp_path, ffmpeg, monkeypatch, failure):
    if failure == "encoder":
        # Simulate a device that became unavailable after a successful probe.
        monkeypatch.setattr(encoding, "_select_encoder", lambda *args:
                            encoding.VideoEncoder("libx264", ffmpeg, True,
                                                  output_args=("-c:v", "missing_encoder")))
    events = []

    def frames():
        try:
            yield np.zeros((36, 64, 3), np.uint8)
            if failure == "renderer":
                raise RuntimeError("render failed")
            yield np.zeros((38 if failure == "dimensions" else 36, 64, 3), np.uint8)
        finally:
            events.append("closed")

    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original")
    audio = tmp_path / "missing.wav" if failure == "audio" else None
    with pytest.raises((RuntimeError, ValueError)):
        save_mp4(frames(), path, encoder="software", ffmpeg=ffmpeg, audio=audio)
    assert events == ["closed"]
    assert path.read_bytes() == b"original" and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")}, {"quality": 11},
                                    {"bitrate": "-1M"}, {"bitrate": "8m"}, {"scale": 0}, {"encoder": "typo"}])
def test_invalid_export_settings_fail_without_creating_file(tmp_path, kwargs):
    path = tmp_path / "invalid.mp4"
    with pytest.raises(ValueError):
        save_mp4([np.zeros((36, 64, 3), np.uint8)], path, **kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("frames,suffix", [([], ".mp4"), ([np.zeros((36, 64, 3))], ".mp4"),
                                          ([np.zeros((36, 64, 3), np.uint8)], ".webm")])
def test_invalid_frame_stream_or_container_fails_early(tmp_path, frames, suffix):
    with pytest.raises(ValueError):
        save_mp4(frames, tmp_path / ("bad" + suffix))
    assert not list(tmp_path.iterdir())


def test_cli_can_check_encoders_without_a_patch(monkeypatch, capsys):
    fake_inventory(monkeypatch, lambda exe, name, *args:
                   encoding.VideoEncoder(name, exe, name == "libx264", "missing driver"))
    main(["encoders", "--size", "64x36", "--fps", "12"])
    output = capsys.readouterr().out
    assert "OK libx264" in output and "-- h264_nvenc" in output and "missing driver" in output


def test_frame_cleanup_failure_preserves_destination(tmp_path, ffmpeg):
    class Frames:
        closed = 0

        def __iter__(self):
            yield np.zeros((36, 64, 3), np.uint8)

        def close(self):
            self.closed += 1
            raise RuntimeError("cleanup failed")

    path = tmp_path / "existing.mp4"
    path.write_bytes(b"original")
    frames = Frames()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        save_mp4(frames, path, encoder="software", ffmpeg=ffmpeg)
    assert frames.closed == 1 and path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_cli_render_routes_encoding_options(tmp_path, ffmpeg, capsys):
    patch = tmp_path / "patch.py"
    patch.write_text("from hydrapy import *\nosc(8).out(o0)\n")
    path = tmp_path / "cli.mp4"
    main(["render", str(patch), "-o", str(path), "--size", "64x36", "--seconds", "0.5",
          "--fps", "12", "--encoder", "software", "--quality", "7", "--bitrate", "500k", "--ffmpeg", ffmpeg])
    assert "encoding with libx264 (software)" in capsys.readouterr().out
    assert path.stat().st_size > 0
