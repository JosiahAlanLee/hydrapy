"""Prerequisites for encoded A/V assertions, using HydraPy's duration-bounded mux path."""
import json
import shutil
import subprocess
import wave

import pytest

from hydrapy.encoding import _ffmpeg_candidates


@pytest.fixture(scope="session")
def av_ffmpeg(tmp_path_factory):
    """Skip missing codecs/tools, but let HydraPy mux regressions fail tests."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe required for encoded A/V synchronization")
    directory = tmp_path_factory.mktemp("av-ffmpeg")
    wav = directory / "audio.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(bytes(44100 * 2))
    reasons = []
    try:
        candidates = _ffmpeg_candidates()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    for executable in candidates:
        version = executable
        try:
            version = subprocess.check_output([executable, "-version"], text=True, timeout=10).splitlines()[0]
            video, muxed = directory / "video.mp4", directory / "muxed.mp4"
            subprocess.run([
                executable, "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=black:s=64x36:r=12:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
            ], check=True, capture_output=True, timeout=30)
            # Check AAC availability independently: missing codecs may skip, but
            # errors or incorrect durations in HydraPy's own mux must fail.
            subprocess.run([
                executable, "-v", "error", "-i", str(wav), "-c:a", "aac",
                "-f", "null", "-",
            ], check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            reasons.append(f"{version}: A/V mux prerequisite failed: {exc}")
            continue
        from hydrapy.export import _mux_audio
        _mux_audio(str(video), wav, str(muxed), duration=1, ffmpeg=executable)
        info = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-show_streams", "-of", "json", str(muxed),
        ], text=True, timeout=10))
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        assert abs(float(audio["duration"]) - 1) <= .001 + 1 / 44100, version
        return executable
    pytest.skip("No FFmpeg supports accurate A/V muxing: " + "; ".join(reasons))
