"""Prerequisites for encoded A/V assertions, tested independently of HydraPy."""
import json
import shutil
import subprocess
import wave

import pytest

from hydrapy.encoding import _ffmpeg_candidates


@pytest.fixture(scope="session")
def av_ffmpeg(tmp_path_factory):
    """Find FFmpeg with working H.264-copy/AAC shortest-stream muxing.

    FFmpeg 4.4 can end AAC early at the video DTS (B-frame reordering),
    even when both inputs are exactly one second. Probe the external tool,
    not save_mp4: a HydraPy regression must fail tests rather than skip them.
    """
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
            subprocess.run([
                executable, "-v", "error", "-y", "-i", str(video), "-i", str(wav),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-shortest", str(muxed),
            ], check=True, capture_output=True, timeout=30)
            info = json.loads(subprocess.check_output([
                ffprobe, "-v", "error", "-show_streams", "-of", "json", str(muxed),
            ], text=True, timeout=10))
            audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
            duration = float(audio["duration"])
            if abs(duration - 1) <= .001 + 1 / 44100:
                return executable
            reasons.append(f"{version}: one-second AAC truncated to {duration}s (-shortest/B-frame bug)")
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration) as exc:
            reasons.append(f"{version}: A/V mux prerequisite failed: {exc}")
    pytest.skip("No FFmpeg supports accurate A/V muxing: " + "; ".join(reasons))
