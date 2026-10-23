"""FFmpeg H.264 selection and streaming, including runtime hardware probes.

Rendering and encoding are independent: these encoders accept CPU RGB frames.
GPU texture/encoder interop is intentionally a separate future optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile


_ALIASES = {"software": "libx264", "nvenc": "h264_nvenc",
            "videotoolbox": "h264_videotoolbox", "qsv": "h264_qsv",
            "amf": "h264_amf", "vaapi": "h264_vaapi"}


@dataclass(frozen=True)
class VideoEncoder:
    """One tested encoder on one FFmpeg executable/device."""
    name: str
    ffmpeg: str
    available: bool
    reason: str = ""
    device: str | None = None
    input_args: tuple[str, ...] = ()
    output_args: tuple[str, ...] = ()

    @property
    def hardware(self):
        return self.name != "libx264"


def _validate(size, fps, quality, bitrate):
    if len(size) != 2 or any(int(v) != v or v < 2 or v % 2 for v in size):
        raise ValueError("H.264 requires even width and height of at least 2 pixels")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(quality) or not 1 <= quality <= 10:
        raise ValueError("quality must be between 1 and 10 (higher is better)")
    if bitrate is not None:
        if not re.fullmatch(r"\d+(?:\.\d+)?[kM]?", str(bitrate)) or float(re.sub("[kM]$", "", str(bitrate))) <= 0:
            raise ValueError("bitrate must be positive, e.g. '8M' or '1500k'")


def _ffmpeg_candidates(executable=None):
    # An explicit override is authoritative; do not silently use another binary.
    override = executable or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if override:
        return [os.fspath(override)]
    candidates = []
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError, OSError):
        pass
    if not candidates:
        raise RuntimeError("FFmpeg not found. Install FFmpeg or pip install imageio-ffmpeg; use --ffmpeg to select a binary.")
    return list(dict.fromkeys(candidates))


def _encoder_names(executable):
    try:
        result = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
        if result.returncode:
            return set(), result.stderr.strip()[-1000:]
        return set(re.findall(r"^\s*V\S{5}\s+(\S+)", result.stdout, re.MULTILINE)), ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return set(), str(exc)


def _options(name, size, fps, quality, bitrate, device):
    # Keep imageio's previous quality -> x264 CRF mapping for software exports.
    q = int((1-quality/10)*51)
    target = str(bitrate) if bitrate is not None else str(max(500_000, round(size[0]*size[1]*fps*.1*(quality/5)**2)))
    inputs, args = [], ["-c:v", name]
    if name == "libx264":
        args += ["-pix_fmt", "yuv420p"] + (["-b:v", target] if bitrate is not None else ["-crf", str(q)])
    elif name == "h264_nvenc":
        args += ["-pix_fmt", "nv12", "-preset", "p4", "-rc", "vbr"]
        args += ["-b:v", target] if bitrate is not None else ["-cq", str(max(1, q)), "-b:v", "0"]
    elif name == "h264_videotoolbox":
        # Without this, VideoToolbox can silently choose a software encoder.
        args += ["-pix_fmt", "nv12", "-allow_sw", "0", "-b:v", target]
    elif name == "h264_qsv":
        args += ["-pix_fmt", "nv12", "-preset", "medium"]
        args += ["-b:v", target] if bitrate is not None else ["-global_quality", str(max(1, q))]
    elif name == "h264_amf":
        args += ["-pix_fmt", "nv12", "-b:v", target]
    elif name == "h264_vaapi":
        inputs = ["-vaapi_device", device]
        args += ["-vf", "format=nv12,hwupload"]
        args += ["-b:v", target] if bitrate is not None else ["-qp", str(max(1, q))]
    return tuple(inputs), tuple(args)


def _raw_input(size, fps):
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{size[0]}x{size[1]}",
            "-r", str(float(fps)), "-i", "pipe:0"]


def _probe(executable, name, size, fps, quality, bitrate, device=None):
    inputs, outputs = _options(name, size, fps, quality, bitrate, device)
    cmd = [executable, "-hide_banner", "-loglevel", "error", *inputs, *_raw_input(size, fps),
           "-an", *outputs, "-frames:v", "1", "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, input=bytes(size[0]*size[1]*3), stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=10)
        reason = result.stderr.decode("utf-8", "replace").strip()[-1600:]
        return VideoEncoder(name, executable, result.returncode == 0, reason if result.returncode else "",
                            device, inputs, outputs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VideoEncoder(name, executable, False, str(exc), device, inputs, outputs)


def _probes(encoder, size, fps, quality, bitrate, ffmpeg, device):
    _validate(size, fps, quality, bitrate)
    size = tuple(map(int, size))
    requested = _ALIASES.get(encoder, encoder)
    if requested != "auto" and requested not in _ALIASES.values():
        raise ValueError("encoder must be auto, software, nvenc, videotoolbox, qsv, amf, vaapi, or the corresponding FFmpeg name")
    if device is not None and requested not in ("auto", "h264_vaapi"):
        raise ValueError("device is a VAAPI render node; use encoder='vaapi' or 'auto'")
    binaries = [(exe, *_encoder_names(exe)) for exe in _ffmpeg_candidates(ffmpeg)]
    hardware = ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_amf", "h264_vaapi"]
    if sys.platform == "darwin":
        hardware.remove("h264_videotoolbox"); hardware.insert(0, "h264_videotoolbox")
    names = hardware + ["libx264"] if requested == "auto" else [requested]
    for name in names:
        for exe, supported, error in binaries:
            if name not in supported:
                yield VideoEncoder(name, exe, False, error or "encoder not compiled into this FFmpeg")
                continue
            devices = [None]
            if name == "h264_vaapi":
                devices = [os.fspath(device)] if device is not None else sorted(glob.glob("/dev/dri/renderD*"))
                if not devices:
                    yield VideoEncoder(name, exe, False, "no VAAPI render node found; use --device /dev/dri/renderD128")
                    continue
            for node in devices:
                yield _probe(exe, name, size, fps, quality, bitrate, node)


def video_encoders(size=(1280, 720), fps=30.0, quality=8, bitrate=None, *, ffmpeg=None, device=None):
    """Test hardware and software H.264 encoders with one frame at the requested size/fps.

    Returns ``VideoEncoder`` results, including failure reasons and executable paths.
    A compiled-in encoder is not reported as available unless its actual encode succeeds.
    ``device`` optionally chooses a Linux VAAPI render node.
    """
    return list(_probes("auto", size, fps, quality, bitrate, ffmpeg, device))


def _select_encoder(encoder, size, fps, quality, bitrate, ffmpeg=None, device=None):
    failures = []
    for result in _probes(encoder, size, fps, quality, bitrate, ffmpeg, device):
        if result.available:
            return result
        failures.append(f"{result.name} [{result.ffmpeg}]: {result.reason}")
    raise RuntimeError(f"No working {encoder} video encoder. " +
                       ("Try encoder='auto' or 'software'.\n" if encoder != "auto" else "\n") + "\n".join(failures))


def _write_video(frames, path, size, fps, encoder):
    """Stream frames once, check FFmpeg's exit status, and keep stderr off the pipe."""
    cmd = [encoder.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *encoder.input_args,
           *_raw_input(size, fps), "-an", *encoder.output_args]
    if os.path.splitext(path)[1].lower() in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(path)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        broken_pipe = False
        try:
            try:
                for frame in frames:
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
            except BrokenPipeError:
                broken_pipe = True
                # The useful device/codec error is in stderr, not this pipe exception.
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{encoder.name} timed out while finishing the video") from exc
            if code or broken_pipe:
                errors.seek(0)
                detail = errors.read().decode("utf-8", "replace")[-4000:]
                raise RuntimeError(f"{encoder.name} encoding failed (FFmpeg exit {code}):\n{detail}")
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
