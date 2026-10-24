"""Render patches/39_bullfrog_layers.py as a music video with the track.

    python docs/bullfrog_layers.py /path/to/Bullfrog.flac -o bullfrog_layers.mp4
    python docs/bullfrog_layers.py track.flac --stills 30,90,150 --contact sheet.png   # look first
    python docs/bullfrog_layers.py track.flac -o part.mp4 --start 100 --seconds 20     # an excerpt

The cover art embedded in the file is extracted and used as the photograph (or pass --cover).
The music is not part of the repository.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def duration(track):
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", track], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError("could not read the track's duration")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("track")
    ap.add_argument("-o", "--out", default="bullfrog_layers.mp4")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None, help="default: to the end of the track")
    ap.add_argument("--cover", default=None, help="the photograph (default: the track's embedded art)")
    ap.add_argument("--stills", default=None, help="comma-separated times: render stills instead of a video")
    ap.add_argument("--contact", default=None, help="with --stills: save them as one sheet")
    ap.add_argument("--warmup", type=float, default=4.0, help="with --stills: seconds of history before each")
    ap.add_argument("--quality", type=int, default=9)
    a = ap.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x"))
    track = os.path.abspath(a.track)
    tmp = tempfile.mkdtemp(prefix="bullfrog_")
    cover = a.cover
    if cover is None:
        cover = os.path.join(tmp, "cover.jpg")
        r = subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", track, "-an", "-c:v", "copy", cover])
        if r.returncode or not os.path.exists(cover):
            cover = None
    os.environ["HYDRAPY_TRACK"] = track
    if cover:
        os.environ["HYDRAPY_COVER"] = os.path.abspath(cover)
    from hydrapy.cli import load_patch
    patch = load_patch(os.path.join(os.path.dirname(HERE), "patches", "39_bullfrog_layers.py"))
    from hydrapy.backend_gl import GLRenderer

    if a.stills:
        import numpy as np
        from PIL import Image as PILImage
        times = [float(t) for t in a.stills.split(",")]
        frames = []
        r = GLRenderer(patch, size=size, fps=a.fps)
        for t in times:
            r.reset() if hasattr(r, "reset") else None
            t0 = max(0.0, t - a.warmup)
            n = int(round((t - t0) * a.fps))
            for i in range(n + 1):
                r.step(t0 + i / a.fps)
            img = r.read()
            frames.append(img)
            PILImage.fromarray(img).save(os.path.splitext(a.out)[0] + f"_{t:06.1f}.png")
        if a.contact:
            cols = 3
            h, w = frames[0].shape[:2]
            rows = (len(frames) + cols - 1) // cols
            sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
            for i, f in enumerate(frames):
                sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = f[..., :3]
            PILImage.fromarray(sheet).save(a.contact)
        print("stills:", ", ".join(f"{t:g}s" for t in times))
        return

    seconds = a.seconds if a.seconds is not None else duration(track) - a.start
    audio = track
    if a.start > 0 or a.seconds is not None:
        audio = os.path.join(tmp, "excerpt.wav")
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", f"{a.start:.3f}", "-t", f"{seconds:.3f}", "-i", track, audio], check=True)
    from hydrapy.backend_gl import iter_frames_gl
    from hydrapy.export import save_mp4
    total = int(round(seconds * a.fps))
    t0 = time.time()

    def frames():
        for i, f in enumerate(iter_frames_gl(patch, seconds=seconds, fps=a.fps, size=size, start=a.start)):
            if i % int(a.fps * 10) == 0:
                el = time.time() - t0
                print(f"  {a.start + i / a.fps:7.1f}s  frame {i}/{total}  {el:6.0f}s elapsed", flush=True)
            yield f

    save_mp4(frames(), a.out, fps=a.fps, quality=a.quality, audio=audio)
    print(f"wrote {a.out}: {total} frames at {size[0]}x{size[1]} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
