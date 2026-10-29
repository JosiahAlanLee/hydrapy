"""Mux audio under a Studio recording exactly, from the recording's timestamps sidecar.

Studio's ``record`` action writes ``take.mp4`` and ``take.timestamps.json`` (``studio.Recorder``):
for every encoded frame, the ``time.monotonic()`` it was presented and, while Studio's JACK client
ran, the JACK frame time then; and ``origin``, the monotonic time of video time 0 (video time ``v`` is
monotonic ``origin + v``). A line through the (present, JACK frame) pairs gives the JACK frame time at
video time 0, so the only other thing needed is when the audio's first sample happened on a clock:

* ``--audio-start-frame N``: the JACK frame time of the capture's first sample (a JACK recorder
  reads ``client.last_frame_time`` in the process callback that writes its first block), or
* ``--audio-start-monotonic T``: its ``time.monotonic()`` (a capture that stamps its blocks).

    python -m hydrapy.recording take.mp4 capture.wav out.mp4 --audio-start-frame 123456
    python -m hydrapy.recording take.mp4 --audio-start-frame 123456          # just print the offset
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

WRAP = 2 ** 32


def load(video_or_sidecar) -> dict:
    """The timestamps for ``take.mp4`` (or the sidecar's own path)."""
    path = Path(video_or_sidecar)
    if not path.name.endswith('.timestamps.json'):
        path = path.with_suffix('.timestamps.json')
    data = json.loads(path.read_text())
    if data.get('format') != 'hydrapy-recording-timestamps-1':
        raise ValueError(f'{path} is not a HydraPy recording timestamps file')
    return data


def _unwrap(values):
    """32-bit JACK frame times as one increasing int64 series."""
    v = np.asarray(values, np.int64)
    steps = np.diff(v)
    steps = (steps + WRAP // 2) % WRAP - WRAP // 2
    return np.concatenate([v[:1], v[0] + np.cumsum(steps)])


def fit(stamps: dict, clock: str = 'jack') -> dict:
    """The clock reading at video time 0 (``start``).

    Video time ``v`` is monotonic time ``origin + v`` (the Recorder fills each slot with what was
    presented within half a frame of it). For ``clock`` 'jack' a line through each rendered frame's
    (present, JACK frame time) pairs maps monotonic time to JACK frames, so ``start`` is in frames,
    ``rate`` is the samplerate, ``drift_ppm`` is how much faster the JACK clock ran than the monotonic
    clock, and ``residual_ms`` the stamps' scatter about the line. For 'monotonic', ``start`` is ``origin``.
    """
    origin = stamps.get('origin')
    if origin is None:
        raise ValueError('This recording has no frames')
    if clock == 'monotonic':
        return dict(clock=clock, start=float(origin), rate=1.0, drift_ppm=0.0, residual_ms=0.0, frames=stamps['frames'])
    if clock != 'jack':
        raise ValueError("clock is 'jack' or 'monotonic'")
    jack = stamps.get('jack')
    if not jack:
        raise ValueError('This recording has no JACK clock: Studio\'s JACK audio was off while it recorded')
    present, raw = stamps['present'], jack['frame_time']
    keep = [k for k, f in enumerate(raw) if f is not None and (k == 0 or present[k] != present[k - 1])]
    if not keep:
        raise ValueError('No frame carries a JACK frame time')
    rate = float(jack['samplerate'])
    t = np.asarray([present[k] for k in keep], np.float64) - origin
    value = _unwrap([raw[k] for k in keep]).astype(np.float64)
    if len(keep) > 1 and np.ptp(t) > 0:
        slope, start = np.polyfit(t, value, 1)
    else:
        slope, start = rate, float(value[0] - t[0] * rate)
    residual = value - (start + slope * t)
    return dict(clock=clock, start=float(start), rate=rate, drift_ppm=float((slope / rate - 1) * 1e6),
                residual_ms=float(np.sqrt(np.mean(residual ** 2)) / rate * 1000), frames=len(keep))


def offset(stamps: dict, *, audio_start_frame=None, audio_start_monotonic=None) -> float:
    """Seconds into the audio that line up with video time 0 (negative: the audio starts later)."""
    if (audio_start_frame is None) == (audio_start_monotonic is None):
        raise ValueError('Give the audio\'s start as a JACK frame time or a monotonic time, not both')
    if audio_start_frame is not None:
        line = fit(stamps, 'jack')
        start = int(round(line['start']))
        frames = (start - int(audio_start_frame) + WRAP // 2) % WRAP - WRAP // 2
        return (frames + line['start'] - start) / line['rate']
    line = fit(stamps, 'monotonic')
    return line['start'] - float(audio_start_monotonic)


def mux(video, audio, out, *, audio_start_frame=None, audio_start_monotonic=None, ffmpeg=None) -> float:
    """Write ``out``: the video untouched, the audio trimmed or delayed onto it (AAC). Returns the offset."""
    from .encoding import _ffmpeg_candidates
    stamps = load(video)
    seconds = offset(stamps, audio_start_frame=audio_start_frame, audio_start_monotonic=audio_start_monotonic)
    shift = ['-ss', f'{seconds:.6f}'] if seconds >= 0 else ['-itsoffset', f'{-seconds:.6f}']
    exe = ffmpeg or _ffmpeg_candidates()[0]
    cmd = [exe, '-y', '-loglevel', 'error', '-i', os.fspath(video), *shift, '-i', os.fspath(audio),
           '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
           '-af', 'apad', '-t', f'{stamps["frames"] / float(stamps["fps"]):.6f}']
    if Path(out).suffix.lower() in ('.mp4', '.mov'):
        cmd += ['-movflags', '+faststart']
    result = subprocess.run([*cmd, os.fspath(out)], capture_output=True)
    if result.returncode:
        raise RuntimeError('FFmpeg mux failed:\n' + result.stderr.decode('utf-8', 'replace')[-4000:])
    return seconds


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m hydrapy.recording', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('audio', nargs='?')
    ap.add_argument('out', nargs='?')
    start = ap.add_mutually_exclusive_group(required=True)
    start.add_argument('--audio-start-frame', type=int)
    start.add_argument('--audio-start-monotonic', type=float)
    a = ap.parse_args(argv)
    stamps = load(a.video)
    kw = dict(audio_start_frame=a.audio_start_frame, audio_start_monotonic=a.audio_start_monotonic)
    line = fit(stamps, 'jack' if a.audio_start_frame is not None else 'monotonic')
    report = dict(offset_s=round(offset(stamps, **kw), 6), drift_ppm=round(line['drift_ppm'], 2),
                  residual_ms=round(line['residual_ms'], 3), frames=stamps['frames'])
    if a.audio and a.out:
        mux(a.video, a.audio, a.out, **kw)
        report['out'] = a.out
    elif a.audio:
        ap.error('give OUT to mux, or only the video to print the offset')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
