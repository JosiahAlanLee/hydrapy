"""Listening and watching: beats from a record, optical flow from footage.

**Beats** finds the beats of a recording (Ellis's dynamic-programming beat tracker, in NumPy; it
uses librosa instead if that is installed and you ask for it) and gives them back as signals, so a
patch can land on the actual downbeats of a record rather than on a Clock:

    beats = Beats.from_file("brahms.wav")
    studie(o0, beat=beats.phase).out(o0)                  # Fischinger, on Brahms's beats
    shape(4, beats.pulse(0.02, 0.4) * 0.3).out(o1)        # an envelope on every beat
    osc(10, 0.1, beats.bar(4)).out(o2)                    # a phase across each four-beat bar
    noise(beats.onset * 20 + 3).out(o3)                   # the onset-strength envelope itself

All of them export to the HTML page (the beat times and the envelope are embedded); play the same
file with ``Audio.from_file`` and the page stays in step with it.

**Flow** measures the motion in a ``Video`` (OpenCV's Farneback optical flow) and serves it as a
texture: red/green = 0.5 + the motion of each pixel per frame (in frame units, times ``gain``),
blue = its speed.  Transport anything along real motion — painted texture over footage, datamosh:

    flow = Flow(Video("driftless.mp4"))
    src(o0).displace(src(flow), -1).blend(src(Video("driftless.mp4")), 0.02).out(o0)

Flow is computed in Python, so it runs on the NumPy and GL backends, not in the HTML export.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .core import Texture, Video
from .signals import Env, Signal, _num


# -----------------------------------------------------------------------------
# audio analysis
# -----------------------------------------------------------------------------

def load_audio(path: str):
    """(mono float32 samples, sample rate) from a WAV, or anything ffmpeg reads."""
    from .signals import _load_audio
    try:
        return _load_audio(path)
    except RuntimeError:
        import subprocess
        import imageio_ffmpeg
        sr = 44100
        out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                              "-ar", str(sr), "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy(), sr


def onset_envelope(samples: np.ndarray, sr: int, hop: int = 512, n_fft: int = 2048, bands: int = 40):
    """Spectral-flux onset strength (positive change in log band energy), one value per hop.
    Returns (envelope normalised to its 99th percentile, frames per second)."""
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    pad = n_fft // 2
    x = np.pad(x, (pad, pad))
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    win = np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))                     # (frames, bins)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(40.0, min(sr / 2, 16000.0), bands + 2)
    fb = np.zeros((bands, len(freqs)))
    for b in range(bands):                                               # triangular, log-spaced
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        fb[b] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    e = np.log1p(1000.0 * (mag ** 2) @ fb.T)
    flux = np.maximum(np.diff(e, axis=0, prepend=e[:1]), 0).sum(axis=1)
    k = max(int(round(0.75 * sr / hop)), 3)                               # remove the trend slower than ~0.75 s
    flux = flux - np.convolve(flux, np.ones(k) / k, mode="same")
    flux = np.maximum(flux, 0)
    top = np.percentile(flux, 99) or 1.0
    return (flux / top).astype(np.float32), sr / hop


def estimate_tempo(env: np.ndarray, fps: float, prior_bpm: float = 120.0, lo: float = 40.0, hi: float = 240.0) -> float:
    """Tempo from the onset envelope's autocorrelation, weighted towards ``prior_bpm`` (log-Gaussian, one octave)."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lags = np.arange(len(ac))
    with np.errstate(divide="ignore"):
        bpm = 60.0 * fps / lags
    ok = (bpm >= lo) & (bpm <= hi)
    w = np.exp(-0.5 * (np.log2(np.where(ok, bpm, 1.0) / prior_bpm)) ** 2)
    score = np.where(ok, ac * w, -np.inf)
    k = int(np.argmax(score))
    # refine with a parabola through the peak
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        k = k + float(np.clip(d, -0.5, 0.5))
    return float(60.0 * fps / k)


def track_beats(env: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Ellis (2007): the beat sequence maximising onset strength plus a penalty for deviating from
    the tempo, by dynamic programming. Returns beat frame indices."""
    period = 60.0 * fps / bpm
    win = np.exp(-0.5 * (np.arange(-int(period), int(period) + 1) * 32.0 / period) ** 2)
    local = np.convolve(env, win, mode="same")
    n = len(local)
    score = np.zeros(n)
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    for t in range(n):
        a, b = max(0, t - hi), t - lo
        if b <= a:
            score[t] = local[t]
            continue
        tau = np.arange(a, b)
        cand = score[tau] - tightness * np.log((t - tau) / period) ** 2
        j = int(np.argmax(cand))
        score[t] = local[t] + cand[j]
        back[t] = tau[j]
    # end on the best-scoring frame in the last period that is a local maximum
    tail = np.arange(max(0, n - int(period) - 1), n)
    t = int(tail[np.argmax(score[tail])])
    beats = [t]
    while back[t] >= 0:
        t = back[t]
        beats.append(t)
    beats = np.array(beats[::-1])
    # drop weak beats at the very edges (silence before or after the music)
    thr = 0.5 * np.median(local[beats]) if len(beats) else 0
    keep = np.ones(len(beats), bool)
    i = 0
    while i < len(beats) and local[beats[i]] < thr:
        keep[i] = False; i += 1
    j = len(beats) - 1
    while j > i and local[beats[j]] < thr:
        keep[j] = False; j -= 1
    return beats[keep]


LATENCY = 0.2      # the tracker hears an onset ~0.2 FFT windows early (the window reaches it before its centre does)


class _BeatPhase(Signal):
    def __init__(self, beats: "Beats", div: int = 1):
        self.beats, self.div = beats, max(int(div), 1)

    def value(self, ctx):
        return self.beats.phase_at(ctx.time, self.div)

    def js(self):
        arr = "[" + ",".join(f"{t:.4f}" for t in self.beats.times) + "]"
        return f"__beatphase({arr},{_num(self.beats.period)},{self.div},time)"


class _Table(Signal):
    def __init__(self, values: np.ndarray, rate: float):
        self.values, self.rate = np.asarray(values, dtype=np.float32), float(rate)

    def value(self, ctx):
        if not len(self.values):
            return 0.0
        x = max(ctx.time, 0.0) * self.rate
        i = int(x)
        if i >= len(self.values) - 1:
            return float(self.values[-1])
        f = x - i
        return float(self.values[i] * (1 - f) + self.values[i + 1] * f)

    def js(self):
        arr = "[" + ",".join(f"{v:.3f}" for v in self.values) + "]"
        return f"__table({arr},{_num(self.rate)},Math.max(time,0))"


class Beats:
    """Beat times of a recording, and signals that follow them."""

    def __init__(self, times, onset: Optional[np.ndarray] = None, onset_fps: float = 0.0, tempo: Optional[float] = None,
                 duration: Optional[float] = None):
        self.times = np.asarray(times, dtype=np.float64)
        if len(self.times) < 2:
            raise ValueError("Beats needs at least two beat times")
        self.tempo = float(tempo) if tempo else 60.0 / float(np.median(np.diff(self.times)))
        self.period = 60.0 / self.tempo
        self.duration = duration
        self._onset = (onset, onset_fps) if onset is not None else None

    @classmethod
    def from_file(cls, path: str, method: str = "auto", prior_bpm: float = 120.0, bpm: Optional[float] = None,
                  tightness: float = 100.0) -> "Beats":
        """``prior_bpm``: where to look for the tempo (one octave either side). ``bpm``: the tempo, if you
        know it — the tracker then only finds where the beats fall. Tempo estimates can land an octave
        off (accents every other beat, busy hi-hats); ``half()`` and ``double()`` fix that afterwards."""
        samples, sr = load_audio(path)
        return cls.from_samples(samples, sr, method=method, prior_bpm=prior_bpm, bpm=bpm, tightness=tightness)

    @classmethod
    def from_samples(cls, samples, sr: int, method: str = "auto", prior_bpm: float = 120.0, bpm: Optional[float] = None,
                     tightness: float = 100.0) -> "Beats":
        env, fps = onset_envelope(samples, sr)
        if method in ("librosa", "auto"):
            try:
                import librosa
                tempo, frames = librosa.beat.beat_track(y=np.asarray(samples, np.float32), sr=sr, hop_length=512,
                                                        start_bpm=bpm or prior_bpm, tightness=tightness, units="frames",
                                                        **({"bpm": bpm} if bpm else {}))
                times = librosa.frames_to_time(frames, sr=sr, hop_length=512)
                return cls(times, env, fps, float(np.atleast_1d(tempo)[0]), len(samples) / sr)
            except ImportError:
                if method == "librosa":
                    raise
        bpm = float(bpm) if bpm else estimate_tempo(env, fps, prior_bpm)
        frames = track_beats(env, fps, bpm, tightness)
        # flux at frame t is the rise from frame t-1 to t: the onset lies about half a hop earlier
        return cls(np.maximum(frames / fps + LATENCY * 2048 / sr, 0.0), env, fps, bpm, len(samples) / sr)

    def half(self) -> "Beats":
        """Every other beat (the tracker found the eighth notes)."""
        return Beats(self.times[::2], *(self._onset or (None, 0.0)), self.tempo / 2, self.duration)

    def double(self) -> "Beats":
        """A beat between every pair (the tracker found the half notes)."""
        mids = (self.times[:-1] + self.times[1:]) / 2
        return Beats(np.sort(np.concatenate([self.times, mids])), *(self._onset or (None, 0.0)), self.tempo * 2, self.duration)

    # -- reading -----------------------------------------------------------------
    def index_at(self, t: float) -> int:
        return int(np.searchsorted(self.times, t, side="right")) - 1

    def phase_at(self, t: float, div: int = 1) -> float:
        """0..1 through the current beat (or ``div`` beats), extrapolating at the tempo before the first
        and after the last beat."""
        T = self.times
        if t < T[0]:
            k = (t - T[0]) / self.period
        elif t >= T[-1]:
            k = len(T) - 1 + (t - T[-1]) / self.period
        else:
            i = self.index_at(t)
            k = i + (t - T[i]) / (T[i + 1] - T[i])
        k = k / max(div, 1)
        return float(k - math.floor(k))

    @property
    def phase(self) -> Signal:
        return _BeatPhase(self, 1)

    def bar(self, beats: int = 4) -> Signal:
        return _BeatPhase(self, beats)

    def pulse(self, attack: float = 0.02, decay: float = 0.4, curve: float = 2.0, lo: float = 0.0, hi: float = 1.0) -> Signal:
        """An envelope on every beat (``attack``/``decay`` as fractions of the beat)."""
        return Env(self.phase, attack, decay, curve, lo, hi)

    @property
    def onset(self) -> Signal:
        if self._onset is None:
            raise ValueError("these Beats carry no onset envelope")
        env, fps = self._onset
        # decimate to ~30 values a second for export
        k = max(int(round(fps / 30.0)), 1)
        n = len(env) // k * k
        dec = env[:n].reshape(-1, k).max(axis=1) if n else env
        return _Table(dec, fps / k)



# -----------------------------------------------------------------------------
# optical flow
# -----------------------------------------------------------------------------

class Flow(Texture):
    """Optical flow of a ``Video`` (OpenCV Farneback) as a texture: rg = 0.5 + motion per frame
    (frame units x ``gain``), b = speed, a = 1.  ``width``: the analysis resolution."""

    _counter = 0

    def __init__(self, video: Video, gain: float = 1.0, width: int = 320, smooth: float = 0.0):
        self.video, self.gain, self.width, self.smooth = video, float(gain), int(width), float(smooth)
        Flow._counter += 1
        self.name = f"flow{Flow._counter}"
        self._cache_index = None
        self._last = None

    def _grey(self, f):
        import cv2
        img = (np.clip(f[::-1, :, :3], 0, 1) * 255).astype(np.uint8)          # Video frames are GL-oriented; cv2 wants rows down
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def flow_at(self, t: float) -> np.ndarray:
        """(h, w, 2) motion in pixels per frame of the analysis image, rows down, +y down."""
        import cv2
        v = self.video
        v.frame_at(t)                                    # decodes and sets fps
        i = int(t * v.fps)
        a = self._grey(v.frame_at((i - 1) / v.fps + 1e-6))
        b = self._grey(v.frame_at(i / v.fps + 1e-6))
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def dynamic_array(self, t: float = 0.0) -> np.ndarray:
        v = self.video
        v.frame_at(t)
        i = int(t * v.fps)
        if self._cache_index == i and self._last is not None:
            return self._last
        fl = self.flow_at(t)
        h, w = fl.shape[:2]
        dx = fl[..., 0] / w * self.gain
        dy = -fl[..., 1] / h * self.gain                 # GL uv: +y up
        out = np.stack([0.5 + dx, 0.5 + dy, np.sqrt(dx * dx + dy * dy) * 10.0, np.ones_like(dx)], -1)[::-1]
        out = out.astype(np.float32)
        if self.smooth and self._last is not None and self._last.shape == out.shape:
            out = (self._last * self.smooth + out * (1 - self.smooth)).astype(np.float32)
        self._cache_index, self._last = i, out
        return out
