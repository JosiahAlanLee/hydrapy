"""Texture synthesis without a network: image quilting (Efros & Freeman, 2001).

Given one photo of an organic texture (a moth wing, lichen, paint on acetate), ``quilt``
assembles a new image of any size from randomly chosen patches of it, stitched along
minimum-error seams.  Every seed gives a different image that shares the statistics of
the original, which is exactly what per-frame hand-painted film needs: a new, related
frame every 1/24 s.  Seconds per frame on a CPU, no training, no GPU.

    q = Quilt("moth_wing.jpg", size=(320, 180), count=24, seed=1)
    solid(1, 1, 1).stain(src(q).alpha(src(q).luma(0.3)), 0.8).out()
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .core import Chain, Texture


def _as_rgb(src) -> np.ndarray:
    """Path, PIL image, array or Chain -> float32 (h, w, 3) in 0..1, row 0 = top."""
    if isinstance(src, str):
        from PIL import Image as PILImage
        return np.asarray(PILImage.open(src).convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(src, Chain):
        from .backend_np import render_chain
        return np.clip(render_chain(src, t=0.0, size=(384, 216))[..., :3], 0, 1).astype(np.float32)
    if hasattr(src, "convert"):  # PIL
        return np.asarray(src.convert("RGB"), dtype=np.float32) / 255.0
    a = np.asarray(src, dtype=np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    return a[..., :3]


def _min_cut_mask(cost: np.ndarray) -> np.ndarray:
    """Vertical minimum-error seam through ``cost`` (rows, cols). True where the NEW block should be used (right of the seam)."""
    rows, cols = cost.shape
    E = cost.astype(np.float64).copy()
    back = np.zeros((rows, cols), dtype=np.int64)
    inf = np.array([np.inf])
    for i in range(1, rows):
        prev = E[i - 1]
        stacked = np.stack([np.concatenate([inf, prev[:-1]]), prev, np.concatenate([prev[1:], inf])])
        j = np.argmin(stacked, axis=0)
        E[i] = cost[i] + stacked[j, np.arange(cols)]
        back[i] = j - 1
    seam = np.zeros(rows, dtype=np.int64)
    seam[-1] = int(np.argmin(E[-1]))
    for i in range(rows - 1, 0, -1):
        seam[i - 1] = seam[i] + back[i, seam[i]]
    return np.arange(cols)[None, :] >= seam[:, None]


def quilt(source, size=(320, 180), block: int = 32, overlap: int = 6, seed: int = 0, tolerance: float = 0.1,
          candidates: int = 1500, augment: bool = True) -> np.ndarray:
    """Synthesise a (H, W, 3) float32 texture of ``size`` = (W, H) from ``source`` by image quilting.

    ``block`` is the patch size in pixels, ``overlap`` the seam width; larger blocks keep more
    structure, smaller ones scramble more.  ``tolerance`` widens the set of acceptable patches
    (more variety, slightly worse seams).  ``augment`` adds flipped copies of the source.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    rng = np.random.default_rng(seed)
    src = _as_rgb(source)
    h, w = src.shape[:2]
    block = int(min(block, h, w))
    overlap = int(max(1, min(overlap, block // 2)))
    sources = [np.ascontiguousarray(src)]
    if augment:
        sources += [np.ascontiguousarray(src[:, ::-1]), np.ascontiguousarray(src[::-1])]
    left_views = [sliding_window_view(s, (block, overlap, 3))[:, :, 0] for s in sources]   # (ny, nx, block, overlap, 3)
    top_views = [sliding_window_view(s, (overlap, block, 3))[:, :, 0] for s in sources]    # (ny, nx, overlap, block, 3)
    corner_views = [sliding_window_view(s, (overlap, overlap, 3))[:, :, 0] for s in sources]

    W, H = int(size[0]), int(size[1])
    step = block - overlap
    nbx = max(1, math.ceil((W - overlap) / step))
    nby = max(1, math.ceil((H - overlap) / step))
    out = np.zeros((nby * step + overlap, nbx * step + overlap, 3), dtype=np.float32)

    for by in range(nby):
        for bx in range(nbx):
            y, x = by * step, bx * step
            n = candidates
            si = rng.integers(0, len(sources), size=n)
            ys = rng.integers(0, h - block + 1, size=n)
            xs = rng.integers(0, w - block + 1, size=n)
            err = np.zeros(n, dtype=np.float64)
            if bx > 0 or by > 0:
                for k in range(len(sources)):
                    m = si == k
                    if not m.any():
                        continue
                    if bx > 0:
                        c = left_views[k][ys[m], xs[m]]
                        err[m] += ((c - out[y:y + block, x:x + overlap][None]) ** 2).sum(axis=(1, 2, 3))
                    if by > 0:
                        c = top_views[k][ys[m], xs[m]]
                        err[m] += ((c - out[y:y + overlap, x:x + block][None]) ** 2).sum(axis=(1, 2, 3))
                    if bx > 0 and by > 0:
                        c = corner_views[k][ys[m], xs[m]]
                        err[m] -= ((c - out[y:y + overlap, x:x + overlap][None]) ** 2).sum(axis=(1, 2, 3))
                best = err.min()
                ok = np.flatnonzero(err <= best * (1.0 + tolerance) + 1e-6)
                pick = int(rng.choice(ok))
            else:
                pick = 0
            s = sources[si[pick]]
            cand = s[ys[pick]:ys[pick] + block, xs[pick]:xs[pick] + block]
            use_new = np.ones((block, block), dtype=bool)
            if bx > 0:
                cost = ((cand[:, :overlap] - out[y:y + block, x:x + overlap]) ** 2).sum(-1)
                use_new[:, :overlap] &= _min_cut_mask(cost)
            if by > 0:
                cost = ((cand[:overlap, :] - out[y:y + overlap, x:x + block]) ** 2).sum(-1)
                use_new[:overlap, :] &= _min_cut_mask(cost.T).T
            region = out[y:y + block, x:x + block]
            out[y:y + block, x:x + block] = np.where(use_new[..., None], cand, region)
    return out[:H, :W]


class Quilt(Texture):
    """A bank of quilted frames synthesised from one texture, used like an Image: ``src(q)``.

    ``count`` frames are generated lazily (different seeds); ``rate`` frames per second are
    shown, each held for ``hold`` frames, in ``mode`` 'cycle' or 'random' order.
    """

    _counter = 0

    def __init__(self, source, size=(320, 180), count: int = 16, block: int = 32, overlap: int = 6, seed: int = 0,
                 rate: float = 12.0, hold: int = 1, mode: str = "random", tolerance: float = 0.1, candidates: int = 1500):
        self.source, self.size, self.count = source, tuple(size), int(count)
        self.block, self.overlap, self.seed, self.tolerance, self.candidates = block, overlap, seed, tolerance, candidates
        self.rate, self.hold, self.mode = float(rate), int(hold), mode
        Quilt._counter += 1
        self.name = f"quilt{Quilt._counter}"
        self._frames: dict[int, np.ndarray] = {}
        self._rgba: dict[int, np.ndarray] = {}

    def frame(self, i: int) -> np.ndarray:
        """uint8 RGBA (H, W, 4), row 0 = top."""
        i = int(i) % self.count
        if i not in self._frames:
            f = quilt(self.source, self.size, self.block, self.overlap, self.seed * 1000 + i, self.tolerance,
                      self.candidates)
            rgb = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
            self._frames[i] = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        return self._frames[i]

    def frames(self) -> list[np.ndarray]:
        return [self.frame(i) for i in range(self.count)]

    def index_at(self, t: float) -> int:
        return bank_index(t, self.count, self.rate, self.hold, self.mode, self.seed)

    def array_at(self, t: float) -> np.ndarray:
        """float32 RGBA, row 0 = bottom (GL convention) for the CPU renderer."""
        i = self.index_at(t)
        if i not in self._rgba:
            self._rgba[i] = (self.frame(i).astype(np.float32) / 255.0)[::-1].copy()
        return self._rgba[i]


def bank_index(t: float, n: int, rate: float, hold: int, mode: str, seed: float) -> int:
    """Which of n frames shows at time t. Mirrors the JS runtime exactly (double precision)."""
    if n <= 0:
        return 0
    k = math.floor(t * rate / max(hold, 1))
    if mode == "random":
        from .signals import _hash11
        return int(math.floor(_hash11(k + seed * 131.0) * n)) % n
    return k % n
