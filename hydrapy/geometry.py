"""Small NumPy-only curve tools and a Generative_Art flora interchange adapter.

These helpers return ordinary point arrays for Lines/to_svg or further geometry
work. They do not import Blender, Shapely, or the Generative_Art repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _points(points, dimensions=(2, 3)):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] not in dimensions or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("points must be a finite Nx2/Nx3 array with at least two points")
    return p


def subdivide(points, iterations=2, closed=False):
    """Chaikin corner cutting; preserve open endpoints and close closed paths once.

    Works in 2D or 3D. Each iteration roughly doubles the number of vertices;
    iterations is limited to 0..8. This smooths a control polygon, not a mesh.
    """
    p = _points(points).copy()
    if int(iterations) != iterations or not 0 <= iterations <= 8:
        raise ValueError("iterations must be an integer in 0..8")
    if closed and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 2:
        raise ValueError("closed paths need at least two distinct vertices")
    for _ in range(int(iterations)):
        a, b = (p, np.roll(p, -1, axis=0)) if closed else (p[:-1], p[1:])
        q = np.stack([.75 * a + .25 * b, .25 * a + .75 * b], axis=1).reshape(-1, p.shape[1])
        p = q if closed else np.vstack([p[0], q, p[-1]])
    return np.vstack([p, p[0]]) if closed else p


def fuzzy_paths(paths, passes=4, jitter=.003, seed=0, time=0, rate=.2, keep_ends=True):
    """Seeded overdraw with continuous motion, in the input's coordinate units.

    Fixed random phases per vertex drive sine offsets, so evaluating at any
    time produces the same result regardless of frame rate or render order.
    Closed paths retain their shared endpoint. Open endpoints stay anchored by
    default. Subdivide first when curves need more vertices.
    """
    if int(passes) != passes or not 1 <= passes <= 64:
        raise ValueError("passes must be an integer in 1..64")
    if not np.isfinite([jitter, time, rate]).all() or jitter < 0:
        raise ValueError("jitter must be nonnegative; time and rate must be finite")
    paths = [_points(p) for p in paths]
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        closed = np.array_equal(p[0], p[-1])
        for _ in range(int(passes)):
            phase = rng.uniform(0, 2 * np.pi, p.shape)
            frequency = rng.uniform(.7, 1.3, p.shape)
            offset = jitter * np.sin(phase + time * rate * 2 * np.pi * frequency)
            if closed:
                offset[-1] = offset[0]
            elif keep_ends:
                offset[[0, -1]] = 0
            out.append(p + offset)
    return out


def flora_lines(source, azimuth=0, elevation=15, detail=2):
    """Orthographic stroke projection of a Generative_Art Plant or flora JSON.

    Accepts a Plant with as_dict(), a mapping, or a JSON path, using schema
    ``wisconsin-flora-1`` (metres, Z up). Angles are degrees; detail filters the
    stroke detail level. Returns unscaled 2D paths for Lines(fit=True).
    This is a wire drawing: surfaces do not hide strokes. For hidden-line
    removal, import an SVG from Generative_Art's visibility renderer instead.
    """
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    elif hasattr(source, "as_dict"):
        source = source.as_dict()
    if not isinstance(source, dict) or source.get("schema") != "wisconsin-flora-1":
        raise ValueError("expected a wisconsin-flora-1 Plant or JSON document")
    if source.get("units") != "metres" or source.get("up") != "Z":
        raise ValueError("flora coordinates must use metres with Z up")
    if not np.isfinite([azimuth, elevation, detail]).all():
        raise ValueError("camera angles and detail must be finite")
    a, e = np.deg2rad([azimuth, elevation])
    right = np.array([np.cos(a), -np.sin(a), 0])
    up = np.array([np.sin(a) * np.sin(e), np.cos(a) * np.sin(e), np.cos(e)])
    projection = np.stack([right, up], axis=1)
    return [_points(s["points"], (3,)) @ projection for s in source.get("strokes", []) if s.get("detail", 0) <= detail]
