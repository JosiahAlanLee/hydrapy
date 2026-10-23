"""Spline sweeps with retained 3D paths and clipped perspective projection.

Wireframes are transparent: no hidden-surface removal is implied.
"""
from __future__ import annotations

import numpy as np

from .geometry import _points
from .ops_3d import _rot3


def catmull_rom(points, samples=12, closed=False):
    """Uniform Catmull-Rom spline; samples per segment, interpolating the controls."""
    p = _points(points).copy()
    if int(samples) != samples or not 1 <= samples <= 256:
        raise ValueError("samples must be an integer in 1..256")
    if closed and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < (3 if closed else 2):
        raise ValueError("closed splines need at least three controls")
    padded = np.vstack([p[-1], p, p[:2]]) if closed else np.vstack([p[0], p, p[-1]])
    t = np.arange(int(samples), dtype=float)[:, None] / samples
    out = []
    for i in range(len(p) if closed else len(p) - 1):
        a, b, c, d = padded[i:i + 4]
        out.append(.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t*t + (-a+3*b-3*c+d)*t*t*t))
    return np.vstack([*out, p[0] if closed else p[-1]])


def sweep_wire(points, radius=.08, sides=12, twist=0, rings=4):
    """Sweep a circular profile along an open 3D path; return 3D polylines.

    radius: scalar or one value per path vertex. twist: total radians.
    rings: draw every Nth cross-section (end caps included), 0 disables rings.
    Frames propagate by projection to minimize unwanted twisting. This helper
    accepts open paths only; use lathe_wire for a closed revolved surface.
    """
    p = _points(points, (3,)).copy()
    if int(sides) != sides or not 3 <= sides <= 128:
        raise ValueError("sides must be an integer in 3..128")
    if int(rings) != rings or rings < 0 or not np.isfinite(twist):
        raise ValueError("rings must be a nonnegative integer; twist must be finite")
    if np.array_equal(p[0], p[-1]):
        raise ValueError("sweep_wire expects an open path")
    r = np.broadcast_to(np.asarray(radius, dtype=float), (len(p),))
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("radius must be finite and nonnegative")
    edges = np.diff(p, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("consecutive path vertices must be distinct")
    edges /= lengths[:, None]
    tangent = np.vstack([edges[0], edges[:-1] + edges[1:], edges[-1]])
    norm = np.linalg.norm(tangent, axis=1)
    tangent[norm < 1e-8] = edges[np.minimum(np.where(norm < 1e-8)[0], len(edges)-1)]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.zeros(3)
    normals = []
    for t in tangent:
        normal = normal - t * np.dot(normal, t)
        if np.linalg.norm(normal) < 1e-8:
            axis = np.eye(3)[np.argmin(np.abs(t))]
            normal = axis - t * np.dot(axis, t)
        normal /= np.linalg.norm(normal)
        normals.append(normal.copy())
    n = np.array(normals)
    b = np.cross(tangent, n)
    distance = np.r_[0, np.cumsum(lengths)]
    angles = np.arange(int(sides)) * 2*np.pi/sides + twist * distance[:, None]/distance[-1]
    mesh = p[:, None, :] + r[:, None, None] * (np.cos(angles)[..., None]*n[:, None, :] + np.sin(angles)[..., None]*b[:, None, :])
    paths = [mesh[:, j].copy() for j in range(int(sides))]
    if rings:
        for i in sorted(set(range(0, len(p), int(rings))) | {len(p)-1}):
            paths.append(np.vstack([mesh[i], mesh[i, 0]]))
    return paths


def lathe_wire(profile, sides=24):
    """Revolve (radius, y) profile controls around Y; return rings and meridians."""
    p = _points(profile, (2,))
    if int(sides) != sides or not 3 <= sides <= 128 or np.any(p[:, 0] < 0):
        raise ValueError("sides must be 3..128 and profile radii nonnegative")
    a = np.arange(int(sides)) * 2*np.pi/sides
    mesh = np.stack([p[:, 0, None]*np.cos(a), np.broadcast_to(p[:, 1, None], (len(p), int(sides))),
                     p[:, 0, None]*np.sin(a)], axis=-1)
    return [mesh[:, j].copy() for j in range(int(sides))] + [np.vstack([row, row[0]]) for row in mesh]


def project_wire(paths, rotX=0, rotY=0, rotZ=0, distance=3, fov=45, aspect=1, near=.05):
    """Project Y-up 3D paths to normalized Y-up 2D, clipping at the near plane.

    Camera looks along -Z from (0,0,distance). Paths may extend off-screen;
    use Lines(fit=False) to preserve the camera framing. No surface occlusion.
    """
    if not np.isfinite([rotX, rotY, rotZ, distance, fov, aspect, near]).all() or not 0 < fov < 179 or aspect <= 0 or near <= 0:
        raise ValueError("finite camera values, 0<fov<179, aspect>0 and near>0 required")
    rotation = _rot3(rotX, rotY, rotZ)
    f = 1/np.tan(np.radians(fov)*.5)
    out = []
    for path in paths:
        p = _points(path, (3,)) @ rotation.T
        depth = distance - p[:, 2]
        current = []
        for i in range(len(p)-1):
            a, b, da, db = p[i].copy(), p[i+1].copy(), depth[i], depth[i+1]
            if da < near and db < near:
                if len(current) > 1: out.append(np.array(current))
                current = []
                continue
            starts_hidden, ends_hidden = da < near, db < near
            if starts_hidden:
                a += (b-a) * ((near-da)/(db-da)); da = near
            elif ends_hidden:
                b = a + (b-a) * ((near-da)/(db-da)); db = near
            pa = .5 + a[:2]*f/(2*da*np.array([aspect, 1]))
            pb = .5 + b[:2]*f/(2*db*np.array([aspect, 1]))
            if not current: current.append(pa)
            current.append(pb)
            if ends_hidden:
                out.append(np.array(current)); current = []
        if len(current) > 1: out.append(np.array(current))
    return out
