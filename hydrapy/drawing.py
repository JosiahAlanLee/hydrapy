"""Text and line drawings as textures, plus a small procedural plant generator.

    src(Text("ANCHORITE", font="DejaVuSerif-Bold", size=0.2))              # alpha = the letters
    leader.stain(src(Text("...", color=(1, 0.2, 0.1))), 1.0)               # text as a dye layer
    src(Lines("drawing.svg", stroke=2)).rotate(0.1)                         # a plotter SVG
    src(Lines(meadow(seed=3), stroke=1.5))                                  # procedural plants

``Text`` and ``Lines`` are static textures (like ``Image``): rendered once with
PIL, straight alpha, usable on all three backends.  Put several in a ``Bank``
for one per frame.
"""
from __future__ import annotations

import math
import os
import re
from typing import Optional, Sequence

import numpy as np

from .core import Texture

Point = tuple[float, float]
Polyline = list[Point]


# -----------------------------------------------------------------------------
# fonts
# -----------------------------------------------------------------------------

_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              os.path.expanduser("~/Library/Fonts"), "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]
_FALLBACKS = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf",
              "Helvetica.ttc", "FreeSans.ttf"]


def find_font(name: Optional[str], px: int):
    """A PIL font: a path, a file name found under the usual font directories, or a fallback."""
    from PIL import ImageFont
    candidates = []
    if name:
        candidates.append(name)
        if not name.lower().endswith((".ttf", ".otf", ".ttc")):
            candidates += [name + ".ttf", name + ".otf"]
    candidates += _FALLBACKS
    for c in candidates:
        if os.path.isfile(c):
            return ImageFont.truetype(c, px)
        for d in _FONT_DIRS:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower() == os.path.basename(c).lower():
                        try:
                            return ImageFont.truetype(os.path.join(root, f), px)
                        except Exception:
                            pass
    return ImageFont.load_default()


class Text(Texture):
    """Text rendered to a transparent canvas; rgb = ``color``, alpha = the letters.

    ``size`` is the cap height as a fraction of the canvas height; ``x``/``y`` place the block
    (0..1, y up); ``align`` is left/center/right; ``width`` wraps at that fraction of the canvas.
    """

    _counter = 0

    def __init__(self, text: str, font: Optional[str] = None, size: float = 0.15, color=(1.0, 1.0, 1.0),
                 x: float = 0.5, y: float = 0.5, align: str = "center", width: Optional[float] = None,
                 line_spacing: float = 1.15, canvas=(1024, 576), bg=None, supersample: int = 2):
        self.text, self.font, self.size, self.color = text, font, float(size), tuple(color)
        self.x, self.y, self.align, self.width = float(x), float(y), align, width
        self.line_spacing, self.canvas, self.bg, self.supersample = float(line_spacing), tuple(canvas), bg, int(supersample)
        Text._counter += 1
        self.name = f"text{Text._counter}"
        self._array = None

    def render(self) -> np.ndarray:
        """uint8 RGBA (H, W, 4), row 0 = top."""
        from PIL import Image as PILImage, ImageDraw
        ss = self.supersample
        W, H = self.canvas[0] * ss, self.canvas[1] * ss
        px = max(int(self.size * H), 4)
        font = find_font(self.font, px)
        lines = self.text.split("\n")
        if self.width:
            lines = _wrap(lines, font, self.width * W)
        img = PILImage.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        lh = px * self.line_spacing
        total = lh * len(lines)
        top = (1.0 - self.y) * H - total / 2
        for i, line in enumerate(lines):
            bbox = d.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            if self.align == "left":
                lx = self.x * W
            elif self.align == "right":
                lx = self.x * W - tw
            else:
                lx = self.x * W - tw / 2
            d.text((lx - bbox[0], top + i * lh - bbox[1]), line, font=font, fill=255)
        if ss > 1:
            img = img.resize(self.canvas, PILImage.LANCZOS)
        a = np.asarray(img, dtype=np.uint8)
        rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
        if self.bg is not None:
            bg = np.array([int(c * 255) for c in self.bg], dtype=np.uint8)
            rgba[..., :3] = bg
            rgba[..., 3] = 255
            fg = np.array([int(c * 255) for c in self.color], dtype=np.uint8)
            t = a[..., None].astype(np.float32) / 255.0
            rgba[..., :3] = (bg * (1 - t) + fg * t + 0.5).astype(np.uint8)
        else:
            rgba[..., :3] = [int(c * 255) for c in self.color]
            rgba[..., 3] = a
        return rgba

    def frame(self, i: int = 0) -> np.ndarray:
        return self.render()

    def array(self) -> np.ndarray:
        """float32 RGBA, row 0 = bottom (GL convention)."""
        if self._array is None:
            self._array = (self.render().astype(np.float32) / 255.0)[::-1].copy()
        return self._array


def _wrap(lines, font, max_w):
    out = []
    for line in lines:
        words = line.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


# -----------------------------------------------------------------------------
# line drawings
# -----------------------------------------------------------------------------

class Lines(Texture):
    """Polylines (or an SVG file / string) stroked onto a transparent canvas.

    ``source``: a list of polylines ``[[(x, y), ...], ...]`` in 0..1 coordinates (y up), or a
    path to an SVG, or SVG text.  SVGs are flattened (paths, polylines, polygons, lines, rects,
    circles, ellipses, with transforms) and fitted into the canvas with ``margin``.
    ``stroke`` is in canvas pixels.  rgb = ``color``, alpha = ink.
    """

    _counter = 0

    def __init__(self, source, stroke: float = 2.0, color=(1.0, 1.0, 1.0), canvas=(1024, 576), fit: bool = True,
                 margin: float = 0.05, supersample: int = 2, bg=None, flip_y: Optional[bool] = None):
        self.source, self.stroke, self.color = source, float(stroke), tuple(color)
        self.canvas, self.fit, self.margin, self.supersample, self.bg = tuple(canvas), fit, float(margin), int(supersample), bg
        self.flip_y = flip_y
        Lines._counter += 1
        self.name = f"lines{Lines._counter}"
        self._array = None
        self._polylines = None

    def polylines(self) -> list[Polyline]:
        if self._polylines is None:
            src = self.source
            if isinstance(src, str):
                text = open(src, encoding="utf-8").read() if os.path.isfile(src) else src
                polys = parse_svg(text)
                flip = True if self.flip_y is None else self.flip_y      # SVG y runs down
            else:
                polys = [list(map(tuple, p)) for p in src]
                flip = False if self.flip_y is None else self.flip_y
            if flip:
                polys = [[(x, -y) for x, y in p] for p in polys]
            self._polylines = polys
        return self._polylines

    def render(self) -> np.ndarray:
        from PIL import Image as PILImage, ImageDraw
        ss = self.supersample
        W, H = self.canvas[0] * ss, self.canvas[1] * ss
        polys = [p for p in self.polylines() if len(p) >= 2]
        img = PILImage.new("L", (W, H), 0)
        if polys:
            if self.fit:
                xs = [x for p in polys for x, _ in p]
                ys = [y for p in polys for _, y in p]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                bw, bh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
                m = self.margin
                sc = min((1 - 2 * m) * W / bw, (1 - 2 * m) * H / bh)
                ox = (W - bw * sc) / 2 - x0 * sc
                oy = (H - bh * sc) / 2 - y0 * sc
                tf = lambda x, y: (x * sc + ox, H - (y * sc + oy))
            else:
                tf = lambda x, y: (x * W, (1 - y) * H)
            d = ImageDraw.Draw(img)
            w = max(int(round(self.stroke * ss)), 1)
            for p in polys:
                pts = [tf(x, y) for x, y in p]
                d.line(pts, fill=255, width=w, joint="curve")
                if w > 2:   # round caps
                    r = w / 2
                    for (x, y) in (pts[0], pts[-1]):
                        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        if ss > 1:
            img = img.resize(self.canvas, PILImage.LANCZOS)
        a = np.asarray(img, dtype=np.uint8)
        rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
        if self.bg is not None:
            bg = np.array([int(c * 255) for c in self.bg], dtype=np.uint8)
            fg = np.array([int(c * 255) for c in self.color], dtype=np.uint8)
            t = a[..., None].astype(np.float32) / 255.0
            rgba[..., :3] = (bg * (1 - t) + fg * t + 0.5).astype(np.uint8)
            rgba[..., 3] = 255
        else:
            rgba[..., :3] = [int(c * 255) for c in self.color]
            rgba[..., 3] = a
        return rgba

    def frame(self, i: int = 0) -> np.ndarray:
        return self.render()

    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = (self.render().astype(np.float32) / 255.0)[::-1].copy()
        return self._array


# --- a small SVG flattener ------------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _mat_mul(a, b):
    """2x3 affine matrices [a, b, c, d, e, f] (SVG order), a ∘ b."""
    return [a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
            a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5]]


def _apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def parse_transform(s: Optional[str]):
    m = [1, 0, 0, 1, 0, 0]
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in _NUM.findall(args)]
        if name == "translate":
            t = [1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0]
        elif name == "scale":
            t = [v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0]
        elif name == "rotate":
            a = math.radians(v[0]); c, s_ = math.cos(a), math.sin(a)
            t = [c, s_, -s_, c, 0, 0]
            if len(v) > 2:
                t = _mat_mul(_mat_mul([1, 0, 0, 1, v[1], v[2]], t), [1, 0, 0, 1, -v[1], -v[2]])
        elif name == "matrix" and len(v) == 6:
            t = v
        elif name == "skewX":
            t = [1, 0, math.tan(math.radians(v[0])), 1, 0, 0]
        elif name == "skewY":
            t = [1, math.tan(math.radians(v[0])), 0, 1, 0, 0]
        else:
            continue
        m = _mat_mul(m, t)
    return m


def _bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def _quad(p0, p1, p2, n=10):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return out


def _arc(p0, rx, ry, phi, large, sweep, p1, n=16):
    """SVG elliptical arc flattened to points (endpoint parameterisation -> centre)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(phi)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1 = cp * dx + sp * dy
    y1 = -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if lam > 1:
        rx *= math.sqrt(lam); ry *= math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    coef = math.sqrt(max(num / den, 0)) if den else 0
    if large == sweep:
        coef = -coef
    cx1 = coef * rx * y1 / ry
    cy1 = -coef * ry * x1 / rx
    cx = cp * cx1 - sp * cy1 + (p0[0] + p1[0]) / 2
    cy = sp * cx1 + cp * cy1 + (p0[1] + p1[1]) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a
    t1 = ang(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    out = []
    for i in range(1, n + 1):
        t = t1 + dt * i / n
        x = rx * math.cos(t); y = ry * math.sin(t)
        out.append((cp * x - sp * y + cx, sp * x + cp * y + cy))
    return out


def parse_path(d: str) -> list[Polyline]:
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|" + _NUM.pattern, d)
    polys: list[Polyline] = []
    cur: Polyline = []
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl = None
    cmd = None
    i = 0

    def nums(k):
        nonlocal i
        v = [float(tokens[i + j]) for j in range(k)]
        i += k
        return v

    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZz]", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur:
                    cur.append(start)
                    polys.append(cur)
                cur = []
                pos = start
                continue
        if cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = nums(2)
            pos = (pos[0] + x, pos[1] + y) if rel else (x, y)
            if cur:
                polys.append(cur)
            cur = [pos]
            start = pos
            cmd = "l" if rel else "L"     # subsequent pairs are implicit lineto
        elif c == "L":
            x, y = nums(2)
            pos = (pos[0] + x, pos[1] + y) if rel else (x, y)
            cur.append(pos)
        elif c == "H":
            (x,) = nums(1)
            pos = (pos[0] + x if rel else x, pos[1])
            cur.append(pos)
        elif c == "V":
            (y,) = nums(1)
            pos = (pos[0], pos[1] + y if rel else y)
            cur.append(pos)
        elif c == "C":
            v = nums(6)
            p1, p2, p3 = (v[0], v[1]), (v[2], v[3]), (v[4], v[5])
            if rel:
                p1 = (pos[0] + p1[0], pos[1] + p1[1]); p2 = (pos[0] + p2[0], pos[1] + p2[1]); p3 = (pos[0] + p3[0], pos[1] + p3[1])
            cur += _bezier(pos, p1, p2, p3)
            last_ctrl, pos = p2, p3
        elif c == "S":
            v = nums(4)
            p2, p3 = (v[0], v[1]), (v[2], v[3])
            if rel:
                p2 = (pos[0] + p2[0], pos[1] + p2[1]); p3 = (pos[0] + p3[0], pos[1] + p3[1])
            p1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
            cur += _bezier(pos, p1, p2, p3)
            last_ctrl, pos = p2, p3
        elif c == "Q":
            v = nums(4)
            p1, p2 = (v[0], v[1]), (v[2], v[3])
            if rel:
                p1 = (pos[0] + p1[0], pos[1] + p1[1]); p2 = (pos[0] + p2[0], pos[1] + p2[1])
            cur += _quad(pos, p1, p2)
            last_ctrl, pos = p1, p2
        elif c == "T":
            v = nums(2)
            p2 = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
            p1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
            cur += _quad(pos, p1, p2)
            last_ctrl, pos = p1, p2
        elif c == "A":
            v = nums(7)
            p1 = (pos[0] + v[5], pos[1] + v[6]) if rel else (v[5], v[6])
            cur += _arc(pos, v[0], v[1], v[2], bool(v[3]), bool(v[4]), p1)
            pos = p1
        if c not in "CSQT":
            last_ctrl = None
    if cur:
        polys.append(cur)
    return polys


def parse_svg(text: str) -> list[Polyline]:
    """Flatten an SVG document into polylines in its user units (y down, as in the file)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    polys: list[Polyline] = []

    def local(tag):
        return tag.split("}")[-1]

    def walk(el, m):
        m = _mat_mul(m, parse_transform(el.get("transform")))
        tag = local(el.tag)
        shapes: list[Polyline] = []
        if tag == "path" and el.get("d"):
            shapes = parse_path(el.get("d"))
        elif tag in ("polyline", "polygon") and el.get("points"):
            v = [float(x) for x in _NUM.findall(el.get("points"))]
            pts = list(zip(v[0::2], v[1::2]))
            if tag == "polygon" and pts:
                pts.append(pts[0])
            shapes = [pts]
        elif tag == "line":
            shapes = [[(float(el.get("x1", 0)), float(el.get("y1", 0))), (float(el.get("x2", 0)), float(el.get("y2", 0)))]]
        elif tag == "rect":
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
            shapes = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
        elif tag in ("circle", "ellipse"):
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            rx = float(el.get("r", el.get("rx", 0)))
            ry = float(el.get("r", el.get("ry", 0)))
            shapes = [[(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 48)]]
        for sh in shapes:
            polys.append([_apply(m, x, y) for x, y in sh])
        for child in el:
            walk(child, m)

    walk(root, [1, 0, 0, 1, 0, 0])
    return polys


def to_svg(polylines: Sequence[Polyline], size=(1024, 576), stroke: float = 1.5, margin: float = 0.05) -> str:
    """Polylines in 0..1 (y up) -> an SVG string, for the pen plotter."""
    W, H = size
    xs = [x for p in polylines for x, _ in p] or [0, 1]
    ys = [y for p in polylines for _, y in p] or [0, 1]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sc = min((1 - 2 * margin) * W / max(x1 - x0, 1e-9), (1 - 2 * margin) * H / max(y1 - y0, 1e-9))
    ox = (W - (x1 - x0) * sc) / 2 - x0 * sc
    oy = (H - (y1 - y0) * sc) / 2 - y0 * sc
    parts = []
    for p in polylines:
        if len(p) < 2:
            continue
        pts = " ".join(f"{x * sc + ox:.2f},{H - (y * sc + oy):.2f}" for x, y in p)
        parts.append(f'<polyline points="{pts}" />')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'<g fill="none" stroke="black" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">\n'
            + "\n".join(parts) + "\n</g>\n</svg>\n")


# -----------------------------------------------------------------------------
# procedural plants (a stand-in for your own line drawings)
# -----------------------------------------------------------------------------

def _rng(seed):
    return np.random.default_rng(int(seed * 1000003) & 0xFFFFFFFF if isinstance(seed, float) else seed)


def plant(kind: str = "branch", seed: int = 0, x: float = 0.5, height: float = 0.8, lean: float = 0.0) -> list[Polyline]:
    """One plant as polylines in 0..1 coordinates (y up), rooted at (x, 0).

    kinds: 'branch' (woody, recursive), 'grass' (a tuft of tapered blades), 'umbel' (a stalk with a
    flat-topped flower head), 'fern' (a frond with pinnae), 'reed' (jointed stem with a seed head).
    """
    rng = _rng(seed)
    polys: list[Polyline] = []

    def stem(p0, ang, length, curve, n=10):
        pts = [p0]
        a = ang
        step = length / n
        px, py = p0
        for i in range(n):
            a += curve / n + rng.normal(0, 0.02)
            px += math.cos(a) * step
            py += math.sin(a) * step
            pts.append((px, py))
        return pts

    base_ang = math.pi / 2 + lean

    if kind == "branch":
        def grow(p0, ang, length, depth):
            s = stem(p0, ang, length, rng.normal(0, 0.3), n=8)
            polys.append(s)
            if depth <= 0 or length < 0.03:
                return
            k = rng.integers(2, 4)
            for j in range(k):
                t = rng.uniform(0.35, 1.0)
                idx = min(int(t * (len(s) - 1)), len(s) - 2)
                p = s[idx]
                a2 = math.atan2(s[idx + 1][1] - s[idx][1], s[idx + 1][0] - s[idx][0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
                grow(p, a2, length * rng.uniform(0.5, 0.72), depth - 1)
        grow((x, 0.0), base_ang, height * 0.5, 4)

    elif kind == "grass":
        for j in range(int(rng.integers(7, 14))):
            a = base_ang + rng.normal(0, 0.45)
            polys.append(stem((x + rng.normal(0, 0.01), 0.0), a, height * rng.uniform(0.4, 1.0), -rng.uniform(0.3, 1.2) * np.sign(a - math.pi / 2 + 1e-6), n=12))

    elif kind == "umbel":
        s = stem((x, 0.0), base_ang, height * 0.75, rng.normal(0, 0.25), n=12)
        polys.append(s)
        top = s[-1]
        ang_top = math.atan2(s[-1][1] - s[-2][1], s[-1][0] - s[-2][0])
        rays = int(rng.integers(7, 12))
        for j in range(rays):
            a = ang_top + (j / (rays - 1) - 0.5) * 2.4
            r = stem(top, a, height * 0.18, 0.0, n=4)
            polys.append(r)
            tip = r[-1]
            for k in range(5):
                a2 = a + (k / 4 - 0.5) * 1.6
                polys.append(stem(tip, a2, height * 0.04, 0.0, n=2))
        for j in range(int(rng.integers(1, 3))):   # a leaf or two on the stalk
            idx = int(rng.uniform(0.2, 0.6) * len(s))
            polys += leaf(s[idx], base_ang + rng.choice([-1, 1]) * 1.1, height * 0.18, rng)

    elif kind == "fern":
        s = stem((x, 0.0), base_ang, height * 0.9, rng.normal(0, 0.5), n=14)
        polys.append(s)
        for i in range(2, len(s) - 1):
            a = math.atan2(s[i + 1][1] - s[i][1], s[i + 1][0] - s[i][0])
            frac = 1 - i / len(s)
            for sign in (-1, 1):
                pin = stem(s[i], a + sign * 1.1, height * 0.22 * frac, sign * 0.4, n=6)
                polys.append(pin)
                for q in range(1, len(pin) - 1):
                    a2 = math.atan2(pin[q + 1][1] - pin[q][1], pin[q + 1][0] - pin[q][0])
                    for s2 in (-1, 1):
                        polys.append(stem(pin[q], a2 + s2 * 1.2, height * 0.05 * frac, 0.0, n=2))

    elif kind == "reed":
        s = stem((x, 0.0), base_ang, height, rng.normal(0, 0.15), n=12)
        polys.append(s)
        for i in range(3, len(s) - 2, 3):      # joints
            a = math.atan2(s[i + 1][1] - s[i][1], s[i + 1][0] - s[i][0])
            polys.append([(s[i][0] - math.sin(a) * 0.008, s[i][1] + math.cos(a) * 0.008), (s[i][0] + math.sin(a) * 0.008, s[i][1] - math.cos(a) * 0.008)])
            polys.append(stem(s[i], a + rng.choice([-1, 1]) * 0.5, height * 0.25, -0.3, n=5))
        head = stem(s[-1], math.atan2(s[-1][1] - s[-2][1], s[-1][0] - s[-2][0]), height * 0.12, 0.3, n=6)
        for q in range(len(head)):
            for s2 in (-1, 1):
                polys.append([head[q], (head[q][0] + s2 * 0.012, head[q][1] - 0.01)])
    else:
        raise ValueError(f"unknown plant kind {kind!r}")
    return polys


def leaf(p0, ang, length, rng, n=10) -> list[Polyline]:
    """A simple leaf outline plus midrib."""
    out = []
    w = length * 0.35
    ca, sa = math.cos(ang), math.sin(ang)
    outline = []
    for sign in (1, -1):
        pts = []
        rngs = range(n + 1) if sign == 1 else range(n, -1, -1)
        for i in rngs:
            t = i / n
            r = w * math.sin(math.pi * t) ** 0.8
            px = p0[0] + ca * length * t - sa * r * sign
            py = p0[1] + sa * length * t + ca * r * sign
            pts.append((px, py))
        outline += pts
    out.append(outline)
    out.append([p0, (p0[0] + ca * length * 0.9, p0[1] + sa * length * 0.9)])
    return out


def meadow(seed: int = 0, n: int = 12, kinds: Sequence[str] = ("grass", "umbel", "branch", "fern", "reed")) -> list[Polyline]:
    """A row of plants across the frame: Driftless roadside, as polylines."""
    rng = _rng(seed)
    polys: list[Polyline] = []
    for i in range(n):
        kind = kinds[int(rng.integers(len(kinds)))]
        x = (i + rng.uniform(0.2, 0.8)) / n
        polys += plant(kind, seed=int(rng.integers(1 << 30)), x=x, height=rng.uniform(0.35, 0.95), lean=rng.normal(0, 0.12))
    return polys
