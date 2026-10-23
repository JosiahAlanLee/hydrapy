"""Mathematical marbling (after Lu, Jaffer, Jin, Zhao and Mao, "Mathematical Marbling", IEEE CG&A 2012).

Suminagashi and ebru are ink floating on water, pushed around by drops and drawn through by
styluses and combs.  Each of those moves has a closed-form, area-preserving, *exactly invertible*
map, so each is a coordinate op here (the op is the inverse: where did the ink at this point come
from), and a chain is the marbling's history in the order it happened:

    water = solid(0.93, 0.9, 0.82)
    water.layer(disc(0.4, 0.5, 0.15).color(0.1, 0.1, 0.3)) \\      # a drop of indigo
         .inkdrop(0.6, 0.5, 0.12).layer(disc(0.6, 0.5, 0.12).color(0.6, 0.1, 0.1)) \\   # a second drop pushes the first aside
         .tine(1.5708, 0, Lfo(0.05, lo=-0.3, hi=0.3), 0.03, 0.12)  # a comb drawn through both, back and forth

No feedback is needed and every parameter is a signal, so a pattern can be combed and un-combed
live.  ``disc`` is a round (aspect-correct) drop of ink: white with alpha = ink, for ``layer``.
Distances are in units of the frame height; positions in 0..1 like the rest of hydrapy.
"""
from __future__ import annotations

import numpy as np

from .core import Op, register
from .ops import F, bc, smoothstep

_ASP = "vec2 asp = vec2(resolution.x / resolution.y, 1.0);"


def _asp(env):
    return np.array([env.res[0] / env.res[1], 1.0], np.float32)


def _np_inkdrop(st, x, y, radius, env):
    asp = _asp(env)
    c = np.stack([bc(x, st), bc(y, st)], axis=-1)
    q = (st - c) * asp
    d2 = (q * q).sum(-1)
    r2 = bc(radius, st) ** 2
    k = np.sqrt(np.clip(1.0 - r2 / np.maximum(d2, 1e-12), 0.0, None))
    k = np.where(d2 <= r2, 0.0, k)
    return (c + q * k[..., None] / asp).astype(np.float32)


register(Op("inkdrop", "coord", [F("x", 0.5), F("y", 0.5), F("radius", 0.1)], np=_np_inkdrop,
            doc="A drop of ink falling on the floating pattern at (x, y): everything is pushed outward, area preserved (|p'-c|^2 = |p-c|^2 + r^2). Inside the drop the pattern's own colour at the centre shows; layer a disc() of ink there. Drops compose: two of radius a and b equal one of sqrt(a^2 + b^2).",
            glsl=_ASP + """
  vec2 c = vec2(x, y);
  vec2 q = (_st - c) * asp;
  float d2 = dot(q, q);
  float r2 = radius * radius;
  float k = d2 <= r2 ? 0.0 : sqrt(max(1.0 - r2 / max(d2, 1e-12), 0.0));
  return c + q * k / asp;"""))


def _np_tine(st, angle, offset, amount, sharpness, period, env):
    asp = _asp(env)
    q = (st - 0.5) * asp
    a = bc(angle, st)
    mx, my = np.cos(a), np.sin(a)
    s = (-np.sin(a)) * q[..., 0] + np.cos(a) * q[..., 1] - bc(offset, st)
    per = bc(period, st)
    s = np.where(per > 0, s - per * np.floor(s / np.where(per > 0, per, 1.0) + 0.5), s)
    disp = bc(amount, st) * np.power(0.5, np.abs(s) / np.maximum(bc(sharpness, st), 1e-6))
    src = q - disp[..., None] * np.stack([mx, my], axis=-1)
    return (src / asp + 0.5).astype(np.float32)


register(Op("tine", "coord", [F("angle", 0), F("offset", 0), F("amount", 0.2), F("sharpness", 0.03), F("period", 0)],
            np=_np_tine,
            doc="A stylus drawn through the ink along a line at `angle` (radians), `offset` from the centre: the ink on the line moves `amount`, falling off by half every `sharpness` away from it. `period` > 0 makes it a comb (a tine every `period`). Exactly invertible, so drawing it back (negative amount) un-marbles.",
            glsl=_ASP + """
  vec2 q = (_st - 0.5) * asp;
  vec2 m = vec2(cos(angle), sin(angle));
  vec2 n = vec2(-m.y, m.x);
  float s = dot(q, n) - offset;
  if (period > 0.0) s -= period * floor(s / period + 0.5);
  float disp = amount * pow(0.5, abs(s) / max(sharpness, 1e-6));
  return (q - disp * m) / asp + 0.5;"""))


def _np_swirl(st, x, y, amount, radius, sharpness, env):
    asp = _asp(env)
    c = np.stack([bc(x, st), bc(y, st)], axis=-1)
    q = (st - c) * asp
    r = np.sqrt((q * q).sum(-1))
    th = -bc(amount, st) * np.power(0.5, np.abs(r - bc(radius, st)) / np.maximum(bc(sharpness, st), 1e-6)) / np.maximum(r, 1e-6)
    cs, sn = np.cos(th), np.sin(th)
    src = np.stack([cs * q[..., 0] - sn * q[..., 1], sn * q[..., 0] + cs * q[..., 1]], axis=-1)
    return (c + src / asp).astype(np.float32)


register(Op("swirl", "coord", [F("x", 0.5), F("y", 0.5), F("amount", 0.2), F("radius", 0.25), F("sharpness", 0.05)],
            np=_np_swirl,
            doc="A stylus drawn round a circle of `radius` about (x, y): ink on the circle moves `amount` along it, falling off by half every `sharpness` inside and out.",
            glsl=_ASP + """
  vec2 c = vec2(x, y);
  vec2 q = (_st - c) * asp;
  float r = length(q);
  float th = -amount * pow(0.5, abs(r - radius) / max(sharpness, 1e-6)) / max(r, 1e-6);
  float cs = cos(th), sn = sin(th);
  return c + vec2(cs * q.x - sn * q.y, sn * q.x + cs * q.y) / asp;"""))


def _np_ripple(st, angle, amplitude, wavelength, phase, env):
    asp = _asp(env)
    q = (st - 0.5) * asp
    a = bc(angle, st)
    m = np.stack([np.cos(a), np.sin(a)], axis=-1)
    s = (-np.sin(a)) * q[..., 0] + np.cos(a) * q[..., 1]
    disp = bc(amplitude, st) * np.sin(6.283185307 * s / np.maximum(bc(wavelength, st), 1e-6) + bc(phase, st))
    return ((q - disp[..., None] * m) / asp + 0.5).astype(np.float32)


register(Op("ripple", "coord", [F("angle", 0), F("amplitude", 0.03), F("wavelength", 0.2), F("phase", 0)], np=_np_ripple,
            doc="A wavy stroke across the whole bath: ink moves along `angle` by a sine of its distance across it (the wavy comb of ebru). Exactly invertible.",
            glsl=_ASP + """
  vec2 q = (_st - 0.5) * asp;
  vec2 m = vec2(cos(angle), sin(angle));
  float s = dot(q, vec2(-m.y, m.x));
  float disp = amplitude * sin(6.283185307 * s / max(wavelength, 1e-6) + phase);
  return (q - disp * m) / asp + 0.5;"""))


def _np_disc(st, x, y, radius, softness, env):
    asp = _asp(env)
    q = (st - np.stack([bc(x, st), bc(y, st)], axis=-1)) * asp
    d = np.sqrt((q * q).sum(-1))
    s = np.maximum(bc(softness, st), 1e-6)
    a = 1.0 - smoothstep(bc(radius, st) - s, bc(radius, st) + s, d)
    return np.stack([np.ones_like(a), np.ones_like(a), np.ones_like(a), a], axis=-1).astype(np.float32)


register(Op("disc", "src", [F("x", 0.5), F("y", 0.5), F("radius", 0.1), F("softness", 0.002)], np=_np_disc,
            doc="A round drop of ink at (x, y), aspect-correct, white with alpha = ink: .color() it and layer() it into the pattern (it matches inkdrop()'s hole).",
            glsl=_ASP + """
  float d = length((_st - vec2(x, y)) * asp);
  float s = max(softness, 1e-6);
  return vec4(vec3(1.0), 1.0 - smoothstep(radius - s, radius + s, d));"""))
