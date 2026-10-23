"""Pixel-feedback glitch ops after aebrer's entropy-locked recursive glitch textures (CC0):
horizontal/vertical tears and samples of the image shredded back onto itself.

They are coordinate ops, so they cost nothing extra and work on any source; the texture is
whatever comes before them.  Drive ``seed`` with an ``EntropyLock`` and feed the output back:

    lock = EntropyLock("ooYourHash", rate=24, lock=0.01)
    src(o0).shred(lock, 4, 0.25, 1.5).tear(lock * 7, 3, 0.1, 0.3).blend(osc(20), 0.05).out(o0)

Because the locked stream keeps returning to earlier values, the same tears and samples recur,
and recurrence on top of feedback is what grows the fractal texture.
"""
from __future__ import annotations

import numpy as np

from .core import Op, register
from .noise import h11, h22
from .ops import F, I, bc

register(Op("tear", "coord", [F("seed", 0), I("count", 3), F("size", 0.12), F("shift", 0.3), F("vertical", 0)],
            np=None, doc="`count` horizontal bands (vertical=1: columns) torn sideways by up to `shift`; which bands and how far come from `seed`.",
            glsl="""vec2 uv = _st;
  for (int k = 0; k < 8; k++) {
    if (k >= count) break;
    vec2 r = _h22(vec2(float(k) * 7.3, seed * 131.0));
    float r2 = _h11(float(k) * 3.1 + seed * 131.0 + 17.0);
    float pos = r.x;
    float h = size * (0.2 + 0.8 * r.y);
    float d = (r2 - 0.5) * 2.0 * shift;
    if (vertical > 0.5) { if (uv.x > pos && uv.x < pos + h) uv.y += d; }
    else { if (uv.y > pos && uv.y < pos + h) uv.x += d; }
  }
  return uv;"""))


def _np_tear(st, seed, count, size, shift, vertical, env):
    uv = st.copy()
    seed = bc(seed, st)
    for k in range(int(count)):
        r = h22(np.stack([np.full_like(seed, k * 7.3), seed * 131.0], axis=-1))
        r2 = h11(k * 3.1 + seed * 131.0 + 17.0)
        pos = r[..., 0]
        h = size * (0.2 + 0.8 * r[..., 1])
        d = (r2 - 0.5) * 2.0 * shift
        if np.ndim(vertical) == 0 and vertical > 0.5 or np.ndim(vertical) and np.mean(vertical) > 0.5:
            m = (uv[..., 0] > pos) & (uv[..., 0] < pos + h)
            uv[..., 1] = np.where(m, uv[..., 1] + d, uv[..., 1])
        else:
            m = (uv[..., 1] > pos) & (uv[..., 1] < pos + h)
            uv[..., 0] = np.where(m, uv[..., 0] + d, uv[..., 0])
    return uv


from .core import REGISTRY  # noqa: E402
REGISTRY["tear"].np = _np_tear


register(Op("shred", "coord", [F("seed", 0), I("count", 4), F("size", 0.2), F("scale", 1.0)],
            np=None, doc="`count` rectangular samples of the image copied onto other places (magnified by `scale`); which, where and how big come from `seed`. Later copies copy from earlier ones.",
            glsl="""vec2 uv = _st;
  for (int k = 0; k < 8; k++) {
    if (k >= count) break;
    vec2 s = _h22(vec2(float(k) * 5.7 + 3.0, seed * 131.0));
    vec2 d = _h22(vec2(float(k) * 9.1 + 11.0, seed * 131.0 + 1.0));
    vec2 wh = size * (0.3 + 0.7 * _h22(vec2(float(k) * 2.3 + 29.0, seed * 131.0 + 2.0)));
    wh.x *= resolution.y / resolution.x;
    if (uv.x >= d.x && uv.x < d.x + wh.x && uv.y >= d.y && uv.y < d.y + wh.y) {
      uv = s + (uv - d) / scale;
    }
  }
  return uv;"""))


def _np_shred(st, seed, count, size, scale, env):
    uv = st.copy()
    seed = bc(seed, st)
    asp = env.res[1] / env.res[0]
    for k in range(int(count)):
        s_ = h22(np.stack([np.full_like(seed, k * 5.7 + 3.0), seed * 131.0], axis=-1))
        d_ = h22(np.stack([np.full_like(seed, k * 9.1 + 11.0), seed * 131.0 + 1.0], axis=-1))
        wh = size * (0.3 + 0.7 * h22(np.stack([np.full_like(seed, k * 2.3 + 29.0), seed * 131.0 + 2.0], axis=-1)))
        wh = np.stack([wh[..., 0] * asp, wh[..., 1]], axis=-1)
        m = (uv[..., 0] >= d_[..., 0]) & (uv[..., 0] < d_[..., 0] + wh[..., 0]) & (uv[..., 1] >= d_[..., 1]) & (uv[..., 1] < d_[..., 1] + wh[..., 1])
        moved = s_ + (uv - d_) / scale
        uv = np.where(m[..., None], moved, uv)
    return uv


REGISTRY["shred"].np = _np_shred
