"""Life-like cellular automata with a rule per pixel, asynchronous updates and spontaneous flips.

``ca`` is one generation of an outer-totalistic automaton on the red channel. Feed it back:

    src(o1).ca(mode=0, p=1.0, flip=0.0005).out(o1)        # Conway's Life, seeded by fluctuation

* ``mode`` picks the rule from ``CA_RULES`` (floor, mod 16). It is a float input, so a chain makes
  it a field: regions of the frame run different rules and meet at moving fronts.
* ``p`` is the chance a cell updates this step; below 1 the automaton is asynchronous and stops
  ticking in lockstep. Also a field.
* ``flip`` is the chance a cell flips (dead to alive, anything else to dead) this step: the
  pixel-level fluctuation that keeps a field from settling.
* Generations rules (states > 2) leave a refractory tail: dying cells step down through levels
  of red before they can be born again.

State: red is the cell's level (1 alive, 0 dead, between = dying), green a trail that fades by
``fade`` per step, blue the age of a living cell (0 at birth, rising to 1). Colour it downstream.
"""
from __future__ import annotations

import re

import numpy as np

from .core import Op, register
from .noise import h31
from .ops import F, I, bc

CA_RULES = (                  # (name, rule, states): rule numbers in the docs are these indices
    ("life", "B3/S23", 2),
    ("highlife", "B36/S23", 2),
    ("day-and-night", "B3678/S34678", 2),
    ("seeds", "B2/S", 2),
    ("brians-brain", "B2/S", 3),
    ("star-wars", "B2/S345", 4),
    ("maze", "B3/S12345", 2),
    ("coral", "B3/S45678", 2),
    ("anneal", "B4678/S35678", 2),
    ("diamoeba", "B35678/S5678", 2),
    ("morley", "B368/S245", 2),
    ("2x2", "B36/S125", 2),
    ("replicator", "B1357/S1357", 2),
    ("frogs", "B34/S12", 3),
    ("walled-cities", "B45678/S2345", 2),
    ("sticks", "B2/S3456", 6),
)
CA_MODES = {name: i for i, (name, _, _) in enumerate(CA_RULES)}


def rule_masks(rule: str) -> tuple[int, int]:
    """'B36/S23' -> (birth bitmask, survival bitmask) over neighbour counts 0..8."""
    m = re.fullmatch(r"B([0-8]*)/S([0-8]*)", rule.replace(" ", "").upper())
    if not m:
        raise ValueError(f"not a B/S rule: {rule!r}")
    return tuple(sum(1 << int(c) for c in set(part)) for part in m.groups())


_TABLE = np.array([(*rule_masks(r), c) for _, r, c in CA_RULES], np.int64)       # (16, 3): born, survive, states


def _glsl_table() -> str:
    lines = ["vec3 _carule(float m) {", "  m = mod(floor(m), 16.0);"]
    for i, (b, s, c) in enumerate(_TABLE):
        lines.append(f"  if (m < {i + 0.5}) return vec3({float(b)}, {float(s)}, {float(c)});")
    lines.append("  return vec3(8.0, 12.0, 2.0);")
    lines.append("}")
    lines.append("float _cabit(float mask, float n) { return mod(floor(mask / exp2(n)), 2.0); }")
    return "\n".join(lines)


def _np_ca(tex, st, mode, p, flip, fade, seed, env):
    cur = env.sample(tex, st)
    n = np.zeros(cur.shape[:-1], np.int64)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i or j:
                n += env.sample_offset(tex, st, (i, j))[..., 0] >= 0.999
    rule = _TABLE[np.mod(np.floor(bc(mode, st)), 16).astype(np.intp)]
    born, survive, C = rule[..., 0], rule[..., 1], rule[..., 2].astype(np.float32)
    q = np.floor(cur[..., 0] * (C - 1) + 0.5)
    alive = cur[..., 0] >= 0.999
    nq = np.where(alive, np.where((survive >> n) & 1, C - 1, C - 2),
                  np.where(q > 0.5, q - 1, np.where((born >> n) & 1, C - 1, 0))).astype(np.float32)
    cell = np.floor(st * env.res).astype(np.float32)
    tick = np.float32(np.fmod(np.floor(np.float32(env.ctx.time) * np.float32(240.0)), 16384.0)
                      + np.float32(seed) * np.float32(17.0) + np.float32(env.pass_index) * np.float32(0.25))
    h1 = h31(np.stack([cell[..., 0], cell[..., 1], np.full_like(cell[..., 0], tick)], -1))
    h2 = h31(np.stack([cell[..., 0] + 517.0, cell[..., 1] + 91.0, np.full_like(cell[..., 0], tick)], -1))
    nq = np.where(h1 >= bc(p, st), q, nq)
    nq = np.where(h2 < bc(flip, st), np.where(nq >= C - 1, 0, C - 1), nq)
    now = nq >= C - 1.5
    r = nq / (C - 1)
    trail = np.maximum(cur[..., 1] * bc(fade, st), now.astype(np.float32))
    age = np.where(now, np.where(alive, np.minimum(cur[..., 2] + 1 / 48, 1.0), 0.0), 0.0)
    return np.stack([r, trail, age, np.ones_like(r)], -1).astype(np.float32)


register(Op("ca", "texture", [F("mode", 0), F("p", 1), F("flip", 0), F("fade", 0.9), F("seed", 0, scalar=True),
                              I("iterations", 1)],
            np=_np_ca, iterations=True, glsl_helpers=_glsl_table(),
            doc="A Life-like automaton on red (1 alive, between = dying), with a rule per pixel: `mode` picks "
                "from CA_RULES (0 Life, 1 HighLife, 2 Day & Night, 3 Seeds, 4 Brian's Brain, 5 Star Wars, 6 Maze, "
                "7 Coral, 8 Anneal, 9 Diamoeba, 10 Morley, 11 2x2, 12 Replicator, 13 Frogs, 14 Walled Cities, "
                "15 Sticks; a chain makes regions). `p`: chance a cell updates (asynchronous below 1). `flip`: "
                "chance a cell flips. Green: a trail fading by `fade`; blue: age of a living cell. Feed it back.",
            glsl="""vec2 px = 1.0 / resolution;
  vec3 rule = _carule(mode);
  float C = rule.z;
  float n = 0.0;
  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
    if (i == 0 && j == 0) continue;
    n += step(0.999, texture2D(_tex, fract(_st + vec2(float(i), float(j))*px)).r);
  }
  vec4 cur = texture2D(_tex, fract(_st));
  float q = floor(cur.r * (C - 1.0) + 0.5);
  bool alive = cur.r >= 0.999;
  float nq;
  if (alive) nq = _cabit(rule.y, n) > 0.5 ? C - 1.0 : C - 2.0;
  else if (q > 0.5) nq = q - 1.0;
  else nq = _cabit(rule.x, n) > 0.5 ? C - 1.0 : 0.0;
  vec2 cell = floor(_st * resolution);
  float tick = mod(floor(time * 240.0), 16384.0) + seed * 17.0 + _pass * 0.25;
  if (_h31(vec3(cell, tick)) >= p) nq = q;                                  // sits this step out
  if (_h31(vec3(cell + vec2(517.0, 91.0), tick)) < flip) nq = nq >= C - 1.0 ? 0.0 : C - 1.0;   // fluctuates
  float now = step(C - 1.5, nq);
  float trail = max(cur.g * fade, now);
  float age = now > 0.5 ? (alive ? min(cur.b + 1.0 / 48.0, 1.0) : 0.0) : 0.0;
  return vec4(nq / (C - 1.0), trail, age, 1.0);"""))
