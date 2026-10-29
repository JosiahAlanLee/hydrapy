# SPDX-License-Identifier: Apache-2.0
# Cellular update semantics adapted from @thi.ng/cellular.
# Copyright 2022-2026 Karsten Schmidt. See NOTICE and LICENSE-THI-NG-APACHE-2.0.txt.
# Python vectorization, time-based source and glyph rendering: hydrapy contributors, 2026.
"""Interacting 1-D automata with memory, as used by the open C-SCAPE engine.

This adapts the public engine, not C-SCAPE/ASCII-SCAPE's curated artwork rules,
palettes, typography, or edition code. NumPy handles one row per generation;
the finished texture can be composed by either hydrapy renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import FrameSource


WOLFRAM3 = ((-1, 0), (0, 0), (1, 0))
WOLFRAM5 = ((-2, 0), *WOLFRAM3, (2, 0))


@dataclass(frozen=True)
class CARule:
    """Bits select occupied neighborhoods; kernel entries are (x offset, history lag).

    The first kernel entry is the least significant neighborhood bit, following
    thi.ng. With ``positional=False`` only the occupied neighbor count matters.
    ``reset=False`` saturates age; ``True`` resets at ``states``. For classic
    binary automata use states=2 and reset=False. Negative history lags are ignored.
    """
    rule: int
    kernel: tuple = WOLFRAM3
    states: int = 2
    reset: bool = False
    positional: bool = True

    def __post_init__(self):
        kernel = tuple(tuple(k) for k in self.kernel)
        if not 1 <= len(kernel) <= 16 or any(len(k) != 2 or any(int(v) != v for v in k) for k in kernel):
            raise ValueError("kernel requires 1..16 integer (offset, lag) pairs")
        if max(k[1] for k in kernel) > 64:
            raise ValueError("history lag must be at most 64 generations")
        if int(self.states) != self.states or not 2 <= self.states <= 256:
            raise ValueError("states must be an integer between 2 and 256")
        rule = int(self.rule, 0) if isinstance(self.rule, str) else int(self.rule)
        if rule < 0 or (not isinstance(self.rule, str) and rule != self.rule):
            raise ValueError("rule must be a nonnegative integer or integer string")
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "kernel", tuple((int(x), int(y)) for x, y in kernel))
        object.__setattr__(self, "states", int(self.states))


class MultiCA1D:
    """Shared cell history with per-cell rule selection; memory is bounded by the kernel."""

    def __init__(self, rules, width, wrap=True):
        self.rules = tuple(r if isinstance(r, CARule) else CARule(**r) for r in rules)
        if not 1 <= len(self.rules) <= 256:
            raise ValueError("provide 1..256 rules")
        if int(width) != width or width < 1:
            raise ValueError("width must be a positive integer")
        self.width, self.wrap = int(width), bool(wrap)
        lag = max(0, max(y for r in self.rules for _, y in r.kernel))
        self.history = np.zeros((lag + 1, self.width), np.uint8)
        self.mask = np.zeros(self.width, np.uint8)
        self.num_states = max(r.states for r in self.rules)
        self._lookups = [np.array([(r.rule >> n) & 1 for n in range(
            (1 << len(r.kernel)) if r.positional else len(r.kernel) + 1)], dtype=bool) for r in self.rules]

    @property
    def current(self):
        return self.history[0]

    def update(self):
        if np.any(self.mask >= len(self.rules)):
            raise ValueError("mask entries must select an existing rule")
        old = self.current.astype(np.uint16)
        result = np.zeros(self.width, np.uint8)
        x = np.arange(self.width)
        for i, (rule, lookup) in enumerate(zip(self.rules, self._lookups)):
            indices = np.zeros(self.width, np.uint32)
            for bit, (dx, lag) in enumerate(rule.kernel):
                if lag < 0:
                    continue
                xx = x + dx
                live = self.history[lag, xx % self.width] != 0
                if not self.wrap:
                    live &= (xx >= 0) & (xx < self.width)
                indices += live.astype(np.uint32) * ((1 << bit) if rule.positional else 1)
            age = old + 1
            age = np.where(age >= rule.states, 0, age) if rule.reset else np.minimum(age, rule.states - 1)
            selected = self.mask == i
            result[selected] = np.where(lookup[indices], age, 0)[selected].astype(np.uint8)
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = result
        return self.current


class Cellular(FrameSource):
    """Scrolling cellular history driven by seconds, independent of video frame rate.

    ``rate`` is generations/second; for 24 generations/beat at 120 BPM use 48.
    ``rules`` are CARule objects, or mappings with the same fields. ``stripes``
    selects the width of alternating rule regions. ``perturb`` is the per-cell
    probability of injecting a new state each generation. Seed and fixed update
    rate make jumps/backwards seeks reproducible (backwards seeks replay).
    ``chars`` optionally renders the history as a glyph grid with cell=(w,h).
    The displayed window is prefilled at t=0, so the first image isn't blank.
    """

    def __init__(self, rules=None, width=512, height=256, rate=96, seed=0,
                 stripes=64, density=.15, perturb=.0005, wrap=True, chars=None, cell=(8, 12)):
        if rules is None:
            rules = [CARule(0x73ed2ac2, WOLFRAM5, 64, True),
                     CARule(0xef14e4ca, WOLFRAM5, 64, True)]
        self.rules = tuple(r if isinstance(r, CARule) else CARule(**r) for r in rules)
        if int(width) != width or width < 1 or int(height) != height or height < 1:
            raise ValueError("width and height must be positive integers")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be positive and finite")
        if int(stripes) != stripes or stripes < 1 or not 0 <= density <= 1 or not 0 <= perturb <= 1:
            raise ValueError("stripes must be positive; density and perturb must be in 0..1")
        self.width, self.height, self.rate = int(width), int(height), float(rate)
        self.seed, self.stripes, self.density, self.perturb = int(seed), int(stripes), float(density), float(perturb)
        self.wrap, self.chars = bool(wrap), chars
        self.cell = tuple(cell)
        if chars is not None and (not isinstance(chars, str) or len(chars) < 2):
            raise ValueError("chars must contain at least two glyphs")
        if len(self.cell) != 2 or any(int(v) != v or v < 2 for v in self.cell):
            raise ValueError("glyph cell must have two integer dimensions >= 2")
        self.cell = tuple(map(int, self.cell))
        self._atlas = None
        super().__init__(self._draw)
        self._reset()

    def _reset(self):
        self.sim = MultiCA1D(self.rules, self.width, self.wrap)
        self.rng = np.random.default_rng(self.seed)
        self.sim.mask[:] = (np.arange(self.width) // self.stripes) % len(self.rules)
        occupied = self.rng.random(self.width) < self.density
        self.sim.current[occupied] = self.rng.integers(1, self.sim.num_states, occupied.sum(), dtype=np.uint8)
        self._rows = np.zeros((self.height, self.width), np.uint8)
        self._head = self._generation = 0
        for _ in range(self.height):
            self._step()
        self._generation = 0

    def _step(self):
        if self.perturb:
            occupied = self.rng.random(self.width) < self.perturb
            self.sim.current[occupied] = self.rng.integers(1, self.sim.num_states, occupied.sum(), dtype=np.uint8)
        self._rows[self._head] = self.sim.update()
        self._head = (self._head + 1) % self.height
        self._generation += 1

    def _glyphs(self):
        if self._atlas is None:
            from PIL import Image, ImageDraw
            from .drawing import find_font
            w, h = self.cell
            font = find_font("DejaVuSansMono", max(2, h - 2))
            atlas = []
            for char in self.chars:
                image = Image.new("L", (w, h))
                d = ImageDraw.Draw(image)
                box = d.textbbox((0, 0), char, font=font)
                d.text(((w - box[2] + box[0]) / 2 - box[0], (h - box[3] + box[1]) / 2 - box[1]),
                       char, font=font, fill=255)
                atlas.append(np.asarray(image))
            self._atlas = np.array(atlas)
        return self._atlas

    def keep_up(self, t):
        """Step to time ``t`` without drawing. The GL renderer calls this every frame, so an automaton whose
        scene is hidden keeps up a generation or two at a time, instead of replaying minutes of them on the
        frame its scene comes back (0.8 s after five minutes at 27 generations a second)."""
        target = max(0, int(math.floor(t * self.rate + 1e-9)))
        if target < self._generation:
            self._reset()
        while self._generation < target:
            self._step()

    def _draw(self, t):
        self.keep_up(t)
        values = np.concatenate((self._rows[self._head:], self._rows[:self._head]))
        tone = (values.astype(np.float32) / (self.sim.num_states - 1)) ** .4
        if self.chars is None:
            pixels = np.round(tone * 255).astype(np.uint8)
        else:
            # Reserve the first glyph for empty cells; young occupied cells must
            # remain visible even when the character ramp is short.
            indices = np.where(values > 0, 1 + (tone * (len(self.chars) - 2)).astype(int), 0)
            w, h = self.cell
            pixels = self._glyphs()[indices].transpose(0, 2, 1, 3).reshape(self.height * h, self.width * w)
        return np.repeat(pixels[..., None], 3, axis=2)
