"""Explicit performance controls; suggested slider ranges never clamp the signal."""
from __future__ import annotations

import math
from .signals import Signal, Lfo, Clock, Env, _num
from .core import default_patch


class Param(Signal):
    """A named, mutable signal exposed by ``hydrapy studio``.

    ``Param('motion', .2, min=0, max=2, unit='Hz')`` works in any signal input.
    Min/max describe the slider; numeric entry may go outside them. A log slider
    requires positive bounds. ``bind(signal, depth)`` adds modulation to the base.
    """
    def __init__(self, name, default=0., *, min=0., max=1., step=.01, unit='',
                 scale='linear', group='Performance', description='', patch=None):
        self.name, self.default = str(name), float(default)
        self.min, self.max, self.step = float(min), float(max), float(step)
        self.unit, self.scale, self.group, self.description = unit, scale, group, description
        if not self.name or not all(math.isfinite(x) for x in (self.default, self.min, self.max, self.step)):
            raise ValueError('Param requires a name and finite numbers')
        if self.min >= self.max or self.step <= 0 or scale not in ('linear', 'log'):
            raise ValueError('Param requires min < max, positive step, and linear/log scale')
        if scale == 'log' and self.min <= 0:
            raise ValueError('log sliders require positive bounds')
        self.base, self.modulation, self.depth = self.default, None, 0.
        self.mod_kind, self.mod_rate = 'none', .1
        p = patch or default_patch()
        if self.name in p.controls:
            raise ValueError(f'duplicate control name: {self.name}')
        p.controls[self.name] = self

    def set(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'{self.name}: value must be finite')
        self.base = value
        return self

    def reset(self):
        return self.set(self.default)

    def bind(self, signal, depth=1.):
        from .signals import as_signal
        depth = float(depth)
        if not math.isfinite(depth):
            raise ValueError('modulation depth must be finite')
        self.modulation, self.depth = as_signal(signal), depth
        self.mod_kind = 'patch'
        return self

    def set_modulation(self, kind='none', depth=0., rate=.1):
        depth, rate = float(depth), float(rate)
        if kind not in ('none', 'lfo', 'beat') or not math.isfinite(depth) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('modulation requires none/lfo/beat, finite depth and a positive rate')
        self.modulation = Lfo(rate, lo=-1, hi=1) if kind == 'lfo' else Env(Clock(rate)) if kind == 'beat' else None
        self.depth, self.mod_rate, self.mod_kind = depth, rate, kind
        return self

    def value(self, ctx):
        return self.base + (self.depth * self.modulation(ctx) if self.modulation is not None else 0.)

    def js(self):
        return f'({_num(self.base)}+{_num(self.depth)}*({self.modulation.js()}))' if self.modulation is not None else _num(self.base)

    def snapshot(self):
        return dict(value=self.base, modulation=self.mod_kind, depth=self.depth, rate=self.mod_rate)

    def describe(self):
        return dict(name=self.name, default=self.default, min=self.min, max=self.max, step=self.step,
                    unit=self.unit, scale=self.scale, group=self.group, description=self.description,
                    **self.snapshot())


def apply_controls(patch, values):
    """Validate a preset before applying any part of it; ignore controls removed by an edit."""
    prepared = []
    for name, entry in values.items():
        if name not in patch.controls:
            continue
        entry = dict(entry) if isinstance(entry, dict) else {'value': entry}
        p = patch.controls[name]
        value = float(entry.get('value', p.base))
        kind = entry.get('modulation', p.mod_kind)
        rate, depth = float(entry.get('rate', p.mod_rate)), float(entry.get('depth', p.depth))
        if not all(math.isfinite(x) for x in (value, rate, depth)) or rate <= 0 or kind not in ('none', 'lfo', 'beat', 'patch'):
            raise ValueError(f'invalid control setting: {name}')
        if kind == 'patch' and p.mod_kind != 'patch':
            raise ValueError(f'{name}: preset requires modulation declared in the patch')
        prepared.append((p, value, kind, depth, rate))
    for p, value, kind, depth, rate in prepared:
        p.set(value)
        if kind != 'patch':
            p.set_modulation(kind, depth, rate)
        else:
            p.depth = depth
