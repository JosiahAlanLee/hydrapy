"""Explicit performance controls; suggested slider ranges never clamp the signal."""
from __future__ import annotations

import math
from .signals import Signal, Lfo, Clock, Env, _num
from .core import control_patch


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
        self._bound = None                # (signal, depth) from bind(): what "Declared in patch" restores
        p = patch or control_patch()          # the scene's own patch while scene(..., patch=) builds
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
        self._bound = (self.modulation, depth)
        return self

    def declaration(self):
        """What the patch source says about this control (not what the performer has done to it)."""
        bound = None if self._bound is None else (type(self._bound[0]).__name__, self._bound[1])
        return dict(default=self.default, bound=bound)

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
                    declared=self._bound is not None, **self.snapshot())


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
        if kind == 'patch' and p._bound is None:
            raise ValueError(f'{name}: preset requires modulation declared in the patch')
        prepared.append((p, value, kind, depth, rate))
    for p, value, kind, depth, rate in prepared:
        p.set(value)
        if kind != 'patch':
            p.set_modulation(kind, depth, rate)
        else:
            p.modulation, p.mod_kind, p.depth = p._bound[0], 'patch', depth


def carry_controls(old, new):
    """Carry the performance across a reload without overriding the edit.

    A control keeps its live value only if the edit left its default alone, and its modulation
    (the performer's choice of none/LFO/beat, or depth on a declared bind) only if the edit left its
    ``bind()`` alone; otherwise the new code wins.  The output on screen stays the one chosen in the
    session unless the edit changed the patch's ``render()``.
    """
    for name, p in new.controls.items():
        q = old.controls.get(name)
        if q is None:
            continue
        was, now = q.declaration(), p.declaration()
        if was['default'] == now['default']:
            p.set(q.base)
        if was['bound'] == now['bound']:
            if q.mod_kind == 'patch':
                p.depth = q.depth
            else:
                p.set_modulation(q.mod_kind, q.depth, q.mod_rate)
    if getattr(old, 'declared_display', None) == getattr(new, 'declared_display', None):
        new.render(new.outputs[old.display.index])
