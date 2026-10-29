"""The Gorgon tracker's lanes as patch signals: ``tracker.gate(1)``, ``tracker.hit(2)``…

This folder also holds the tracker's browser files; being a package means
``import hydrapy.tracker`` and ``from hydrapy import tracker`` are the same
thing. See ``hydrapy.scenes`` for what each signal means.
"""
from ..scenes import _Tracker as _Lanes

_lanes = _Lanes()
gate, hit, note, vel, beat, row = _lanes.gate, _lanes.hit, _lanes.note, _lanes.vel, _lanes.beat, _lanes.row
ahead = _lanes.ahead

__all__ = ['gate', 'hit', 'note', 'vel', 'beat', 'row', 'ahead']
