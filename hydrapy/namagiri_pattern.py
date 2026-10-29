"""Deterministic Namagiri sequence adapter for score/event generation.

This intentionally does not mutate the Gorgon tracker or presume its event
schema. It provides an explicit event stream for a future bridge and for VCV
comparison. It maps convergent changes to event velocity and phase to pitch.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .namagiri import convergent_controls

@dataclass(frozen=True)
class NamagiriEvent:
    step: int
    note: int
    velocity: float
    phase: float
    magnitude: float

def pattern(q: complex, steps: int = 16, root: int = 48,
            span: int = 24, threshold: float = 0.0) -> list[NamagiriEvent]:
    """Map convergents to a bounded note-event sequence.

    Depth equals steps; values represent convergent index, not elapsed time.
    Event scheduling is left to Gorgon/VCV. No randomness or wall clock.
    """
    if not 1 <= steps <= 64:
        raise ValueError("steps must be 1..64")
    if not 1 <= span <= 96 or not 0 <= root <= 127:
        raise ValueError("invalid pitch mapping")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be 0..1")
    c = convergent_controls(q,steps)
    out=[]
    for i in range(steps):
        phase=float(c["phase"][i])
        mag=float(c["magnitude"][i])
        change=float(c["delta"][i])
        velocity=float(np.clip(change/(1+change),0,1))
        if velocity < threshold:
            continue
        note=int(np.clip(root+np.floor((phase+np.pi)/(2*np.pi)*span),0,127))
        out.append(NamagiriEvent(i,note,velocity,phase,mag))
    return out
