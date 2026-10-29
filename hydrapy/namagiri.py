"""Pure NumPy Rogers–Ramanujan convergent controls for Gorgon and HydraPy.

No rendering dependency: the returned series can drive score parameters,
CV values, or visual controls. Values are finite approximants, not an
assertion of convergence on the unit circle.
"""
from __future__ import annotations
import numpy as np

def convergents(q: complex, depth: int = 16) -> np.ndarray:
    """R_1(q)..R_depth(q) using the finite continued-fraction definition.

    Restrict |q| < 1 to avoid ambiguous boundary convergence.
    The principal branch of q**(1/5) is used.
    """
    q = complex(q)
    if not np.isfinite(q.real) or not np.isfinite(q.imag) or abs(q) >= 1:
        raise ValueError("q must be finite and satisfy |q| < 1")
    if not 1 <= int(depth) <= 64:
        raise ValueError("depth must be between 1 and 64")
    values = np.empty(int(depth), dtype=np.complex128)
    root = q ** (1/5)
    for n in range(1, int(depth)+1):
        tail = 1+0j
        for k in range(n, 0, -1):
            if abs(tail) < 1e-12:
                raise ArithmeticError("near-zero continued-fraction denominator")
            tail = 1 + q**k/tail
        values[n-1] = root/tail
    return values

def convergent_controls(q: complex, depth: int = 16) -> dict[str, np.ndarray]:
    """Intermediate convergents as bounded, deterministic compositional data."""
    z = convergents(q, depth)
    phase = np.angle(z)
    magnitude = np.abs(z)
    delta = np.abs(np.diff(z, prepend=z[0]))
    return {"real":z.real, "imag":z.imag, "magnitude":magnitude,
            "phase":phase, "delta":delta}

__all__ = ["convergents", "convergent_controls"]
