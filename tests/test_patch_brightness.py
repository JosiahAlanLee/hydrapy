"""Every example patch keeps its brightness (catches a whole class of alpha/display regressions)."""
import json
import os

import pytest

from patch_brightness import BASELINE, measure, patch_files


def _gl():
    try:
        from hydrapy.backend_gl import _create_context
        return _create_context()
    except Exception:
        return None


CTX = _gl()
BASE = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}


@pytest.mark.skipif(CTX is None, reason="no GL context")
@pytest.mark.parametrize("name", patch_files())
def test_patch_brightness_matches_baseline(name):
    if name not in BASE:
        pytest.skip(f"no baseline for {name}: run python tests/patch_brightness.py")
    got = measure(name, CTX)
    for out, want in BASE[name].items():
        v = got.get(out)
        assert v is not None, f"{name} {out} no longer renders"
        assert abs(v - want) < 0.02 or 0.8 < (v + 1e-3) / (want + 1e-3) < 1.25, \
            f"{name} {out}: mean brightness {v:.3f}, baseline {want:.3f}"
