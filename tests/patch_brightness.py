"""Mean displayed brightness of every example patch, per output, on the GL backend.

    python tests/patch_brightness.py            # rewrite tests/baselines/brightness.json

test_patch_brightness.py renders the same way and compares. A change that makes a patch go dark
(or blow out) is either a regression or a deliberate look change; in the second case, rerun this.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASELINE = os.path.join(HERE, "baselines", "brightness.json")
SIZE, FPS, FRAMES = (96, 54), 24, 36
SKIP = ("08_", "09_", "17_", "18_")     # CPU-only op, audio file analysis, p5 externals, live hardware


def patch_files():
    d = os.path.join(REPO, "patches")
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith(SKIP))


def measure(name, ctx=None) -> dict:
    sys.path.insert(0, REPO)
    from hydrapy import default_patch, reset_patch
    from hydrapy.backend_gl import GLRenderer
    from hydrapy.live import clear_hooks
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)                                  # patches may write WAVs next to themselves
        try:
            reset_patch(); clear_hooks()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                runpy.run_path(os.path.join(REPO, "patches", name + ".py"))
            p = default_patch()
            r = GLRenderer(p, size=SIZE, fps=FPS, ctx=ctx)
            try:
                for i in range(FRAMES):
                    r.step(i / FPS)
                out = {}
                for o in p.active_outputs():
                    p.render(o)
                    out[o.name] = round(float(r.read().mean()) / 255.0, 4)
                return out
            finally:
                r.release()
        finally:
            clear_hooks()
            os.chdir(cwd)


if __name__ == "__main__":
    from hydrapy.backend_gl import _create_context
    ctx = _create_context()
    data = {n: measure(n, ctx) for n in patch_files()}
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    with open(BASELINE, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print(f"wrote {BASELINE}: {len(data)} patches")
