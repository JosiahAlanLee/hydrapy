"""Reproducible renderer timings; see PERFORMANCE.md for interpretation.

OPENBLAS_NUM_THREADS=1 python docs/benchmark.py --backend cpu --size 320x180
python docs/benchmark.py --backend gl --size 1920x1080 --readback --json timing.json
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "patches"))
import numpy as np
import hydrapy as H
from hydrapy.glsl import compile_patch


def cases(size):
    image = H.Image("benchmark-memory")
    image._array = np.random.default_rng(11).random((size[1], size[0], 4), dtype=np.float32)
    image._array[..., 3] = 1
    base = lambda: H.src(image)
    return {
        "osc": lambda: H.osc(),
        "blur": lambda: base().blur(),
        "rd": lambda: base().rd(),
        "bleed": lambda: base().bleed(),
        "drops": lambda: H.drops(),
        "harmonograph": lambda: H.harmonograph(),
        "wave": lambda: H.wave(),
        "arc": lambda: H.arc(),
        "yiq": lambda: base().yiq(),
        "normalMap": lambda: base().normalMap(),
        "raymarch": lambda: H.raymarch(H.SDF3.box(.45, .45, .45).fold(
            3, scale=2.1, x=.7, y=.6, z=.65, angle=.2).scale(.85),
            rotX=.35, rotY=H.Time()*.16, steps=160, mode=1),
        "nca": lambda: H.src(H.NCA(H.NCAModel.random(), size=(160, 90), speed=30)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gl"), default="gl")
    parser.add_argument("--size", default="320x180")
    parser.add_argument("--case", action="append", help="repeat for specific cases; defaults to all")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--readback", action="store_true", help="include GPU RGB download for video export")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        size = tuple(int(v) for v in args.size.lower().split("x"))
        if len(size) != 2 or min(size) < 1 or args.frames < 1 or args.warmup < 1:
            raise ValueError
    except ValueError:
        parser.error("size must be WIDTHxHEIGHT; frames and warmup must be positive")
    factories = cases(size)
    names = args.case or [*factories, "bullfrog_surface"]
    if any(n not in factories and n != "bullfrog_surface" for n in names):
        parser.error("unknown case; choose from " + ", ".join([*factories, "bullfrog_surface"]))
    ctx = None
    result = {"backend": args.backend, "size": size, "fps": 30, "readback": args.readback,
              "frames": args.frames, "warmup": args.warmup, "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(), "cases": []}
    if args.backend == "gl":
        from hydrapy.backend_gl import GLRenderer, _create_context
        ctx = _create_context()
        result["renderer"] = ctx.info.get("GL_RENDERER")
        result["vendor"] = ctx.info.get("GL_VENDOR")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    try:
        for name in names:
            if name == "bullfrog_surface":
                from bullfrog import make_scene
                patch = make_scene(5)
                patch.size = size
            else:
                patch = H.Patch(size=size)
                factories[name]().out(patch.o0, patch=patch)
            started = time.perf_counter()
            renderer = H.Renderer(patch, fps=30) if ctx is None else GLRenderer(patch, fps=30, ctx=ctx)
            setup_ms = (time.perf_counter() - started) * 1000
            samples, first_ms = [], None
            try:
                for i in range(args.warmup + args.frames):
                    started = time.perf_counter()
                    renderer.step(i / 30)
                    if ctx is not None:
                        if args.readback: renderer.read()
                        ctx.finish()  # include execution, not just command submission
                    elapsed = (time.perf_counter() - started) * 1000
                    if i == 0: first_ms = elapsed
                    if i >= args.warmup: samples.append(elapsed)
                row = {"name": name, "passes": sum(len(p.passes) for p in compile_patch(patch)),
                       "setup_ms": setup_ms, "first_ms": first_ms,
                       "median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
                       "samples_ms": samples}
                result["cases"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
            finally:
                if ctx is not None: renderer.release()
                del renderer
                gc.collect()
            if args.json:
                args.json.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if ctx is not None: ctx.release()


if __name__ == "__main__":
    main()
