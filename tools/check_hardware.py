"""Explicit local GPU regression check; never opens audio devices or loads models.

python tools/check_hardware.py --expect-renderer 'RTX 4090' --label idle \
    --output .hydrapy/checks/idle-before
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
SUITES = {
    "operators": [
        "tests/test_hydrapy.py::test_cpu_and_gpu_agree",
        "tests/test_hydrapy.py::test_gl_variants_skip_faded_branches_and_match_the_full_program",
        "tests/test_hydrapy.py::test_scaled_outputs_on_gl_match_numpy_and_read_at_their_size",
        "tests/test_hydrapy.py::test_pipelined_readback_yields_the_same_frames",
    ],
    "automata": ["tests/test_automata.py"],
    "equivalence": ["tests/test_performance_equivalence.py"],
    "render-timing": ["tests/test_render_timing.py"],
    "audio-timing": ["tests/test_audio_timing.py"],
}
BENCHMARK_CASES = ("osc", "rd", "wave", "nca")


def junit_summary(path):
    """A missing report, empty collection or skipped coverage cannot count as a pass."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    result = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            result[key] += case.find(tag) is not None
    result["passed"] = bool(cases) and not any(result[k] for k in ("failures", "errors", "skipped"))
    return result


def gpu_info(expected):
    from hydrapy.backend_gl import _create_context
    ctx = _create_context()
    try:
        info = {key: ctx.info.get(key) for key in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}
        renderer = str(info["GL_RENDERER"])
        if expected.casefold() not in renderer.casefold():
            raise RuntimeError(f"expected renderer containing {expected!r}, got {renderer!r}")
        if any(name in renderer.casefold() for name in ("llvmpipe", "softpipe", "swrast", "software rasterizer", "swiftshader")):
            raise RuntimeError(f"software renderer is not a hardware result: {renderer}")
        return info
    finally:
        ctx.release()


def run_command(argv, log, timeout):
    started = time.monotonic()
    result = {"argv": argv, "log": log.name}
    with log.open("w", encoding="utf-8") as stream:
        try:
            with subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=os.name == "posix") as child:
                try:
                    result["returncode"] = child.wait(timeout=timeout)
                    result["passed"] = child.returncode == 0
                except BaseException:
                    # A timed-out pytest may still own an encoder. Stop this command's
                    # process group only; never signal other render/training jobs.
                    try:
                        if os.name == "posix":
                            os.killpg(child.pid, signal.SIGKILL)
                        else:
                            child.kill()
                    except ProcessLookupError:
                        pass
                    child.wait()
                    raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.update(passed=False, error=str(exc))
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
                                       timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new report directory; never overwritten")
    parser.add_argument("--expect-renderer", required=True, help="required substring of GL_RENDERER")
    parser.add_argument("--label", required=True, help="conditions, e.g. idle-before or render-with-inference")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        width, height = (int(x) for x in args.size.lower().split("x"))
        if min(width, height, args.frames, args.warmup) < 1 or not args.expect_renderer.strip():
            raise ValueError
    except ValueError:
        parser.error("size must be WIDTHxHEIGHT; frames/warmup positive; renderer nonempty")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(THREADS)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    report = {
        "schema": "hydrapy-hardware-check-1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"), "python": sys.version,
        "platform": platform.platform(), "thread_limits": THREADS,
        "expected_renderer": args.expect_renderer, "status": "running", "checks": [],
        "not_measured": ["audio devices / ADAT / CV", "RAVE inference", "end-to-end latency", "hardware encoding"],
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
    }
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        report["tools"] = {}
        for name in ("ffmpeg", "ffprobe", "node"):
            executable = shutil.which(name)
            if not executable:
                raise RuntimeError(f"{name} is required; install it before the hardware check")
            version = subprocess.check_output([executable, "--version" if name == "node" else "-version"],
                                              text=True, stderr=subprocess.STDOUT, timeout=10)
            report["tools"][name] = {"path": executable, "version": version.splitlines()[0]}
        report["gpu"] = gpu_info(args.expect_renderer)
        save()
        # A separate process per group releases all GL contexts between test files.
        # These tests use file/simulated audio; none opens a live audio output.
        for name, tests in SUITES.items():
            print(f"Checking {name}…", flush=True)
            xml = output / f"{name}.xml"
            result = run_command([sys.executable, "-m", "pytest", "-q", "-ra", *tests,
                                  f"--junitxml={xml}"], output / f"{name}.log", 600)
            result["name"] = name
            try:
                result["tests"] = junit_summary(xml)
                result["passed"] &= result["tests"]["passed"]
            except (OSError, ET.ParseError) as exc:
                result.update(passed=False, report_error=str(exc))
            report["checks"].append(result)
            save()
        print("Measuring frame times with GPU readback…", flush=True)
        command = [sys.executable, "docs/benchmark.py", "--backend", "gl", "--readback",
                   "--size", args.size, "--frames", str(args.frames), "--warmup", str(args.warmup),
                   "--json", str(output / "benchmark.json")]
        for name in BENCHMARK_CASES:
            command.extend(["--case", name])
        result = run_command(command, output / "benchmark.log", 600)
        result["name"] = "benchmark"
        # benchmark.py checkpoints partial output; success requires every requested case.
        try:
            timing = json.loads((output / "benchmark.json").read_text())
            complete = [c["name"] for c in timing["cases"]] == list(BENCHMARK_CASES)
            same_gpu = timing["renderer"] == report["gpu"]["GL_RENDERER"]
            result["passed"] &= complete and same_gpu
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(passed=False, report_error=str(exc))
        report["checks"].append(result)
        report["status"] = "passed" if all(c["passed"] for c in report["checks"]) else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
