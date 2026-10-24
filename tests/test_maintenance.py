"""Maintenance reports must distinguish tested hardware from missing coverage."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("hardware_check", Path(__file__).resolve().parents[1] / "tools/check_hardware.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.mark.parametrize("cases,passed", [
    ('<testcase name="ok"/>', True),
    ('<testcase name="no-gpu"><skipped message="missing GL"/></testcase>', False),
    ('<testcase name="bad"><failure message="pixels differ"/></testcase>', False),
    ('<testcase name="collect"><error message="import failed"/></testcase>', False),
    ('', False),
])
def test_junit_requires_executed_passing_tests(tmp_path, cases, passed):
    path = tmp_path / "tests.xml"
    path.write_text(f'<testsuites><testsuite>{cases}</testsuite></testsuites>')
    assert check.junit_summary(path)["passed"] is passed


@pytest.mark.parametrize("actual,expected,passes", [
    ("NVIDIA GeForce RTX 4090", "rtx 4090", True),
    ("Intel Graphics", "RTX 4090", False),
    ("llvmpipe (LLVM)", "llvmpipe", False),
])
def test_wrong_or_software_gpu_cannot_pass(monkeypatch, actual, expected, passes):
    from hydrapy import backend_gl
    released = []
    ctx = SimpleNamespace(info={"GL_RENDERER": actual}, release=lambda: released.append(True))
    monkeypatch.setattr(backend_gl, "_create_context", lambda: ctx)
    if passes:
        assert check.gpu_info(expected)["GL_RENDERER"] == actual
    else:
        with pytest.raises(RuntimeError):
            check.gpu_info(expected)
    assert released == [True]


def test_failed_preflight_writes_report_without_starting_work(tmp_path, monkeypatch):
    def absent(_):
        raise RuntimeError("expected GPU is unavailable")
    def forbidden(*args):
        pytest.fail("preflight failure started a test or benchmark")
    monkeypatch.setattr(check, "gpu_info", absent)
    monkeypatch.setattr(check, "run_command", forbidden)
    monkeypatch.setattr(check, "git", lambda *args: "test-commit")
    monkeypatch.setattr(check.shutil, "which", lambda name: name)
    monkeypatch.setattr(check.subprocess, "check_output", lambda *a, **kw: "version\n")
    for key in (*check.THREADS, "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        monkeypatch.setenv(key, "1")
    output = tmp_path / "run"
    args = ["--output", str(output), "--expect-renderer", "RTX 4090", "--label", "idle"]
    assert check.main(args) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and "GPU is unavailable" in report["error"]
    assert report["checks"] == [] and report["finished_utc"]
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        check.main(args)
    assert (output / "report.json").read_bytes() == before


def test_subprocess_failure_retains_output(tmp_path):
    log = tmp_path / "failed.log"
    result = check.run_command([sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"], log, 5)
    assert not result["passed"] and result["returncode"] == 7
    assert "diagnostic" in log.read_text()


def test_subprocess_timeout_is_not_a_pass(tmp_path):
    result = check.run_command([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "timeout.log", .1)
    assert not result["passed"] and "timed out" in result["error"]
