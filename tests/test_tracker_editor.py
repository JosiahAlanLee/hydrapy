"""Cross-lane editing, pitch limits and history must preserve the transport contract."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_score_editor_operations():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for editor checks')
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, '--test', str(root / 'tests/tracker_editor.mjs')], check=True, timeout=30)
