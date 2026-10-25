"""Exercise the actual browser scheduling functions with deterministic transport updates."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_audio_survives_pattern_and_song_wraps():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the browser scheduler regression')
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, str(root/'tests/tracker_scheduler.cjs'), str(root/'hydrapy/tracker/app.js')],
                   check=True, timeout=30)
