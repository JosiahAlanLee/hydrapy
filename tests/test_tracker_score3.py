"""The editor's score-format-3 module: its mapping, text forms, and evaluate() against tracker_v3's."""
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from hydrapy.tracker_v3 import TICKS, Timeline, evaluate
from test_tracker_v3 import random_score, random_transitions


def test_the_browser_evaluator_matches_hydrapy(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the editor checks')
    rng = random.Random(20261027)
    cases = []
    for _ in range(60):
        score = random_score(rng)
        cycle = Timeline(score).cycle
        transport = {'tick': rng.randrange(cycle), 'fill': random_transitions(rng, 2 * cycle)}
        if rng.random() < .7:
            transport['bus'] = {'fill': random_transitions(rng, 2 * cycle),
                                'mutes': [random_transitions(rng, 2 * cycle) for _ in range(6)]}
        u0, u1 = rng.choice([-8 * TICKS, 0, rng.randrange(cycle)]), rng.randrange(cycle, 3 * cycle)
        cases.append(dict(score=score, transport=transport, u0=u0, u1=u1, candidates=evaluate(score, transport, u0, u1)))
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(cases))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, '--test', str(root / 'tests/tracker_score3.mjs')], check=True, timeout=120,
                   env=dict(os.environ, GORGON_V3_CASES=str(path)))
