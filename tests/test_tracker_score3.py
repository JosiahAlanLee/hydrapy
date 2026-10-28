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


def test_seeded_legacy_upgrade_and_editor_roundtrip(tmp_path):
    """Public upgrades implement the spec; editor imports also retain colliding metadata."""
    from hydrapy import studio_tracker, tracker_v3

    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the editor checks')
    rng = random.Random(20261028)
    root = Path(__file__).resolve().parents[1]
    demo = json.loads((root / 'hydrapy/tracker/demo.json').read_text())
    cases = []
    for index in range(200):
        old = json.loads(json.dumps(demo))
        old.update(format=f'hydrapy-tracker-{1 + index % 2}', bpm=rng.randint(60, 200),
                   pattern=rng.randrange(4), loop=bool(rng.getrandbits(1)),
                   mutes=[bool(rng.getrandbits(1)) for _ in range(4)],
                   extra={'index': index, 'nested': [None, True, rng.random(), 'metadata']})
        for key in ('rate', 'bar', 'bars', 'audio', 'chain', 'seed', 'lanes', 'legacy', 'legacyTop'):
            if index % 4:
                old[key] = {'metadata': [key, index]}
        if index % 2:
            old['sounds'] = [{'machine': rng.randrange(192), 'params': {'p1': rng.randrange(128)}} for _ in range(4)]
        for p in old['patterns']:
            p['name'] = f'Random-{index}-{rng.randrange(10000)}'
            for name, (lo, hi, _) in studio_tracker.LANES.items():
                p['keys'][name] = [{'r': row, 'v': rng.uniform(lo, hi)}
                                   for row in sorted(rng.sample(range(64), rng.randrange(6)))]
                p['easing'][name] = rng.choice(['hold', 'linear', 'smooth'])
            p['scenes'] = [{'r': row, 'name': f'scene-{row}', 'fade': rng.randrange(65)}
                           for row in sorted(rng.sample(range(64), rng.randrange(4)))]
            p['extra'] = {'audio': ['unknown', rng.randrange(100)]}
            for key in ('rate', 'bar', 'bars', 'lanes', 'legacyMeta'):
                if index % 4 and rng.random() < .7:
                    p[key] = {'unknown': [key, rng.randrange(100)]}
            p['audio'] = [[None if rng.random() < .8 else
                           dict(note=rng.choice(['C-4', 'D#3', 'AIR']), vel=rng.randrange(128),
                                extra={'audio': 'kept'}) for _ in range(64)] for _ in range(4)]
            if index % 2:
                p['audio'][0][0] = {'note': 'C-4', 'vel': 90, 'locks': {'p2': rng.randrange(128)}}
        old = studio_tracker.validate(old)
        cases.append({'old': old, 'upgraded': tracker_v3.upgrade(old), 'reversible': index % 4 == 0})
    path = tmp_path / 'upgrades.json'
    path.write_text(json.dumps(cases))
    subprocess.run([node, '--test', str(root / 'tests/tracker_score3.mjs')], check=True, timeout=120,
                   env=dict(os.environ, GORGON_UPGRADE_CASES=str(path)))
