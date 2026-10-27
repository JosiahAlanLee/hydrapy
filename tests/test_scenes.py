"""Named scenes chosen by the tracker's SCENE lane, and the lanes as patch signals."""
import copy

import numpy as np
import pytest

import hydrapy.core as core
from hydrapy import Param, noise, o0, o1, osc, scene, solid, tracker, voronoi
from hydrapy.scenes import check_scenes, scene_names
from hydrapy.signals import Signal
from hydrapy.studio import Session
from hydrapy.studio_tracker import Tracker, scene_fades, scene_weights, validate


@pytest.fixture
def patch():
    previous = core._default_patch
    p = core.Patch()
    core._default_patch = p
    yield p
    core._default_patch = previous


def three(patch):
    scene('drift')(lambda: osc(8, .1, 1.2))
    scene('fracture')(lambda: voronoi(5, .3, .3).kaleid(6))
    scene('third')(lambda: noise(3, .2))
    return patch


def score_with(*changes, loop=False, pattern=0):
    d = copy.deepcopy(Tracker().document)
    d['loop'], d['pattern'] = loop, pattern
    for p, row, name, fade in changes:
        d['patterns'][p].setdefault('scenes', []).append({'r': row, 'name': name, 'fade': fade})
    for p in d['patterns']:
        p.get('scenes', []).sort(key=lambda k: k['r'])
    return validate(d)


def weights(document, beat, patch):
    return {k: round(v, 6) for k, v in scene_weights(document, beat, patch.scenes).items()}


def test_scenes_compose_one_output_and_the_first_plays_undriven(patch):
    three(patch)
    assert scene_names(patch) == ['drift', 'fracture', 'third']
    blends = [t for t in patch.o0.chain.transforms if t.op.name == 'blend']
    assert len(blends) == 3 and patch.o0.chain.transforms[0].op.name == 'solid'
    assert [b.args[1](None) for b in blends] == [1., 0., 0.]
    check_scenes(patch)


def test_reassigning_an_output_after_its_scenes_is_refused(patch):
    three(patch)
    osc(2).out(o0)
    with pytest.raises(ValueError, match='reassigned after its scenes'):
        check_scenes(patch)


def test_existing_output_chain_is_the_base_and_other_outputs_are_independent(patch):
    noise(2).out(o0)
    scene('a')(lambda: osc(3))
    scene('b', o1)(lambda: osc(4))
    assert patch.o0.chain.transforms[0].op.name == 'noise'
    assert patch.scenes['b'][0] == 1 and patch.o1.chain.transforms[0].op.name == 'solid'


@pytest.mark.parametrize('bad, message', [
    (lambda: scene('has space'), 'Scene names'),
    (lambda: scene('x' * 25), 'Scene names'),
])
def test_scene_names_are_checked(patch, bad, message):
    with pytest.raises(ValueError, match=message):
        bad()


def test_duplicate_and_non_chain_scenes_are_refused(patch):
    scene('a')(lambda: osc(3))
    with pytest.raises(ValueError, match='twice'):
        scene('a')
    with pytest.raises(TypeError, match='must return a chain'):
        scene('b')(lambda: 3)


def test_old_scores_load_and_scene_keys_are_validated():
    d = copy.deepcopy(Tracker().document)
    for p in d['patterns']:
        p.pop('scenes', None)
    assert all(p['scenes'] == [] for p in validate(d)['patterns'])
    good = score_with((1, 8, 'drift', 4))
    assert good['patterns'][1]['scenes'] == [{'r': 8, 'name': 'drift', 'fade': 4}]
    for key, message in [({'r': 64, 'name': 'a'}, 'Scene rows'), ({'r': 1, 'name': 'a b'}, 'Scene names'),
                         ({'r': 1, 'name': 'a', 'fade': 65}, 'fade'), ({'r': 1, 'name': 'a', 'x': 1}, 'row, a name')]:
        d = copy.deepcopy(Tracker().document)
        d['patterns'][0]['scenes'] = [key]
        with pytest.raises(ValueError, match=message):
            validate(d)
    d = copy.deepcopy(Tracker().document)
    d['patterns'][0]['scenes'] = [{'r': 4, 'name': 'a'}, {'r': 4, 'name': 'b'}]
    with pytest.raises(ValueError, match='unique, sorted'):
        validate(d)


def test_scene_changes_hold_wrap_and_crossfade_in_declaration_order(patch):
    three(patch)
    d = score_with((0, 0, 'drift', 0), (0, 32, 'third', 8), (2, 0, 'fracture', 4))
    assert weights(d, 4, patch) == {'drift': 1, 'fracture': 0, 'third': 0}
    # drift -> third over 2 beats: third is later in the chain, so it fades in over drift.
    assert weights(d, 9, patch) == {'drift': 1, 'fracture': 0, 'third': .5}
    assert weights(d, 10, patch) == {'drift': 0, 'fracture': 0, 'third': 1}
    # third -> fracture over 1 beat: fracture is earlier, so third fades out over it.
    assert weights(d, 32.5, patch) == {'drift': 0, 'fracture': 1, 'third': .5}
    assert weights(d, 40, patch) == {'drift': 0, 'fracture': 1, 'third': 0}
    # Wraps: before the first change of the next pass, the last change holds.
    looped = score_with((1, 8, 'fracture', 0), loop=True, pattern=1)
    assert weights(looped, 16, patch)['fracture'] == 1 and weights(looped, 31.9, patch)['fracture'] == 1
    assert scene_fades(d, patch.scenes) == [('drift', 'third'), ('third', 'fracture')]


def test_unknown_names_are_ignored_and_outputs_without_changes_keep_their_first(patch):
    scene('a')(lambda: osc(3))
    scene('b')(lambda: osc(4))
    scene('c', o1)(lambda: osc(5))
    d = score_with((0, 0, 'b', 0), (0, 8, 'missing', 0))
    assert weights(d, 3, patch) == {'a': 0, 'b': 1, 'c': 1}


def test_a_fade_never_cuts_to_black_or_jumps(patch):
    scene('a')(lambda: osc(3))
    scene('b')(lambda: osc(4))
    scene('c')(lambda: osc(5))
    # One change with a fade: every pass returns to the same scene, so it never fades from black.
    single = score_with((0, 0, 'a', 8))
    assert weights(single, 0, patch)['a'] == 1 and weights(single, 63.99, patch)['a'] == 1
    # A change to the scene already showing doesn't fade either.
    same = score_with((0, 0, 'a', 0), (0, 16, 'a', 8))
    assert weights(same, 4.5, patch)['a'] == 1
    # A 32-row fade cut short by a change 12 rows later is clamped to finish there: no jump at beat 4.
    cut = score_with((0, 0, 'a', 0), (0, 4, 'b', 32), (0, 16, 'c', 8))
    before, after = weights(cut, 3.999, patch), weights(cut, 4.0, patch)
    assert before['b'] > .999 and after['b'] == 1 and after['c'] == 0
    assert weights(cut, 2.5, patch)['b'] == .5          # halfway through the clamped 3-beat fade


def test_apply_drives_weights_and_disabling_keeps_the_last_scene(patch):
    three(patch)
    t = Tracker(score_with((0, 0, 'drift', 0), (1, 0, 'third', 0)))
    t.enabled = True
    t.seek(20, 0.)
    t.apply(patch, 0.)
    assert patch.scene_weights['third'] == 1
    t.enabled = False
    t.seek(2, 0.)
    t.apply(patch, 0.)
    assert patch.scene_weights['third'] == 1


def test_lane_signals_follow_hits_mutes_key_and_seeks(patch):
    d = copy.deepcopy(Tracker().document)
    for p in d['patterns']:
        p['audio'] = [[None] * 64 for _ in range(4)]
    d['key'], d['bpm'] = 7, 120                    # G: notes rise five semitones from the D-based score
    d['patterns'][0]['audio'][0][4] = {'note': 'A-4', 'vel': 127}
    d['patterns'][0]['audio'][1][4] = {'note': 'C-4', 'vel': 64}
    d['mutes'][1] = True
    t = Tracker(validate(d))
    t.enabled = True
    gate, hit, note, vel = tracker.gate(1), tracker.hit(1, decay=.5), tracker.note(1), tracker.vel(1)
    signals = dict(gate=gate, hit=hit, note=note, vel=vel, muted=tracker.gate(2), beat=tracker.beat())
    def at(beat):
        t.seek(beat, 0., discontinuity=False)
        t.apply(patch, 0.)
        return {k: round(s(None), 4) for k, s in signals.items()}
    assert at(0)['note'] == 60 and at(0)['gate'] == 0
    v = at(1.)                                    # row 4 starts: the hit
    assert v['gate'] == 1 and v['hit'] == 1 and v['note'] == 69 + 5 and v['vel'] == 1 and v['muted'] == 0
    assert at(1.15)['gate'] == 0                  # past half a row
    assert at(1.25)['hit'] == .75                 # 0.125 s into a 0.5 s decay at 120 BPM
    later = at(3.)
    assert later['hit'] == 0 and later['note'] == 74 and later['beat'] == 3
    t.seek(3., 0.)                                # a real seek clears the holds
    t.apply(patch, 0.)
    assert note(None) == 60 and vel(None) == 0


def test_lane_signals_default_without_the_tracker(patch):
    assert tracker.note(2)(None) == 60 and tracker.gate(3)(None) == 0 and tracker.hit(4)(None) == 0
    assert tracker.gate(6)(None) == 0             # version 3's sixth lane
    with pytest.raises(ValueError):
        tracker.gate(7)


def test_session_publishes_scenes_and_rejects_a_reassigned_patch(tmp_path):
    (tmp_path / 'p.py').write_text("from hydrapy import *\n@scene('a')\ndef _(): return osc(3)\n"
                                   "@scene('b', o1)\ndef _(): return osc(4)\n")
    (tmp_path / 'bad.py').write_text("from hydrapy import *\n@scene('a')\ndef _(): return osc(3)\nosc(2).out(o0)\n")
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        import time
        deadline = time.monotonic() + 5
        while not session.tracker_state.get('scenes') and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.tracker_state['scenes'] == [{'name': 'a', 'output': 'o0'}, {'name': 'b', 'output': 'o1'}]
        with pytest.raises(ValueError, match='reassigned'):
            session.call('load', path='bad.py')
        assert session.path.name == 'p.py'
    finally:
        session.close()


def test_tracker_export_follows_scenes_deterministically(patch):
    three(patch)
    t = Tracker(score_with((0, 0, 'drift', 0), (0, 8, 'third', 4)))
    t.enabled = True
    run = lambda: [f.copy() for f in t.frames(patch, seconds=1.5, fps=8, size=(16, 9), backend='cpu')]
    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[-1])         # drift at the start, third by the end


def test_gl_precompiles_every_scene_and_fade_and_never_compiles_while_playing(patch):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer, _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')
    three(patch)
    t = Tracker(score_with((0, 0, 'drift', 0), (0, 16, 'third', 8), (0, 40, 'fracture', 4)))
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
    r.settle = LIVE_SETTLE
    try:
        assert r.prewarm_scenes(scene_fades(t.document, patch.scenes)) == 5
        t.enabled = True
        compiles, full = 0, 0
        for i in range(int(64 * 60 / t.document['bpm'] * 30)):
            before = len(r._variants)
            t.apply(patch, i / 30)
            r.step(i / 30)
            compiles += len(r._variants) - before
            full += r._active.get('o0') is None
        assert compiles == 0 and full == 0
    finally:
        r.release()


def test_gl_scene_output_matches_a_plain_blend_chain(patch):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import GLRenderer, _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')
    three(patch)
    plain = core.Patch()
    for w in [{'drift': 1}, {'drift': 1, 'fracture': .5}, {'third': 1, 'fracture': .25}]:
        patch.scene_weights = w
        solid(0, 0, 0, 1).blend(osc(8, .1, 1.2), w.get('drift', 0)).blend(
            voronoi(5, .3, .3).kaleid(6), w.get('fracture', 0)).blend(noise(3, .2), w.get('third', 0)).out(o0, patch=plain)
        a = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
        b = GLRenderer(plain, size=(32, 18), fps=30, ctx=ctx)
        try:
            a.prewarm_scenes()
            a.step(.7); b.step(.7)
            assert np.array_equal(a.read(), b.read())
        finally:
            a.release(); b.release()


def test_gl_new_idle_combinations_fall_back_to_a_precompiled_scene_not_the_full_program(patch):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer, _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')
    warp = Param('Warp', .5)
    scene('a')(lambda: osc(8).modulate(noise(3), warp))       # a gate of the patch's own
    scene('b')(lambda: voronoi(5))
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
    r.settle = LIVE_SETTLE
    try:
        r.prewarm_scenes()
        patch.scene_weights = {'a': 1, 'b': 0}
        warp.set(0)                               # a new idle set: b's blend and the modulate
        r.step(0.)
        chosen = r._active['o0']
        assert chosen is not None and len(chosen) == 1   # a's precompiled variant, not every scene
    finally:
        r.release()


def lane_score(cells):
    d = copy.deepcopy(Tracker().document)
    for p in d['patterns']:
        p['audio'] = [[None] * 64 for _ in range(4)]
    d['bpm'] = 120
    for (p, lane, row), hit in cells.items():
        d['patterns'][p]['audio'][lane][row] = hit
    return d


def play(t, patch, beats, step=.05):
    """Apply frame by frame from the current beat, returning hit(1) at each frame."""
    out = []
    for b in beats:
        t.seek(b, 0., discontinuity=False)
        t.apply(patch, 0.)
        out.append(tracker.hit(1, decay=.2)(None))
    return out


def test_hits_fire_once_per_crossing_and_never_after_a_mute_delete_or_loop_change(patch):
    d = lane_score({(0, 0, 4): {'note': 'C-4', 'vel': 100}})
    d['loop'] = True
    t = Tracker(validate(d))
    t.enabled = True
    frames = [k * .05 for k in range(0, 700)]            # beats 0-35: three passes of a 16-beat loop
    hits = play(t, patch, frames)
    onsets = [frames[k] for k in range(1, len(frames)) if hits[k] == 1 and hits[k - 1] < 1]
    assert [round(x, 6) for x in onsets] == [1.0, 17.0, 33.0]   # once per pass, not again in between
    # Muted: the next crossing doesn't fire.
    d2 = copy.deepcopy(t.document); d2['mutes'][0] = True
    t.replace(d2, 0., t.revision)
    assert max(play(t, patch, [k * .05 for k in range(340, 400)])) == 0
    # Deleted, unmuted: still nothing.
    d3 = copy.deepcopy(t.document); d3['mutes'][0] = False; d3['patterns'][0]['audio'][0][4] = None
    t.replace(d3, 0., t.revision)
    assert max(play(t, patch, [k * .05 for k in range(400, 700)])) == 0


def test_a_hit_crossed_between_frames_still_fires(patch):
    t = Tracker(validate(lane_score({(0, 0, 5): {'note': 'C-4', 'vel': 100}})))
    t.enabled = True
    play(t, patch, [1.0])                                  # row 4
    assert play(t, patch, [1.6])[0] > 0                     # jumped past row 5 (beat 1.25) in one frame
    assert tracker.vel(1)(None) == pytest.approx(100 / 127)


def test_editing_the_playing_cell_updates_note_and_velocity(patch):
    t = Tracker(validate(lane_score({(0, 0, 4): {'note': 'C-4', 'vel': 100}})))
    t.enabled = True
    play(t, patch, [1.0])
    d = copy.deepcopy(t.document); d['patterns'][0]['audio'][0][4] = {'note': 'G-4', 'vel': 50}
    t.replace(d, 0., t.revision)
    play(t, patch, [1.05])
    assert tracker.note(1)(None) == 67 and tracker.vel(1)(None) == pytest.approx(50 / 127)


def test_gl_precompiled_scenes_include_the_outputs_other_idle_gates(patch):
    pytest.importorskip('moderngl')
    from hydrapy.backend_gl import LIVE_SETTLE, GLRenderer, _create_context
    try:
        ctx = _create_context()
    except Exception as exc:
        pytest.skip(f'no GL context: {exc}')
    scene('a')(lambda: osc(8).modulate(noise(3), 0))       # always idle
    scene('b')(lambda: voronoi(5).modulate(noise(2), Param('Warp', 0)))   # idle at load
    t = Tracker(score_with((0, 0, 'a', 0), (0, 16, 'b', 0)))
    r = GLRenderer(patch, size=(32, 18), fps=30, ctx=ctx)
    r.settle = LIVE_SETTLE
    try:
        r.prewarm_scenes(scene_fades(t.document, patch.scenes))
        t.enabled = True
        compiles = 0
        for i in range(int(64 * 60 / t.document['bpm'] * 30)):
            before = len(r._variants)
            t.apply(patch, i / 30)
            r.step(i / 30)
            compiles += len(r._variants) - before
        assert compiles == 0
    finally:
        r.release()


def test_reload_with_automation_off_keeps_the_scene_showing(tmp_path):
    (tmp_path / 'p.py').write_text("from hydrapy import *\n@scene('a')\ndef _(): return osc(3)\n"
                                   "@scene('b')\ndef _(): return osc(4)\n")
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        d = copy.deepcopy(session.tracker.document)
        d['patterns'][0]['scenes'] = [{'r': 0, 'name': 'b', 'fade': 0}]
        session.call('tracker_score', document=d, revision=session.tracker.revision)
        session.call('tracker_transport', enabled=True, playing=True, beat=1)
        import time
        deadline = time.monotonic() + 5
        while (session.renderer.patch.scene_weights or {}).get('b') != 1 and time.monotonic() < deadline:
            time.sleep(.01)
        session.call('tracker_transport', enabled=False)
        (tmp_path / 'p.py').write_text((tmp_path / 'p.py').read_text() + "# edited\n")
        session.call('load', path='p.py')
        assert session.renderer.patch.scene_weights == {'a': 0.0, 'b': 1.0}
    finally:
        session.close()


def test_exports_outside_the_tracker_show_the_scene_showing(tmp_path):
    import json
    (tmp_path / 'p.py').write_text("from hydrapy import *\n@scene('a')\ndef _(): return solid(1,0,0)\n"
                                   "@scene('b')\ndef _(): return solid(0,0,1)\n")
    session = Session(tmp_path, 'p.py', backend='cpu', size=(32, 18), fps=30)
    try:
        session.renderer.patch.scene_weights = {'a': 0., 'b': 1.}
        job = session.call('export', format='png', backend='cpu', size=[32, 18], fps=10, seconds=1)
        import time
        deadline = time.monotonic() + 60
        status = {}
        while time.monotonic() < deadline:
            jobs = list((tmp_path / '.hydrapy' / 'jobs').glob('*/status.json'))
            status = json.loads(jobs[0].read_text()) if jobs else {}
            if status.get('state') in ('done', 'failed', 'error'):
                break
            time.sleep(.2)
        assert status.get('state') == 'done', status
        from PIL import Image
        out = json.loads(jobs[0].with_name('job.json').read_text())['output']
        pixel = np.asarray(Image.open(out))[0, 0]
        assert pixel[2] > 200 and pixel[0] < 50              # scene b (blue), not the first declared
    finally:
        session.close()


class Counted(Signal):
    """A constant that counts how often a renderer evaluates it."""

    def __init__(self, v):
        self.v, self.calls = v, 0

    def value(self, ctx):
        self.calls += 1
        return self.v


def test_cpu_skips_scenes_that_are_not_showing(patch):
    from hydrapy.backend_np import Renderer
    freq = {name: Counted(4 + k) for k, name in enumerate(('drift', 'fracture', 'third'))}
    for name, f in freq.items():
        scene(name)(lambda f=f: osc(f, .1))
    r = Renderer(patch, size=(16, 9), fps=10)
    for w, drawn in [({'drift': 1}, {'drift'}), ({'drift': 1, 'fracture': .5}, {'drift', 'fracture'}),
                     ({'third': 1}, {'third'})]:
        patch.scene_weights = w
        for f in freq.values():
            f.calls = 0
        r.step(.5)
        assert {name for name, f in freq.items() if f.calls} == drawn


@pytest.mark.parametrize('amount', [0, .4])
def test_cpu_gated_combines_match_the_ungated_result(patch, amount):
    # A chain-valued amount is per pixel, so it is never gated: the original evaluation path.
    from hydrapy.backend_np import render_chain
    a = render_chain(osc(6).blend(voronoi(5), amount).modulate(noise(3), amount), t=.3, size=(24, 12))
    b = render_chain(osc(6).blend(voronoi(5), solid(amount)).modulate(noise(3), solid(amount)), t=.3, size=(24, 12))
    assert np.array_equal(a, b)
