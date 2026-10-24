"""The edit can be sampled in any order; cellular updates use a fixed clock."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/"docs"/"bell_riots_video.py"
    spec=importlib.util.spec_from_file_location("bell_video_test",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def score(m):
    data=dict(duration=64.,tempo=120.,rate=100.,beats=np.arange(.1,64.,.5),
              energy=np.tile([.4,.3,.6],(6401,1)),hit=np.full(6401,.8))
    return m.Score(data)


def test_score_covers_track_and_locks_reproducibly():
    m=module();a,b=score(m),score(m)
    assert a.cuts==b.cuts and a.starts[0]==0
    assert np.all(np.diff(a.starts)>0) and a.starts[-1]<a.duration
    np.testing.assert_array_equal(a.locked,b.locked)
    assert sum(a.lock_counts)>0
    for c in a.cuts[1:]:
        assert abs(c["start"]-a.beat_time(c["beat"]))<1e-10
    expected=[a.at(t) for t in (0.,12.3,40.8,63.99)]
    for t in np.linspace(0,63.9,600):a.at(t)
    assert expected==[a.at(t) for t in (0.,12.3,40.8,63.99)]


def test_excerpt_controls_use_absolute_track_time():
    m=module();s=score(m)
    s.data["energy"][2000:]=[.8,.7,.1]
    before,after=s.at(19.9),s.at(20.1)
    assert before["bass"]==.4 and after["bass"]==.8
    assert 0<=after["glitch"]<=.86
    assert len(after["locked"])==4


def test_all_visual_systems_and_fixed_simulation_steps():
    m=module()
    try:ctx=m._create_context()
    except Exception as exc:pytest.skip(str(exc))
    a=b=None
    try:
        a=m.Renderer(score(m),(160,96),ctx=ctx)
        b=m.Renderer(score(m),(160,96),ctx=ctx)
        for t in np.arange(0,2.001,1/30):a.advance(float(t))
        b.advance(2.)
        np.testing.assert_array_equal(a.field.read_rgba(),b.field.read_rgba())
        for scene in range(len(m.SCENES)):
            frame=a.draw(12.,force_scene=scene)
            assert frame.shape==(96,160,3) and frame.dtype==np.uint8
            assert frame.max()>60 and frame.std()>3, m.SCENES[scene]
        with pytest.raises(ValueError):a.advance(1.)
    finally:
        if a:a.release()
        if b:b.release()
        ctx.release()
