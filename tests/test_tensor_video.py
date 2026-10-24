"""Score invariants for the reproducible Tensor Junglist music-video edit."""
import importlib.util
from pathlib import Path

import numpy as np

import hydrapy as H


def score():
    before=set(H.REGISTRY)
    path=Path(__file__).resolve().parents[1]/"docs"/"tensor_junglist_video.py"
    spec=importlib.util.spec_from_file_location("tensor_video_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(H.REGISTRY)==before
    data=dict(duration=40.,tempo=120.,rate=100.,beats=np.arange(.25,40,.5),
              energy=np.tile([.3,.6,.9],(4001,1)),hit=np.full(4001,.8))
    return module.Score(data)


def test_edit_is_deterministic_and_covers_track():
    a,b=score(),score()
    assert a.cuts==b.cuts
    assert np.all(np.diff(a.starts)>0)
    assert 0<=a.starts[0]<a.period
    assert a.starts[-1]<a.duration
    for t in np.linspace(0,a.duration-1e-5,1000):
        cut,index,glitch=a.update(t)
        assert a.cuts[index]==cut and 0<=cut["scene"]<9
        assert 0<=glitch<=1
        assert cut["start"]<=t or index==0
        if index+1<len(a.cuts): assert a.cuts[index+1]["start"]>t


def test_subbeat_cuts_interpolate_the_detected_beats():
    s=score()
    assert s.beat_time(.5)==.5
    assert s.beat_time(-1)==-.25
    assert s.beat_time(82)==41.25
    for c in s.cuts:
        assert abs(((c["start"]-.25)/.5)*2-round(((c["start"]-.25)/.5)*2))<1e-8


def test_audio_controls_use_track_time_and_stay_finite():
    s=score()
    s.update(12.5)
    np.testing.assert_allclose([s.controls[k] for k in ("bass","mid","high","hit")],[.3,.6,.9,.8])
    values=[s.update(t)[2] for t in (0,.25,12.5,39.999)]
    assert np.isfinite(values).all()
