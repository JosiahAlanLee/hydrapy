"""Kick discrimination, power-of-two folds and the strict layer budget."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/"docs/bell_riots_sparse.py"
    spec=importlib.util.spec_from_file_location("sparse_video_test",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def score(m):
    d=dict(duration=361.,tempo=137.,rate=10.,beats=np.arange(0,361,60/137),
           energy=np.tile([.4,.3,.6],(3611,1)),hit=np.full(3611,.8))
    t=np.arange(.07,361,60/137)
    return m.SparseScore(d,t,np.ones(len(t)),start=0.)


def test_kicks_reject_high_hats_and_missing_beats():
    module()
    from bell_riots_kicks import detect_kicks
    sr=8000;t=np.arange(sr*5)/sr;x=np.zeros_like(t)
    expected=np.array([.35,.85,1.35,2.35,2.85,3.35,3.85,4.35])
    for hit in expected:
        d=t-hit;v=np.maximum(d,0)
        x+=np.where(d>=0,np.sin(2*np.pi*65*v)*np.exp(-v/.07),0)
    # Include a hi-hat at the deliberately missing kick, and eighth-note hats.
    for hit in np.arange(.1,4.9,.25):
        d=t-hit;v=np.maximum(d,0)
        x+=np.where(d>=0,.2*np.sin(2*np.pi*2800*v)*np.exp(-v/.008),0)
    found,_=detect_kicks(x,sr,.5,np.arange(.35,5,.5))
    assert len(found)==len(expected)
    assert np.max(abs(found-expected))<.025
    assert len(detect_kicks(np.zeros(sr),sr)[0])==0


def test_generation_counts_and_no_early_pulse():
    m=module();s=score(m);k=s.clock
    assert k.pulse(k.times[0]-.001)==0
    assert k.pulse(k.times[0])>.9
    assert set(np.diff(sorted(k.fold_generations)))<={32,64}
    np.testing.assert_allclose(k.generation_times[::8],k.times[:len(k.generation_times[::8])])
    for i,t in enumerate(k.fold_times):
        assert t in k.times
        frame=np.ceil(t*30-1e-9)/30
        assert -1e-9<=frame-t<1/30+1e-9
    assert np.all(np.diff(k.generation_times)>0)


def test_two_layers_including_titles_and_transition_gaps():
    m=module();s=score(m)
    times=list(np.arange(0,s.duration,.041))
    times += [t for c in s.cues for t in (c["start"],c["end"],c["start"]+.001,c["end"]-.001)]
    assert max(len(s.visible_layers(t)) for t in times)==2
    for a,b in zip(s.cues,s.cues[1:]):
        assert b["start"]-a["end"]>=4.2
        assert s.visible_layers((a["end"]+b["start"])/2)==("unfolding",)
    assert all(s.at(t)["amount"]==0 for t in (1.,3.,357.,360.))


def test_state_catchup_and_all_accents(tmp_path):
    m=module()
    try:ctx=m._create_context()
    except Exception as exc:pytest.skip(str(exc))
    import shutil
    if not shutil.which("inkscape") and importlib.util.find_spec("cairosvg") is None:
        ctx.release();pytest.skip("SVG rasterizer unavailable")
    art=tmp_path/"art"
    for path in m.DRAWINGS:
        p=art/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                     '<path fill="none" stroke="black" stroke-width="2" d="M10 90L50 10L90 90ZM10 50H90"/></svg>')
    a=b=None
    try:
        s=score(m);a=m.SparseRenderer(s,(160,96),art,tmp_path/"cache",ctx)
        b=m.SparseRenderer(s,(160,96),art,tmp_path/"cache",ctx)
        for t in np.arange(0,12,1/30):a.draw(float(t),read=False)
        first=a.draw(12.);second=b.draw(12.)
        np.testing.assert_array_equal(first,second)
        for cue in s.cues[:6]:
            t=(cue["start"]+cue["end"])/2;frame=a.draw(t)
            assert frame.shape==(96,160,3) and frame.dtype==np.uint8
            assert frame.max()>100 and frame.std()>5
            dark=np.mean(frame.astype(float)@np.array([.2126,.7152,.0722])<20.4)
            assert dark>.45
        with pytest.raises(ValueError):a.draw(1.)
    finally:
        if a:a.release()
        if b:b.release()
        ctx.release()
