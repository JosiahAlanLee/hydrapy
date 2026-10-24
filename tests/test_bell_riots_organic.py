"""Edit/audio alignment and persistent, independent visual clocks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/"docs/bell_riots_organic.py"
    spec=importlib.util.spec_from_file_location("bell_organic_test",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def data():
    return dict(duration=795.55,tempo=137.,rate=10.,beats=np.arange(.019,795.55,60/137),
                energy=np.tile([.4,.3,.6],(7956,1)),hit=np.full(7956,.8))


def test_phrase_edit_and_audio_controls():
    m=module();d=data();s=m.OrganicScore(d)
    phrase_beats=d["beats"][::32]
    assert s.start in phrase_beats
    assert abs(s.duration-360)<16*60/137
    assert s.start+s.duration==float(d["duration"])
    d["energy"][:int((s.start+20)*10)]=[.1,.2,.3]
    assert s.at(19)["bass"]==.1 and s.at(21)["bass"]==.4
    assert s.cuts[0]["start"]==0
    assert all(a["start"]<b["start"] for a,b in zip(s.cuts,s.cuts[1:]))
    for cut in s.cuts[1:]:
        assert abs(cut["start"]-s.beat_time(cut["beat"]))<1e-8
    assert s.at(22)==m.OrganicScore(d).at(22)
    with pytest.raises(ValueError):m.OrganicScore(d,800)


def test_layer_clocks_do_not_restart_at_cuts():
    m=module();s=m.OrganicScore(data())
    # Layer transitions lie between edit points, instead of being bar-reset clips.
    for name,offset in (("paint",9.1),("drawings",3.7)):
        period=m.CLOCKS[name]
        boundaries=np.arange(period-offset,s.duration,period)
        assert all(np.min(abs(s.starts-t))>1e-3 for t in boundaries)
    values=[m.cycle(t,m.CLOCKS["paint"],9.1) for t in (0,19,80,210)]
    assert len(set(values))==4
    a,b,w=m.cycle(m.CLOCKS["paint"]-1e-6,m.CLOCKS["paint"])
    c,d,v=m.cycle(m.CLOCKS["paint"]+1e-6,m.CLOCKS["paint"])
    assert b==c and w>1-1e-9 and v==0
    assert len(set(m.CLOCKS.values()))==len(m.CLOCKS)


def test_gl_layers_and_simulation_replay(tmp_path):
    m=module()
    try:ctx=m._create_context()
    except Exception as exc:pytest.skip(str(exc))
    if not m.shutil.which("inkscape") and importlib.util.find_spec("cairosvg") is None:
        ctx.release();pytest.skip("SVG rasterizer is not installed")
    a=b=None
    try:
        s=m.OrganicScore(data())
        art=tmp_path/"art"
        # Integration fixtures contain no copied user artwork.
        for path in m.DRAWINGS:
            target=art/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                              '<path fill="none" stroke="black" stroke-width="2" '
                              'd="M10 90L50 10L90 90ZM10 50H90M30 20L70 80"/></svg>')
        a=m.OrganicRenderer(s,(160,96),tmp_path/"cache",ctx,art)
        b=m.OrganicRenderer(s,(160,96),tmp_path/"cache",ctx,art)
        for t in np.arange(0,2.001,1/30):a.advance(float(t))
        b.advance(2.)
        aa,at=a.mould.gl_state(a.network);ba,bt=b.mould.gl_state(b.network)
        np.testing.assert_array_equal(aa,ba);np.testing.assert_array_equal(at,bt)
        for scene in (0,2,3,8):
            frame=a.draw(12.,force_scene=scene)
            assert frame.shape==(96,160,3) and frame.dtype==np.uint8
            assert frame.max()>90 and frame.std()>10
        # The incoming paint must keep its crop when it becomes the current paint.
        boundary=m.CLOCKS["paint"]-9.1
        before=a.draw(boundary-1e-5,force_scene=0)
        after=a.draw(boundary+1e-5,force_scene=0)
        assert np.abs(after.astype(float)-before).mean()<1.
        with pytest.raises(ValueError):a.draw(1.)
        for tex in a.paint:
            rgba=np.frombuffer(tex.read(),np.float16).astype(np.float32).reshape(-1,4)
            assert np.isfinite(rgba).all() and rgba[:,:3].std()>.1
    finally:
        if a:a.release()
        if b:b.release()
        ctx.release()
