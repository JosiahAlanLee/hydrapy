"""Timing, real delay history and scope of the dancer matte."""
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'docs'),str(ROOT/'patches')]
import bell_riots_dancer as d


def context():
    from hydrapy.backend_gl import _create_context
    try:return _create_context()
    except Exception as exc:pytest.skip(str(exc))


def test_measured_beats_keep_phase_across_missing_kicks_and_long_spans():
    period=60/137
    beats=np.array([0,1,2,4,5,8,16,32,300,824])
    jitter=np.array([.012,.003,-.005,.009,.001,-.008,.004,.006,-.004,.002])
    times=beats*period+jitter
    clock=d.DanceClock(times,beat0=0)
    np.testing.assert_allclose([clock.beat_at(t) for t in times],beats,atol=1e-10)
    assert [clock.tick_at(t) for t in times]==list(beats*15)
    # The missing beat between indices 2 and 4 stays beat 3, not beat 2.5.
    assert clock.beat_at((times[2]+times[3])/2)==3
    assert abs(clock.beat_at(times[-1]+32*period)-856)<1e-9
    assert d.HALF_BPM==68.5
    assert abs(d.PRINT_PERIOD-32*period)<1e-12


def test_duplicate_and_off_grid_onsets_do_not_renumber_the_dance():
    period=60/137
    clock=d.DanceClock([period,period+.015,1.5*period,2*period],beat0=0)
    np.testing.assert_array_equal(clock.beats,[1,2])
    assert clock.tick_at(period)==15
    assert d.DanceClock([],beat0=0).tick_at(8*period)==120
    with pytest.raises(ValueError):clock.beat_at(float('nan'))


def test_hydrapy_delay_taps_are_exact_and_seek_preroll_matches_history():
    ctx=context();a=b=None
    # Constant images remove spatial shifts from this check, making each tap's
    # value identify its source frame unambiguously.
    bank=np.broadcast_to((np.arange(480)%240).astype('u1')[:,None,None],(480,18,32)).copy()
    clock=d.DanceClock([],beat0=0)
    try:
        a=d.DelayedDancer(bank,clock,ctx);b=d.DelayedDancer(bank,clock,ctx)
        a.advance(95/(137/4))
        pixels=a.renderer.read()
        assert abs(pixels[9,16,0]-95)<=1
        expected=.60*65+.34*35+.18*5
        assert abs(float(pixels[9,16,1])-expected)<=1
        assert len(a.renderer.outs['o0']['ring'])==92
        a.advance(100/(137/4));b.advance(100/(137/4))
        np.testing.assert_array_equal(a.renderer.read(),b.renderer.read())
        # Jumping across a hidden passage still reconstructs the same last 90 ticks.
        a.advance(700/(137/4));b.advance(700/(137/4))
        np.testing.assert_array_equal(a.renderer.read(),b.renderer.read())
        steps=a.steps;a.advance(700/(137/4));assert a.steps==steps
        with pytest.raises(ValueError):a.advance(0)
    finally:
        if a:a.release()
        if b:b.release()
        ctx.release()


def test_only_paint_passages_gain_the_dancer(tmp_path):
    if not shutil.which('inkscape') and importlib.util.find_spec('cairosvg') is None:
        pytest.skip('SVG rasterizer unavailable')
    from bell_riots_organic import DRAWINGS
    art=tmp_path/'art'
    for relative in DRAWINGS:
        p=art/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path fill="none" stroke="black" stroke-width="2" d="M10 90L50 10L90 90ZM10 50H90"/></svg>')
    controls={k:np.zeros(361*200+1,dtype='f4') for k in ('hat','cow','stab','high')}
    controls.update(duration=361.,rate=200.,source_start=434.46189569161)
    kicks=np.arange(.015,361,60/137)
    score=d.CounterpointScore(controls,kicks,np.ones(len(kicks)))
    bank=np.zeros((480,36,64),dtype='u1');bank[:,6:32,25:38]=255
    ctx=context();base=changed=None
    try:
        base=d.CounterpointRenderer(score,(160,96),art,tmp_path/'cache',ctx)
        changed=d.DancerRenderer(score,(160,96),art,tmp_path/'cache',ctx,matte=bank)
        for t in (3.,10.,17.):
            a=base.draw(t);b=changed.draw(t)
            if score.at(t)['mode']==1:
                assert np.mean(abs(a.astype(float)-b))>2
                assert np.mean(b>175)>np.mean(a>175)
            else:np.testing.assert_array_equal(a,b)
        assert max(len(score.visible_layers(t)) for t in np.arange(0,361,.25))==2
    finally:
        if base:base.release()
        if changed:changed.release()
        ctx.release()
