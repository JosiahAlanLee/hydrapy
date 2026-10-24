"""Audio processing and composition contracts for the fourth closing edit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def modules():
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'docs'),str(root/'patches')]
    import bell_riots_mix as mix
    import bell_riots_counterpoint as video
    return mix,video


def score(video):
    n=round(361.093*200)+1
    controls={key:np.zeros(n,dtype='f4') for key in ('hat','cow','stab','high')}
    controls.update(duration=361.093,rate=200.,source_start=434.46189569161)
    kicks=np.arange(.01,361.093,60/137)
    return video.CounterpointScore(controls,kicks,np.ones(len(kicks)))


def test_bass_shelf_changes_balance_without_shifting_channels():
    mix,_=modules();sr=48000;t=np.arange(sr*2)/sr
    for hz,expected in ((35,1.8),(4000,0.)):
        x=np.tile((.2*np.sin(2*np.pi*hz*t))[:,None],(1,2)).astype('f4')
        y=mix.shelf(x)
        gain=10*np.log10(np.mean(y[sr:]**2)/np.mean(x[sr:]**2))
        assert abs(gain-expected)<.15
        np.testing.assert_array_equal(y[:,0],y[:,1])


def test_parallel_saturation_compensates_gain_and_reduces_aliases():
    mix,_=modules();sr=48000;t=np.arange(sr)/sr
    x=np.tile((.3*np.sin(2*np.pi*440*t))[:,None],(1,2)).astype('f4')
    y=mix.saturate(x)
    gain=10*np.log10(np.mean(y[1000:-1000]**2)/np.mean(x[1000:-1000]**2))
    assert -3.5<gain<-2.9 and len(y)==len(x)
    x=np.tile((.7*np.sin(2*np.pi*18000*t))[:,None],(1,2)).astype('f4')
    y=mix.saturate(x,drive=7,blend=1,reduction=0)[:,0]
    naive=np.tanh(x[:,0]*7)
    window=np.hanning(sr)
    alias=abs(np.fft.rfft(y*window))[6000]
    reference=abs(np.fft.rfft(naive*window))[6000]
    # 4x oversampling should reduce this extreme 18 kHz overload by >20 dB;
    # finite oversampling does not remove every high-order tanh harmonic.
    assert alias<reference*.10


def test_upper_envelope_tracks_real_transient():
    mix,_=modules();sr=48000;t=np.arange(sr)/sr
    d=np.maximum(t-.4,0)
    x=np.where(t>=.4,np.sin(2*np.pi*7000*d)*np.exp(-d/.025),0)
    env=mix.control_envelope(np.tile(x[:,None],(1,2)),band=(3500,16000))
    assert env.max()>.5 and env[:77].max()<.001
    assert .395<=np.argmax(env)/200<=.44
    assert env[150:].max()<.001


def test_story_has_continuity_brief_bridges_and_two_layers():
    _,v=modules();s=score(v)
    assert s.cues[0]['start']==0 and s.cues[-1]['end']==s.duration
    for a,b in zip(s.cues,s.cues[1:]):assert abs(a['end']-b['start'])<1e-10
    assert s.cues[-1]['mode']==0
    bridges=sum(c['end']-c['start'] for c in s.cues if c['mode']==4)
    assert 0<bridges/s.duration<.03
    assert max(len(s.visible_layers(t)) for t in np.arange(0,s.duration,1/30))==2


def test_all_pairs_render_and_upper_controls_change_the_image(tmp_path):
    _,v=modules()
    from hydrapy.backend_gl import _create_context
    from bell_riots_organic import DRAWINGS
    try:ctx=_create_context()
    except Exception as exc:pytest.skip(str(exc))
    import shutil
    if not shutil.which('inkscape') and importlib.util.find_spec('cairosvg') is None:
        ctx.release();pytest.skip('SVG rasterizer unavailable')
    art=tmp_path/'art'
    for relative in DRAWINGS:
        p=art/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path fill="none" stroke="black" stroke-width="2" d="M10 90L50 10L90 90ZM10 50H90"/></svg>')
    r=None
    try:
        s=score(v);r=v.CounterpointRenderer(s,(160,96),art,tmp_path/'cache',ctx)
        modes=set()
        for t in (10.,17.,25.,28.3,32.):
            a=r.draw(t);mode=s.at(t)['mode'];modes.add(mode)
            assert a.shape==(96,160,3) and a.max()>40 and a.std()>3
            if mode!=4:
                for key in ('hat','cow','stab','high'):s.controls[key][:]=1.
                b=r.draw(t)
                assert np.mean(abs(a.astype(float)-b))>.05
                for key in ('hat','cow','stab','high'):s.controls[key][:]=0.
        assert modes=={0,1,2,3,4}
    finally:
        if r:r.release()
        ctx.release()
