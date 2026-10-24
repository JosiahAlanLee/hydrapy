"""Bella Mort: the beat-locked harmonograph thread, its acts, the cut and the coda."""
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs'))
import bella_mort_video as b


def synthetic(seconds=470.,bpm=140.,rate=100.):
    period=60/bpm;beats=np.arange(.47,seconds,period)
    n=int(seconds*rate);t=np.arange(n)/rate
    hat=.4+.3*np.sin(2*np.pi*t/40)**2;onset=(np.sin(2*np.pi*t/period)>.95).astype(float)
    return dict(beats=beats,rate=rate,energy=np.stack([hat,hat,hat],1),hit=onset,onset=onset,tempo=bpm,
                duration=seconds,sha256='x')


def test_phrase_grid_and_acts():
    s=b.MortScore(synthetic(),465.6)
    start=s.time_at(b.PHRASE0)
    assert abs(start-(.47+15*60/140))<1e-9 and abs(s.phrase_at(start))<1e-9
    assert abs(s.phrase_at(s.time_at(b.PHRASE0+32*7))-7)<1e-9
    g=np.linspace(-.5,34,2000);w=s.acts(g)
    for key in ('sym','iris','taut','dilation'):assert np.all(np.diff(w[key])>=-1e-12)
    assert w['sym'][g<b.ACTS['lachesis']].max()==0 and w['iris'][g<b.ACTS['atropos']].max()==0
    assert w['taut'][g>=b.ACTS['taut']+1].min()==1
    assert s.end==pytest.approx(465.6+b.TAIL)


def test_pen_never_lifts():
    """One continuous thread: sampled at 600 Hz, no step is a jump, through every act and phrase."""
    s=b.MortScore(synthetic(),465.6);t=np.arange(0,465.6,1/600)
    steps=[]
    for chunk in np.array_split(t,40):
        pos,weight,_=s.pen(np.r_[chunk,chunk[-1]+1/600])
        steps.append(np.linalg.norm(np.diff(pos,axis=0),axis=-1).max(axis=1))
    steps=np.concatenate(steps)
    assert steps.max()<.02                       # under 22 px per 1/600 s at 1080 wide
    around=np.maximum(np.r_[steps[2:],0,0],np.r_[0,0,steps[:-2]])
    assert (steps/(around+2e-4)).max()<2.5       # a lifted pen would be one step far longer than its neighbours
    pos,weight,_=s.pen(np.array([30.,200.,400.]))
    assert weight[0,0]==1 and weight[0,1:].max()==0          # Clotho: one thread
    assert weight[1].min()>0 and weight[2].min()>0           # Lachesis and Atropos: ten copies
    r=np.linalg.norm(pos[2],axis=-1)
    assert np.all(r>=s.pupil(np.array([400.]),np.array([s.phrase_at(400.)]))[0]-1e-9)
    assert np.all(r<=b.IRIS*1.08)


def test_eye_closes_into_a_line():
    s=b.MortScore(synthetic(),465.6);late=s.time_at(b.PHRASE0+32*(b.ACTS['taut']+1)+4)
    pos,_,_=s.pen(np.linspace(late,late+1,200))
    assert np.abs(pos[...,1]).max()<.01 and np.abs(pos[...,0]).max()>.5


def test_coda_is_short_quiet_and_ends_silent():
    y=b.coda(465.6);sr=44100
    assert y.shape==(round(b.TAIL*sr),2)
    assert 20*np.log10(np.abs(y).max())<-6
    assert 20*np.log10(np.sqrt((y[-sr//2:]**2).mean())+1e-12)<-50


def test_find_stop_and_mix(tmp_path):
    sr=44100;t=np.arange(int(2.5*sr))/sr;x=.5*np.sin(2*np.pi*220*t)*(t<1.7)
    path=tmp_path/'tone.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((x*32767).astype('<i2').tobytes())
    stop=b.find_stop(path);assert abs(stop-1.7)<.01
    hit=x.copy();hit[(t>=1.62)&(t<1.62+.005)]*=0;hit[t>=1.7]=0
    hit[(t>=1.65)&(t<1.7)]=.9*np.exp(-(t[(t>=1.65)&(t<1.7)]-1.65)/.01)*np.sin(2*np.pi*90*t[(t>=1.65)&(t<1.7)])
    hit[(t>=1.62)&(t<1.65)]*=.05
    with wave.open(str(tmp_path/'hit.wav'),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((hit*32767).astype('<i2').tobytes())
    assert abs(b.find_stop(tmp_path/'hit.wav')-1.65)<.006   # the cut lands on the last hit, not its decay
    report=b.build_mix(path,tmp_path/'mix.flac',stop)
    assert report['end']==pytest.approx(stop+b.TAIL,abs=1e-4)
    out=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',
                        str(tmp_path/'mix.flac')],capture_output=True,text=True,check=True).stdout
    assert float(out)==pytest.approx(stop+b.TAIL,abs=.01)


def test_renderer_draws_thread_then_black_after_the_cut():
    try:
        from hydrapy.backend_gl import _create_context
        _create_context().release()
    except Exception as exc:
        pytest.skip(f'no OpenGL context: {exc}')
    s=b.MortScore(synthetic(),465.6);r=b.MortRenderer(s,(216,270),30.)
    try:
        lit=[r.draw(30+i/30) for i in range(20)][-1]
        assert lit.max()>120 and lit.mean()<40
        r.last=None
        for i in range(10):r.draw(s.stop+3+i/30)
        dark=r.draw(s.stop+3.4)
        assert dark.max()<12
    finally:r.release()
