"""Re-scoring the external hat patch to the edit 4/5 cuts, and the render checks."""
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'docs'),str(ROOT/'patches')]
import bell_riots_rescore as r
import bell_riots_checks as c

CUES=[(0,16,0),(16,32,1),(32,48,0),(48,64,3),(64,66,4),(66,82,2),(82,89,3),(89,90,4),(90,98,2),
      (98,130,1),(130,134,2),(134,138,0),(138,146,1),(146,826,0)]


def cut_beats(cues):
    return {a for a,_,_ in cues if a>1}


def test_patterns_rolls_and_stabs_follow_the_cut_list():
    arrays,shots=r.score(CUES)
    cuts=cut_beats(CUES)
    assert all(len(v)==826 for v in arrays.values())
    hv=arrays['hv']
    changes={b for b in range(10,826) if hv[b]!=hv[b-1]}
    assert changes<=cuts                                   # patterns only change where the picture cuts
    assert {b+1 for b in np.where(arrays['fill']>0)[0]}<=cuts
    long_starts={a+5 for a,b in shots if b-a>=16}
    assert set(np.where(arrays['stab']>0)[0])<={b-1 for b in cuts}|long_starts
    assert set(np.unique(arrays['stab']))<={0,2,3} and set(np.unique(arrays['chord']))<={0,1,2,3}
    assert not hv[:9].any() and np.isclose(arrays['gain'][0],.4) and arrays['gain'][-1]==0
    assert not arrays['stab'][-9:].any() and not arrays['cb'][-9:].any() and not arrays['cb'][:33].any()


def test_bridge_rolls_are_added_without_changing_the_other_draws():
    plain,_=r.score(CUES);bridged,_=r.score(CUES,roll_into_bridges=True)
    bridges=[a for a,_,m in CUES if m==r.BRIDGE]
    assert all(bridged['fill'][b-1]==1 for b in bridges)
    other=np.ones(826,bool);other[[b-1 for b in bridges]]=False
    np.testing.assert_array_equal(plain['fill'][other],bridged['fill'][other])
    for k in ('hv','stab','chord','cb','gain'):np.testing.assert_array_equal(plain[k],bridged[k])


@pytest.mark.parametrize('offset',range(4))
def test_bar_offset_moves_the_pattern_start_as_the_patch_reads_it(offset):
    original=np.arange(32,dtype=float)                     # two 16-step variants
    moved=r.rotate(original,offset)
    for variant in range(2):
        for beat in range(8):
            for step in range(4):
                # bell_hats.pd reads table[variant*16 + (beat mod 4)*4 + step]
                got=moved[variant*16+(beat%4)*4+step]
                assert got==original[variant*16+((beat-offset)%4)*4+step]


def fake_patch(folder,n=826):
    folder.mkdir()
    tables=[(k,n) for k in r.SCORE_ARRAYS]+[('hclosed',80),('hopen',80),('cowpat',32)]
    (folder/'bell_hats.pd').write_text('#N canvas 40 40 1300 1200 12;\n'+''.join(
        f'#X obj {20+i*150} 90 table {k} {size};\n' for i,(k,size) in enumerate(tables)))
    (folder/'fmhat~.pd').write_text('#N canvas 0 0 10 10 12;\n')
    for name,size in (('closed',80),('open',80),('cowpat',32)):
        (folder/f'{name}.txt').write_text(' '.join(str(i%7) for i in range(size))+'\n')
    for name in r.SCORE_ARRAYS:(folder/f'{name}.txt').write_text('0 '*n+'\n')
    return folder


def test_rescore_writes_a_new_folder_and_leaves_the_supplied_one_alone(tmp_path):
    src=fake_patch(tmp_path/'bell_hats');before={p.name:p.read_bytes() for p in src.iterdir()}
    out=tmp_path/'bell_hats_edit5'
    summary=r.rescore(src,out,CUES,bar_offset=2)
    assert {p.name:p.read_bytes() for p in src.iterdir()}==before
    assert (out/'bell_hats.pd').read_bytes()==before['bell_hats.pd'] and (out/'fmhat~.pd').exists()
    for name in r.SCORE_ARRAYS:assert len((out/f'{name}.txt').read_text().split())==826
    closed=np.array((out/'closed.txt').read_text().split(),dtype=float)
    np.testing.assert_array_equal(closed,r.rotate(np.array(before['closed.txt'].split(),dtype=float),2))
    assert summary['rolls']==summary['rolls_into_cuts'] and summary['bar_offset']==2
    with pytest.raises(ValueError):r.rescore(src,src,CUES)


def test_rescore_rejects_a_patch_with_other_tables(tmp_path):
    src=fake_patch(tmp_path/'bell_hats',n=700)
    (src/'bell_hats.pd').write_text((src/'bell_hats.pd').read_text().replace('table cb 700','table cb 826'))
    with pytest.raises(ValueError):r.rescore(src,tmp_path/'out',CUES)


def test_edit_cues_come_from_the_renderer_score():
    try:cues=r.score_cues()
    except Exception as exc:pytest.skip(f'counterpoint renderer unavailable: {exc}')
    assert len(cues)==76 and cues[:5]==[(0,16,0),(16,32,1),(32,48,0),(48,64,3),(64,66,4)]
    assert sum(m==r.BRIDGE for _,_,m in cues)>=5


def test_cut_detection_finds_abrupt_changes_but_not_motion():
    rng=np.random.default_rng(3);thumbs=[];level=rng.uniform(20,200,(18,32))
    for i in range(300):
        if i in (60,150,151,240):level=rng.uniform(20,200,(18,32))     # 150/151 is a one-frame flash
        thumbs.append(level+rng.normal(0,1.5,(18,32))+np.sin(i/9)*3)
    cuts=c.detect_cuts(np.array(thumbs,np.float32))
    assert {60,240}<=set(cuts.tolist())
    assert set(cuts.tolist())<={60,150,151,152,240}


def test_roll_detection_on_synthetic_hats(tmp_path):
    if shutil.which('ffmpeg') is None:pytest.skip('ffmpeg unavailable')
    from scipy.io import wavfile
    sr=48000;x=np.zeros(int(40*sr),np.float32);noise=np.random.default_rng(1).normal(0,1,240).astype(np.float32)
    def hit(t,v=.5):i=int(t*sr);x[i:i+240]+=noise*np.exp(-np.arange(240)/40)*v
    rolls={20,41,60}
    for b in range(1,90):
        t0=c.BEAT0+b*c.PERIOD
        for k in range(8 if b in rolls else 0):hit(t0+k*c.PERIOD/8)
        hit(t0+c.PERIOD/2)                                 # ordinary off-beat hats
    wavfile.write(tmp_path/'hats.wav',sr,x)
    res=c.check_rolls(tmp_path/'hats.wav',[21,61,80])
    assert set(res['roll_beats'])==rolls and res['rolls_into_cuts']==2 and res['cuts_with_roll']==2
