"""Measure a Bell Riots render against the 137 BPM score grid.

    python docs/bell_riots_checks.py all Bell_Riots_Edit_5_1080p.mp4 [--json report.json]

Subcommands (each takes the rendered MP4; ``audio`` also accepts an audio file):

  cuts     picture cuts found in the video, their offset from the beat grid and
           agreement with CounterpointScore's cue starts
  rolls    32nd-note hat rolls in the audio and whether each one leads into a cut
  dancer   the dancer silhouette per 480-tick cycle: whether poses repeat with the
           cycle, and where its holds fall relative to the half-time beats
  audio    EBU R128 loudness, true peak, loudness range and the levels at both edges
  encode   video encoder settings recorded in the stream

These are measurements, not pass/fail tests. Thresholds that pick out the dancer
assume edit 5's chalk-white figure; adjust --figure-min and --figure-sat for others.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import butter, sosfilt

ROOT=Path(__file__).resolve().parents[1]
PERIOD=60/137.
BEAT0=-.008301162003440554
TICKS_PER_BEAT,TICKS_PER_CYCLE=15,480


def probe(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    video=next((s for s in info['streams'] if s['codec_type']=='video'),None)
    return info,video


def score_cut_beats():
    """Cue start beats from CounterpointScore, or None if the renderer can't be imported."""
    try:
        sys.path[:0]=[str(ROOT),str(ROOT/'docs'),str(ROOT/'patches')]
        from bell_riots_counterpoint import CounterpointScore
    except Exception:
        return None
    kicks=BEAT0+np.arange(1,826)*PERIOD
    s=CounterpointScore(dict(duration=361.093),kicks,np.ones_like(kicks))
    return sorted({int(round((c['start']-BEAT0)/PERIOD)) for c in s.cues if c['start']>0})


def scan_video(path,fig_min=160,fig_sat=.32,size=(256,144)):
    """One decoding pass: gray thumbnails, brightness and the central figure silhouette."""
    _,v=probe(path);num,den=map(int,v['avg_frame_rate'].split('/'));fps=num/den
    w,h=size;cx0,cx1=w//4,3*w//4
    proc=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-vf',f'scale={w}:{h}:flags=area',
                           '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    thumbs,bright,area,motion,sils=[],[],[],[],[]
    prev=None;nbytes=w*h*3
    while True:
        b=proc.stdout.read(nbytes)
        if len(b)<nbytes:break
        f=np.frombuffer(b,np.uint8).reshape(h,w,3).astype(np.float32)
        g=f.mean(2);thumbs.append(g.reshape(18,h//18,32,w//32).mean((1,3)));bright.append(float(g.mean()))
        c=f[:,cx0:cx1];mx,mn=c.max(2),c.min(2)
        m=(mx>fig_min)&((mx-mn)/(mx+1e-3)<fig_sat)
        s=None
        if m.sum()>200 and g.mean()<90:
            m=ndi.binary_closing(m,np.ones((5,5)));lab,k=ndi.label(m)
            if k:
                s=ndi.binary_fill_holes(lab==1+int(np.argmax(ndi.sum(m,lab,range(1,k+1)))))
                if s.sum()<400:s=None
        area.append(0 if s is None else int(s.sum()))
        motion.append(np.nan if s is None or prev is None else float((s^prev).sum()/s.sum()))
        sils.append(None if s is None else ndi.zoom(s.astype(np.float32),(48/h,32/(cx1-cx0)),order=1).astype(np.float16))
        prev=s
    proc.wait()
    return dict(fps=fps,thumbs=np.array(thumbs,np.float32),bright=np.array(bright),area=np.array(area),
                motion=np.array(motion),sils=sils)


def detect_cuts(thumbs):
    d=np.abs(np.diff(thumbs,axis=0)).mean((1,2))
    med=np.array([np.median(d[max(0,i-15):i+16]) for i in range(len(d))])
    dist=lambda a,b:float(np.abs(thumbs[a]-thumbs[b]).mean())
    cuts=[]
    for i in range(3,len(d)-4):
        if d[i]>max(4*med[i],5.) and d[i]>=d[i-1] and d[i]>=d[i+1]:
            if dist(i-1,i+3)>2*.5*(dist(i-3,i-1)+dist(i+2,i+4))+2:cuts.append(i+1)
    return np.array(cuts,int)


def check_cuts(scan,expected=None):
    fps=scan['fps'];cuts=detect_cuts(scan['thumbs']);t=cuts/fps
    beats=(t-BEAT0)/PERIOD;k=np.round(beats).astype(int)
    six=np.round((t-BEAT0)/(PERIOD/4));off16=(t-(BEAT0+six*PERIOD/4))*1000
    r=dict(cuts=len(cuts),cut_times=np.round(t,3).tolist(),cut_beats=k.tolist(),
           median_offset_ms=float(np.median(off16)) if len(t) else None,
           within_one_frame=float(np.mean(np.abs(off16)<=1000/fps+.5)) if len(t) else None,
           on_a_beat=float(np.mean(six%4==0)) if len(t) else None,
           beat_in_bar_counts=np.bincount(k%4,minlength=4).tolist())
    if expected is not None:
        found=set(k.tolist());r['score_cues']=len(expected);r['score_cues_found']=len(found&set(expected))
        r['unexpected_cuts']=sorted(found-set(expected))
    return r


def audio_mono(path,sr=48000):
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(sr),'-f','f32le','-'],
                       capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.float32),sr


def check_rolls(path,cut_beats):
    """Beats where hats fill at least six of the eight 32nd-note positions."""
    x,sr=audio_mono(path)
    y=sosfilt(butter(6,4500,'highpass',fs=sr,output='sos'),x)
    w=sr//1000;e=np.log(np.sqrt(np.convolve(y*y,np.ones(w)/w,'same'))[::w]+1e-6)
    d=np.diff(e);thr=np.percentile(d,99.)
    on=np.where((d>thr)&(np.r_[thr,d[:-1]]<=thr))[0]
    keep=[]
    for i in on:
        if not keep or i-keep[-1]>25:keep.append(i)
    on=np.array(keep)/1000.
    n=int((len(x)/sr-BEAT0)/PERIOD)
    rolls=[]
    for b in range(1,n):
        pos=BEAT0+b*PERIOD+np.arange(8)*PERIOD/8
        if sum(np.any(np.abs(on-p)<.012) for p in pos)>=6:rolls.append(b)
    cuts=set(int(c) for c in cut_beats)
    into=[b for b in rolls if b+1 in cuts]
    return dict(rolls=len(rolls),roll_beats=rolls,rolls_into_cuts=len(into),
                cuts_with_roll=len({b+1 for b in into}),cuts=len(cuts),chance=len(cuts)/max(n,1))


def check_dancer(scan):
    fps=scan['fps'];m=scan['motion'].copy();edge=np.zeros(len(m),bool)
    for c in detect_cuts(scan['thumbs']):
        m[max(0,c-2):c+3]=np.nan                                  # no motion across a cut
        edge[max(0,c-15):c+15]=True                               # the figure fades in/out here
    ok=np.isfinite(m)
    if ok.sum()<60:return dict(dancer_frames=int(ok.sum()),note='no dancer silhouette found')
    t=(np.arange(len(m))-.5)/fps
    tick=np.floor((t-BEAT0)/PERIOD*TICKS_PER_BEAT).astype(int)%TICKS_PER_CYCLE
    prof=np.full(TICKS_PER_CYCLE,np.nan)
    for k in range(TICKS_PER_CYCLE):
        sel=ok&(tick==k)
        if sel.any():prof[k]=m[sel].mean()
    good=np.isfinite(prof);p=prof.copy()
    p[~good]=np.interp(np.where(~good)[0],np.where(good)[0],p[good],period=TICKS_PER_CYCLE)
    ps=np.convolve(np.r_[p[-4:],p,p[:4]],np.ones(5)/5,'same')[4:-4]
    holds=[k for k in range(TICKS_PER_CYCLE) if ps[k]==ps[np.arange(k-12,k+13)%TICKS_PER_CYCLE].min()]
    half=2*TICKS_PER_BEAT;offsets=[int((k+half/2)%half-half/2) for k in holds]
    # pose repetition: frames on the same cycle tick vs frames at unrelated ticks
    idx=[i for i,s in enumerate(scan['sils']) if s is not None and not edge[i]]
    vec={i:(lambda v:(v-v.mean())/(np.linalg.norm(v-v.mean())+1e-6))(scan['sils'][i].astype(np.float32).ravel()) for i in idx}
    rng=np.random.default_rng(0);same,other=[],[]
    ftick=np.floor(((np.arange(len(m)))/fps-BEAT0)/PERIOD*TICKS_PER_BEAT).astype(int)
    by_tick={}
    for i in idx:by_tick.setdefault(int(ftick[i]%TICKS_PER_CYCLE),[]).append(i)
    for i in rng.choice(idx,min(1500,len(idx)),replace=False):
        partners=[j for j in by_tick[int(ftick[i]%TICKS_PER_CYCLE)] if abs(int(ftick[j])-int(ftick[i]))>=TICKS_PER_CYCLE//2]
        if partners:same.append(float(vec[i]@vec[rng.choice(partners)]))
        other.append(float(vec[i]@vec[rng.choice(idx)]))
    fold=p.reshape(-1,half).reshape(-1,6,5).mean((0,2))
    return dict(dancer_frames=int(ok.sum()),cycle_ticks_seen=float(good.mean()),
                same_tick_pose_similarity=float(np.median(same)) if same else None,
                random_pose_similarity=float(np.median(other)),
                holds=len(holds),hold_offsets_from_half_beat_ticks=offsets,
                holds_within_3_ticks_of_half_beat=float(np.mean(np.abs(offsets)<=3)),
                motion_by_sixth_of_half_beat=np.round(fold/fold.mean(),3).tolist(),
                motion_per_half_beat=np.round(p.reshape(-1,half).mean(1)/p.mean(),3).tolist())


def check_audio(path):
    r=subprocess.run(['ffmpeg','-hide_banner','-nostats','-i',str(path),'-vn','-af','ebur128=peak=true','-f','null','-'],
                     capture_output=True,text=True).stderr
    tail=r[r.rfind('Summary:'):]
    grab=lambda key:float(re.search(key+r':\s+(-?[\d.]+)',tail).group(1))
    x,sr=audio_mono(path)
    win=lambda a,b:round(float(20*np.log10(np.sqrt(np.mean(x[int(a*sr):int(b*sr)].astype(np.float64)**2))+1e-12)),1)
    end=len(x)/sr
    return dict(integrated_lufs=grab('I'),loudness_range_lu=grab('LRA'),true_peak_dbtp=grab('Peak'),
                duration=round(end,3),first_second_dbfs=win(0,1),
                last_two_seconds_per_100ms=[win(end-2+i*.1,end-1.9+i*.1) for i in range(20)],
                median_level_dbfs=float(np.median([win(s,s+1) for s in range(int(end)-1)])))


def check_encode(path):
    info,v=probe(path)
    head=open(path,'rb').read(4_000_000)
    m=re.search(rb'x264 - core [^\x00]*',head)
    opts=m.group(0).decode('ascii','replace') if m else ''
    pick=lambda k:(re.search(rf'\b{k}=([^ ]+)',opts) or [None,None])[1]
    return dict(codec=v['codec_name'],size=f"{v['width']}x{v['height']}",fps=v['avg_frame_rate'],
                video_kbps=round(int(v.get('bit_rate',0))/1000),rate_control=pick('rc'),bitrate=pick('bitrate'),
                crf=pick('crf'),duration=float(info['format']['duration']))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('check',choices=('all','cuts','rolls','dancer','audio','encode'))
    p.add_argument('media',type=Path);p.add_argument('--json',type=Path)
    p.add_argument('--figure-min',type=float,default=160);p.add_argument('--figure-sat',type=float,default=.32)
    a=p.parse_args();report={}
    want=lambda c:a.check in ('all',c)
    expected=score_cut_beats()
    scan=scan_video(a.media,a.figure_min,a.figure_sat) if want('cuts') or want('dancer') or (want('rolls') and expected is None) else None
    if want('cuts'):report['cuts']=check_cuts(scan,expected)
    if want('rolls'):
        beats=expected if expected is not None else check_cuts(scan)['cut_beats']
        report['rolls']=check_rolls(a.media,beats)
    if want('dancer'):report['dancer']=check_dancer(scan)
    if want('audio'):report['audio']=check_audio(a.media)
    if want('encode'):report['encode']=check_encode(a.media)
    for name,r in report.items():
        print(f'[{name}]')
        for k,v in r.items():
            if isinstance(v,list) and len(v)>24:v=f'{len(v)} values (see --json)'
            print(f'  {k}: {v}')
    if a.json:a.json.write_text(json.dumps(report,indent=1)+'\n')


if __name__=='__main__':main()
