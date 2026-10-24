"""Fifth Bell Riots pass: a half-time dancer through HydraPy's frame delay.

The delayed silhouette is a matte for the existing paint layer. Moire stays
behind it; all other fourth-pass scenes and the mixed audio keep their timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'docs'),str(ROOT/'patches')]
import hydrapy as H
from hydrapy.backend_gl import GLRenderer
from hydrapy.glsl import VERTEX_DESKTOP
from bell_riots_counterpoint import CounterpointRenderer, CounterpointScore, FRAGMENT, PAIRS
from bell_riots_video import smooth

TRACK_BPM=137.
HALF_BPM=TRACK_BPM/2
TICKS_PER_BEAT=15
TICKS_PER_CYCLE=480
DELAY_TICKS=(30,60,90)
DELAY_GAINS=(.60,.34,.18)
PRINT_PERIOD=1920/137


class DanceClock:
    """Measured kick anchors on the score's beat numbers, with gap interpolation.

    Missing kicks do not shorten the choreography or shift the beat count.
    Outside the measured interval the 137 BPM grid continues at constant speed.
    """
    def __init__(self,kicks,period=60/137,beat0=-.008301162003440554):
        self.period=float(period);self.beat0=float(beat0)
        times=np.asarray(kicks,dtype=float)
        if times.ndim!=1 or not np.all(np.isfinite(times)):
            raise ValueError('kick times must be a finite one-dimensional array')
        nearest={}
        for t in times:
            beat=int(round((t-beat0)/period))
            error=abs(t-(beat0+beat*period))
            if error<period*.25 and (beat not in nearest or error<nearest[beat][0]):
                nearest[beat]=(error,float(t))
        self.beats=np.array(sorted(nearest),dtype=float)
        self.times=np.array([nearest[int(b)][1] for b in self.beats],dtype=float)

    def beat_at(self,t):
        if not math.isfinite(t):raise ValueError('time must be finite')
        if not len(self.times):return (t-self.beat0)/self.period
        if t<self.times[0]:return self.beats[0]+(t-self.times[0])/self.period
        if t>self.times[-1]:return self.beats[-1]+(t-self.times[-1])/self.period
        return float(np.interp(t,self.times,self.beats))

    def tick_at(self,t):
        return math.floor(self.beat_at(t)*TICKS_PER_BEAT+1e-8)


def delay_patch(source,size):
    """Actual HydraPy output history, with one/two/three half-beat taps.

    Red carries today's matte, green its displaced echoes. These channels
    control the paint material in the final shader, not extra image layers.
    """
    p=H.Patch(size=size)
    H.src(source).scale(1.10).out(p.o0,patch=p)
    c=H.src(p.o0).color(1,0,0)
    for delay,gain,shift in zip(DELAY_TICKS,DELAY_GAINS,(.115,-.135,.245)):
        c=c.add(H.delay(p.o0,delay).scroll(shift,0).color(0,1,0),gain)
    c.out(p.o1,patch=p);p.render(p.o1)
    return p


class DelayedDancer:
    def __init__(self,matte,clock,ctx):
        self.matte=np.load(matte,mmap_mode='r') if isinstance(matte,(str,Path)) else np.asarray(matte)
        if self.matte.dtype!=np.uint8 or self.matte.ndim!=3 or min(self.matte.shape)<2:
            raise ValueError('matte must be uint8 [frame,height,width], with at least two frames')
        self.clock=clock;self.tick=0;self.last_tick=None;self.steps=0
        self.rgba=np.empty((*self.matte.shape[1:],4),dtype=np.uint8);self.rgba[:,:,3]=255
        source=H.FrameSource(self.sample)
        self.renderer=GLRenderer(delay_patch(source,(self.matte.shape[2],self.matte.shape[1])),
                                 ctx=ctx,fps=TRACK_BPM/4)

    def sample(self,_):
        at=(self.tick%TICKS_PER_CYCLE)*len(self.matte)/TICKS_PER_CYCLE
        i=math.floor(at);f=at-i
        if f<1e-8:
            gray=self.matte[i]
        else:
            gray=np.rint(self.matte[i].astype('f4')*(1-f)+self.matte[(i+1)%len(self.matte)]*f).astype('u1')
        self.rgba[:,:,:3]=gray[:,:,None]
        return self.rgba

    def advance(self,t):
        target=self.clock.tick_at(t)
        if self.last_tick is not None and target<self.last_tick:
            raise ValueError('dancer time must increase')
        # A bounded pre-roll reconstructs precisely the same history after an
        # excerpt seek or a hidden interval. No full-song replay is necessary.
        start=target-max(DELAY_TICKS) if self.last_tick is None else max(self.last_tick+1,target-max(DELAY_TICKS))
        for tick in range(start,target+1):
            self.tick=tick
            self.renderer.step(self.steps/(TRACK_BPM/4));self.steps+=1
        self.last_tick=target
        return self.renderer.displayed_texture()

    def release(self):self.renderer.release()


PAINT_MATTE=r"""
    // The dancer gates and lights the existing pigment; the moire remains
    // the other visible material. Echoes are delayed versions of this matte.
    vec2 tap=texture(dancerTap,uv).rg;
    float body=clamp(tap.r,0.,1.);
    float echo=clamp(tap.g-body*.55,0.,1.);
    vec3 moireOnly=bg+cool*moire*(.30+.09*hat)*window;
    rgb=mix(rgb,moireOnly,danceAmount*.48*(1.-body));
    vec3 chalk=mix(warm*.48,vec3(.94,.96,.90),.78);
    chalk=mix(chalk,dye,.15+.20*shape)+warm*grain*.18;
    vec3 ghost=mix(cool*.48,warm*.65,.30+.25*sin(time*.047));
    rgb=mix(rgb,ghost,echo*danceAmount*.72);
    rgb=mix(rgb,chalk,body*danceAmount*(.89+.07*stab));
"""
DANCER_FRAGMENT=FRAGMENT.replace('uniform sampler2D cells,',
    'uniform sampler2D dancerTap;\nuniform float danceAmount;\nuniform sampler2D cells,',1).replace(
    '  }else if(mode==2){',PAINT_MATTE+'\n  }else if(mode==2){',1)


class DancerRenderer(CounterpointRenderer):
    def __init__(self,*args,matte,**kwargs):
        super().__init__(*args,**kwargs)
        self.dance_clock=DanceClock(self.score.clock.times)
        self.dancer=DelayedDancer(matte,self.dance_clock,self.ctx)
        self.vao.release();self.program.release()
        self.program=self.ctx.program(vertex_shader=VERTEX_DESKTOP,
            fragment_shader=DANCER_FRAGMENT.replace('__PIXEL_CLOCK__',H.REGISTRY['pixelOsc'].glsl))
        self.vao=self.ctx.vertex_array(self.program,[(self.vbo,'2f','position')])

    def draw(self,t,read=True):
        self.advance(t);s=self.score.at(t)
        amount=0.
        if s['mode']==1:
            self.dancer.advance(t).use(7)
            amount=(.78,.88,.96,.88,.98,.90)[s['act']]
            amount*=smooth((t-s['start'])/.38)*smooth((s['end']-t)/.42)
        self.program['dancerTap'].value=7
        self.program['danceAmount'].value=amount
        return super().draw(t,read)

    def release(self):
        self.dancer.release();super().release()


def frames(score,size,fps,offset,length,report,art_root,cache,matte,checkpoints=None):
    r=DancerRenderer(score,size,art_root,cache,matte=matte)
    report['renderer']=r.ctx.info['GL_RENDERER'];wall=last=time.monotonic()
    count=math.ceil(length*fps);dark=[]
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    try:
        for i in range(count):
            t=offset+i/fps;frame=r.draw(t)
            if i%round(fps*15)==0 or i==90:
                luma=frame.astype('f4')@np.array([.2126,.7152,.0722],dtype='f4')
                dark.append(dict(time=t,fraction=float(np.mean(luma<20.4))))
                if checkpoints:Image.fromarray(frame).save(checkpoints/f'{t:08.2f}.jpg',quality=94)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f'{t:.1f}/{offset+length:.1f}s | {(i+1)/(now-wall):.1f} fps | {PAIRS[score.at(t)["mode"]]}',flush=True)
                last=now
        report['render_fps']=count/(time.monotonic()-wall);report['dark_fraction_samples']=dark
    finally:r.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mix',type=Path);p.add_argument('--matte',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--kicks',type=Path,required=True)
    p.add_argument('--art-root',type=Path,required=True);p.add_argument('--assets-cache',type=Path,required=True)
    p.add_argument('-o','--out',type=Path,required=True);p.add_argument('--size',default='1920x1080')
    p.add_argument('--fps',type=float,default=30.);p.add_argument('--offset',type=float,default=0.)
    p.add_argument('--seconds',type=float);p.add_argument('--bitrate',default='7M')
    p.add_argument('--encoder',default='auto');p.add_argument('--checkpoints',type=Path)
    a=p.parse_args();size=tuple(map(int,a.size.lower().split('x')))
    if len(size)!=2 or min(size)<96 or any(x%2 for x in size):p.error('Use even dimensions >=96')
    if not math.isfinite(a.fps) or a.fps<=0 or not math.isfinite(a.offset) or a.offset<0:p.error('Invalid fps/offset')
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0):p.error('Invalid duration')
    with np.load(a.controls) as z:controls={k:z[k] for k in z.files}
    with np.load(a.kicks) as z:score=CounterpointScore(controls,z['times'],z['strength'])
    offset=round(a.offset*a.fps)/a.fps
    if offset>=score.duration:p.error('Offset exceeds edit')
    length=min(a.seconds or score.duration,score.duration-offset)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(mix=a.mix.name,mix_sha256=hashlib.sha256(a.mix.read_bytes()).hexdigest(),duration=score.duration,
        offset=offset,seconds=length,source_start=float(controls['source_start']),size=size,fps=a.fps,cues=score.cues,
        max_visible_layers=2,matte_sha256=hashlib.sha256(a.matte.read_bytes()).hexdigest(),
        track_bpm=TRACK_BPM,half_bpm=HALF_BPM,full_beats_per_cycle=32,
        clock='measured kicks assigned to score beat numbers; missing beats interpolated',
        delay_operator='HydraPy delay(o0, frames)',delay_ticks=DELAY_TICKS,
        ticks_per_half_beat=30,delay_half_beats=[1,2,3],delay_gains=DELAY_GAINS,
        paint_matte_seconds=sum(c['end']-c['start'] for c in score.cues if c['mode']==1))
    print(f'{score.duration:.6f}s | dancer at {HALF_BPM:g} BPM half time | HydraPy delay',flush=True)
    with tempfile.TemporaryDirectory(prefix='bell-dancer-',dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(score,size,a.fps,offset,length,report,a.art_root,a.assets_cache,a.matte,a.checkpoints),
            tmp/'video.mp4',fps=a.fps,bitrate=a.bitrate,encoder=a.encoder,
            on_encoder=lambda e:print('Encoder:',e.name,flush=True))
        subprocess.run(['ffmpeg','-v','error','-i',str(tmp/'video.mp4'),'-ss',str(offset),'-t',str(length),'-i',str(a.mix),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','320k','-movflags','+faststart',
            '-metadata','title=Bell Riots Pt 2 — Half-Time Dancer Edit',str(tmp/'complete.mp4')],check=True)
        (tmp/'complete.mp4').replace(a.out)
    a.out.with_suffix('.cues.json').write_text(json.dumps(report,indent=2)+'\n');print(a.out,flush=True)


if __name__=='__main__':main()
