"""Render Tensor Junglist: a full-track, percussion-led demoscene edit.

python docs/tensor_junglist_video.py /path/to/track.flac -o tensor.mp4
--start 112 --seconds 16 renders an excerpt with correctly trimmed audio.
--stills writes a contact sheet without rendering the whole video.
The supplied recording stays local and is not included in the repository.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "patches"))
from hydrapy import save_mp4
from hydrapy.analysis import track_beats
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.drawing import find_font
from hydrapy.glsl import VERTEX_DESKTOP
from tensor_junglist import make_scene, SCENES


def analyse(path, cache):
    """Bounded FFT batches; a cached score is tied to the recording's bytes."""
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["sha256"]) == digest:
                return {k:saved[k] for k in saved.files}
    sr,hop,nfft=22050,220,1024
    raw=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-vn","-ac","1",
                        "-ar",str(sr),"-f","f32le","-"],capture_output=True,check=True).stdout
    x=np.frombuffer(raw,np.float32)
    duration=len(x)/sr
    frames=np.lib.stride_tricks.sliding_window_view(np.pad(x,(nfft//2,nfft//2)),nfft)[::hop]
    freq=np.fft.rfftfreq(nfft,1/sr)
    energy,flux,prev=[],[],None
    for start in range(0,len(frames),512):
        mag=np.abs(np.fft.rfft(frames[start:start+512]*np.hanning(nfft),axis=1))
        log=np.log1p(mag*15)
        flux.extend(np.maximum(np.diff(log,axis=0,prepend=log[:1] if prev is None else prev),0).mean(axis=1))
        prev=log[-1:]
        energy.extend(np.stack([np.sqrt((mag[:,(freq>=lo)&(freq<hi)]**2).mean(axis=1))
                               for lo,hi in ((35,180),(180,2000),(2000,10000))],axis=1))
    rate=sr/hop
    energy,flux=np.asarray(energy),np.asarray(flux)
    onset=np.maximum(flux-np.convolve(flux,np.ones(101)/101,mode="same"),0)
    onset/=max(np.percentile(onset,98),1e-6)
    e=onset-onset.mean()
    n=1 << (2*len(e)-1).bit_length()
    ac=np.fft.irfft(np.abs(np.fft.rfft(e,n))**2,n)[:len(e)]
    lo,hi=int(rate*60/210),int(rate*60/110)
    lag=int(np.argmax(ac[lo:hi]))+lo
    a,b,c=ac[lag-1:lag+2]
    lag=lag+float(np.clip(.5*(a-c)/(a-2*b+c),-.5,.5))
    tempo=60*rate/lag
    beats=track_beats(onset,rate,tempo,tightness=90)/rate+.2*nfft/sr
    pulse=np.empty_like(onset)
    previous=0.
    for i,v in enumerate(onset):
        previous=max(v,previous*math.exp(-1/rate/.065))
        pulse[i]=previous
    energy=np.clip((energy / np.maximum(np.percentile(energy,92,axis=0),1e-6)-.18)/.82,0,1.3)
    data=dict(sha256=digest,energy=energy.astype("f4"),hit=np.clip(pulse,0,1.4).astype("f4"),
              onset=onset.astype("f4"),beats=beats,rate=rate,tempo=tempo,duration=duration)
    np.savez_compressed(cache,**data)
    return data


class Score:
    def __init__(self,data):
        self.data=data
        self.duration=float(data["duration"])
        self.tempo=float(data["tempo"])
        self.rate=float(data["rate"])
        self.beats=np.asarray(data["beats"])
        self.period=60/self.tempo
        self.sections=np.array([0,28,56,82,112,154,172,196,224,244,278,304,336,363,379.13044])*self.duration/379.13044
        pools=[(0,6,3),(2,0,4),(3,1,0),(1,8,3),(4,2,5),(6,0,7),(5,3,2),
               (4,0,1),(6,3,7),(1,5,4),(2,0,8),(4,1,3),(8,0,5),(6,3,0)]
        rng=np.random.default_rng(6137)
        self.cuts=[]
        patterns=[[8,4,4,4,2,2,4,4], [4,2,2,4,1,1,2,4,2,2,1,1,4,2],
                  [4,2,1,1,2,2,1,1,4,2,1,1,2,2,.5,.5,1,.5,.5,1]]
        position=0.
        previous=-1
        while True:
            t=max(0.,self.beat_time(position))
            if t>=self.duration: break
            section=min(len(pools)-1,int(np.searchsorted(self.sections,t,side="right")-1))
            level=0 if t<16 or section in (5,13) else (2 if section in (4,7,9,11,12) else 1)
            count=len(self.cuts)
            duration=patterns[level][count % len(patterns[level])]
            # A sustained geometry shot at some phrase starts gives the fast
            # inserts contrast without changing or stretching the recording.
            if count%29==20 and level: duration=8
            pool=pools[section]
            scene=pool[(count+section)%len(pool)]
            if scene==previous: scene=pool[(count+section+1)%len(pool)]
            self.cuts.append(dict(start=t,scene=int(scene),section=section,
                                  variant=int(rng.integers(0,7)),beats=duration))
            previous=scene
            position+=duration
        self.starts=np.array([c["start"] for c in self.cuts])
        self.controls=dict(bass=0.,mid=0.,high=0.,hit=0.,beat=0.,turn=0.)

    def beat_time(self,b):
        i=int(math.floor(b));f=b-i
        if i<0: return self.beats[0]+b*self.period
        if i>=len(self.beats)-1: return self.beats[-1]+(b-len(self.beats)+1)*self.period
        return float(self.beats[i]*(1-f)+self.beats[i+1]*f)

    def update(self,t):
        i=int(np.clip(np.searchsorted(self.starts,t,side="right")-1,0,len(self.cuts)-1))
        cut=self.cuts[i]
        j=int(np.clip(t*self.rate,0,len(self.data["energy"])-1))
        b=int(np.searchsorted(self.beats,t,side="right")-1)
        phase=(t-self.beat_time(b))/max(self.beat_time(b+1)-self.beat_time(b),.1)
        bass,mid,high=self.data["energy"][j]
        self.controls.update(bass=float(bass),mid=float(mid),high=float(high),
                             hit=float(self.data["hit"][j]),beat=float(np.exp(-max(0,phase)*6)),
                             turn=float(cut["variant"]))
        age=t-cut["start"]
        burst=(.15 if b%8<6 else .65)*max(0,self.controls["hit"]-.35)
        glitch=min(1.,burst+(max(0,1-age/.085)*.45 if i else 0))
        return cut,i,glitch


POST = """#version 330 core
uniform sampler2D scene, previous, title;
uniform vec2 resolution;
uniform float time, bass, hit, beat, glitch, cut, variant, palette, fade, titleMix;
out vec4 colour;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
void main(){
    vec2 uv=gl_FragCoord.xy/resolution;
    vec2 p=uv;
    float frame=floor(time*30.);
    float row=floor(p.y*27.);
    float tear=step(.73,hash(vec2(row,floor(frame/2.))))*glitch;
    p.x+=tear*(hash(vec2(row+4.,cut))-.5)*.32;
    p=(p-.5)/(1.0+beat*.025)+.5;
    if(variant>4.5 && variant<5.5){p=abs(p-.5)*2.;}
    if(variant>5.5){p=fract(p*vec2(2.,2.));}
    float aberration=.0006+glitch*.014;
    vec4 centre=texture(scene,fract(p));
    vec3 rgb=vec3(texture(scene,fract(p+vec2(aberration,0))).r,
                  centre.g,texture(scene,fract(p-vec2(aberration,0))).b)*centre.a;
    // The tiny chromatic split is persistent; large tearing and block holds
    // follow transients. Avoid full-screen white flashes.
    vec2 block=floor(uv*vec2(26.,18.));
    float keep=step(.67,hash(block+cut))*glitch*.8;
    rgb=mix(rgb,texture(previous,uv+vec2(.006,0)).rgb,keep);
    if(palette>.5 && palette<1.5) rgb=mix(rgb,rgb.brg,.7);
    if(palette>1.5) rgb=mix(rgb,rgb.gbr,.78);
    float scan=.97+.03*sin(gl_FragCoord.y*3.14159);
    float vignette=1.-.28*dot((uv-.5)*vec2(1.,1.4),(uv-.5)*vec2(1.,1.4));
    rgb=pow(max(rgb,0.),vec3(.88))*scan*vignette;
    // Fine crosshairs and sparse edge marks establish a stable frame of reference.
    vec2 q=abs(uv-.5);
    float vertical=(1.-smoothstep(1./resolution.x,2./resolution.x,abs(q.x-.46)))
                    *step(.38,q.y)*(1.-step(.43,q.y));
    float horizontal=(1.-smoothstep(1./resolution.y,2./resolution.y,abs(q.y-.43)))
                      *step(.4,q.x)*(1.-step(.46,q.x));
    rgb+=vec3(.2,.35,.28)*clamp(vertical+horizontal,0.,1.);
    float grain=(hash(gl_FragCoord.xy+frame)-.5)*.012;
    rgb+=grain;
    rgb*=1.-titleMix*.72*(1.-smoothstep(.55,.85,uv.x));
    vec4 text=texture(title,uv);
    rgb=mix(rgb,text.rgb,text.a*titleMix);
    colour=vec4(clamp(rgb*fade,0.,1.),1.);
}
"""


class Post:
    def __init__(self,ctx,size):
        self.ctx,self.size=ctx,size
        self.program=ctx.program(vertex_shader=VERTEX_DESKTOP,fragment_shader=POST)
        self.vbo=ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype="f4").tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,"2f","position")])
        self.targets=[]
        for _ in range(2):
            tex=ctx.texture(size,3,dtype="f1");tex.filter=(ctx.LINEAR,ctx.LINEAR)
            fbo=ctx.framebuffer(color_attachments=[tex]);fbo.use();ctx.clear(0,0,0,1)
            self.targets.append((tex,fbo))
        self.current=0
        im=Image.new("RGBA",size,(0,0,0,0));d=ImageDraw.Draw(im)
        h=size[1];x=int(size[0]*.075);y=int(h*.28)
        mono=find_font("DejaVuSansMono",round(h*.025))
        large=find_font("DejaVuSans-Bold",round(h*.105))
        d.text((x,y-round(h*.07)),"A N C H O R   R I T E",font=mono,fill=(184,224,201,255))
        d.text((x,y),"TENSOR",font=large,fill=(228,237,222,255))
        d.text((x,y+round(h*.12)),"JUNGLIST",font=large,fill=(183,239,111,255))
        d.line((x,y+round(h*.27),x+round(size[0]*.35),y+round(h*.27)),fill=(162,226,128,255),width=max(1,round(h*.003)))
        d.text((x,y+round(h*.3)),"PINNED     /     06",font=mono,fill=(196,215,210,255))
        self.title=ctx.texture(size,4,np.asarray(im)[::-1].copy().tobytes())

    def read(self,renderer,t,score,cut,index,glitch):
        dest=1-self.current
        tex,fbo=self.targets[dest];fbo.use();self.ctx.viewport=(0,0,*self.size)
        renderer.displayed_texture().use(0)
        self.targets[self.current][0].use(1);self.title.use(2)
        smooth=lambda x: (lambda v:v*v*(3-2*v))(float(np.clip(x,0,1)))
        title=smooth(t/.6)*smooth((8-t)/2) if t<8 else smooth((t-score.duration+6)/2)
        values=dict(scene=0,previous=1,title=2,resolution=tuple(map(float,self.size)),time=t,
                    **{k:score.controls[k] for k in ("bass","hit","beat")},
                    glitch=glitch,cut=float(index),variant=float(cut["variant"]),
                    palette=float((cut["section"]//2)%3),
                    fade=smooth(t/.5)*smooth((score.duration-t)/2.2),titleMix=title)
        for name,value in values.items():
            if name in self.program: self.program[name].value=value
        self.vao.render(self.ctx.TRIANGLES)
        self.current=dest
        return np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(self.size[1],self.size[0],3)[::-1].copy()

    def release(self):
        for tex,fbo in self.targets: fbo.release();tex.release()
        self.vao.release();self.vbo.release();self.program.release();self.title.release()


def frames(score,size,fps,start=0,seconds=None,stills=None):
    ctx=_create_context()
    print("Renderer:",ctx.info.get("GL_RENDERER"),flush=True)
    post=Post(ctx,size)
    renderers={}
    length=min(score.duration-start,seconds if seconds is not None else score.duration)
    if stills:
        times=[12,41,67,91,122,161,188,214,253,286,324,356]
        times=[t*score.duration/379.13044 for t in times]
    else:
        times=start+np.arange(int(math.ceil(length*fps)))/fps
    snapshots=[];wall=time.monotonic();last_report=wall
    try:
        for i,t in enumerate(times):
            cut,index,glitch=score.update(float(t));scene=cut["scene"]
            if scene not in renderers:
                renderers[scene]=GLRenderer(make_scene(scene,score.controls,size),size=size,ctx=ctx,fps=fps)
                if scene==7:
                    for k in range(8):
                        renderers[scene].step(max(0,float(t)-(8-k)/fps))
            r=renderers[scene]
            r.step(float(t))
            frame=post.read(r,float(t),score,cut,index,glitch)
            if stills:
                snapshots.append((float(t),scene,Image.fromarray(frame)))
            else:
                now=time.monotonic()
                if now-last_report>15:
                    print(f"{t:.1f}s / {score.duration:.1f}s | {(i+1)/(now-wall):.1f} fps | {SCENES[scene]}",flush=True)
                    last_report=now
                yield frame
        if stills:
            thumb=(480,270);sheet=Image.new("RGB",(1440,4*300),(7,12,17));d=ImageDraw.Draw(sheet)
            font=find_font("DejaVuSansMono",14)
            for i,(t,scene,im) in enumerate(snapshots):
                x,y=(i%3)*480,(i//3)*300
                sheet.paste(im.resize(thumb),(x,y));d.text((x+12,y+278),f"{int(t)//60:02d}:{int(t)%60:02d}  {SCENES[scene]}",font=font,fill=(181,206,194))
            sheet.save(stills)
    finally:
        for r in renderers.values(): r.release()
        post.release();ctx.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio",type=Path)
    parser.add_argument("-o","--out",type=Path,default=Path("Tensor_Junglist.mp4"))
    parser.add_argument("--size",default="1920x1080")
    parser.add_argument("--fps",type=float,default=30)
    parser.add_argument("--start",type=float,default=0)
    parser.add_argument("--seconds",type=float)
    parser.add_argument("--stills",type=Path)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--encoder",default="auto")
    parser.add_argument("--bitrate",default="6M")
    args=parser.parse_args()
    size=tuple(map(int,args.size.lower().split("x")))
    if len(size)!=2 or min(size)<64 or any(n%2 for n in size): parser.error("use even WIDTHxHEIGHT, at least 64x64")
    if not math.isfinite(args.fps) or args.fps<=0 or not math.isfinite(args.start) or args.start<0: parser.error("invalid fps/start")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0): parser.error("seconds must be positive")
    cache=args.cache or args.out.with_suffix(".analysis.npz")
    score=Score(analyse(args.audio,cache))
    if args.start>=score.duration: parser.error("start is beyond the track")
    print(f"{score.duration:.3f} seconds; estimated pulse {score.tempo:.2f} BPM; {len(score.cuts)} cuts",flush=True)
    if args.stills:
        list(frames(score,size,args.fps,stills=args.stills))
    else:
        with tempfile.TemporaryDirectory(prefix="tensor-render-",dir=args.out.parent) as tmp:
            audio=args.audio
            if args.start or args.seconds:
                audio=Path(tmp)/"excerpt.wav"
                length=min(args.seconds or score.duration,score.duration-args.start)
                subprocess.run(["ffmpeg","-v","error","-ss",str(args.start),"-i",str(args.audio),
                                "-t",str(length),"-vn","-c:a","pcm_s16le",str(audio)],check=True)
            video=Path(tmp)/"video.mp4"
            save_mp4(frames(score,size,args.fps,args.start,args.seconds),video,fps=args.fps,
                     encoder=args.encoder,bitrate=args.bitrate,
                     on_encoder=lambda e:print("Encoder:",e.name,flush=True))
            muxed=Path(tmp)/"complete.mp4"
            subprocess.run(["ffmpeg","-v","error","-i",str(video),"-i",str(audio),
                            "-map","0:v:0","-map","1:a:0","-map_metadata","1",
                            "-c:v","copy","-c:a","aac","-b:a","256k","-shortest",
                            "-movflags","+faststart",str(muxed)],check=True)
            muxed.replace(args.out)
        print(args.out,flush=True)
    args.out.with_suffix(".cues.json").write_text(json.dumps(dict(
        track=args.audio.name,sha256=str(score.data["sha256"]),duration=score.duration,
        estimated_bpm=score.tempo,size=size,fps=args.fps,cuts=[{**c,"name":SCENES[c["scene"]]} for c in score.cuts]),indent=2)+"\n")


if __name__=="__main__": main()
