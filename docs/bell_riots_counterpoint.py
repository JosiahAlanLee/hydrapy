"""Fourth Bell Riots edit: upper-register counterpoint and an authored return.

Requires the mixed closing edit and upper-controls.npz from bell_riots_mix.py.
"""
from __future__ import annotations

import argparse
import colorsys
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
from hydrapy.glsl import VERTEX_DESKTOP
from hydrapy.backend_gl import GLRenderer
from bell_riots_sparse import SparseRenderer
from bell_riots_kicks import KickClock
from bell_riots_video import smooth

PAIRS=('TRANSIT / SIGNAL','PAINT / MOIRE','GLYPHS / PHYSARUM','REGISTERS / DRIFTLESS','UNFOLDING BRIDGE')


class CounterpointScore:
    def __init__(self,controls,kicks,strength):
        self.controls=controls;self.duration=float(controls['duration']);self.start=0.
        self.clock=KickClock(kicks,strength,duration=self.duration)
        self.period=60/137.;self.beat0=-.008301162003440554
        self.cues=[];i=0
        # Establish the distant signal, map the network, dissolve it in pigment,
        # cross the interference, then return down the same tunnel.
        acts=((0,64,(0,1,0,3),(16,16,16,16)),
              (64,224,(2,3,2,1,0),(16,8,8,16,16)),
              (224,416,(1,1,2,1,3),(32,16,16,32,16)),
              (416,656,(2,0,3,1,2,3,0,1),(8,8,4,4,8,4,8,16)),
              (656,776,(1,2,1,3,0),(32,16,16,16,40)),
              (776,826,(0,),(50,)))
        for act,(first,last,modes,lengths) in enumerate(acts):
            beat=first;j=0
            if act:
                self.add(beat,beat+2,4,act,i);i+=1;beat+=2
            while beat<last:
                end=min(last,beat+lengths[j%len(lengths)])
                bridge=1 if i%7==6 and end-beat>=8 and act<5 else 0
                self.add(beat,end-bridge,modes[j%len(modes)],act,i);i+=1
                if bridge:self.add(end-1,end,4,act,i);i+=1
                beat=end;j+=1
        self.starts=np.array([c['start'] for c in self.cues])

    def add(self,first,last,mode,act,variant):
        start=max(0.,self.beat0+first*self.period);end=min(self.duration,self.beat0+last*self.period)
        if end>start:self.cues.append(dict(start=start,end=end,mode=mode,act=act,variant=variant))

    def at(self,t):
        i=max(0,min(len(self.cues)-1,int(np.searchsorted(self.starts,t,side='right')-1)))
        cue=self.cues[i]
        j=min(int(t*float(self.controls['rate'])),len(self.controls['hat'])-1)
        hits={name:float(np.max(self.controls[name][max(0,j-6):j+1])) for name in ('hat','cow','stab','high')}
        return dict(**cue,**hits,kick=self.clock.pulse(t),title=smooth(t/.5)*smooth((5.-t)/1.2),
                    progress=(t-cue['start'])/(cue['end']-cue['start']))

    def visible_layers(self,t):
        s=self.at(t)
        if s['title']>0:return ('transit','title')
        if s['mode']==4:return ('unfolding',)
        return tuple(PAIRS[s['mode']].split(' / '))


FRAGMENT=r"""#version 330 core
uniform sampler2D cells,mould,toxi,glyphs,paintTex,drawingTex,title;
uniform vec2 resolution;
uniform vec3 cool,warm;
uniform float time,kick,hat,cow,stab,high,fade,titleMix,progress,paintLow,paintHigh;
uniform int mode,variant,act;
out vec4 colour;
const float TAU=6.28318530718;
float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
vec4 pixelClock(vec2 _st,float rate,float spread,float density,float pixel,int geometry,float phase,float threshold){
__PIXEL_CLOCK__
}
float clockField(vec2 uv){
  vec2 p=(uv-.5)*rot(.035*sin(time*.021))+.5;
  return pixelClock(p,.031,.82,96.+float(variant%4)*9.,3.,3,.137+.012*stab,0.).r;
}
float glyphField(vec2 p){
  vec2 g=vec2(80.,45.),cell=floor(p*g),within=fract(p*g);
  float value=texture(toxi,fract((cell+.5)/g+vec2(time*.002,0.))).r;
  return texture(glyphs,vec2((floor(clamp(value,0.,.999)*15.)+within.x)/16.,within.y)).r;
}
float drawing(vec2 q){
  float n=6.+float(variant%4)*2.,a=atan(q.y,q.x)+time*.023;
  a=abs(mod(a+TAU/n*.5,TAU/n)-TAU/n*.5);
  vec2 p=vec2(cos(a),sin(a))*length(q)*rot(-time*.017)+.5;
  return clamp(texture(drawingTex,fract(p+vec2(.12*sin(time*.031),.09*cos(time*.027)))).r*2.,0.,1.);
}
vec3 unfold(vec2 p){
  vec2 grid=vec2(320.,180.);p=(p-.5)/(1.+.018*kick)+.5;
  vec3 c=texture(cells,(floor(p*grid)+.5)/grid).rgb;
  vec2 f=abs(fract(p*grid)-.5)*2.;float gap=1.-smoothstep(.84,.99,max(f.x,f.y));
  return (c.r*vec3(.94,.94,.87)+c.g*cool*.10)*gap;
}
void main(){
  vec2 uv=gl_FragCoord.xy/resolution;
  vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);
  vec3 bg=vec3(.008,.012,.020),rgb=bg;
  if(mode==0){
    // A continuous perspective tunnel; the small monochrome signal is at its far end.
    vec2 p=q*rot(.035*sin(time*.049));float r=max(length(p),.025);
    float a=atan(p.y,p.x)/TAU,z=.23/r+time*.38;
    vec2 g=abs(fract(vec2(a*18.,z*4.))-.5);
    float lines=1.-smoothstep(.012,.030+.006*hat,min(g.x,g.y));
    float grid=lines*smoothstep(.115,.19,r);
    float packets=step(.69,texture(toxi,fract(vec2(a*2.,z*.07))).r);
    float gates=step(.7,fract(z*.5))*step(.35,hat);
    rgb+=cool*grid*(.50+.30*hat)+warm*grid*gates*.28;
    rgb+=cool*packets*.065*smoothstep(.18,.28,r);
    float radius=.115*(1.+.065*kick);
    if(act==5)radius*=1.+.20*sin(progress*3.14159)-.60*smoothstep(.82,1.,progress);
    float aperture=1.-smoothstep(radius-.002,radius+.002,r);
    if(titleMix==0.){
      vec2 local=.5+q/radius*.24;
      vec3 signal=unfold(local);
      // A black disc keeps tunnel lines from running through the far-end signal.
      rgb=mix(rgb,signal*(.90+.10*kick),aperture);
    }
  }else if(mode==1){
    // Two layers: the continuously evolving moire and one physical paint texture.
    vec2 p=(uv-.5)*rot(.08*sin(time*.017))+.5;
    vec3 pigment=texture(paintTex,fract(p*vec2(1.24,.94)+vec2(time*.006,-time*.004))).rgb;
    float ink=clamp(1.-lum(pigment),0.,1.);
    float shape=smoothstep(paintLow,paintHigh,ink);
    float moire=clockField(uv);
    float window=1.-smoothstep(.62,.94,length(q*vec2(.85,1.15)));
    rgb+=cool*moire*(.30+.09*hat)*window;
    vec3 dye=mix(warm*.55,cool*.58,clamp((pigment.b-pigment.r)*2.+.45,0.,1.));
    float grain=abs(ink-(1.-lum(texture(paintTex,fract(p*vec2(1.24,.94)+vec2(time*.006+.008,-time*.004))).rgb)));
    dye+=warm*grain*(.9+.65*stab);
    float opacity=shape*(.56+.22*sin(time*.071))*(.88+.12*stab);
    rgb=mix(rgb,dye,opacity*window);
    // Leave moire visible through the thin pigment, and around its boundaries.
    rgb+=warm*moire*grain*.25;
  }else if(mode==2){
    // A network becomes a register field: one Physarum material and one glyph plane.
    vec2 p=vec2(abs(uv.x-.5)*1.65,uv.y)+vec2(time*.002,-time*.0014);
    float network=pow(clamp(texture(mould,fract(p)).r*2.6,0.,1.),1.15);
    float space=1.-smoothstep(.60,1.04,length(q));
    rgb+=cool*network*(.44+.13*cow)*space;
    vec2 mirror=vec2(abs(uv.x-.5)*2.,abs(uv.y-.5)*2.);
    mirror.x+=step(.88,h(vec2(floor(uv.y*28.),float(variant))))*hat*.035;
    float glyph=glyphField(mirror);
    vec2 block=floor(mirror*vec2(12.,8.));
    float selected=step(.58-.08*hat,h(block+float(variant)));
    float lanes=1.-smoothstep(.34,.48,abs(uv.y-.5));
    rgb=mix(rgb,mix(warm,vec3(.88,.91,.84),.65),glyph*selected*lanes*(.72+.15*hat));
  }else if(mode==3){
    ivec2 p=ivec2(floor(abs(uv-.5)*resolution/4.));
    p.x+=int(floor(time*4.))%128+int(cow*3.);
    int bits=((p.x&p.y)^((p.x>>2)*(p.y>>2))^((p.x+p.y)>>3));
    float value=float((bits>>(variant%4))&1);
    float keep=step(.52,h(floor(uv*vec2(18.,12.))+float(variant)));
    rgb+=cool*value*keep*(.20+.10*hat);
    float lines=drawing(q);
    float space=1.-smoothstep(.62,.96,length(q));
    rgb=mix(rgb,warm*(.66+.16*stab),lines*.82*space);
  }else{
    rgb=bg+unfold(uv)*(.84+.10*kick);
  }
  rgb*=1.-.22*smoothstep(.48,1.05,length(q));
  vec4 label=texture(title,uv);rgb=mix(rgb,label.rgb,label.a*titleMix);
  colour=vec4(clamp(rgb*fade,0.,1.),1.);
}
"""


class CounterpointRenderer(SparseRenderer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.network.release()
        patch=H.Patch(size=(384,216))
        # A slowly changing nutrient field keeps the network from coarsening
        # into a single broad vein during a six-minute continuous simulation.
        H.noise(11,.035).thresh(.18,.24).out(patch.o1,patch=patch)
        self.mould=H.Physarum(size=patch.size,agents=(128,128),seed=29,colour=(1,1,1),
            sensor_angle=H.Lfo(1/43.7,lo=.85,hi=1.35),sensor_distance=H.Lfo(1/57.1,lo=6,hi=11),
            turn=H.Lfo(1/37.3,lo=.70,hi=1.05),step=1.5,deposit=.035,decay=.94,diffuse=.65,
            exposure=1.1,speed=30,max_steps=2,food=patch.o1,food_weight=.8)
        H.src(self.mould).out(patch.o0,patch=patch);patch.render(patch.o0)
        self.network=GLRenderer(patch,size=patch.size,ctx=self.ctx,fps=15)
        for i in range(180):self.network.step(i/15)
        self.paint_windows=[]
        for tex in self.paint:
            a=np.frombuffer(tex.read(),np.float16).reshape(512,512,4).astype('f4')
            ink=1.-a[:,:,:3]@np.array([.2126,.7152,.0722],dtype='f4')
            self.paint_windows.append(tuple(map(float,np.percentile(ink,[62,88]))))
        self.vao.release();self.program.release()
        self.program=self.ctx.program(vertex_shader=VERTEX_DESKTOP,
            fragment_shader=FRAGMENT.replace('__PIXEL_CLOCK__',H.REGISTRY['pixelOsc'].glsl))
        self.vao=self.ctx.vertex_array(self.program,[(self.vbo,'2f','position')])

    def draw(self,t,read=True):
        self.advance(t);s=self.score.at(t);ctx=self.ctx
        if s['mode'] in (0,2):
            tick=int(t*15)
            if tick!=self.glyph_tick:
                frame=self.cellular.frame_at(tick/15+4)
                self.toxi.write(np.ascontiguousarray(frame[::-1,:,0]).tobytes(),alignment=1);self.glyph_tick=tick
        tex=(self.field.displayed_texture(),self.mould.gl_texture(self.network),self.toxi,self.glyphs,
             self.paint[(s['variant']+s['act'])%6],self.drawings[s['variant']%2],self.title)
        self.fbo.use();ctx.viewport=(0,0,*self.size)
        for i,x in enumerate(tex):x.use(i)
        hue=.555+.075*math.sin(math.tau*t/251.)
        low,high=self.paint_windows[(s['variant']+s['act'])%6]
        values=dict(cells=0,mould=1,toxi=2,glyphs=3,paintTex=4,drawingTex=5,title=6,
                    resolution=tuple(map(float,self.size)),time=float(t),mode=s['mode'],variant=s['variant'],act=s['act'],
                    kick=s['kick'],hat=s['hat'],cow=s['cow'],stab=s['stab'],high=s['high'],progress=s['progress'],
                    paintLow=low,paintHigh=max(low+.001,high),
                    cool=colorsys.hsv_to_rgb(hue,.58,.78),warm=colorsys.hsv_to_rgb((hue+.5)%1,.60,.94),
                    titleMix=s['title'],fade=smooth(t/.4)*smooth((self.score.duration-t)/1.6))
        for name,value in values.items():
            if name in self.program:self.program[name].value=value
        self.vao.render(ctx.TRIANGLES)
        if read:return np.frombuffer(self.fbo.read(components=3,alignment=1),np.uint8).reshape(self.size[1],self.size[0],3)[::-1].copy()


def frames(score,size,fps,offset,length,report,art_root,cache,checkpoints=None):
    renderer=CounterpointRenderer(score,size,art_root,cache)
    report['renderer']=renderer.ctx.info['GL_RENDERER'];wall=last=time.monotonic()
    count=math.ceil(length*fps);dark=[]
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    try:
        renderer.advance(offset)
        for i in range(count):
            t=offset+i/fps;frame=renderer.draw(t)
            if i%round(fps*15)==0 or i==90:
                luma=frame.astype('f4')@np.array([.2126,.7152,.0722],dtype='f4')
                dark.append(dict(time=t,fraction=float(np.mean(luma<20.4))))
                if checkpoints:Image.fromarray(frame).save(checkpoints/f'{t:08.2f}.jpg',quality=94)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f'{t:.1f}/{offset+length:.1f}s | {(i+1)/(now-wall):.1f} fps | {PAIRS[score.at(t)["mode"]]}',flush=True);last=now
        report['render_fps']=count/(time.monotonic()-wall);report['dark_fraction_samples']=dark
    finally:renderer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mix',type=Path);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--kicks',type=Path,required=True);p.add_argument('--art-root',type=Path,required=True)
    p.add_argument('--assets-cache',type=Path,required=True);p.add_argument('-o','--out',type=Path,required=True)
    p.add_argument('--size',default='1920x1080');p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--offset',type=float,default=0.);p.add_argument('--seconds',type=float)
    p.add_argument('--bitrate',default='7M');p.add_argument('--encoder',default='auto');p.add_argument('--checkpoints',type=Path)
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
                max_visible_layers=2,upper_controls_sha256=hashlib.sha256(a.controls.read_bytes()).hexdigest())
    print(f'{score.duration:.6f}s | {len(score.cues)} shots | upper-register response',flush=True)
    with tempfile.TemporaryDirectory(prefix='bell-counterpoint-',dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(score,size,a.fps,offset,length,report,a.art_root,a.assets_cache,a.checkpoints),tmp/'video.mp4',
                   fps=a.fps,bitrate=a.bitrate,encoder=a.encoder,on_encoder=lambda e:print('Encoder:',e.name,flush=True))
        subprocess.run(['ffmpeg','-v','error','-i',str(tmp/'video.mp4'),'-ss',str(offset),'-t',str(length),'-i',str(a.mix),
                        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','320k','-movflags','+faststart',
                        '-metadata','title=Bell Riots Pt 2 — Counterpoint Edit',str(tmp/'complete.mp4')],check=True)
        (tmp/'complete.mp4').replace(a.out)
    a.out.with_suffix('.cues.json').write_text(json.dumps(report,indent=2)+'\n');print(a.out,flush=True)


if __name__=='__main__':main()
