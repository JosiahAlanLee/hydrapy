"""Bell Riots: kick-led Unfolding, negative space, at most two visible layers.

python docs/bell_riots_sparse.py track.flac --art-root /path/to/Generative_Art -o Bell_Riots_Sparse.mp4
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
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"docs"),str(ROOT/"patches")]
import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.drawing import find_font
from hydrapy.glsl import VERTEX_DESKTOP
from bell_riots_video import smooth
from bell_riots_organic import OrganicRenderer, DRAWINGS, phrase_start, raster_assets
from bell_riots_kicks import KickClock, analyse_kicks
from tensor_junglist_video import analyse
from unfolding_score import make_field, INNER, OUTER

ACCENTS={1:"GLYPH RELAY",2:"PIXEL CLOCKS",3:"BIT REGISTERS",4:"PHYSARUM",5:"DRIFTLESS LINES",6:"PAINT"}


class SparseScore:
    def __init__(self,data,kicks,strength,start=None):
        self.data=data;self.start=phrase_start(data) if start is None else float(start)
        self.duration=float(data["duration"])-self.start
        if not 0<=self.start<float(data["duration"]):raise ValueError("start must be within the recording")
        self.clock=KickClock(kicks,strength,duration=self.duration)
        # These accents are deliberately not cut to the beat. They withdraw fully
        # before the next source enters, with a base-only interval between them.
        modes=(1,5,4,3,6,2,5,1,6,4,2,3)
        lengths=(9.7,6.3,8.1,11.4,7.6,10.2)
        gaps=(4.3,6.7,5.1,7.9,4.9,6.1)
        fx=H.Fx(2024,lock=.72)
        self.cues=[];t=9.;i=0
        while t<self.duration-8:
            end=min(t+lengths[i%len(lengths)],self.duration-6.)
            self.cues.append(dict(start=t,end=end,mode=modes[i%len(modes)],variant=i,
                                  seed=int(fx.rand()*10000)))
            t=end+gaps[i%len(gaps)];i+=1
        self.starts=np.array([c["start"] for c in self.cues])

    def at(self,t):
        title=max(smooth(t/.5)*smooth((5.8-t)/1.5),smooth((t-self.duration+5.)/1.5))
        index=int(np.searchsorted(self.starts,t,side="right")-1)
        mode=variant=seed=0;amount=0.
        if index>=0 and title==0:
            c=self.cues[index]
            if t<c["end"]:
                amount=.68*smooth((t-c["start"])/1.8)*smooth((c["end"]-t)/2.2)
                mode,variant,seed=c["mode"],c["variant"],c["seed"]
        return dict(mode=mode,variant=variant,seed=seed,amount=amount,title=title,kick=self.clock.pulse(t))

    def visible_layers(self,t):
        s=self.at(t)
        return ("unfolding","title") if s["title"]>0 else (("unfolding",ACCENTS[s["mode"]]) if s["amount"]>0 else ("unfolding",))


FRAGMENT=r"""#version 330 core
uniform sampler2D cells, accentTex, glyphs, title;
uniform vec2 resolution;
uniform float time, kick, amount, fade, titleMix, maskSeed;
uniform int mode, variant;
uniform vec3 cool, warm;
out vec4 colour;
const float TAU=6.28318530718;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
vec4 pixelClock(vec2 _st,float rate,float spread,float density,float pixel,int geometry,float phase,float threshold){
__PIXEL_CLOCK__
}
float accent(vec2 uv){
  if(mode==1){
    vec2 p=vec2(abs(uv.x-.5)*2.,abs(uv.y-.5)*2.);
    vec2 cell=floor(p*vec2(96.,54.)),f=fract(p*vec2(96.,54.));
    float v=texture(accentTex,fract((cell+.5)/vec2(96.,54.)+vec2(time*.003,0))).r;
    return texture(glyphs,vec2((floor(clamp(v,0.,.999)*15.)+f.x)/16.,f.y)).r;
  }
  if(mode==2)return pixelClock(uv,.035,1.3,80.+float(variant%4)*11.,4.,3,.137,0.).r;
  if(mode==3){
    ivec2 p=ivec2(floor(abs(uv-.5)*resolution/4.));p.x+=int(floor(time*4.))%128;
    int bits=((p.x&p.y)^((p.x>>2)*(p.y>>2))^((p.x+p.y)>>3));
    return float((bits>>(variant%4))&1);
  }
  if(mode==4){
    vec2 p=vec2(abs(uv.x-.5)*1.7,uv.y)+vec2(time*.002,-time*.0014);
    return smoothstep(.12,.58,texture(accentTex,fract(p)).r);
  }
  if(mode==5){
    vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);
    float n=6.+2.*float(variant%4),a=atan(q.y,q.x)+time*.027;
    a=abs(mod(a+TAU/n*.5,TAU/n)-TAU/n*.5);
    vec2 p=vec2(cos(a),sin(a))*length(q)*rot(-time*.019)+.5;
    return clamp(texture(accentTex,fract(p+vec2(.12*sin(time*.031),.09*cos(time*.027)))).r*2.,0.,1.);
  }
  if(mode==6){
    vec2 p=uv*vec2(1.15,.93)+vec2(time*.008,-time*.005);
    float c=lum(texture(accentTex,fract(p)).rgb);
    float edge=abs(c-lum(texture(accentTex,fract(p+vec2(.012,.008))).rgb));
    return smoothstep(.025,.16,edge)*smoothstep(.12,.60,1.-c);
  }
  return 0.;
}
void main(){
  vec2 uv=gl_FragCoord.xy/resolution;
  vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);
  vec3 bg=vec3(.008,.011,.017);
  vec2 sampleUV=(uv-.5)/(1.+.018*kick)+.5;
  vec2 grid=vec2(320.,180.),cellUV=(floor(sampleUV*grid)+.5)/grid;
  vec3 cell=texture(cells,cellUV).rgb;
  vec3 paper=vec3(.93,.92,.82);
  vec3 ink=cell.g*cool*.12+cell.r*mix(paper,cool,cell.b*.68);
  vec2 tile=abs(fract(uv*grid)-.5)*2.;
  float gaps=1.-smoothstep(.82,.99,max(tile.x,tile.y));
  vec2 px=1./grid;
  float edge=max(max(abs(cell.r-texture(cells,cellUV+vec2(px.x,0)).r),
                     abs(cell.r-texture(cells,cellUV-vec2(px.x,0)).r)),
                 max(abs(cell.r-texture(cells,cellUV+vec2(0,px.y)).r),
                     abs(cell.r-texture(cells,cellUV-vec2(0,px.y)).r)));
  float core=1.-smoothstep(.18,.30,length(q));
  float contour=mix(smoothstep(.16,.75,edge)*.65,1.,core);
  // Open the interiors of large outer shapes; preserve the central rosette.
  // Negative space follows the cells rather than a geometric crop window.
  vec3 base=ink*gaps*contour*(.95+.10*kick);
  vec3 rgb=bg+base;
  if(mode>0 && amount>0.){
    vec2 tiles=vec2(16.,10.),block=floor(uv*tiles);
    block=min(block,tiles-1.-block);
    float selected=step(.82,hash(block+vec2(maskSeed,maskSeed*.371)));
    vec2 f=abs(fract(uv*tiles)-.5)*2.;
    float inset=(1.-smoothstep(.72,.96,f.x))*(1.-smoothstep(.68,.94,f.y));
    // Only sparse islands, mostly in the dark gaps; there is one accent sampler.
    float mask=selected*inset*(1.-smoothstep(.07,.27,lum(base)));
    if(mask>0.){
      float a=accent(uv)*amount*mask;
      rgb=mix(rgb,warm*.88,a);
    }
  }
  vec4 text=texture(title,uv);
  rgb=mix(rgb,text.rgb,text.a*titleMix);
  colour=vec4(clamp(rgb*fade,0.,1.),1.);
}
"""


class SparseRenderer:
    def __init__(self,score,size,art_root,cache,ctx=None):
        paths=raster_assets(cache,art_root)
        self.score,self.size=score,size
        self.owns_context=ctx is None;self.ctx=ctx or _create_context();ctx=self.ctx
        self.controls=dict(outer=8.,inner=16.,radius=.22,fold=0.,drop=0.,seed=0.)
        self.field=GLRenderer(make_field(self.controls),size=(320,180),ctx=ctx,fps=30)
        for i in range(96):
            self.controls.update(fold=float(i%32==0),drop=float(i==0),seed=(i//97)*.731)
            self.field.step(i/30)
        self.generation=-1;self.physics_tick=-1;self.glyph_tick=-1;self.last_time=-1.
        patch=H.Patch(size=(384,216))
        self.mould=H.Physarum(size=patch.size,agents=(128,128),seed=29,colour=(1,1,1),
                             sensor_angle=H.Lfo(1/33.1,lo=.25,hi=.87),
                             sensor_distance=H.Lfo(1/46.7,lo=8,hi=24),
                             turn=H.Lfo(1/33.1,lo=.22,hi=.54),deposit=.065,decay=.961,
                             diffuse=.72,exposure=1.4,speed=30,max_steps=2)
        H.src(self.mould).out(patch.o0,patch=patch);patch.render(patch.o0)
        self.network=GLRenderer(patch,size=patch.size,ctx=ctx,fps=15)
        for i in range(180):self.network.step(i/15)
        self.paint=OrganicRenderer.paint_bank(self)
        self.drawings=[]
        for path in paths:
            im=ImageOps.invert(Image.open(path).convert("L"));im.thumbnail((1900,1900),Image.Resampling.LANCZOS)
            canvas=Image.new("L",(2048,2048));canvas.paste(im,((2048-im.width)//2,(2048-im.height)//2))
            tex=ctx.texture(canvas.size,1,np.asarray(canvas)[::-1].copy().tobytes())
            tex.filter=(ctx.LINEAR,ctx.LINEAR);self.drawings.append(tex)
        self.cellular=H.Cellular(width=384,height=216,rate=34.25,seed=2024,stripes=24,density=.24,perturb=.0012)
        self.toxi=ctx.texture((384,216),1,dtype="f1");self.toxi.filter=(ctx.NEAREST,ctx.NEAREST)
        atlas=Image.new("L",(128,12));d=ImageDraw.Draw(atlas);font=find_font("DejaVuSansMono",11)
        for i,char in enumerate(" .:;=+*ox#%@/\\[]"):d.text((i*8,-1),char,font=font,fill=255)
        self.glyphs=ctx.texture(atlas.size,1,np.asarray(atlas)[::-1].copy().tobytes());self.glyphs.filter=(ctx.NEAREST,ctx.NEAREST)
        self.title=self.title_texture()
        self.program=ctx.program(vertex_shader=VERTEX_DESKTOP,
                                 fragment_shader=FRAGMENT.replace("__PIXEL_CLOCK__",H.REGISTRY["pixelOsc"].glsl))
        self.vbo=ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype="f4").tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,"2f","position")])
        self.texture=ctx.texture(size,3,dtype="f1");self.fbo=ctx.framebuffer(color_attachments=[self.texture])

    def title_texture(self):
        w,h=self.size;im=Image.new("RGBA",self.size);d=ImageDraw.Draw(im)
        small=find_font("DejaVuSansMono",max(10,round(h*.013)))
        large=find_font("DejaVuSans-Bold",max(16,round(h*.037)))
        x=round(w*.047);y=round(h*.43)
        d.text((x,y),"ANCHOR RITE",font=small,fill=(171,190,192,255))
        d.text((x,y+round(h*.038)),"BELL RIOTS",font=large,fill=(228,228,211,255))
        d.text((x,y+round(h*.089)),"PT. 2",font=small,fill=(208,146,91,255))
        return self.ctx.texture(self.size,4,np.asarray(im)[::-1].copy().tobytes())

    def advance(self,t):
        if t<self.last_time-1e-8:raise ValueError("time must increase")
        self.last_time=t;k=self.score.clock
        target=int(np.searchsorted(k.generation_times,t+1e-9,side="right")-1)
        while self.generation<target:
            self.generation+=1;i=self.generation;now=float(k.generation_times[i])
            self.controls.update(outer=OUTER[int(now/67)%len(OUTER)],inner=INNER[int(now/41)%len(INNER)],
                                 radius=.14+.16*(.5+.5*math.sin(math.tau*now/173)),
                                 fold=float(i in k.fold_generations),drop=float(i in k.drop_generations),
                                 seed=((96+i)//97)*.731)
            self.field.step((96+i)/30)
        target=int(math.floor(t*15+1e-8))
        while self.physics_tick<target:
            self.physics_tick+=1;self.network.step((180+self.physics_tick)/15)

    def draw(self,t,read=True):
        self.advance(t);s=self.score.at(t);ctx=self.ctx
        mode,variant=s["mode"],s["variant"]
        if mode==1:
            tick=int(t*15)
            if tick!=self.glyph_tick:
                im=self.cellular.frame_at(tick/15+4)
                self.toxi.write(np.ascontiguousarray(im[::-1,:,0]).tobytes(),alignment=1);self.glyph_tick=tick
            tex=self.toxi
        elif mode==4:tex=self.mould.gl_texture(self.network)
        elif mode==5:tex=self.drawings[variant%2]
        elif mode==6:tex=self.paint[(variant+variant//6)%len(self.paint)]
        else:tex=self.toxi
        self.fbo.use();ctx.viewport=(0,0,*self.size)
        for i,texture in enumerate((self.field.displayed_texture(),tex,self.glyphs,self.title)):texture.use(i)
        hue=.55+.085*math.sin(math.tau*t/311)
        values=dict(cells=0,accentTex=1,glyphs=2,title=3,resolution=tuple(map(float,self.size)),
                    time=float(t),kick=s["kick"],amount=s["amount"],mode=mode,variant=variant,
                    maskSeed=float(s["seed"]),cool=colorsys.hsv_to_rgb(hue,.56,.75),
                    warm=colorsys.hsv_to_rgb((hue+.5)%1,.60,.95),titleMix=s["title"],
                    fade=smooth(t/.3)*smooth((self.score.duration-t)/1.8))
        for name,value in values.items():
            if name in self.program:self.program[name].value=value
        self.vao.render(ctx.TRIANGLES)
        if read:return np.frombuffer(self.fbo.read(components=3,alignment=1),np.uint8).reshape(self.size[1],self.size[0],3)[::-1].copy()

    def release(self):
        self.field.release();self.network.release()
        for x in self.paint+self.drawings+[self.toxi,self.glyphs,self.title,self.vao,self.vbo,self.program,self.fbo,self.texture]:x.release()
        if self.owns_context:self.ctx.release()


def frames(score,size,fps,offset,length,report,art_root,cache,checkpoints=None):
    r=SparseRenderer(score,size,art_root,cache)
    report["renderer"]=r.ctx.info["GL_RENDERER"];print("Renderer:",report["renderer"],flush=True)
    wall=last=time.monotonic();count=math.ceil(length*fps)
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    dark=[]
    try:
        # No composited-image feedback: hidden simulations can catch up without
        # drawing thousands of full-resolution frames before an excerpt.
        r.advance(offset)
        for i in range(count):
            t=offset+i/fps;frame=r.draw(t)
            if i%round(fps*15)==0 or i==90:
                luma=frame.astype(np.float32)@np.array([.2126,.7152,.0722],dtype="f4")
                dark.append(dict(time=t,fraction=float(np.mean(luma<255*.08))))
                if checkpoints:Image.fromarray(frame).save(checkpoints/f"{t:08.2f}.jpg",quality=94)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f"{t:.1f}/{offset+length:.1f}s | {(i+1)/(now-wall):.1f} fps | {' + '.join(score.visible_layers(t))}",flush=True);last=now
        report["render_fps"]=count/(time.monotonic()-wall);report["dark_fraction_samples"]=dark
    finally:r.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio",type=Path);p.add_argument("--art-root",type=Path,required=True)
    p.add_argument("-o","--out",type=Path,default=Path("Bell_Riots_Sparse.mp4"))
    p.add_argument("--cache",type=Path);p.add_argument("--kick-cache",type=Path)
    p.add_argument("--assets-cache",type=Path,default=Path(tempfile.gettempdir())/"hydrapy-bell-sparse")
    p.add_argument("--size",default="1920x1080");p.add_argument("--fps",type=float,default=30.)
    p.add_argument("--edit-start",type=float);p.add_argument("--offset",type=float,default=0.)
    p.add_argument("--seconds",type=float);p.add_argument("--bitrate",default="8M")
    p.add_argument("--encoder",default="auto");p.add_argument("--checkpoints",type=Path)
    a=p.parse_args()
    try:size=tuple(map(int,a.size.lower().split("x")))
    except ValueError:p.error("size must be WIDTHxHEIGHT")
    if len(size)!=2 or min(size)<96 or any(x%2 for x in size):p.error("use even dimensions >=96")
    if not math.isfinite(a.fps) or a.fps<=0 or not math.isfinite(a.offset) or a.offset<0:p.error("invalid fps/offset")
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0):p.error("invalid duration")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=analyse(a.audio,a.cache or a.out.with_suffix(".analysis.npz"))
    start=phrase_start(data) if a.edit_start is None else a.edit_start
    if not math.isfinite(start) or not 0<=start<float(data["duration"]):p.error("invalid edit start")
    kicks,strength=analyse_kicks(a.audio,data,start,a.kick_cache or a.out.with_suffix(".kicks.npz"))
    score=SparseScore(data,kicks,strength,start)
    offset=round(a.offset*a.fps)/a.fps
    if offset>=score.duration:p.error("offset is beyond the edit")
    length=min(a.seconds or score.duration,score.duration-offset)
    report=dict(track=a.audio.name,sha256=str(data["sha256"]),source_start=start,source_end=float(data["duration"]),
                edit_duration=score.duration,offset=offset,seconds=length,size=size,fps=a.fps,
                kick_times=kicks.tolist(),fold_times=score.clock.fold_times,
                cues=score.cues,max_visible_layers=2,cell_grid=[320,180],generations_per_kick=8,
                drawings={path:hashlib.sha256((a.art_root/path).read_bytes()).hexdigest() for path in DRAWINGS})
    print(f"{score.duration:.6f}s | {len(kicks)} kick candidates | {len(score.cues)} accents | at most two layers",flush=True)
    with tempfile.TemporaryDirectory(prefix="bell-sparse-",dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(score,size,a.fps,offset,length,report,a.art_root,a.assets_cache,a.checkpoints),
                   tmp/"video.mp4",fps=a.fps,bitrate=a.bitrate,encoder=a.encoder,
                   on_encoder=lambda e:print("Encoder:",e.name,flush=True))
        subprocess.run(["ffmpeg","-v","error","-i",str(tmp/"video.mp4"),"-ss",str(start+offset),"-t",str(length),
                        "-i",str(a.audio),"-map","0:v:0","-map","1:a:0","-map_metadata","1",
                        "-metadata","title=Bell Riots Pt 2 — Unfolding Edit","-c:v","copy","-c:a","aac",
                        "-b:a","320k","-movflags","+faststart",str(tmp/"complete.mp4")],check=True)
        (tmp/"complete.mp4").replace(a.out)
    a.out.with_suffix(".cues.json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.out,flush=True)


if __name__=="__main__":main()
