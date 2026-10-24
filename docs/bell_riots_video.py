"""Full-track Bell Riots video: pixel clocks, rule fields and locked glitches.

python docs/bell_riots_video.py track.flac -o Bell_Riots.mp4
The soundtrack is never stretched. Excerpts replay cellular history from zero.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(ROOT), str(ROOT / "patches"), str(ROOT / "docs")]
import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.drawing import find_font
from hydrapy.glsl import VERTEX_DESKTOP
from bell_riots import make_field
from tensor_junglist_video import analyse

SCENES = ("PHASE LATTICE", "DISTRICT FIELD", "GLYPH RELAY", "PHASE APERTURE",
          "CELLULAR LOOM", "RULE FRONTS", "ORBIT DIVIDER", "BIT REGISTERS",
          "LOCKED MEMORY", "TRANSIT GRID")


class Score:
    """A deterministic edit and random-access entropy-locked control score."""
    def __init__(self, data, seed=2024):
        self.data, self.seed = data, seed
        self.duration, self.tempo, self.rate = (float(data[k]) for k in ("duration", "tempo", "rate"))
        self.beats = np.asarray(data["beats"])
        self.period = 60 / self.tempo
        # These landmarks follow Bell Riots' gradual spectral changes. Scaling
        # supports exploratory use with other durations, not automatic sections.
        landmarks = np.array([0, 112, 224, 336, 420, 504, 568, 680, 752]) * self.duration / 795.554694
        self.sections = np.array([0.] + [float(self.beats[np.argmin(abs(self.beats-t))]) for t in landmarks[1:]])
        self.locks = (.10, .20, .42, .62, .16, .37, .72, .49, .28)
        pools = ((1, 0, 4, 7), (2, 3, 1, 4), (7, 2, 5, 9), (0, 6, 8, 3),
                 (3, 4, 6, 1), (5, 2, 1, 9), (8, 7, 2, 5), (0, 3, 6, 4), (1, 0, 8, 2))
        patterns = ((16, 8, 8, 4, 4, 8, 4, 4, 8),
                    (8, 4, 4, 2, 2, 4, 2, 2, 8, 4, 2, 2, 4),
                    (8, 2, 2, 1, 1, 2, 1, 1, 4, 2, 1, .5, .5, 1, 1, 4, 2, 2))
        self.cuts = []; position = 0.; previous = -1
        while True:
            t = max(0., self.beat_time(position)) if position else 0.
            if t >= self.duration: break
            section = self.section(t); count = len(self.cuts)
            level = 0 if section in (0, 4, 7) else (2 if section in (2, 3, 6, 8) else 1)
            duration = patterns[level][count % len(patterns[level])]
            if count % 23 == 0: duration = 16  # let the emergent fields breathe
            scene = pools[section][(count + count//7) % len(pools[section])]
            if scene == previous: scene = pools[section][(count + 1) % len(pools[section])]
            self.cuts.append(dict(start=t, beat=position, beats=duration, scene=scene,
                                  section=section, variant=(count//3 + section*2) % 8))
            previous = scene; position += duration
        self.starts = np.array([c["start"] for c in self.cuts])
        self.control_times = np.array([self.beat_time(i*.5) for i in range(len(self.beats)*2+4)])
        streams = [H.Fx(seed+i*109, lock=lock) for i, lock in enumerate(self.locks)]
        self.locked = np.array([[streams[self.section(t)].rand() for _ in range(4)]
                               for t in self.control_times], dtype="f4")
        self.lock_counts = [f.locks for f in streams]

    def section(self, t):
        return int(np.clip(np.searchsorted(self.sections, t, side="right")-1, 0, 8))

    def beat_time(self, b):
        i = int(math.floor(b)); f = b-i
        if i < 0: return float(self.beats[0]+b*self.period)
        if i >= len(self.beats)-1: return float(self.beats[-1]+(b-len(self.beats)+1)*self.period)
        return float(self.beats[i]*(1-f)+self.beats[i+1]*f)

    def at(self, t):
        index = int(np.clip(np.searchsorted(self.starts, t, side="right")-1, 0, len(self.cuts)-1))
        cut = self.cuts[index]
        j = int(np.clip(t*self.rate, 0, len(self.data["energy"])-1))
        b = int(np.searchsorted(self.beats, t, side="right")-1)
        phase = (t-self.beat_time(b))/max(.1, self.beat_time(b+1)-self.beat_time(b))
        bass, mid, high = map(float, self.data["energy"][j])
        hit = float(self.data["hit"][j]); beat = float(np.exp(-max(0., phase)*7))
        k = int(np.clip(np.searchsorted(self.control_times, t, side="right")-1, 0, len(self.locked)-1))
        locked = tuple(map(float, self.locked[k]))
        section = self.section(t)
        gate = .18 if b % 16 < 12 else .9
        glitch = min(.86, gate*max(0., hit-.3) + max(0., 1-(t-cut["start"])/.10)*.48)
        return dict(cut=cut, index=index, section=section, bass=bass, mid=mid, high=high,
                    hit=hit, beat=beat, locked=locked, glitch=glitch,
                    bank=(2, 8, 5, 9, 6, 3, 11, 8, 2)[section] + math.floor(locked[3]*3))


FRAGMENT = r"""#version 330 core
uniform sampler2D cells, toxi, glyphs, memory, title, hud;
uniform vec2 resolution;
uniform float time, bass, mid, high, hit, beat, glitch, fade, titleMix, progress;
uniform vec4 locked;
uniform int scene, variant, palette;
uniform vec2 sites[15];
out vec4 color;
const float TAU=6.28318530718;
float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
vec4 pixelClock(vec2 _st,float rate,float spread,float density,float pixel,int geometry,float phase,float threshold){
__PIXEL_CLOCK__
}
float loom(vec2 p){return texture(toxi,fract(p)).r;}
float glyph(vec2 p,vec2 grid){
  vec2 cell=floor(p*grid), within=fract(p*grid);
  float value=loom((cell+.5)/grid);
  float symbol=floor(clamp(value,0.,.999)*15.);
  return texture(glyphs,vec2((symbol+within.x)/16.,within.y)).r;
}
vec3 front(vec2 p){return texture(cells,fract(p)).rgb;}
float orbit(vec2 p){
  vec2 q=(p-.5)*vec2(resolution.x/resolution.y,1.);
  float a=1.e6,b=1.e6;
  for(int i=0;i<15;i++){float d=dot(q-sites[i],q-sites[i]);if(d<a){b=a;a=d;}else b=min(b,d);}
  float edge=1.-smoothstep(.002,.007,sqrt(b)-sqrt(a));
  float bands=step(.56,fract(sqrt(a)*37.-time*.15));
  return max(edge,bands*.72);
}
void main(){
  vec2 uv=gl_FragCoord.xy/resolution;
  vec2 p=uv, q=(uv-.5)*vec2(resolution.x/resolution.y,1.);
  float row=floor(p.y*54.);
  float tear=step(.88,h(vec2(row,floor(locked.x*97.))))*glitch;
  p.x+=tear*(locked.y-.5)*.18;
  p.y+=step(.965,h(vec2(floor(p.x*64.),floor(locked.z*73.))))*glitch*.035;
  float v=0., accent=0., shade=1.;
  vec2 mirror=vec2(abs(p.x-.5)*2.,p.y);
  if(scene==0){
    v=pixelClock(p,.035,1.3,80.+float(variant)*11.,3.+float(variant%2),3,.137,0.).r;
  }else if(scene==1){
    vec3 c=front(mirror);
    v=max(step(.97,c.r),c.g*.32);
    accent=step(.98,c.r)*(1.-c.b)*.65;
    vec2 grid=fract(p*vec2(24.,16.));
    v*=.86+.14*step(.08,min(grid.x,grid.y));
  }else if(scene==2){
    vec2 m=vec2(abs(p.x-.5)*2.,abs(p.y-.5)*2.);
    v=glyph(m,vec2(96.,54.));
    float band=step(.42,abs(p.y-.5));
    v=mix(v,loom(p*vec2(2.,.5)),band*.6);
    accent=v*step(.94,h(floor(m*vec2(24.,14.))))*beat;
  }else if(scene==3){
    vec2 s=(p-.5)*rot(time*.012)+.5;
    v=pixelClock(s,.025,.65,100.+float(variant)*9.,3.,1,.137,0.).r;
    float r=length(q), ring=step(.955,fract(r*14.-time*.025));
    v=mix(v*.20,max(v,ring),1.-smoothstep(.38,.385,r));
    accent=ring*.4;
  }else if(scene==4){
    vec2 m=vec2(abs(p.x-.5)*2.,abs(p.y-.5)*2.);
    m.x+=floor(m.y*32.)*.003*float(variant%3-1);
    float a=loom(m),b=loom(m*vec2(2.,1.)+vec2(.25,0.));
    v=step(.23,mix(a,b,step(.5,fract(p.x*4.))));
    accent=step(.80,a)*step(.8,b)*.4;
  }else if(scene==5){
    vec2 m=(p-.5)/(.52+float(variant%3)*.24)+.5;
    vec3 c=front(m);
    vec2 g=fract(m*vec2(480.,270.));
    v=max(step(.97,c.r),c.g*.28)*(.75+.25*step(.14,min(g.x,g.y)));
    accent=clamp(c.g-c.r,0.,1.)*.8;
    v*=1.-smoothstep(.47,.5,abs(p.y-.5));
  }else if(scene==6){
    v=orbit((p-.5)*rot(time*.025)+.5);
    accent=step(.8,v)*loom(p)*.55;
  }else if(scene==7){
    ivec2 a=ivec2(floor(abs(p-.5)*resolution/4.));
    a.x+=int(floor(time*4.))%128;
    int bit=((a.x&a.y)^((a.x>>2)*(a.y>>2))^((a.x+a.y)>>3));
    v=float((bit>>(variant%4))&1);
    float seam=step(.97,fract(p.y*18.));
    accent=seam*glitch;
  }else if(scene==8){
    vec3 c=front(mirror);
    float clock=pixelClock(p,.04,.7,65.,4.,0,.137,.1).r;
    v=max(c.r*.75,clock*.36);
    vec2 s=(p-.5)*rot((locked.x-.5)*.012)*1.008+.5;
    float prev=dot(texture(memory,fract(s)).rgb,vec3(.3,.5,.2));
    v=max(v,prev*.88);
    accent=step(.96,c.r)*.8;
  }else{
    float r=max(length(q),.035),a=atan(q.y,q.x)/TAU;
    float z=.23/r+time*.38;
    vec2 g=abs(fract(vec2(a*18.,z*4.))-.5);
    float lines=1.-smoothstep(.014,.035,min(g.x,g.y));
    float registers=loom(vec2(a*2.,z*.07));
    v=max(lines,step(.7,registers)*.45)*smoothstep(.04,.15,r);
    accent=lines*step(.65,fract(z*.5))*.8;
  }
  vec3 bg=vec3(.025,.036,.033),fg=vec3(.83,.85,.78),ac=vec3(.95,.32,.17);
  if(palette==1){bg=vec3(.027,.032,.045);fg=vec3(.80,.62,.36);ac=vec3(.55,.81,.79);}
  if(palette==2){bg=vec3(.025,.035,.049);fg=vec3(.64,.83,.82);ac=vec3(.91,.50,.32);}
  if(palette==3){bg=vec3(.045,.028,.029);fg=vec3(.84,.76,.65);ac=vec3(.91,.21,.16);}
  vec3 rgb=mix(bg,fg,clamp(v,0.,1.));
  rgb=mix(rgb,ac,clamp(accent,0.,.75));
  float keep=step(.84,h(floor(p*vec2(32.,20.))+floor(locked.xy*64.)))*glitch*.78;
  rgb=mix(rgb,texture(memory,fract(p+vec2((locked.z-.5)*.014,0))).rgb,keep);
  // Pixel-precise framing and sparse readout; no random full-frame grain.
  float border=step(.965,abs(uv.x-.5)*2.)+step(.94,abs(uv.y-.5)*2.);
  rgb=mix(rgb,bg,.88*clamp(border,0.,1.));
  float line=step(.988,fract(uv.x*40.))*step(.965,uv.y)*step(uv.y,.977);
  rgb=mix(rgb,fg,line*.55);
  float timeline=step(uv.y,.017)*step(.014,uv.y)*step(uv.x,progress);
  rgb=mix(rgb,ac,timeline*.75);
  vec4 label=texture(hud,uv);rgb=mix(rgb,label.rgb,label.a*.8);
  vec4 text=texture(title,uv);
  rgb*=1.-titleMix*.80*(1.-smoothstep(.4,.8,uv.x));
  rgb=mix(rgb,text.rgb,text.a*titleMix);
  color=vec4(clamp(rgb*fade,0.,1.),1.);
}
"""


def smooth(x):
    v=float(np.clip(x,0,1));return v*v*(3-2*v)


class Renderer:
    SIM_RATE = 15
    def __init__(self, score, size=(1920,1080), ctx=None):
        self.score,self.size=score,size
        self.owns_context=ctx is None;self.ctx=ctx or _create_context();ctx=self.ctx
        self.controls=dict(init=1.,bass=0.,hit=0.,bank=2.)
        self.field=GLRenderer(make_field(self.controls,size),size=size,ctx=ctx,fps=self.SIM_RATE)
        for tick in range(48):
            self.field.step(tick/self.SIM_RATE);self.controls["init"]=0.
        self.tick=-1
        self.cellular=H.Cellular(width=384,height=216,rate=34.25,seed=2024,stripes=24,
                                density=.24,perturb=.0012)
        self.toxi=ctx.texture((384,216),1,dtype="f1");self.toxi.filter=(ctx.NEAREST,ctx.NEAREST)
        self.toxi_tick=-1
        atlas=Image.new("L",(16*8,12));d=ImageDraw.Draw(atlas);font=find_font("DejaVuSansMono",11)
        for i,char in enumerate(" .:;=+*ox#%@/\\[]"):
            d.text((i*8, -1),char,font=font,fill=255)
        self.glyphs=ctx.texture(atlas.size,1,np.asarray(atlas)[::-1].copy().tobytes());self.glyphs.filter=(ctx.NEAREST,ctx.NEAREST)
        self.title=self.text_texture(True);self.hud=self.text_texture(False)
        shader=FRAGMENT.replace("__PIXEL_CLOCK__",H.REGISTRY["pixelOsc"].glsl)
        self.program=ctx.program(vertex_shader=VERTEX_DESKTOP,fragment_shader=shader)
        self.vbo=ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype="f4").tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,"2f","position")])
        self.targets=[]
        for _ in range(2):
            tex=ctx.texture(size,3,dtype="f1");tex.filter=(ctx.NEAREST,ctx.NEAREST)
            fbo=ctx.framebuffer(color_attachments=[tex]);fbo.use();ctx.clear(.025,.036,.033,1)
            self.targets.append((tex,fbo))
        self.current=0

    def text_texture(self,title):
        w,h=self.size;im=Image.new("RGBA",self.size);d=ImageDraw.Draw(im)
        small=find_font("DejaVuSansMono",max(10,round(h*.014)))
        if title:
            big=find_font("DejaVuSans-Bold",round(h*.105));x,y=int(w*.067),int(h*.31)
            d.text((x,y-int(h*.065)),"A N C H O R   R I T E",font=small,fill=(216,220,204,255))
            d.text((x,y),"BELL RIOTS",font=big,fill=(224,223,207,255))
            d.text((x,y+int(h*.13)),"PT. 2",font=big,fill=(215,133,71,255))
            d.text((x,y+int(h*.29)),"P I N N E D    /    0 4",font=small,fill=(216,220,204,255))
        else:
            d.text((int(w*.04),int(h*.035)),"SANCTUARY / DISTRICT A",font=small,fill=(210,217,200,255))
            label="2024  /  ARCHIVE 04";bbox=d.textbbox((0,0),label,font=small)
            d.text((w-int(w*.04)-bbox[2],int(h*.035)),label,font=small,fill=(210,217,200,255))
        return self.ctx.texture(self.size,4,np.asarray(im)[::-1].copy().tobytes())

    def advance(self,t):
        target=int(math.floor(t*self.SIM_RATE+1e-8))
        if target<self.tick:raise ValueError("Cellular history requires monotonically increasing times")
        while self.tick<target:
            self.tick+=1;s=self.score.at(self.tick/self.SIM_RATE)
            self.controls.update(bass=s["bass"],hit=s["hit"],bank=s["bank"])
            self.field.step((self.tick+48)/self.SIM_RATE)
        k=int(t*15+1e-8)
        if k!=self.toxi_tick:
            im=self.cellular.frame_at(k/15)
            self.toxi.write(np.ascontiguousarray(im[::-1,:,0]).tobytes(),alignment=1);self.toxi_tick=k

    def draw(self,t,force_scene=None,read=True):
        self.advance(t);s=self.score.at(t);cut=s["cut"]
        dest=1-self.current;tex,fbo=self.targets[dest];fbo.use();self.ctx.viewport=(0,0,*self.size)
        field=self.field.displayed_texture();field.filter=(self.ctx.NEAREST,self.ctx.NEAREST)
        for i,texture in enumerate((field,self.toxi,self.glyphs,self.targets[self.current][0],self.title,self.hud)):texture.use(i)
        title=smooth(t/.4)*smooth((9-t)/2) if t<9 else smooth((t-self.score.duration+7)/2)
        sites=[(0.,0.)]
        for ring in range(2):
            a=t*.21*(1 if ring==0 else -1)+ring*math.pi
            sites.extend((.51*.5**ring*math.cos(a+math.tau*i/7),.51*.5**ring*math.sin(a+math.tau*i/7)) for i in range(7))
        values=dict(cells=0,toxi=1,glyphs=2,memory=3,title=4,hud=5,resolution=tuple(map(float,self.size)),time=float(t),
                    **{k:s[k] for k in ("bass","mid","high","hit","beat","glitch","locked")},
                    scene=cut["scene"] if force_scene is None else force_scene,variant=cut["variant"],
                    palette=(0,0,1,2,0,1,3,2,0)[s["section"]],
                    fade=smooth(t/.18)*smooth((self.score.duration-t)/1.6),titleMix=title,
                    progress=t/self.score.duration)
        for name,value in values.items():
            if name in self.program:self.program[name].value=value
        self.program["sites"].write(np.array(sites,dtype="f4").tobytes())
        self.vao.render(self.ctx.TRIANGLES);self.current=dest
        if read:return np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(self.size[1],self.size[0],3)[::-1].copy()

    def release(self):
        self.field.release()
        for tex,fbo in self.targets:fbo.release();tex.release()
        for x in (self.toxi,self.glyphs,self.title,self.hud,self.vao,self.vbo,self.program):x.release()
        if self.owns_context:self.ctx.release()


def frames(score,size,fps,start,length,report,checkpoints=None):
    renderer=Renderer(score,size);report["renderer"]=renderer.ctx.info["GL_RENDERER"]
    print("Renderer:",report["renderer"],flush=True)
    wall=last=time.monotonic();count=math.ceil(length*fps)
    captures={round((t-start)*fps) for t in [3.,*np.arange(28,score.duration,56),score.duration-4.] if start<=t<start+length}
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    try:
        # Excerpts reconstruct CA from zero; two seconds settle image feedback.
        if start:
            for t in np.arange(max(0,start-2),start,1/fps):renderer.draw(float(t),read=False)
        for i in range(count):
            t=start+i/fps
            frame=renderer.draw(t)
            if checkpoints and i in captures:
                Image.fromarray(frame).save(checkpoints/f"{t:08.2f}.jpg",quality=94)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f"{t:.1f}/{start+length:.1f}s | {(i+1)/(now-wall):.1f} fps | {SCENES[score.at(t)['cut']['scene']]}",flush=True);last=now
        report["render_fps"]=count/(time.monotonic()-wall)
    finally:renderer.release()


def gallery(score,size,path):
    r=Renderer(score,size);thumb=(480,270);sheet=Image.new("RGB",(2400,600),(7,10,9));d=ImageDraw.Draw(sheet)
    try:
        for i,name in enumerate(SCENES):
            t=12+i*.5;frame=r.draw(t,force_scene=i)
            sheet.paste(Image.fromarray(frame).resize(thumb,Image.Resampling.NEAREST),((i%5)*480,(i//5)*300))
            d.text(((i%5)*480+12,(i//5)*300+278),name,font=find_font("DejaVuSansMono",14),fill=(216,220,204))
        sheet.save(path)
    finally:r.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio",type=Path);p.add_argument("-o","--out",type=Path,default=Path("Bell_Riots_Pt_2.mp4"))
    p.add_argument("--cache",type=Path);p.add_argument("--size",default="1920x1080");p.add_argument("--fps",type=float,default=30)
    p.add_argument("--start",type=float,default=0);p.add_argument("--seconds",type=float);p.add_argument("--seed",type=int,default=2024)
    p.add_argument("--gallery",type=Path);p.add_argument("--bitrate",default="8M");p.add_argument("--encoder",default="auto")
    p.add_argument("--checkpoints",type=Path,help="save occasional source frames during a long render")
    a=p.parse_args()
    try:size=tuple(map(int,a.size.lower().split("x")))
    except ValueError:p.error("size must be WIDTHxHEIGHT")
    if len(size)!=2 or min(size)<96 or any(v%2 for v in size):p.error("use even dimensions >=96")
    if not math.isfinite(a.fps) or a.fps<=0 or not math.isfinite(a.start) or a.start<0:p.error("invalid fps/start")
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0):p.error("invalid duration")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    score=Score(analyse(a.audio,a.cache or a.out.with_suffix(".analysis.npz")),a.seed)
    if a.start>=score.duration:p.error("start is beyond the track")
    length=min(a.seconds or score.duration,score.duration-a.start)
    print(f"{score.duration:.3f}s; {score.tempo:.3f} BPM; {len(score.cuts)} cuts; entropy resets {score.lock_counts}",flush=True)
    report=dict(track=a.audio.name,sha256=str(score.data["sha256"]),duration=score.duration,
                start=a.start,seconds=length,seed=a.seed,size=size,fps=a.fps,estimated_bpm=score.tempo,
                sections=score.sections.tolist(),entropy_lock=list(score.locks),entropy_resets=score.lock_counts,
                cuts=[dict(c,name=SCENES[c["scene"]]) for c in score.cuts])
    if a.gallery:gallery(score,size,a.gallery);return
    with tempfile.TemporaryDirectory(prefix="bell-riots-",dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(score,size,a.fps,a.start,length,report,a.checkpoints),tmp/"video.mp4",fps=a.fps,
                   encoder=a.encoder,bitrate=a.bitrate,on_encoder=lambda e:print("Encoder:",e.name,flush=True))
        subprocess.run(["ffmpeg","-v","error","-i",str(tmp/"video.mp4"),"-ss",str(a.start),"-t",str(length),
                        "-i",str(a.audio),"-map","0:v:0","-map","1:a:0","-map_metadata","1","-c:v","copy",
                        "-c:a","aac","-b:a","320k","-movflags","+faststart",str(tmp/"complete.mp4")],check=True)
        (tmp/"complete.mp4").replace(a.out)
    a.out.with_suffix(".cues.json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.out,flush=True)


if __name__=="__main__":main()
