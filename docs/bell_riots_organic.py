"""Bell Riots, second pass: a phrase-aligned closing edit with independent layers.

python docs/bell_riots_organic.py track.flac --art-root /path/to/Generative_Art -o Bell_Riots_Organic.mp4
--offset and --seconds select an excerpt within the new edit, including its audio.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "docs"), str(ROOT / "patches")]
import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.drawing import find_font
from hydrapy.glsl import VERTEX_DESKTOP
from hydrapy.recipes import handpainted
from bell_riots_video import Renderer as PixelRenderer, Score, SCENES, smooth
from tensor_junglist_video import analyse

DRAWINGS = ("studies/wisconsin-flora/output/leadplant.svg",
            "studies/farm-remnants/output/barn-22-d72.svg")
CLOCKS = dict(drawings=23.7, paint=31.3, physarum=33.1, roles=41.9, palette=73.9)


def phrase_start(data, seconds=360.):
    """Nearest 32-beat phrase on the estimated beat grid; never trim the tail."""
    beats = np.asarray(data["beats"])
    candidates = beats[::32]
    target = max(0., float(data["duration"]) - seconds)
    return 0. if target == 0 else float(candidates[np.argmin(abs(candidates - target))])


def cycle(t, period, offset=0., count=6, overlap=.22):
    """Independent, continuous crossfade; no music or other layer clock input."""
    position = (t + offset) / period
    index = math.floor(position)
    return index % count, (index + 1) % count, smooth((position-index-1+overlap)/overlap)


class OrganicScore:
    def __init__(self, data, start=None, seed=2024):
        self.source = Score(data, seed)
        self.start = phrase_start(data) if start is None else float(start)
        if not 0 <= self.start < self.source.duration:
            raise ValueError("edit start must be within the track")
        self.duration = self.source.duration-self.start
        self.data, self.seed, self.tempo = data, seed, self.source.tempo
        self.first_beat = float(np.interp(self.start, self.source.beats, np.arange(len(self.source.beats))))
        self.cuts = []
        beat = 0.; previous = -1
        # Fewer cuts than the first pass. The dense passage still has short bursts.
        pools = ((3, 4, 6, 1), (4, 2, 1, 0), (8, 7, 2, 5), (0, 3, 6, 4), (1, 0, 8, 2))
        while True:
            t = 0. if not self.cuts else self.beat_time(beat)
            if t >= self.duration: break
            section = self.source.section(self.start+t)
            n = len(self.cuts)
            pattern = (8, 8, 16, 4, 4, 8, 4, 8) if section != 6 else (8, 4, 4, 2, 2, 4, 8, 2, 2, 4, 16)
            length = pattern[n % len(pattern)]
            pool = pools[max(0, section-4) % len(pools)]
            scene = pool[(n+n//9) % len(pool)]
            if scene == previous: scene = pool[(pool.index(scene)+1) % len(pool)]
            self.cuts.append(dict(start=t, beat=beat, beats=length, scene=scene,
                                  section=section, variant=(n//3+section*2) % 8))
            previous = scene; beat += length
        self.starts = np.array([c["start"] for c in self.cuts])

    def beat_time(self, beat):
        return self.source.beat_time(self.first_beat+beat)-self.start

    def at(self, t):
        s = self.source.at(self.start+t)
        index = int(np.clip(np.searchsorted(self.starts, t, side="right")-1, 0, len(self.cuts)-1))
        cut = self.cuts[index]
        # Retain locked registers, but reduce the first pass's tearing duty cycle.
        s.update(cut=cut, index=index,
                 glitch=min(.64, s["glitch"]*.55+max(0., 1-(t-cut["start"])/.08)*.24))
        return s


class MatteRenderer(PixelRenderer):
    def text_texture(self, title):
        return self.ctx.texture((1, 1), 4, bytes(4))


def raster_assets(cache, art_root):
    """Cache the actual Generative_Art vectors; never synthesize substitutes."""
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    if art_root is None:
        raise ValueError("supply the Generative_Art checkout with --art-root")
    for relative in DRAWINGS:
        source = Path(art_root) / relative
        name = source.stem
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        path = cache / f"{name}-{digest}.png"
        if not path.exists():
            if shutil.which("inkscape"):
                subprocess.run(["inkscape", str(source), "--export-type=png", "--export-width=1800",
                                "--export-background=white", "--export-background-opacity=1",
                                f"--export-filename={path}"], check=True, capture_output=True)
            else:
                try: import cairosvg
                except ImportError as exc:
                    raise RuntimeError("Install Inkscape or cairosvg to rasterize the source drawings") from exc
                cairosvg.svg2png(url=str(source), write_to=str(path), output_width=1800, background_color="white")
        result.append(path)
    return result


FRAGMENT = r"""#version 330 core
uniform sampler2D pixelField, mould, plant, barn, paintA, paintB, title;
uniform vec2 resolution;
uniform float time, bass, beat, paintMix, drawingMix, foldsA, foldsB;
uniform float organicWeight, pixelWeight, fade, titleMix;
uniform vec3 cool, warm;
out vec4 colour;
const float TAU=6.28318530718;
mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
float luma(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
float drawing(vec2 p,float n,float angle){
  vec2 q=(p-.5)*vec2(resolution.x/resolution.y,1.);
  float r=length(q), a=atan(q.y,q.x)+angle;
  a=abs(mod(a+TAU/n*.5,TAU/n)-TAU/n*.5);
  vec2 uv=vec2(cos(a),sin(a))*r;
  uv=uv*rot(-angle*.73)*(.86+.10*sin(time*.071))+.5;
  float x=texture(plant,fract(uv+vec2(.12*sin(time*.031),.09*cos(time*.027)))).r;
  float y=texture(barn,fract((uv-.5)*rot(time*.011)*1.27+.5)).r;
  float blend=.5+.5*sin(time*.083+1.7);
  return max(x*(.55+.45*blend),y*(1.-.5*blend));
}
void main(){
  vec2 uv=gl_FragCoord.xy/resolution;
  vec2 q=(uv-.5)*vec2(resolution.x/resolution.y,1.);
  vec2 p=(uv-.5)*rot(.035*sin(time*.021))+.5;
  vec2 paintUV=p*vec2(1.15,.93)+vec2(time*.008,-time*.005);
  vec3 pigment=texture(paintA,fract(paintUV)).rgb;
  if(paintMix>0.)pigment=mix(pigment,texture(paintB,fract(paintUV)).rgb,paintMix);
  float ink=clamp(1.-luma(pigment),0.,1.);
  vec2 mp=vec2(abs(p.x-.5)*1.5,p.y)+vec2(time*.002,-time*.0014);
  float network=texture(mould,fract(mp)).r;
  network=clamp(network*2.8,0.,1.);
  float lines=drawing(uv,foldsA,time*.027);
  if(drawingMix>0.)lines=mix(lines,drawing(uv,foldsB,time*.027),drawingMix);
  lines=clamp(lines*1.55,0.,1.);
  float breath=.5+.5*sin(time*.113+.8);
  float dyeMask=smoothstep(.10,.65,ink);
  vec3 dark=mix(vec3(.012,.022,.034),cool*.065,.35);
  // Thick dye absorbs light; complementary colours belong to different layers.
  vec3 base=mix(dark,cool*(.22+.21*breath),dyeMask);
  base=mix(base,warm*.35,clamp(pigment.r-pigment.b,0.,1.)*.6);
  base+=cool*pow(network,1.4)*(.36+.30*(1.-breath));
  base=mix(base,warm*(.78+.18*ink),lines*(.63+.28*breath));
  float field=luma(texture(pixelField,uv).rgb);
  float matte=smoothstep(.12,.63,field);
  float aperture=.5+.5*sin(length(q)*8.5-time*.091+sin(atan(q.y,q.x)*3.)*.65);
  float reveal=mix(.22,.95,smoothstep(.18,.85,aperture))*organicWeight;
  // Pixel cells reveal the image below. Fine complementary edges remain legible.
  vec3 within=base*(1.25+.75*organicWeight)+cool*.17+warm*.06;
  vec3 rgb=mix(base*(.34+.45*reveal),within,matte*pixelWeight);
  float px=1./resolution.x;
  float adjacent=luma(texture(pixelField,uv+vec2(px*2.,0.)).rgb);
  float edge=clamp(abs(field-adjacent)*2.,0.,1.);
  rgb+=mix(warm,vec3(.91,.91,.82),.27)*edge*.23*pixelWeight;
  rgb=mix(rgb,base,clamp(reveal*.65,0.,.8));
  // The slow structure is always present; percussion only nudges exposure.
  rgb*=.96+.055*beat+.035*bass;
  float vignette=1.-.22*smoothstep(.30,1.0,length(q));
  rgb=pow(max(rgb,vec3(0.)),vec3(.88))*vignette;
  float border=smoothstep(.982,.994,abs(uv.x-.5)*2.);
  rgb=mix(rgb,dark,border);
  vec4 text=texture(title,uv);
  rgb*=1.-titleMix*.58*(1.-smoothstep(.28,.8,uv.x));
  rgb=mix(rgb,text.rgb,text.a*titleMix);
  colour=vec4(clamp(rgb*fade,0.,1.),1.);
}
"""


class OrganicRenderer:
    SIM_RATE = 15
    def __init__(self, score, size=(1920,1080), cache=None, ctx=None, art_root=None):
        self.score, self.size = score, size
        paths = raster_assets(cache or Path(tempfile.gettempdir())/"hydrapy-bell-organic",art_root)
        self.owns_context = ctx is None
        self.ctx = ctx or _create_context(); ctx = self.ctx
        self.pixel = MatteRenderer(score, size, ctx=ctx)
        self.tick = -1; self.last_time = -1.
        self.controls = dict(angle=.55, distance=15., turn=.42)
        patch = H.Patch(); patch.size = (384,216)
        self.mould = H.Physarum(size=(384,216), agents=(128,128), seed=29,
                               sensor_angle=H.Fn(lambda _:self.controls["angle"]),
                               sensor_distance=H.Fn(lambda _:self.controls["distance"]),
                               turn=H.Fn(lambda _:self.controls["turn"]),
                               deposit=.065, decay=.961, diffuse=.72, exposure=1.4,
                               colour=(1,1,1), speed=30, max_steps=2)
        H.src(self.mould).out(patch.o0, patch=patch); patch.render(patch.o0)
        self.network = GLRenderer(patch, size=(384,216), fps=self.SIM_RATE, ctx=ctx)
        for i in range(180): self.network.step(i/self.SIM_RATE)
        self.paint = self.paint_bank()
        self.drawings = []
        for path in paths:
            im = ImageOps.invert(Image.open(path).convert("L"))
            canvas = Image.new("L",(2048,2048))
            im.thumbnail((1900,1900),Image.Resampling.LANCZOS)
            canvas.paste(im,((2048-im.width)//2,(2048-im.height)//2))
            tex = ctx.texture(canvas.size,1,np.asarray(canvas)[::-1].copy().tobytes())
            tex.filter=(ctx.LINEAR,ctx.LINEAR);tex.repeat_x=tex.repeat_y=True
            self.drawings.append(tex)
        self.title = self.title_texture()
        self.program=ctx.program(vertex_shader=VERTEX_DESKTOP,fragment_shader=FRAGMENT)
        self.vbo=ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype="f4").tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,"2f","position")])
        self.texture=ctx.texture(size,3,dtype="f1"); self.fbo=ctx.framebuffer(color_attachments=[self.texture])

    def title_texture(self):
        w,h=self.size; im=Image.new("RGBA",self.size); d=ImageDraw.Draw(im)
        x,y=round(w*.065),round(h*.34)
        small=find_font("DejaVuSansMono",max(10,round(h*.016)))
        big=find_font("DejaVuSans-Bold",max(20,round(h*.090)))
        d.text((x,y-round(h*.055)),"A N C H O R   R I T E",font=small,fill=(205,226,213,255))
        d.text((x,y),"BELL RIOTS",font=big,fill=(225,230,210,255))
        d.text((x,y+round(h*.11)),"PT. 2",font=big,fill=(226,144,85,255))
        d.text((x,y+round(h*.25)),"PINNED  /  CLOSING TRANSMISSION",font=small,fill=(205,226,213,255))
        return self.ctx.texture(self.size,4,np.asarray(im)[::-1].copy().tobytes())

    def paint_bank(self):
        """Bake real dye/drop/bleed operators once; animate their textures at display rate."""
        ctx=self.ctx; textures=[]
        for seed in (3,17,29,43,71,89):
            p=H.Patch();p.size=(512,512)
            handpainted(seed=seed,hues=(.51,.055,.65),density=(.66,.48,.35),
                        scale=(2.6,4.4,7.8),stretch=(2.7,1.1,3.3),size=.72,
                        bleed=2,cracks=.13,texture=.32).out(p.o0,patch=p)
            p.render(p.o0)
            r=GLRenderer(p,size=p.size,ctx=ctx)
            try:
                r.step(0)
                tex=ctx.texture(p.size,4,dtype="f2");tex.filter=(ctx.LINEAR,ctx.LINEAR)
                tex.repeat_x=tex.repeat_y=True
                fb=ctx.framebuffer(color_attachments=[tex])
                srcfb=ctx.framebuffer(color_attachments=[r.displayed_texture()])
                ctx.copy_framebuffer(fb,srcfb);srcfb.release();fb.release()
                textures.append(tex)
            finally:r.release()
        return textures

    def advance(self,t):
        if t < self.last_time-1e-8: raise ValueError("render time must be monotonically increasing")
        self.last_time=t
        target=int(math.floor(t*self.SIM_RATE+1e-8))
        while self.tick<target:
            self.tick+=1; now=self.tick/self.SIM_RATE
            phase=math.tau*(now+8.4)/CLOCKS["physarum"]
            self.controls.update(angle=.56+.31*math.sin(phase),distance=16.+8.*math.cos(phase*.71),
                                 turn=.38+.16*math.sin(phase+1.3))
            self.network.step((self.tick+180)/self.SIM_RATE)

    def draw(self,t,read=True,force_scene=None):
        self.advance(t);self.pixel.draw(t,read=False,force_scene=force_scene)
        ctx=self.ctx;s=self.score.at(t)
        pa,pb,pm=cycle(t,CLOCKS["paint"],9.1)
        da,db,dm=cycle(t,CLOCKS["drawings"],3.7)
        folds=(6.,10.,8.,12.,4.,16.)
        # A continuous hue trajectory keeps the two inks opposite each other.
        hue=.51+.19*(.5-.5*math.cos(math.tau*(t+7.)/CLOCKS["palette"]))
        cool=colorsys.hsv_to_rgb(hue,.64,.85)
        warm=colorsys.hsv_to_rgb((hue+.5)%1.,.68,.94)
        role=.5+.5*math.sin(math.tau*(t+2.8)/CLOCKS["roles"])
        organic=.42+.53*role
        pixel=.68+.30*(1.-role)
        if s["section"]==6: pixel=min(1.,pixel+.10)
        self.fbo.use();ctx.viewport=(0,0,*self.size)
        textures=(self.pixel.targets[self.pixel.current][0],self.mould.gl_texture(self.network),
                  *self.drawings,self.paint[pa],self.paint[pb],self.title)
        for i,tex in enumerate(textures):tex.use(i)
        intro=smooth(t/.7)*smooth((7.-t)/2.) if t<7 else 0.
        outro=smooth((t-self.score.duration+6)/2.)
        values=dict(pixelField=0,mould=1,plant=2,barn=3,paintA=4,paintB=5,title=6,
                    resolution=tuple(map(float,self.size)),time=float(t),
                    **{k:s[k] for k in ("bass","beat")},
                    paintMix=pm,drawingMix=dm,foldsA=folds[da],foldsB=folds[db],
                    organicWeight=organic,pixelWeight=pixel,cool=cool,warm=warm,
                    fade=smooth(t/.25)*smooth((self.score.duration-t)/1.8),
                    titleMix=max(intro,outro))
        for name,value in values.items():
            if name in self.program:self.program[name].value=value
        self.vao.render(ctx.TRIANGLES)
        if read:return np.frombuffer(self.fbo.read(components=3,alignment=1),np.uint8).reshape(self.size[1],self.size[0],3)[::-1].copy()

    def release(self):
        self.pixel.release();self.network.release()
        for tex in self.paint+self.drawings+[self.title]:tex.release()
        for obj in (self.vao,self.vbo,self.program,self.fbo,self.texture):obj.release()
        if self.owns_context:self.ctx.release()


def frames(score,size,fps,offset,length,report,cache,art_root,checkpoints=None):
    r=OrganicRenderer(score,size,cache,art_root=art_root)
    report["renderer"]=r.ctx.info["GL_RENDERER"]
    print("Renderer:",report["renderer"],flush=True)
    wall=last=time.monotonic(); count=math.ceil(length*fps)
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    try:
        # Replay the whole edit at the requested cadence, including image memory.
        # This makes an excerpt agree with the corresponding full-render frames.
        first=round(offset*fps)
        for i in range(first):
            r.draw(i/fps,read=False)
            if time.monotonic()-last>15:
                print(f"Preroll {i/fps:.1f}/{offset:.1f}s",flush=True);last=time.monotonic()
        for i in range(count):
            t=(first+i)/fps
            frame=r.draw(t)
            if checkpoints and (i==90 or i%round(fps*30)==0):
                Image.fromarray(frame).save(checkpoints/f"{t:08.2f}.jpg",quality=94)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f"{t:.1f}/{offset+length:.1f}s | {(first+i+1)/(now-wall):.1f} fps | {SCENES[score.at(t)['cut']['scene']]}",flush=True);last=now
        report["render_fps"]=(first+count)/(time.monotonic()-wall)
    finally:r.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio",type=Path);p.add_argument("-o","--out",type=Path,default=Path("Bell_Riots_Organic.mp4"))
    p.add_argument("--cache",type=Path);p.add_argument("--assets-cache",type=Path)
    p.add_argument("--art-root",type=Path,required=True,help="existing Generative_Art checkout containing the two source vectors")
    p.add_argument("--size",default="1920x1080");p.add_argument("--fps",type=float,default=30.)
    p.add_argument("--edit-start",type=float);p.add_argument("--offset",type=float,default=0.)
    p.add_argument("--seconds",type=float);p.add_argument("--seed",type=int,default=2024)
    p.add_argument("--bitrate",default="8M");p.add_argument("--encoder",default="auto")
    p.add_argument("--checkpoints",type=Path)
    a=p.parse_args()
    try:size=tuple(map(int,a.size.lower().split("x")))
    except ValueError:p.error("size must be WIDTHxHEIGHT")
    if len(size)!=2 or min(size)<96 or any(x%2 for x in size):p.error("use even dimensions >=96")
    if not math.isfinite(a.fps) or a.fps<=0 or not math.isfinite(a.offset) or a.offset<0:p.error("invalid fps/offset")
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0):p.error("invalid duration")
    score=OrganicScore(analyse(a.audio,a.cache or a.out.with_suffix(".analysis.npz")),a.edit_start,a.seed)
    # One common clock for the requested video excerpt and its sample-accurate audio trim.
    offset=round(a.offset*a.fps)/a.fps
    if offset>=score.duration:p.error("offset is beyond the edit")
    length=min(a.seconds or score.duration,score.duration-offset)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(track=a.audio.name,sha256=str(score.data["sha256"]),source_start=score.start,
                source_end=score.source.duration,edit_duration=score.duration,offset=offset,seconds=length,
                size=size,fps=a.fps,seed=a.seed,estimated_bpm=score.tempo,clocks=CLOCKS,
                drawings={path:hashlib.sha256((a.art_root/path).read_bytes()).hexdigest() for path in DRAWINGS},
                cuts=[dict(c,name=SCENES[c["scene"]],source_time=score.start+c["start"]) for c in score.cuts])
    print(f"Source {score.start:.6f}–{score.source.duration:.6f}s | edit {score.duration:.6f}s | {len(score.cuts)} cuts",flush=True)
    with tempfile.TemporaryDirectory(prefix="bell-organic-",dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(score,size,a.fps,offset,length,report,a.assets_cache,a.art_root,a.checkpoints),
                   tmp/"video.mp4",fps=a.fps,bitrate=a.bitrate,encoder=a.encoder,
                   on_encoder=lambda e:print("Encoder:",e.name,flush=True))
        subprocess.run(["ffmpeg","-v","error","-i",str(tmp/"video.mp4"),"-ss",str(score.start+offset),
                        "-t",str(length),"-i",str(a.audio),"-map","0:v:0","-map","1:a:0",
                        "-map_metadata","1","-metadata","title=Bell Riots Pt 2 — Organic Edit",
                        "-c:v","copy","-c:a","aac","-b:a","320k","-movflags","+faststart",str(tmp/"complete.mp4")],check=True)
        (tmp/"complete.mp4").replace(a.out)
    a.out.with_suffix(".cues.json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.out,flush=True)


if __name__=="__main__":main()
