"""Two-tone cellular / bit-shift study, using a local music recording.

python docs/two_tone_study.py track.flac -o study.mp4 --start 112 --seconds 42
Original rules/compositions, informed by the user's mono / nicthib references.
The GPU holds a binary state; entropy locking controls recurring local edits.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hydrapy import Fx, save_mp4
from hydrapy.backend_gl import _create_context
from hydrapy.glsl import VERTEX_DESKTOP
from tensor_junglist_video import analyse


PALETTE = np.array([[17, 20, 18], [216, 220, 204]], dtype=np.uint8)
NAMES = ("REGISTER", "LATTICE", "QUADRANTS")


def initial_state(size, variant, seed):
    """Integer patterns at cell resolution; no external images or glyphs."""
    w, h = size
    y, x = np.indices((h, w), dtype=np.uint32)
    if variant == 0:
        # An elementary automaton's history forms a textile-like starting field.
        row = (((np.arange(w) % 72) < 18) ^ ((np.arange(w) % 19) == 0)).astype(np.uint8)
        state = np.empty((h, w), np.uint8)
        for j in range(h):
            state[j] = row
            code = (np.roll(row, 1) << 2) | (row << 1) | np.roll(row, -1)
            rule = (90, 150, 60, 102)[(j // 30) % 4]
            row = ((rule >> code) & 1).astype(np.uint8)
    elif variant == 1:
        # Bitwise arithmetic makes seams and nested rectangular interference.
        a = (x * 3) ^ (y * 5)
        b = ((x + y) & (x - y)) >> 2
        state = (((a ^ b) >> 3) & 1).astype(np.uint8)
    else:
        # Sparse marks and blocked grids leave room for local changes to register.
        a = (x & y) ^ ((x >> 1) & (y >> 1))
        state = (((a & 7) < 2) ^ (((x // 48) + (y // 30)) % 2)).astype(np.uint8)
    # Deterministic small defects distinguish editions without obscuring the rule.
    rng = np.random.default_rng(seed)
    state ^= (rng.random((h, w)) < .001).astype(np.uint8)
    return np.ascontiguousarray(state)


EVOLVE = """#version 330 core
uniform usampler2D state;
uniform ivec2 grid;
uniform int variant, tick, rule, operation, row, column, shift, extent;
uniform int cellX, cellY, gate, edit;
layout(location=0) out uint nextState;
uint cell(ivec2 p) {
    // All neighbor/shift offsets are at most two cells. Keep the operands
    // positive before integer remainder, including at the wrapped edges.
    return texelFetch(state, (p + grid) % grid, 0).r;
}
void main() {
    ivec2 p=ivec2(gl_FragCoord.xy);
    uint old=cell(p), v=old;
    // Local edits persist in the state. Most of the frame is left untouched.
    if(edit == 1) {
        if(operation == 0 && abs(p.y-row)<extent) v=cell(p+ivec2(shift,0));
        else if(operation == 1 && abs(p.x-column)<extent) v=cell(p+ivec2(0,shift));
        else if(operation == 2 && abs(p.x-column)<extent*4 && abs(p.y-row)<extent*3)
            v=cell(p+ivec2(shift,shift));
        else if(operation == 3 && abs(p.x-column)<extent*3 && abs(p.y-row)<extent) {
            uint code=(cell(p+ivec2(-1,-1))<<2u) |
                       (cell(p+ivec2(0,-1))<<1u) | cell(p+ivec2(1,-1));
            v=(uint(rule)>>code)&1u;
        }
    }
    // A moving, narrow active region gives the pattern its own quiet clock.
    // Three update modes; the output always stays exactly one bit per cell.
    ivec2 tile=p/ivec2(24,20);
    if(edit == 0 && gate == 1 && tile.x == cellX && tile.y == cellY) {
        if(variant == 0) {
            uint code=(cell(p+ivec2(-1,-1))<<2u) |
                       (cell(p+ivec2(0,-1))<<1u) | cell(p+ivec2(1,-1));
            v=(uint(rule)>>code)&1u;
        } else if(variant == 1) {
            v=cell(p+ivec2(1,0));
            if((p.y+tick)%19==0) v^=cell(p+ivec2(0,1));
        } else {
            uint n=cell(p+ivec2(-1,-1))+cell(p+ivec2(0,-1))+cell(p+ivec2(1,-1))+
                   cell(p+ivec2(-1,0))+cell(p+ivec2(1,0))+
                   cell(p+ivec2(-1,1))+cell(p+ivec2(0,1))+cell(p+ivec2(1,1));
            v=uint(n==3u || (old==1u && n==2u));
        }
    }
    nextState=v&1u;
}
"""


DISPLAY = """#version 330 core
uniform usampler2D state;
uniform ivec2 grid, resolution;
uniform int pixel, variant;
uniform vec3 background, foreground;
out vec4 color;
void main() {
    ivec2 offset=(resolution-grid*pixel)/2;
    ivec2 q=ivec2(gl_FragCoord.xy)-offset;
    uint bit=0u;
    if(all(greaterThanEqual(q,ivec2(0))) && all(lessThan(q,grid*pixel))) {
        ivec2 p=q/pixel;
        // Reflection is stable for the whole shot, at integer cell boundaries.
        p.x=min(p.x,grid.x-1-p.x);
        if(variant!=1) p.y=min(p.y,grid.y-1-p.y);
        bit=texelFetch(state,p,0).r;
    }
    color=vec4(bit==0u ? background : foreground,1.);
}
"""


class BitField:
    """Small integer ping-pong state plus one nearest-cell presentation pass."""
    def __init__(self, ctx, size=(1920,1080), pixel=4, seed=56):
        self.ctx, self.size, self.pixel, self.seed = ctx, size, pixel, seed
        self.grid = ((size[0] // pixel * 9 // 10) // 2 * 2,
                     (size[1] // pixel * 8 // 9) // 2 * 2)
        self.vbo = ctx.buffer(np.array([-1,-1,3,-1,-1,3], dtype="f4").tobytes())
        self.programs = [ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=s)
                         for s in (EVOLVE, DISPLAY)]
        self.vaos = [ctx.vertex_array(p, [(self.vbo,"2f","position")]) for p in self.programs]
        self.textures = [ctx.texture(self.grid,1,dtype="u1") for _ in range(2)]
        for t in self.textures: t.filter=(ctx.NEAREST,ctx.NEAREST)
        self.fbos = [ctx.framebuffer(color_attachments=[t]) for t in self.textures]
        self.image = ctx.texture(size,3,dtype="f1")
        self.output = ctx.framebuffer(color_attachments=[self.image])
        self.current = 0

    def reset(self, variant):
        self.variant = variant
        raw = initial_state(self.grid, variant, self.seed+variant).tobytes()
        for t in self.textures: t.write(raw, alignment=1)
        self.current = 0

    def step(self, controls, edit=False):
        dest = 1-self.current
        self.fbos[dest].use(); self.ctx.viewport=(0,0,*self.grid)
        self.textures[self.current].use(0)
        values=dict(state=0,grid=self.grid,variant=self.variant,edit=int(edit),**controls)
        for k,v in values.items():
            if k in self.programs[0]: self.programs[0][k].value=int(v) if isinstance(v,np.integer) else v
        self.vaos[0].render(self.ctx.TRIANGLES)
        self.current=dest

    def read(self):
        self.output.use(); self.ctx.viewport=(0,0,*self.size)
        self.textures[self.current].use(0)
        values=dict(state=0,grid=self.grid,resolution=self.size,pixel=self.pixel,variant=self.variant,
                    background=tuple(PALETTE[0]/255),foreground=tuple(PALETTE[1]/255))
        for k,v in values.items(): self.programs[1][k].value=v
        self.vaos[1].render(self.ctx.TRIANGLES)
        return np.frombuffer(self.output.read(components=3,alignment=1),np.uint8).reshape(
            self.size[1],self.size[0],3)[::-1].copy()

    def release(self):
        for x in [*self.vaos,*self.fbos,*self.textures,self.output,self.image,*self.programs,self.vbo]:
            x.release()


def render(data, size, fps, start, seconds, seed, pixel, report, stills=None):
    ctx=_create_context()
    print("Renderer:",ctx.info["GL_RENDERER"],flush=True)
    field=BitField(ctx,size,pixel,seed)
    beats=np.asarray(data["beats"])
    first=int(np.searchsorted(beats,start))
    local_beats=beats[first:]-start
    # Cut only twice, at tracked beats nearest one-third and two-thirds.
    boundaries=[0.]+[float(local_beats[np.argmin(abs(local_beats-seconds*f))])
                     for f in (1/3,2/3)]+[seconds]
    report.update(start=start,seconds=seconds,seed=seed,size=size,fps=fps,palette=PALETTE.tolist(),
                  scenes=[dict(name=NAMES[i],start=boundaries[i],end=boundaries[i+1],
                               lock=(.08,.3,.65)[i]) for i in range(3)])
    controls=dict(tick=0,rule=90,operation=0,row=0,column=0,shift=1,extent=1,
                  cellX=0,cellY=0,gate=0)
    variant=-1; previous_beat=-1; previous_hit=-100.; fx=None
    frames=int(math.ceil(seconds*fps)); wall=time.monotonic(); last=wall
    snapshots=[]
    try:
        for frame in range(frames):
            t=frame/fps
            which=min(2,int(np.searchsorted(boundaries,t,side="right")-1))
            if which!=variant:
                variant=which; field.reset(variant)
                fx=Fx(seed+variant*101,lock=(.08,.3,.65)[variant])
                previous_hit=-100.
            tick=int(t*12)
            j=int(np.clip((start+t)*float(data["rate"]),0,len(data["hit"])-1))
            beat=int(np.searchsorted(local_beats,t,side="right")-1)
            onset=float(data["onset"][j])
            hit=(beat!=previous_beat) or (onset>.6 and t-previous_hit>.16)
            previous_beat=beat
            # The control stream locks; the visual state is retained between edits.
            if hit:
                previous_hit=t
                controls.update(operation=fx.rnd_int(0,3),row=fx.rnd_int(0,field.grid[1]//2-1),
                                column=fx.rnd_int(0,field.grid[0]//2-1),
                                shift=fx.pick([-2,-1,1,2]),extent=fx.pick([1,1,2,3]),
                                rule=fx.pick([54,60,90,102,110,150]))
                field.step(controls,edit=True)
            # Independent, low-rate cellular clock. No repeated update at 30fps.
            gate=frame==0 or int((frame-1)/fps*12)!=tick
            controls.update(tick=tick,gate=int(gate),
                            cellX=(tick//3) % max(1,field.grid[0]//48),
                            cellY=(tick//max(1,field.grid[0]//48)) % max(1,field.grid[1]//40))
            if gate: field.step(controls)
            rgb=field.read()
            if stills is not None and frame in {int(frames*f) for f in (.04,.2,.39,.54,.72,.91)}:
                snapshots.append(Image.fromarray(rgb))
            now=time.monotonic()
            if now-last>10:
                print(f"{t:.1f}/{seconds:.1f}s | {(frame+1)/(now-wall):.1f} fps | {NAMES[variant]}",flush=True)
                last=now
            yield rgb
        report["render_fps"]=frames/(time.monotonic()-wall)
        if stills is not None:
            sheet=Image.new("RGB",(960*2,540*3),tuple(PALETTE[0]))
            for i,im in enumerate(snapshots):
                sheet.paste(im.resize((960,540),Image.Resampling.NEAREST),((i%2)*960,(i//2)*540))
            sheet.save(stills)
    finally:
        field.release(); ctx.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio",type=Path)
    p.add_argument("-o","--out",type=Path,default=Path("Two_Tone_Study.mp4"))
    p.add_argument("--start",type=float,default=112)
    p.add_argument("--seconds",type=float,default=42)
    p.add_argument("--size",default="1920x1080")
    p.add_argument("--fps",type=int,default=30)
    p.add_argument("--pixel",type=int,default=4)
    p.add_argument("--seed",type=int,default=56)
    p.add_argument("--cache",type=Path)
    p.add_argument("--stills",type=Path)
    a=p.parse_args()
    try: size=tuple(map(int,a.size.lower().split("x")))
    except ValueError: p.error("size must be WIDTHxHEIGHT")
    if len(size)!=2 or min(size)<96 or any(n%2 for n in size): p.error("use even dimensions >=96")
    if a.pixel<1 or a.pixel>min(size)//16 or a.fps<1: p.error("invalid pixel size or fps")
    if not math.isfinite(a.start) or a.start<0 or not math.isfinite(a.seconds) or a.seconds<3:
        p.error("start must be >=0 and seconds >=3")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=analyse(a.audio,a.cache or a.out.with_suffix(".analysis.npz"))
    seconds=min(a.seconds,float(data["duration"])-a.start)
    if seconds<3: p.error("not enough audio after start")
    report=dict(audio=a.audio.name,sha256=str(data["sha256"]))
    with tempfile.TemporaryDirectory(prefix="two-tone-",dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        save_mp4(render(data,size,a.fps,a.start,seconds,a.seed,a.pixel,report,a.stills),
                 tmp/"video.mp4",fps=a.fps,quality=9,encoder="auto",
                 on_encoder=lambda e: print("Encoder:",e.name,flush=True))
        subprocess.run(["ffmpeg","-v","error","-i",str(tmp/"video.mp4"),
                        "-ss",str(a.start),"-t",str(seconds),"-i",str(a.audio),
                        "-map","0:v:0","-map","1:a:0","-c:v","copy", "-c:a","aac","-b:a","256k",
                        "-metadata","title=Tensor Junglist — Two Tone Study",
                        "-metadata","artist=Anchor Rite","-movflags","+faststart",str(tmp/"complete.mp4")],check=True)
        (tmp/"complete.mp4").replace(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.out,flush=True)


if __name__=="__main__": main()
