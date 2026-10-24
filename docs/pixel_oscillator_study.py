"""Export an uninterrupted independent-oscillator study and a live HTML player.

python docs/pixel_oscillator_study.py --out Pixel_Clocks.mp4 --seconds 90 \
    --audio track.flac --audio-start 112 --html Pixel_Clocks.html
The optional recording is a soundtrack only: it never drives the oscillators.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"patches")]
import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from pixel_oscillators import make_patch, PRESETS


def live_html(preset="saddle"):
    """Same registered GLSL body, one WebGL2 pass, no external dependencies."""
    shader="""#version 300 es
precision highp float;
precision highp int;
uniform vec2 resolution;
uniform float time, rate, spread, density, pixel, phase, threshold;
uniform int geometry;
out vec4 color;
vec4 oscillator(vec2 _st) {
"""+H.REGISTRY["pixelOsc"].glsl+"""
}
void main() {
  float value=oscillator(gl_FragCoord.xy/resolution).r;
  color=vec4(mix(vec3(17.,20.,18.),vec3(216.,220.,204.),value)/255.,1.);
}
"""
    defaults={k:dict(v,phase=.137,threshold=0) for k,v in PRESETS.items()}
    return HTML.replace("__SHADER__",json.dumps(shader)).replace("__PRESETS__",json.dumps(defaults)).replace("__PRESET__",json.dumps(preset))


HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pixel Clocks — HydraPy</title><style>
:root{color-scheme:dark;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#d8dccc;background:#111412}
*{box-sizing:border-box}body{margin:0;overflow:hidden}canvas{display:block;width:100vw;height:100vh;image-rendering:pixelated}
button,select,input{font:inherit;accent-color:#d8dccc}button,select{color:inherit;background:#111412;border:1px solid #586053;padding:8px 10px;border-radius:0}
button:hover,select:hover{border-color:#d8dccc}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #d8dccc;outline-offset:3px}
#bar{position:fixed;bottom:16px;left:16px;display:flex;gap:6px;align-items:center;padding:8px;background:#111412ed;flex-wrap:wrap;max-width:calc(100vw - 32px)}
#bar strong{font-size:11px;letter-spacing:.14em;margin:0 10px 0 4px}#clock{font-variant-numeric:tabular-nums;min-width:62px;text-align:right;color:#9da895}
#panel{position:fixed;bottom:82px;left:16px;background:#111412f5;border:1px solid #586053;padding:18px;width:min(430px,calc(100vw - 32px));max-height:65vh;overflow:auto}
[hidden]{display:none!important}.row{display:grid;grid-template-columns:108px 1fr 55px;gap:9px;align-items:center;margin:10px 0}.row output{text-align:right;color:#b9c6ad}.row input{width:100%;min-width:0}
#panel p{line-height:1.6;color:#aeb9a3;margin:15px 0 0}#panel a{color:#d8dccc}#error{position:fixed;top:30%;left:10%;right:10%;background:#111412;padding:24px;border:1px solid #d8dccc;line-height:1.7}
@media(max-width:560px){#bar{bottom:8px;left:8px;max-width:calc(100vw - 16px);gap:4px}#bar strong{display:none}#panel{left:8px;bottom:112px}.row{grid-template-columns:96px 1fr 42px}}
</style></head><body>
<canvas id="view" aria-label="Two-color independent pixel oscillator field"></canvas>
<div id="bar"><strong>PIXEL CLOCKS</strong><select id="preset" aria-label="Pattern preset"><option value="saddle">Saddle</option><option value="radial">Radial</option><option value="folded">Folded</option><option value="lattice">Lattice</option></select><button id="pause">Pause</button><button id="restart">Restart</button><button id="toggle" aria-expanded="false" aria-controls="panel">Controls</button><button id="save">Save PNG</button><button id="full">Fullscreen</button><span id="clock" aria-label="Elapsed time">0.0s</span></div>
<section id="panel" hidden aria-label="Oscillator controls">
<div id="sliders"></div>
<div class="row"><label for="seek">Time / s</label><input id="seek" type="range" min="0" max="1800" step=".1" value="0"><output id="seek-value">0.0</output></div>
<p id="resolution"></p>
<p>Each cell runs its own fixed clock. Hold the settings still and watch their phases drift. Scrubbing time or changing controls jumps directly to the corresponding state.</p>
<p>Original HydraPy study inspired by <a href="https://www.memo.tv/projects/2021/pixlflippr/" target="_blank" rel="noopener">Memo Akten’s PixlFlippr description</a>. No feedback or neighboring-cell updates.</p>
<p>Space: pause · R: restart · H: hide/show controls</p></section>
<pre id="error" hidden role="alert"></pre>
<script>
(()=>{'use strict';
const canvas=document.getElementById('view'),presets=__PRESETS__,select=document.getElementById('preset');
select.value=__PRESET__;
let params={...presets[select.value]},running=true,offset=0,epoch=performance.now(),dirty=true;
const clock=()=>offset+(running?(performance.now()-epoch)/1000:0);
const fail=e=>{let box=document.getElementById('error');box.hidden=false;box.textContent='This browser could not run the pixel shader.\n'+e;};
try {
const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
if(!gl)throw Error('WebGL2 is unavailable.');
function compile(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
const vertex=compile(gl.VERTEX_SHADER,'#version 300 es\nin vec2 position;void main(){gl_Position=vec4(position,0.,1.);}');
const fragment=compile(gl.FRAGMENT_SHADER,__SHADER__);
const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
const position=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
const uniforms={};for(const name of ['resolution','time',...Object.keys(params)])uniforms[name]=gl.getUniformLocation(program,name);
function resize(){const dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.floor(innerWidth*dpr));canvas.height=Math.max(1,Math.floor(innerHeight*dpr));gl.viewport(0,0,canvas.width,canvas.height);document.getElementById('resolution').textContent=canvas.width+' × '+canvas.height+' · native canvas pixels';dirty=true;}
addEventListener('resize',resize);resize();
const definitions=[['rate','Rate (Hz)',0,2,.005],['spread','Spread (Hz)',0,6,.01],['density','Density',1,300,.25],['pixel','Cell (px)',1,12,1],['phase','Phase (cyc)',0,1,.001],['threshold','Threshold',-.95,.95,.01]];
for(const [name,label,min,max,step] of definitions){let row=document.createElement('div');row.className='row';const l=document.createElement('label');l.htmlFor=name;l.textContent=label;const input=document.createElement('input');Object.assign(input,{id:name,type:'range',min,max,step,value:params[name]});const output=document.createElement('output');output.id=name+'-value';output.textContent=params[name];row.append(l,input,output);document.getElementById('sliders').append(row);input.addEventListener('input',()=>{params[name]=Number(input.value);output.textContent=input.value;dirty=true;});}
function setTime(t){offset=t;epoch=performance.now();dirty=true;}
function pause(){offset=clock();epoch=performance.now();running=!running;document.getElementById('pause').textContent=running?'Pause':'Play';dirty=true;}
function panel(){const p=document.getElementById('panel');p.hidden=!p.hidden;document.getElementById('toggle').setAttribute('aria-expanded',String(!p.hidden));}
select.addEventListener('change',()=>{params={...presets[select.value]};for(const [name] of definitions){document.getElementById(name).value=params[name];document.getElementById(name+'-value').textContent=params[name];}setTime(0);});
document.getElementById('pause').onclick=pause;document.getElementById('restart').onclick=()=>setTime(0);document.getElementById('toggle').onclick=panel;
document.getElementById('seek').oninput=e=>setTime(Number(e.target.value));
document.getElementById('full').onclick=()=>{const p=document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();if(p)p.catch(e=>fail(e.message));};
document.getElementById('save').onclick=()=>{draw();canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='Pixel_Clocks.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});};
addEventListener('keydown',e=>{if(/INPUT|SELECT|BUTTON/.test(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();pause();}if(e.key.toLowerCase()==='r')setTime(0);if(e.key.toLowerCase()==='h')panel();});
function draw(){const t=clock();gl.uniform2f(uniforms.resolution,canvas.width,canvas.height);gl.uniform1f(uniforms.time,t);for(const [name,value] of Object.entries(params)){if(name==='geometry')gl.uniform1i(uniforms[name],value);else gl.uniform1f(uniforms[name],value);}gl.drawArrays(gl.TRIANGLES,0,3);document.getElementById('clock').textContent=t.toFixed(1)+'s';document.getElementById('seek-value').textContent=t.toFixed(1);if(document.activeElement.id!=='seek')document.getElementById('seek').value=t;dirty=false;}
function frame(){if(running||dirty)draw();requestAnimationFrame(frame);}frame();
}catch(e){fail(e.message||String(e));}
})();
</script></body></html>'''


def frames(size, fps, seconds, preset, report):
    ctx=_create_context()
    print("Renderer:",ctx.info["GL_RENDERER"],flush=True)
    renderer=GLRenderer(make_patch(size,preset),ctx=ctx,fps=fps)
    report.update(renderer=ctx.info["GL_RENDERER"],source_passes=len(renderer.programs))
    started=last=time.monotonic()
    try:
        for frame in range(math.ceil(seconds*fps)):
            renderer.step(frame/fps)
            yield renderer.read()
            now=time.monotonic()
            if now-last>10:
                print(f"{frame/fps:.1f}/{seconds:.1f}s | {(frame+1)/(now-started):.1f} fps",flush=True)
                last=now
        report["render_fps"]=math.ceil(seconds*fps)/(time.monotonic()-started)
    finally:
        renderer.release();ctx.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("Pixel_Clocks.mp4"))
    p.add_argument("--html",type=Path)
    p.add_argument("--seconds",type=float,default=90)
    p.add_argument("--size",default="1920x1080")
    p.add_argument("--fps",type=int,default=30)
    p.add_argument("--preset",choices=PRESETS,default="saddle")
    p.add_argument("--audio",type=Path)
    p.add_argument("--audio-start",type=float,default=0)
    a=p.parse_args()
    try:size=tuple(map(int,a.size.lower().split("x")))
    except ValueError:p.error("size must be WIDTHxHEIGHT")
    if len(size)!=2 or min(size)<64 or any(x%2 for x in size):p.error("use even dimensions >=64")
    if not math.isfinite(a.seconds) or a.seconds<=0 or a.fps<1:p.error("invalid duration/fps")
    if not math.isfinite(a.audio_start) or a.audio_start<0:p.error("invalid audio start")
    if a.audio:
        duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(a.audio)]))
        if a.audio_start+a.seconds>duration+.001:p.error("the requested excerpt exceeds the audio duration")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(preset=a.preset,controls=dict(PRESETS[a.preset],phase=.137,threshold=0),
                size=size,fps=a.fps,seconds=a.seconds,autonomous=True,
                audio=a.audio.name if a.audio else None,audio_start=a.audio_start if a.audio else None)
    with tempfile.TemporaryDirectory(prefix="pixel-clocks-",dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        H.save_mp4(frames(size,a.fps,a.seconds,a.preset,report),tmp/"video.mp4",fps=a.fps,
                   quality=9,encoder="auto",on_encoder=lambda e:print("Encoder:",e.name,flush=True))
        if a.audio:
            subprocess.run(["ffmpeg","-v","error","-i",str(tmp/"video.mp4"),"-ss",str(a.audio_start),
                            "-t",str(a.seconds),"-i",str(a.audio),"-map","0:v:0","-map","1:a:0",
                            "-c:v","copy","-c:a","aac","-b:a","256k","-movflags","+faststart",
                            "-metadata",f"title={a.audio.stem} — Pixel Clocks",
                            str(tmp/"complete.mp4")],check=True)
            (tmp/"complete.mp4").replace(a.out)
        else:(tmp/"video.mp4").replace(a.out)
    if a.html:
        a.html.parent.mkdir(parents=True,exist_ok=True);a.html.write_text(live_html(a.preset))
    a.out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.out,flush=True)


if __name__=="__main__":main()
