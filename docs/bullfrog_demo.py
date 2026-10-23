"""Render the complete Bullfrog demo with two live scenes during every transition.

python docs/bullfrog_demo.py /path/to/Bullfrog.flac -o bullfrog.mp4
--seconds 20 makes a short excerpt; --contact writes a scene contact sheet.
Music is supplied locally and is never copied into the repository.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'patches'))
from bullfrog import make_scene, track_audio, SCENES
from hydrapy import save_mp4
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.drawing import find_font

FRACTIONS = np.array([0.,.13,.27,.41,.57,.72,.86,1.])


def smooth(x):
    x = np.clip(x,0,1)
    return x*x*(3-2*x)


def frames(audio, size=(1280,720), fps=24, seconds=None, contact=None, preview_only=False):
    duration = audio.duration()
    length = min(seconds if seconds is not None else duration,duration)
    boundaries = FRACTIONS*duration
    transition = 10.
    ctx = _create_context()
    renderers, snapshots = {}, {}
    font = find_font('DejaVuSans',max(14,round(size[1]*.022)))
    title_font = find_font('DejaVuSans',max(30,round(size[1]*.08)))
    small_font = find_font('DejaVuSans',max(11,round(size[1]*.016)))
    wall_start = time.monotonic()
    last_chapter = -1
    def renderer(index,t):
        if index not in renderers:
            r = GLRenderer(make_scene(index,audio),size=size,fps=fps,ctx=ctx)
            # Warm stateful sources before they enter the dissolve, without advancing the music.
            for k in range(round(2*fps)):
                r.step(max(0,t-2)+k/fps)
            renderers[index]=r
        return renderers[index]
    count = max(1,round(length*fps))
    try:
        times = [(i/fps,i) for i in range(count)] if not preview_only else [(float((boundaries[i]+boundaries[i+1])/2),i) for i in range(len(SCENES))]
        for t,i in times:
            chapter = min(len(SCENES)-1,max(0,int(np.searchsorted(boundaries,t,side='right')-1)))
            active = [chapter]
            mix = 0.
            if chapter and t < boundaries[chapter]+transition:
                active=[chapter-1,chapter]
                mix=float(smooth((t-boundaries[chapter])/transition))
            images=[]
            for j in active:
                r=renderer(j,t)
                r.step(t)
                images.append(r.read())
            image=images[-1]
            if len(images)==2:
                image=np.clip(images[0].astype(np.float32)*(1-mix)+image.astype(np.float32)*mix,0,255).astype(np.uint8)
            for j in list(renderers):
                if j not in active:
                    renderers.pop(j).release()
            fade=float(smooth(t/2.5)*smooth((duration-t)/5))
            if fade<1:
                image=(image.astype(np.float32)*fade).astype(np.uint8)
            im=Image.fromarray(image)
            draw=ImageDraw.Draw(im,'RGBA')
            x,y=round(size[0]*.045),round(size[1]*.075)
            if t<10:
                opacity=int(255*smooth(t/1.5)*smooth((10-t)/3))
                draw.text((x,y),'ANCHOR RITE  /  PINNED',font=font,fill=(224,232,216,opacity))
                draw.text((x,y+font.size*1.8),'BULLFROG',font=title_font,fill=(240,245,231,opacity))
            local=t-boundaries[chapter]
            if t>12 and local<18:
                opacity=int(210*smooth((local-8)/2)*smooth((18-local)/3))
                draw.text((x,size[1]-round(size[1]*.09)),f'{chapter+1:02d}  {SCENES[chapter][0]}',font=font,fill=(235,241,225,opacity))
                draw.text((x,size[1]-round(size[1]*.05)),SCENES[chapter][1],font=small_font,fill=(180,202,185,opacity))
            if preview_only or (chapter not in snapshots and local>20):
                snapshots[chapter]=im.copy()
            if chapter!=last_chapter:
                print(f'{t:6.1f}s / {duration:.1f}s — {SCENES[chapter][0]}',flush=True)
                last_chapter=chapter
            if not preview_only and i and i%(24*fps)==0:
                elapsed=time.monotonic()-wall_start
                print(f'  rendered {t:.0f}s, {i/elapsed:.1f} frames/s',flush=True)
            yield np.asarray(im)
        if contact:
            thumb=(480,270)
            sheet=Image.new('RGB',(thumb[0]*2, (thumb[1]+38)*4),(14,20,17))
            draw=ImageDraw.Draw(sheet)
            for j,snapshot in snapshots.items():
                snapshot=snapshot.resize(thumb)
                x,y=(j%2)*thumb[0],(j//2)*(thumb[1]+38)
                sheet.paste(snapshot,(x,y))
                draw.text((x+12,y+thumb[1]+9),f'{j+1:02d} / {SCENES[j][0]}',font=find_font('DejaVuSans',16),fill=(210,227,209))
            sheet.save(contact)
    finally:
        for r in renderers.values():
            r.release()
        ctx.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio')
    parser.add_argument('-o','--out',default='bullfrog.mp4')
    parser.add_argument('--size',default='1280x720')
    parser.add_argument('--fps',type=float,default=24)
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--encoder',default='auto')
    parser.add_argument('--contact')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    size=tuple(map(int,args.size.lower().split('x')))
    if len(size)!=2 or min(size)<64 or args.fps<=0 or not math.isfinite(args.fps):
        parser.error('Use a size of at least 64×64 and a positive fps')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be a positive finite number')
    audio=track_audio(args.audio)
    if args.preview_only:
        list(frames(audio,size,args.fps,contact=args.contact,preview_only=True))
    else:
        save_mp4(frames(audio,size,args.fps,args.seconds,args.contact),args.out,fps=args.fps,
                 audio=args.audio,encoder=args.encoder,quality=6,
                 on_encoder=lambda e:print(f'Encoder: {e.name} ({"hardware" if e.hardware else "software"})',flush=True))
        print(args.out,flush=True)
    cue_path=Path(args.out).with_suffix('.cues.json')
    cue_path.write_text(json.dumps(dict(track=Path(args.audio).name,duration=audio.duration(),fps=args.fps,
        size=size,transition_seconds=10,scenes=[dict(start=round(float(FRACTIONS[i]*audio.duration()),3),name=title,technique=technique)
        for i,(title,technique) in enumerate(SCENES)]),indent=2))


if __name__=='__main__':
    main()
