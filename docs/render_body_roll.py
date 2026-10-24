"""Print the CC0 Body Roll scene on the Bell Riots half-time clock.

blender -b Dancer_Overlay.blend -t 6 -P docs/render_body_roll.py -- --output /tmp/body-roll [--warp]
The input scene is the self-contained Mesh2Motion mannequin kit; provenance is
in bell-riots-dancer.md. Original action keys remain available in the scene.

By default the action is stretched linearly over the 480-frame cycle, so its
holds fall wherever the stretch puts them. --warp measures the holds (minima of
mean bone speed) and uses a monotonic piecewise-linear time map to move each one
onto a half-time beat (a multiple of 30 frames), keeping 480 frames per cycle.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

FRAMES=480
HALF_BEAT=30


def find_holds(speed,step,cycle,window,max_rel=.7):
    """Local speed minima over one looping cycle as (source offset, speed / mean).

    ``speed`` is sampled every ``step`` source frames from offset 0; ``window``
    is the half-width, in source frames, a minimum must hold against.
    """
    speed=np.asarray(speed,dtype=float);n=len(speed)
    kernel=np.exp(-.5*(np.arange(-6,7)/2.)**2);kernel/=kernel.sum()
    smooth=np.real(np.fft.ifft(np.fft.fft(speed)*np.fft.fft(np.roll(np.r_[kernel,np.zeros(n-13)],-6))))
    w=max(1,int(round(window/step)));rel=smooth/smooth.mean();holds=[]
    for i in range(n):
        # argmin keeps the first of equal samples: a minimum falling exactly between
        # two samples is one hold, not two.
        near=np.arange(i-w,i+w+1)%n
        if rel[i]<max_rel and near[np.argmin(rel[near])]==i:
            holds.append(((i+.5)*step%cycle,float(rel[i])))
    return holds


def hold_map(holds,cycle,frames=FRAMES,step=HALF_BEAT,max_ratio=2.):
    """Breakpoints (source offsets, output offsets) placing holds on multiples of ``step``.

    The strongest holds are placed first, each on its nearest free multiple, and a
    hold is skipped if it would land on the loop point, cross a placed hold, or make
    any segment faster or slower than ``max_ratio`` times the linear stretch.
    """
    scale=frames/cycle;placed={0.:0.,float(cycle):float(frames)}
    for src,_ in sorted(holds,key=lambda h:h[1]):
        target=round(src*scale/step)*step
        if not 0<target<frames or target in placed.values():continue
        trial=dict(placed);trial[float(src)]=float(target)
        xs=np.array(sorted(trial));ys=np.array([trial[x] for x in xs])
        ratio=np.diff(ys)/np.diff(xs)/scale
        if np.all(np.diff(ys)>0) and np.all((1/max_ratio<=ratio)&(ratio<=max_ratio)):placed=trial
    xs=np.array(sorted(placed));return xs,np.array([placed[x] for x in xs])


def apply_map(x,xs,ys):
    """Piecewise-linear map, extended past both ends with the end slopes (for handles)."""
    if x<xs[0]:return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
    if x>xs[-1]:return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
    return float(np.interp(x,xs,ys))


def bone_speed(scene,rig,first,last,step=.25):
    """Mean world-space speed of every pose bone's head and tail, per source frame."""
    points=[]
    for x in np.arange(first,last+step/2,step):
        f=int(np.floor(x));scene.frame_set(f,subframe=float(x-f));m=rig.matrix_world
        points.append([tuple(m@b.head)+tuple(m@b.tail) for b in rig.pose.bones])
    points=np.array(points).reshape(len(points),-1,3)
    return np.linalg.norm(np.diff(points,axis=0),axis=2).mean(1)/step


def main():
    import bpy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',default='1920x1080')
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--warp',action='store_true',help='move the action holds onto half-time beats')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    width,height=map(int,a.size.split('x'))
    a.output.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    rig=next(o for o in scene.objects if o.type=='ARMATURE')
    source=bpy.data.actions['Dance Body Roll']
    first,last=source.frame_range;cycle=last-first;scale=FRAMES/cycle
    xs,ys=np.array([0.,cycle]),np.array([0.,float(FRAMES)]);holds=[]
    if a.warp:
        rig.animation_data.action=source;rig.animation_data.action_slot=source.slots[0]
        for track in rig.animation_data.nla_tracks:track.mute=True
        # Minima must hold for ±12 output frames, the window bell_riots_checks.py uses.
        holds=find_holds(bone_speed(scene,rig,first,last),.25,cycle,12/scale)
        xs,ys=hold_map(holds,cycle)
    action=source.copy();action.name='Body Roll - 68.5 BPM - 16 half-time beats'
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        for co in (key.co,key.handle_left,key.handle_right):
                            co.x=1+apply_map(co.x-first,xs,ys)
                    curve.update()
    action.use_fake_user=True
    rig.animation_data.action=action
    rig.animation_data.action_slot=action.slots[0]
    for track in rig.animation_data.nla_tracks:track.mute=True
    scene.frame_start=1;scene.frame_end=FRAMES;scene.frame_set(1)
    # 137/4 fps yields exactly 30 frames per half-time beat and 480 per cycle.
    # The full music video resamples this phase continuously onto its 30 fps grid.
    scene.render.fps=137;scene.render.fps_base=4
    scene.render.resolution_x=width;scene.render.resolution_y=height
    scene.render.resolution_percentage=100
    scene.render.engine='CYCLES';scene.cycles.device='CPU'
    scene.cycles.samples=a.samples;scene.cycles.use_denoising=False
    scene.render.film_transparent=True
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene.render.filepath='//frames/'
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'Body_Roll_Half_Time.blend'))
    (a.output/'timing.json').write_text(json.dumps(dict(
        track_bpm=137,half_bpm=68.5,half_beats_per_cycle=16,
        full_beats_per_cycle=32,frames=FRAMES,fps_numerator=137,fps_denominator=4,
        cycle_seconds=1920/137,source_frames=[float(first),float(last)],
        speed_relative_to_native=(last-first)/24/(1920/137),
        warp=dict(holds=[[round(first+h,3),round(r,3)] for h,r in holds],
                  source_frames=[round(first+x,3) for x in xs],output_frames=[1+y for y in ys])),indent=2)+'\n')
    out=a.output/'frames';out.mkdir(exist_ok=True)
    for frame in range(1,FRAMES+1):
        scene.frame_set(frame);scene.render.filepath=str(out/f'{frame:04}.png')
        bpy.ops.render.render(write_still=True)
    print('BODY_ROLL_PRINT_DONE',flush=True)


if __name__=='__main__':main()
