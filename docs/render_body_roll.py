"""Print the CC0 Body Roll scene on the Bell Riots half-time clock.

blender -b Dancer_Overlay.blend -t 6 -P docs/render_body_roll.py -- --output /tmp/body-roll
The input scene is the self-contained Mesh2Motion mannequin kit; provenance is
in bell-riots-dancer.md. Original action keys remain available in the scene.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',default='1920x1080')
    p.add_argument('--samples',type=int,default=8)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    width,height=map(int,a.size.split('x'))
    a.output.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    rig=next(o for o in scene.objects if o.type=='ARMATURE')
    source=bpy.data.actions['Dance Body Roll']
    action=source.copy();action.name='Body Roll - 68.5 BPM - 16 half-time beats'
    first,last=source.frame_range
    frames=480;scale=frames/(last-first)
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        for co in (key.co,key.handle_left,key.handle_right):
                            co.x=1+(co.x-first)*scale
                    curve.update()
    action.use_fake_user=True
    rig.animation_data.action=action
    rig.animation_data.action_slot=action.slots[0]
    for track in rig.animation_data.nla_tracks:track.mute=True
    scene.frame_start=1;scene.frame_end=frames;scene.frame_set(1)
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
        full_beats_per_cycle=32,frames=frames,fps_numerator=137,fps_denominator=4,
        cycle_seconds=1920/137,source_frames=[float(first),float(last)],
        speed_relative_to_native=(last-first)/24/(1920/137)),indent=2)+'\n')
    out=a.output/'frames';out.mkdir(exist_ok=True)
    for frame in range(1,frames+1):
        scene.frame_set(frame);scene.render.filepath=str(out/f'{frame:04}.png')
        bpy.ops.render.render(write_still=True)
    print('BODY_ROLL_PRINT_DONE',flush=True)


if __name__=='__main__':main()
