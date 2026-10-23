"""Bullfrog — seven audio-reactive scenes for Anchor Rite's track.

Set HYDRAPY_TRACK to your local audio file; HYDRAPY_DEMO_SCENE selects 0–6.
Without a track, slow synthetic signals let you audition the visual controls.
The full-track render and moving crossfades live in docs/bullfrog_demo.py.
"""
from pathlib import Path
import os
from hydrapy import *

STUDIO = {'title': 'Bullfrog / scene instrument', 'tags': ['Sound & rhythm', 'Paint & texture', 'Geometry'],
          'preview': 'docs/studio/bullfrog.jpg'}
SCENES = [
    ('PRESSURE', 'Oscillators / feedback'),
    ('CURRENT', 'Mathematical marbling'),
    ('GROWTH', 'Neural cellular automata'),
    ('NETWORK', 'Physarum trails'),
    ('CONTOUR', 'Morphing distance fields'),
    ('SURFACE', 'Normal maps / moving light'),
    ('RETURN', 'Orbiting cells / oscillator interference'),
]


def track_audio(path):
    return Audio.from_file(str(path), smooth=.65, scale=5, cutoff=.5)


def make_scene(index, audio=None):
    index = int(index)
    if not 0 <= index < len(SCENES):
        raise ValueError('scene must be 0–6')
    p = Patch()
    motion = Param('Motion', 1, min=.2, max=2, step=.01, patch=p)
    response = Param('Audio response', 1, min=0, max=2, step=.01, patch=p)
    colour = Param('Colour', .1, min=0, max=1, step=.01, patch=p)
    detail = Param('Detail', 1, min=.3, max=2, step=.01, patch=p)
    low = (audio.fft[0]*1.2).clip(0,1) if audio else Lfo(.37,lo=.2,hi=.8)
    mid = (audio.fft[1]*2).clip(0,1) if audio else Lfo(.19,lo=.1,hi=.7)
    high = (audio.fft[3]*3).clip(0,1) if audio else Lfo(.71,lo=0,hi=.6)
    low, mid, high = low*response, mid*response, high*response
    slow = Time()*motion
    if index == 0:
        base = osc(7*detail+low*3,.027,.5).rotate(slow*.014).modulate(noise(2.2,.045), .08+mid*.08)
        base = base.kaleid(3).palette(.02,.15,.28,colour+slow*.007,.8).color(.72,.88,.8)
        base.out(p.o0,patch=p)
        src(p.o1).scale(1.002).rotate(.001).blend(src(p.o0),.12+high*.06).out(p.o1,patch=p)
        p.render(p.o1)
    elif index == 1:
        base = osc(12*detail+low*3,.025,.5).rotate(.4)
        base = base.tine(.8, .05, (slow*.13).sin()*.15+mid*.05,.025,.13)
        base = base.swirl(.42,.5,(slow*.08).sin()*.18,.3,.06).inkdrop(.6,.53,.09+low*.035)
        base.palette(.03,.17,.3,colour+slow*.004,.8).color(.95,.8,.6).out(p.o0,patch=p)
    elif index == 2:
        model = Path(__file__).parent/'models'/'gravel.npz'
        grown = NCA(str(model),size=(128,72),speed=30,rate=(motion*.5).clip(.05,.95),perturb=.001+high*.004,seed=31)
        src(grown).modulate(noise(1.8,.025),mid*.025).scale(1.1/detail).contrast(1.2).saturate(.5).color(.62,.92,.78).hue(colour*.25).out(p.o0,patch=p)
    elif index == 3:
        trails = Physarum(size=(256,144),agents=(128,64),sensor_angle=.45+mid*.3,
                          sensor_distance=(9+low*10)*detail,turn=.35,step=1.1*motion,decay=.95,
                          exposure=.7,colour=(.7,1,.75),speed=48,seed=17)
        src(trails).scale(1.05).hue(colour*.2).contrast(1.2).out(p.o0,patch=p)
    elif index == 4:
        field = sdfMetaballs(5,.14+low*.025,.34,.13,speed=.22,seed=8)
        cut = sdfCircle(.13, .5+(slow*.1).sin()*.16,.5)
        field.sdfSubtract(cut).sdfBands(16*detail,.24+high*.05,slow*.18).palette(.04,.19,.35,colour,.8).color(.9,.82,.7).out(p.o0,patch=p)
    elif index == 5:
        height = fbm(3.2*detail, octaves=3).add(veins(2.5,speed=.06),.28).bake()
        normals = height.normalMap(.04+low*.025)
        light = normals.relight(azimuth=slow*.14,elevation=.65+mid*.2,ambient=.26,diffuse=.85,specular=.55,shininess=32)
        environment = osc(3,.035,.5).rotate(.4).palette(.04,.18,.3,colour).bake()
        environment.envMap(normals,slow*.006).blend(solid(.35,.55,.42),.65).mult(light).contrast(1.1).out(p.o0,patch=p)
    else:
        cells = orbitVoronoi(points=8,rings=3,radius=.4,speed=.07,phase=slow*.01,width=.006,mode=2)
        base = cells.sdfBands(14*detail,.22,slow*.12).palette(.02,.18,.32,colour,.8)
        base.blend(osc(9+low*5,.025,.6).polar(1.2).kaleid(5).palette(.04,.2,.35,colour),.25+mid*.2).color(.7,.9,.82).out(p.o0,patch=p)
    return p


if __name__ in ('__main__','__hydrapy_patch__','<run_path>'):
    audio_path = os.environ.get('HYDRAPY_TRACK')
    audio = track_audio(audio_path) if audio_path else None
    demo_patch = make_scene(int(os.environ.get('HYDRAPY_DEMO_SCENE','0')), audio)
