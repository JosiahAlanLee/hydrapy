"""Record genuine Gorgon and Rack interfaces on a disposable Linux desktop."""
import json
import os
from pathlib import Path
import shutil
import subprocess as sp
import sys
import threading
import time

ROOT = Path.cwd()
HERE = ROOT / 'tools/gorgon_demo'
OUT = ROOT / 'demo_capture'
OUT.mkdir(exist_ok=True)
RACK = Path('/tmp/Rack2Free')
USER = OUT / 'rack-user'

def run(*args, **kwargs):
    return sp.run(args, check=True, **kwargs)

def prepare():
    plugin = ROOT / '.demo-dsp/vcv/gorgon'
    shutil.copy(HERE/'Capture.cpp', plugin/'src/Capture.cpp')
    p = plugin/'src/plugin.cpp'
    p.write_text(p.read_text().replace('void init(', 'extern Model* modelDemoCapture;\n\nvoid init(').replace('p->addModel(modelGorgon);','p->addModel(modelGorgon);\n\tp->addModel(modelDemoCapture);'))
    p = plugin/'Makefile'
    p.write_text(p.read_text().replace('SOURCES += src/plugin.cpp src/Gorgon.cpp','SOURCES += src/plugin.cpp src/Gorgon.cpp src/Capture.cpp'))
    p = plugin/'plugin.json'; manifest=json.loads(p.read_text())
    manifest['modules'].append(dict(slug='DemoCapture',name='Recording fixture',hidden=True,tags=[]))
    p.write_text(json.dumps(manifest))
    run('make','-C',str(plugin),'-j2','RACK_DIR=/tmp/Rack-SDK')
    dest=USER/'plugins-lin-x64/GorgonBridge';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy(plugin/'plugin.so',dest/'plugin.so');shutil.copy(plugin/'plugin.json',dest/'plugin.json')
    shutil.copytree(plugin/'res',dest/'res',dirs_exist_ok=True)
    shutil.copy(next(RACK.glob('Fundamental*.vcvplugin')),USER/'plugins-lin-x64')
    (USER/'settings.json').write_text(json.dumps(dict(windowSize=[1600,900],windowPos=[0,0],sampleRate=48000,threadCount=1,frameRateLimit=24,showTipsOnLaunch=False,autoCheckUpdates=False,preferDarkPanels=True,cableOpacity=.55,tooltips=False,rackBrightness=.9,autosaveInterval=10)))
    modules=[];cables=[]
    def mod(i,slug,x,params=None,data=None,plugin='Fundamental',y=0):
        m=dict(id=i,plugin=plugin,model=slug,version='2.0.0',pos=[x,y],params=[dict(id=k,value=v) for k,v in (params or {}).items()])
        if data is not None:m['data']=data
        modules.append(m)
    mod(1,'Gorgon',0,data={'bindings':['Hue','__none__','__none__','__none__'],'pitchZero':60},plugin='GorgonBridge')
    mod(2,'VCO',12,{2:0})
    mod(3,'ADSR',21,{0:.09,1:.4,2:.13,3:.35})
    mod(4,'VCA-1',30,{0:.7,1:1})
    mod(5,'VCA-1',36,{0:.75,1:1})
    mod(6,'Scope',42,{0:0,1:-5,2:0,3:-5,4:0,7:1})
    mod(7,'LFO',55,{0:1,2:-3.0})
    mod(99,'DemoCapture',0,plugin='GorgonBridge',y=2)
    colors=['#8fdcc4','#e7b375','#97bde0','#c4a3c6']
    def cable(a,ao,b,bi,col=0):cables.append(dict(id=len(cables)+1,outputModuleId=a,outputId=ao,inputModuleId=b,inputId=bi,color=colors[col]))
    cable(1,12,2,0,1);cable(1,11,3,4);cable(2,1,4,1,1);cable(3,0,4,0);cable(4,0,5,1,1);cable(1,3,5,0,3)
    cable(1,0,6,0);cable(1,8,6,1,1);cable(7,0,1,2,3)
    cable(5,0,99,0,1)
    for out,inp in [(0,1),(1,2),(2,3),(4,4),(12,5),(3,6),(8,7)]:cable(1,out,99,inp)
    cable(7,0,99,8,3);cable(99,0,1,0);cable(99,1,1,1)
    patch=dict(version='2.6.4',zoom=1.5,gridOffset=[-3,-.12],modules=modules,cables=cables)
    (OUT/'Gorgon-Demo.vcv').write_text(json.dumps(patch,indent=2))
    print('Prepared native Rack patch',flush=True)

def capture():
    from hydrapy.studio import Session,make_server,advertise,withdraw
    from playwright.sync_api import sync_playwright
    import mss
    session=Session(ROOT,'tools/gorgon_demo/visual.py',backend='gl',size=(640,360),fps=30)
    server,token=make_server(session,0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    advertised=advertise(server.server_port,token,ROOT)
    session.call('tracker_transport',enabled=True,playing=False,beat=0)
    env=dict(os.environ,DISPLAY=':99',LD_LIBRARY_PATH=str(RACK),GORGON_CAPTURE=str(OUT/'rack-audio'))
    rack=sp.Popen([str(RACK/'Rack'),'-u',str(USER),str(OUT/'Gorgon-Demo.vcv')],cwd=RACK,env=env,stdout=open(OUT/'rack.stdout','w'),stderr=sp.STDOUT)
    time.sleep(8)
    run('xdotool','search','--name','VCV Rack','windowsize','1600','900','windowmove','0','0',env=env)
    time.sleep(2)
    with mss.mss(display=':99') as grab:grab.shot(output=str(OUT/'rack-initial.png'))
    if rack.poll() is not None:raise RuntimeError('Rack exited')
    videos=[];rec=[];meta={'events':[],'contexts':{}}
    with sync_playwright() as pw:
      browser=pw.chromium.launch(headless=True,ignore_default_args=['--mute-audio'],args=['--autoplay-policy=no-user-gesture-required'])
      pages={};contexts={}
      for name,size in [('tracker',(1440,1000)),('program',(960,540)),('aux',(960,540))]:
        context=browser.new_context(viewport=dict(width=size[0],height=size[1]),record_video_dir=str(OUT/name),record_video_size=dict(width=size[0],height=size[1]))
        page=context.new_page();page.set_default_timeout(12000)
        url=f'http://127.0.0.1:{server.server_port}/tracker/'
        if name!='tracker':url+=f'?output={name}'
        page.goto(url+'#'+token)
        page.wait_for_function('gorgonPreview.frames>=1')
        if name!='tracker':page.locator('#outputDismiss').click()
        pages[name]=page;contexts[name]=context
      page=pages['tracker']
      # A cursor is useful in a screencast and does not alter application behavior.
      page.evaluate('''() => {const e=document.createElement('div');e.id='demo-pointer';e.style='position:fixed;width:24px;height:24px;border:2px solid #f0c088;border-radius:50%;pointer-events:none;z-index:99999;left:-50px';document.body.append(e);addEventListener('mousemove',v=>{e.style.left=(v.clientX-12)+'px';e.style.top=(v.clientY-12)+'px'});}''')
      rack_start=time.monotonic()
      rec.append(sp.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','x11grab','-framerate','24','-video_size','1600x900','-i',':99','-c:v','libx264','-preset','ultrafast','-crf','19',str(OUT/'rack.mp4')],env=env,stdin=sp.PIPE))
      sound_start=time.monotonic()
      rec.append(sp.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','pulse','-i','demo.monitor','-ac','2','-ar','48000',str(OUT/'browser.wav')],stdin=sp.PIPE))
      t0=time.monotonic();meta.update(t0=t0,rack_start=rack_start,sound_start=sound_start)
      def at(t,name,action=None):
        while time.monotonic()-t0<t:page.wait_for_timeout(50)
        print(f'{time.monotonic()-t0:.2f}: {name}',flush=True)
        meta['events'].append(dict(time=time.monotonic()-t0,name=name))
        if action:action()
      def click(sel):page.locator(sel).click()
      def select(sel,value):page.locator(sel).select_option(value)
      at(1,'Enable browser synth',lambda:click('#audioBtn'))
      at(2,'Play score',lambda:click('#playBtn'))
      at(7,'Four patterns; one clock',lambda:page.locator('#patterns').hover())
      at(12,'Loop DRIFT',lambda:(click('[data-pattern="0"]'),click('#loopBtn')))
      at(16,'Inspect pitched note',lambda:(click('#followBtn'),click('[data-row="6"][data-col="1"]')))
      at(19,'Write E4 at velocity 110',lambda:(page.locator('#noteInput').fill('E-4'),page.locator('#velocityInput').fill('110'),click('#insertNoteBtn')))
      at(22,'Audition edited note',lambda:click('#auditionBtn'))
      at(25,'Transpose to G minor',lambda:select('#key','7'))
      at(28,'Return to D minor',lambda:select('#key','2'))
      at(30,'Roll automation',lambda:click('[data-curve="roll"]'))
      at(34,'Feedback automation',lambda:click('[data-curve="feedback"]'))
      at(38,'Dolly Z automation',lambda:click('[data-curve="z"]'))
      at(41,'Hold interpolation',lambda:click('[data-ease="hold"]'))
      at(44,'Smooth interpolation',lambda:click('[data-ease="smooth"]'))
      at(47,'Independent edge output',lambda:select('#auxMode','o2'))
      at(52,'Blackout monitors',lambda:click('#blackoutBtn'))
      at(54,'Restore monitors',lambda:click('#blackoutBtn'))
      at(57,'Switch soundtrack to native Rack',lambda:click('#audioBtn'))
      at(59,'Show full arrangement',lambda:(click('#loopBtn'),click('#followBtn')))
      at(65,'Polyphonic pitch, gates, velocity')
      at(73,'Rack LFO returns to Hue')
      at(81,'Pause from Rack PLAY input',lambda:(OUT/'rack-audio.command').write_text('1'))
      at(84,'Resume from Rack PLAY input',lambda:(OUT/'rack-audio.command').write_text('11'))
      at(88,'Reset from Rack RESET input',lambda:(OUT/'rack-audio.command').write_text('12'))
      at(93,'Save the score',lambda:click('#saveBtn'))
      at(97,'Final performance')
      at(105,'End')
      for name in pages:
        pages[name].screenshot(path=str(OUT/(name+'-final.png')))
        videos.append((name,pages[name].video))
      for name,context in contexts.items():
        meta['contexts'][name]={'end':time.monotonic()}
        context.close()
      for name,video in videos:
        shutil.move(str(video.path()),OUT/(name+'.webm'))
      browser.close()
    for p in rec:
      p.communicate(b'q',timeout=30)
    rack.terminate()
    try:rack.wait(timeout=15)
    except sp.TimeoutExpired:rack.kill();rack.wait()
    withdraw(advertised,token);server.shutdown();server.server_close();session.close()
    (OUT/'timeline.json').write_text(json.dumps(meta,indent=2))
    (OUT/'score.json').write_text(json.dumps(session.tracker.document,indent=2))
    # Crop the PCM capture to the demonstration interval. Rack's native fallback
    # engine supplies audio-rate processing without a physical sound device.
    native=json.loads((OUT/'rack-audio.json').read_text())
    run('ffmpeg','-hide_banner','-loglevel','error','-y','-f','f32le','-ar',str(int(native['sample_rate'])),'-ac','1','-i',str(OUT/'rack-audio.f32'),'-ss',str(t0-native['start']),'-t','106',str(OUT/'rack.wav'))
    (OUT/'rack-audio.f32').unlink()
    print('CAPTURE COMPLETE',flush=True)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:capture()
