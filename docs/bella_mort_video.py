"""Bella Mort — one harmonograph thread, spun, measured and cut.

Anchor Rite's opening track, after the three Fates. Atropa belladonna is named
for Atropos, who cuts the thread; "bella donna" for the dilated pupils it gave.

* Clotho (spinning): one bone-white thread from a damped harmonograph whose
  pendulums are locked to the measured beat and re-struck every 32-beat phrase.
* Lachesis (measuring): the thread is repeated five times round the centre,
  fanning out into a belladonna flower as the track opens up.
* Atropos (cutting): the flower becomes an iris and the pupil dilates; in the
  last phrase the figure is pulled into one taut, vibrating thread, which is cut
  on the frame where the recording stops. The only added audio is that cut.

The pen never lifts: phrase changes cross-fade the old and new pendulums over two
beats. Ink is laid as Gaussian splats spaced along the path's arc length, weighted
partly by the time the pen spends there, so turnarounds glow as in a long exposure.

    python docs/bella_mort_video.py "01 Bella Mort.flac" -o Bella_Mort.mp4 \
      --cache bella-analysis.npz [--offset 140 --seconds 20] [--stills 30,200,400]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hydrapy as H
from hydrapy.backend_gl import _create_context
from hydrapy.drawing import find_font
from tensor_junglist_video import analyse

PHRASE_BEATS=32
PHRASE0=15        # beat index of the first phrase downbeat (measured: section changes land on 15 mod 32)
TAIL=3.6          # seconds after the cut: recoil, then black
IRIS=.44          # iris radius in frame widths
ACTS=dict(lachesis=10,atropos=23,taut=32)   # first phrase of each act
BONE=np.array([1.,.93,.82])
TIERS=[[(1,1)],[(1,2),(2,1),(1,1)],[(2,3),(3,2),(3,4),(4,3)],[(3,4),(4,5),(3,5),(5,4),(2,5)]]


def smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3-2*x)


def find_stop(path):
    """The recording's hard stop, in seconds.

    The end of sound is the end of the last 5 ms block above -45 dBFS. If a hit starts
    within the 120 ms before it (a block at least 9 dB above the one before), the stop
    is that hit, so the cut lands on the final strike rather than after its decay.
    """
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar','44100','-f','f32le','-'],
                       capture_output=True,check=True).stdout
    x=np.frombuffer(raw,np.float32);n=220;blocks=x[:len(x)//n*n].reshape(-1,n)
    level=20*np.log10(np.sqrt((blocks**2).mean(1))+1e-9)
    end=int(np.nonzero(level>-45)[0][-1])+1
    for i in range(end-1,max(1,end-24),-1):
        if level[i]-level[i-1]>=9:return i*n/44100
    return end*n/44100


class MortScore:
    def __init__(self,data,stop):
        self.beats=np.asarray(data['beats'],float);self.period=float(np.median(np.diff(self.beats)))
        self.rate=float(data['rate']);self.hat=np.asarray(data['energy'])[:,2];self.hit=np.asarray(data['hit'])
        self.onset=np.asarray(data['onset']);self.stop=float(stop);self.end=self.stop+TAIL
        # The pupil's light reflex: constriction follows the upper register with a 0.6 s lag.
        a=math.exp(-1/(self.rate*.6));self.reflex=np.empty_like(self.hat);level=0.
        for i,v in enumerate(np.clip(self.hat,0,1.3)):level=a*level+(1-a)*v;self.reflex[i]=level
        self.count=int(math.ceil(self.phrase_at(self.stop)))+1
        def mean(x,k):
            i=self.index(self.time_at(PHRASE0+PHRASE_BEATS*k))
            return x[i:max(i+1,self.index(self.time_at(PHRASE0+PHRASE_BEATS*(k+1))))].mean()
        density=np.array([mean(self.onset,k) for k in range(self.count)]);self.density=density/np.median(density)
        bright=np.array([mean(self.hat,k) for k in range(self.count)])
        self.bright=np.argsort(np.argsort(bright))/max(1,self.count-1)      # rank, 0 darkest .. 1 brightest
        self.params={k:self.phrase_params(k) for k in range(-1,self.count+1)}

    def index(self,t):
        return int(np.clip(t*self.rate,0,len(self.hat)-1))

    def beat_at(self,t):
        t=np.asarray(t,float);b=self.beats
        return np.where(t<b[0],(t-b[0])/self.period,np.where(t>b[-1],len(b)-1+(t-b[-1])/self.period,
                        np.interp(t,b,np.arange(len(b)))))

    def time_at(self,beat):
        b=self.beats
        if beat<0:return b[0]+beat*self.period
        if beat>len(b)-1:return b[-1]+(beat-len(b)+1)*self.period
        return float(np.interp(beat,np.arange(len(b)),b))

    def phrase_at(self,t):
        """Position in phrases: phrase k spans [k, k+1); the prologue is negative."""
        return (self.beat_at(t)-PHRASE0)/PHRASE_BEATS

    def phrase_params(self,k):
        rng=np.random.default_rng(1729+k)
        if k<2:tier=0
        elif k<5:tier=1
        elif k<=ACTS['lachesis']:tier=2
        else:tier=3 if k>=19 else 2
        if 0<=k<self.count and k!=ACTS['lachesis'] and self.density[k]>1.08 and tier<3:tier+=1
        rx,ry=TIERS[tier][rng.integers(len(TIERS[tier]))]
        f0=.5;ph=rng.uniform(0,2*np.pi,4)
        if tier==0:ph[2]=ph[0]+np.pi/2
        x=[(1.,f0*rx+rng.normal(0,.004),ph[0])];y=[(1.,f0*ry+rng.normal(0,.004),ph[2])]
        if tier>=2:
            extra=.22+.14*(tier-2)
            x.append((extra,f0*(rx+1)+rng.normal(0,.003),ph[1]));y.append((extra,f0*(ry+1)+rng.normal(0,.003),ph[3]))
        damp=.06 if k<ACTS['lachesis'] else .028
        bright=float(self.bright[k]) if 0<=k<self.count else .5
        return dict(x=x,y=y,damp=damp,amp=.30/(1+(x[1][0] if len(x)>1 else 0)),tier=tier,ratio=(rx,ry),
                    open=bright,tilt=float(rng.uniform(-.5,.5)),squeeze=float(rng.uniform(.46,.66)),
                    fibre=float(rng.choice([1.25,1.5,2.,2.5])),crypts=int(rng.choice([5,7,9,11])),
                    depth=float(rng.uniform(.02,.07)))

    def param(self,key,k,u,beats=4.):
        """A per-phrase value, cross-faded over the first beats of each phrase."""
        cur=np.array([self.params[int(np.clip(x,-1,self.count))][key] for x in k],float)
        prev=np.array([self.params[int(np.clip(x-1,-1,self.count))][key] for x in k],float)
        return prev+(cur-prev)*smooth(u/beats)

    def harmonograph(self,k,u):
        p=self.params[int(np.clip(k,-1,self.count))]
        x=sum(a*np.sin(2*np.pi*f*u+ph) for a,f,ph in p['x'])
        y=sum(a*np.sin(2*np.pi*f*u+ph) for a,f,ph in p['y'])
        return np.stack([x,y],-1)*(p['amp']*np.exp(-p['damp']*u))[...,None]

    def acts(self,g):
        """Act weights at phrase position g (scalar or array)."""
        a,i,z=ACTS['lachesis'],ACTS['atropos'],ACTS['taut']
        return dict(sym=smooth(g-a),iris=smooth((g-i)*2),taut=smooth(g-z),
                    dilation=.075+.27*smooth((g-i)/(z-i)),prologue=smooth((g+.47)/.47))

    def pupil(self,t,g):
        """Pupil radius: slow dilation, hippus (its natural unrest) and the light reflex."""
        t=np.asarray(t,float);i=np.clip((t*self.rate).astype(int),0,len(self.reflex)-1)
        hippus=1+.018*np.sin(2*np.pi*t/2.9)+.008*np.sin(2*np.pi*t/1.13)
        return self.acts(g)['dilation']*hippus*(1-.07*self.reflex[i])

    def iris(self,k,beat,theta,rp,j,mirror,q):
        p=self.params[int(np.clip(k,-1,self.count))]
        phi=mirror*(theta+2*np.pi*beat/16+.05*np.sin(2*np.pi*beat/3))+2*np.pi*q/5
        r=rp+(IRIS-rp)*(.5+.5*np.sin(2*np.pi*p['fibre']*beat+1.7*j))*(1+p['depth']*np.sin(p['crypts']*phi))
        return np.stack([r*np.cos(phi),r*np.sin(phi)],-1)

    def pen(self,t):
        """Pen positions for times t: (len(t), 10 copies, 2) in frame widths, copy weights, and the
        positions before the closing eye flattens them (which set the ink colour).

        Copies 0-4 are rotations by 72 degrees and 5-9 their mirror images (dihedral
        symmetry, as in the flower). Before Lachesis only copy 0 draws.
        """
        t=np.asarray(t,float);beat=self.beat_at(t);g=(beat-PHRASE0)/PHRASE_BEATS
        k=np.floor(g).astype(int);u=beat-PHRASE0-PHRASE_BEATS*k
        base=np.empty((len(t),2))
        for kk in np.unique(k):
            m=k==kk;new=self.harmonograph(kk,u[m])
            if kk>=0:
                w=smooth(u[m]/2.)[:,None];new=(1-w)*self.harmonograph(kk-1,u[m]+PHRASE_BEATS)+w*new
            base[m]=new
        w=self.acts(g);base*=w['prologue'][:,None]
        sym,iris_w,taut=w['sym'][:,None],w['iris'][:,None],w['taut'][:,None]
        theta=2*np.pi*beat/(PHRASE_BEATS*9);j=np.arange(10)[None,:];q=j%5;mirror=1-2*(j//5)
        # Lachesis: the copies fan out from one thread into a five-petalled flower, opening
        # wider in the track's brighter phrases.
        opening=self.param('open',k,u)[:,None];tilt=(self.param('tilt',k,u)[:,None]*mirror)*sym
        squeeze=self.param('squeeze',k,u)[:,None]
        lx=base[:,None,0]*(1-squeeze*sym)*mirror;ly=base[:,None,1]*(1-.38*sym)
        ct,st=np.cos(tilt),np.sin(tilt)
        px=ct*lx-st*ly;py=st*lx+ct*ly+(.12+.13*opening)*sym
        ang=theta[:,None]+2*np.pi*q/5*sym;c,s=np.cos(ang),np.sin(ang)
        flower=np.stack([c*px-s*py,s*px+c*py],-1)
        # Atropos: radial fibres from the pupil edge to the iris rim.
        rp=self.pupil(t,g)[:,None];iris=np.empty_like(flower)
        for kk in np.unique(k):
            m=k==kk;args=(beat[m,None],theta[m,None],rp[m],j,mirror,q)
            fresh=self.iris(kk,*args);old=self.iris(kk-1,*args);wv=smooth(u[m]/2.)[:,None,None]
            iris[m]=old+(fresh-old)*wv
        pos=flower+(iris-flower)*iris_w[...,None]
        # The last phrase: the eye closes; the iris flattens into one taut, vibrating thread.
        x=pos[...,0]*(1+.41*taut);amp=.055*(1-smooth((g-32.5)/.7))[:,None]+.004
        y=pos[...,1]*(1-taut)+taut**4*amp*np.cos(np.pi*x/1.24)*np.sin(2*np.pi*2*beat)[:,None]
        others=np.maximum(sym**2,iris_w)*(1-.75*taut)
        weight=np.where(j==0,1.,others)*(1-.66*sym*(1-iris_w)-.45*iris_w)
        return np.stack([x,y],-1),weight,pos

    def colours(self,pos,t):
        """Ink colour per splat: bone, the flower's throat-to-plum gradient, then the iris."""
        g=float(self.phrase_at(t));w=self.acts(np.array([g]));r=np.linalg.norm(pos,axis=-1)[...,None]
        throat,petal,tip=np.array([.80,.88,.42]),np.array([.66,.36,.86]),np.array([.52,.20,.50])
        flower=np.where(r<.12,throat+(petal-throat)*smooth(r/.12),petal+(tip-petal)*smooth((r-.12)/.3))
        hazel,violet,limbal=np.array([.78,.80,.38]),np.array([.52,.34,.82]),np.array([.30,.14,.34])
        rp=float(self.pupil(np.array([t]),np.array([g]))[0]);v=np.clip((r-rp)/max(IRIS-rp,1e-3),0,1)
        iris=np.where(v<.4,hazel+(violet-hazel)*smooth(v/.4),violet+(limbal-violet)*smooth((v-.4)/.6))
        c=BONE+(flower-BONE)*.86*float(w['sym'][0])
        c=c+(iris-c)*.82*float(w['iris'][0])
        return c+(BONE-c)*float(w['taut'][0])**3

    def at(self,t):
        g=float(self.phrase_at(t));w={k:float(np.ravel(v)[0]) for k,v in self.acts(np.array([g])).items()}
        w['pupil']=float(self.pupil(np.array([t]),np.array([g]))[0])
        i=self.index(t);hat=float(np.clip(self.hat[max(0,i-3):i+1].max(),0,1.3))
        fade=5.+1.5*w['sym']-1.5*w['iris']-3.8*w['taut']
        title=float(smooth((self.time_at(PHRASE0)-.4-t)/1.1))   # on from frame 0: the post's thumbnail
        return dict(g=g,acts=w,hat=hat,hit=float(np.clip(self.hit[i],0,1.4)),fade=fade,title=title,
                    phrase=int(math.floor(g)),cut=t>=self.stop)


SPLAT_VS='''#version 330
in vec2 corner;in vec2 pos;in float weight;in vec3 colour;
uniform vec2 resolution;uniform float radius;
out vec2 offset;out vec3 tint;out float w;
void main(){offset=corner*radius;tint=colour;w=weight;
  gl_Position=vec4((pos+offset)/resolution*2.-1.,0.,1.);}'''
SPLAT_FS='''#version 330
in vec2 offset;in vec3 tint;in float w;uniform float core,halo,haloGain;out vec4 frag;
void main(){float d=dot(offset,offset);
  float k=exp(-d/(2.*core*core))+haloGain*exp(-d/(2.*halo*halo));frag=vec4(tint*w*k,0.);}'''
FULL_VS='''#version 330
in vec2 position;void main(){gl_Position=vec4(position,0.,1.);}'''
FADE_FS='''#version 330
uniform float keep;out vec4 frag;void main(){frag=vec4(keep);}'''
COMPOSITE_FS='''#version 330
uniform sampler2D canvas,title,tip;uniform vec2 resolution;
uniform float exposure,breathe,pupil,taut,catchlight,titleMix,grain;
out vec4 frag;
void main(){
  vec2 px=gl_FragCoord.xy,c=resolution*.5;
  vec3 ink=texture(canvas,((px-c)/(1.+breathe)+c)/resolution).rgb+texture(tip,px/resolution).rgb;
  float lum=dot(ink,vec3(.2126,.7152,.0722));
  vec3 col=mix(ink*(1.-exp(-lum*exposure))/max(lum,1e-4),1.-exp(-ink*exposure),.35);
  vec2 p=(px-c)/resolution.x;float r=length(p);
  float soft=.004+.01*pupil;
  float open=max(1.-taut,1e-3);vec2 e=vec2(p.x,p.y/open);
  float inside=(1.-smoothstep(pupil-soft,pupil+soft,length(e)))*step(.002,open);
  col*=1.-inside*(1.-taut);
  vec2 q=vec2(p.x,p.y/open)-vec2(-.34,.36)*pupil;
  float cr=.06*pupil;col+=vec3(.92,.93,1.)*catchlight*inside*(1.-smoothstep(cr*.55,cr,length(q*vec2(1.,1.25))));
  vec3 bg=vec3(.011,.009,.013)*(1.-.4*r*r)*(1.-.85*inside);
  col=bg+col*(1.-bg);
  col*=1.-.24*smoothstep(.38,.92,r);
  vec4 t=texture(title,px/resolution);col=mix(col,t.rgb,t.a*titleMix);
  float n=fract(sin(dot(px+grain,vec2(12.9898,78.233)))*43758.5453)-.5;
  frag=vec4(clamp(col+n*(1.6/255.),0.,1.),1.);}'''


class MortRenderer:
    def __init__(self,score,size,fps):
        import moderngl
        self.mgl=moderngl;self.score,self.size,self.fps=score,size,fps
        self.ctx=_create_context();ctx=self.ctx;w,h=size;self.unit=w/1080.
        self.canvas=ctx.texture(size,4,dtype='f2');self.canvas.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.canvas.repeat_x=self.canvas.repeat_y=False
        self.canvas_fbo=ctx.framebuffer(color_attachments=[self.canvas])
        self.tip=ctx.texture(size,4,dtype='f2');self.tip_fbo=ctx.framebuffer(color_attachments=[self.tip])
        self.out=ctx.texture(size,3);self.out_fbo=ctx.framebuffer(color_attachments=[self.out])
        self.splat=ctx.program(vertex_shader=SPLAT_VS,fragment_shader=SPLAT_FS)
        self.fade=ctx.program(vertex_shader=FULL_VS,fragment_shader=FADE_FS)
        self.comp=ctx.program(vertex_shader=FULL_VS,fragment_shader=COMPOSITE_FS)
        self.corner=ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4').tobytes())
        self.capacity=1<<18;self.instances=ctx.buffer(reserve=self.capacity*24,dynamic=True)
        self.splat_vao=ctx.vertex_array(self.splat,[(self.corner,'2f','corner'),
                                                    (self.instances,'2f 1f 3f/i','pos','weight','colour')])
        self.tri=ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype='f4').tobytes())
        self.fade_vao=ctx.vertex_array(self.fade,[(self.tri,'2f','position')])
        self.comp_vao=ctx.vertex_array(self.comp,[(self.tri,'2f','position')])
        self.title=self.title_texture()
        core,halo=.95*self.unit,4.2*self.unit
        self.splat['resolution'].value=tuple(map(float,size));self.splat['radius'].value=3*halo
        self.splat['core'].value=core;self.splat['halo'].value=halo;self.splat['haloGain'].value=.07
        self.comp['resolution'].value=tuple(map(float,size))
        self.canvas_fbo.clear(0,0,0,0);self.last=None;self.frame=0

    def title_texture(self):
        w,h=self.size;im=Image.new('RGBA',self.size);d=ImageDraw.Draw(im)
        small=find_font('LiberationSerif-Regular',round(w*.030));big=find_font('LiberationSerif-Italic',round(w*.105))
        note=find_font('LiberationSerif-Italic',round(w*.032))
        def centred(y,text,font,fill,spacing=0):
            if spacing:
                widths=[d.textlength(ch,font=font) for ch in text];x=(w-sum(widths)-spacing*(len(text)-1))/2
                for ch,cw in zip(text,widths):d.text((x,y),ch,font=font,fill=fill);x+=cw+spacing
            else:d.text(((w-d.textlength(text,font=font))/2,y),text,font=font,fill=fill)
        y=round(h*.74)
        centred(y,'ANCHOR RITE',small,(190,182,166,255),spacing=round(w*.012))
        centred(y+round(h*.03),'Bella Mort',big,(236,226,206,255))
        centred(y+round(h*.125),'Atropa belladonna',note,(168,120,190,255))
        return self.ctx.texture(self.size,4,np.asarray(im)[::-1].copy().tobytes())

    def splats(self,pts,times,weights,colours,density):
        """Resample a polyline at even arc length; weight by length and dwell time."""
        seg=np.linalg.norm(np.diff(pts,axis=0),axis=1);L=np.r_[0,np.cumsum(seg)]
        spacing=.55*self.unit;n=max(1,int(L[-1]/spacing))
        if L[-1]<1e-6:
            return np.c_[pts[-1:],[[weights[-1]*density*.3]],colours[-1:]]
        s=(np.arange(n)+.5)*L[-1]/n;ds=L[-1]/n
        x=np.interp(s,L,pts[:,0]);y=np.interp(s,L,pts[:,1]);tt=np.interp(s,L,times)
        dwell=(np.gradient(tt) if n>1 else np.array([times[-1]-times[0]]))*self.unit*900.
        w=(.62*ds+.38*np.minimum(dwell,6*ds))*np.interp(s,L,weights)*density
        c=np.stack([np.interp(s,L,colours[:,i]) for i in range(3)],-1)
        return np.c_[x,y,w,c]

    def string_halves(self,tau):
        """After the cut: two halves recoil to their anchors, whipping as they go."""
        w,h=self.size;out=[];s=np.linspace(0,1,400)[:,None]
        reach=.62*(.35*np.exp(-tau/.12)+.65*np.exp(-tau/1.4))*(1+.12*np.sin(tau*15)*np.exp(-tau/.5))
        for side in (-1,1):
            x=side*(.62-reach*s[:,0])
            y=.05*np.exp(-tau/.6)*np.sin(np.pi*s[:,0]*(1+3*tau))*s[:,0]**1.5*np.sin(2*np.pi*4.5*tau+side)
            pts=np.c_[w/2+x*w,h/2+y*w];bright=np.exp(-tau/1.)*(.55+.45*s[:,0])
            out.append(self.splats(pts,np.zeros(len(pts)),bright,np.tile(BONE,(len(pts),1)),1.25))
        return np.concatenate(out)

    def draw(self,t,read=True):
        sc=self.score;s=sc.at(t);w,h=self.size;ctx=self.ctx;m=self.mgl
        self.canvas_fbo.use();ctx.enable(m.BLEND)
        dt=1/self.fps;beads=[]
        if s['cut']:
            self.canvas_fbo.clear(0,0,0,0);data=self.string_halves(t-sc.stop)
        else:
            ctx.blend_func=m.ZERO,m.SRC_COLOR;self.fade['keep'].value=math.exp(-dt/s['fade'])
            self.fade_vao.render()
            t0=self.last if self.last is not None and 0<t-self.last<.2 else t-dt
            n=96;times=np.linspace(t0,t,n+1)
            pos,weight,unflattened=sc.pen(times);col=sc.colours(unflattened,t)
            density=(.55+.85*s['hat'])
            data=[]
            for j in range(10):
                if weight[:,j].max()<1e-3:continue
                pts=np.c_[w/2+pos[:,j,0]*w,h/2+pos[:,j,1]*w]
                data.append(self.splats(pts,times,weight[:,j],col[:,j] if col.ndim==3 else col,density))
                # The spindle: a bead at the pen tip that flares with each kick.
                bead=weight[-1,j]*(.2+.9*s['hit'])*(1-s['acts']['taut'])*7*self.unit
                beads.append(np.r_[pts[-1],bead,(BONE*.6+col[-1,j]*.4)])
            data=np.concatenate(data)
        data=data[:self.capacity].astype('f4')
        ctx.blend_func=m.ONE,m.ONE;self.instances.write(data.tobytes())
        self.splat_vao.render(m.TRIANGLE_STRIP,instances=len(data))
        self.tip_fbo.use();self.tip_fbo.clear(0,0,0,0)
        if len(beads):
            beads=np.asarray(beads,'f4');self.instances.write(beads.tobytes())
            self.splat_vao.render(m.TRIANGLE_STRIP,instances=len(beads))
        ctx.disable(m.BLEND);self.last=t
        if not read:return None
        a=s['acts'];self.out_fbo.use();self.canvas.use(0);self.title.use(1);self.tip.use(2)
        u=self.comp;u['canvas'].value=0;u['title'].value=1;u['tip'].value=2
        u['exposure'].value=1.9-.45*a['sym']*(1-a['iris']);u['breathe'].value=.006*s['hit']*(1-a['taut'])
        u['pupil'].value=a['pupil'] if a['iris']>0 else 0.
        u['taut'].value=a['taut'] if not s['cut'] else 1.
        u['catchlight'].value=.95*a['iris']*float(1-smooth(a['taut']*2.5))*(0. if s['cut'] else 1.)
        u['titleMix'].value=s['title'];u['grain'].value=float(self.frame%977);self.frame+=1
        self.comp_vao.render()
        return np.frombuffer(self.out_fbo.read(components=3),np.uint8).reshape(h,w,3)[::-1]

    def release(self):
        for x in (self.canvas,self.canvas_fbo,self.tip,self.tip_fbo,self.out,self.out_fbo,self.splat,self.fade,self.comp,self.corner,
                  self.instances,self.tri,self.splat_vao,self.fade_vao,self.comp_vao,self.title):x.release()
        self.ctx.release()


def coda(stop,sr=44100,seconds=TAIL):
    """The cut: dry shears closing, then the two slack halves of a C string recoiling left and right."""
    n=int(round(seconds*sr));t=np.arange(n)/sr;rng=np.random.default_rng(466)
    click=np.diff(np.r_[0,rng.normal(0,1,n)])*np.exp(-t/.0016)*.11
    ring=sum(a*np.sin(2*np.pi*f*t)*np.exp(-t/d) for a,f,d in ((.035,3120,.04),(.025,4710,.025),(.018,6930,.015)))
    voices=np.zeros((n,2))
    for ch,(f0,delay) in enumerate(((130.81,.0),(130.81*1.0035,.011))):
        tt=np.clip(t-delay,0,None);f=f0*(1-.12*(1-np.exp(-tt/.12)))
        phase=2*np.pi*np.cumsum(f)/sr
        voice=sum(np.sin(p*phase+.3*p)*np.exp(-tt*(1.1+.55*p))/p**1.15 for p in range(1,13))
        voices[:,ch]=voice*np.minimum(1,tt/.002)*(t>=delay)*.16
    ir_t=np.arange(int(1.6*sr))/sr;ir=rng.normal(0,1,(len(ir_t),2))*np.exp(-ir_t/.42)[:,None]*.012
    wet=np.stack([np.convolve(voices[:,c],ir[:,c])[:n] for c in range(2)],-1)
    return voices+wet+(click+ring)[:,None]*np.array([[1.,.92]])


def build_mix(audio,out,stop,sr=44100):
    raw=subprocess.run(['ffmpeg','-v','error','-i',str(audio),'-ar',str(sr),'-ac','2','-f','f32le','-'],
                       capture_output=True,check=True).stdout
    track=np.frombuffer(raw,np.float32).reshape(-1,2).astype(float)
    end=int(round((stop+TAIL)*sr));mix=np.zeros((max(end,len(track)),2));mix[:len(track)]=track
    start=int(round(stop*sr));tail=coda(stop,sr);mix[start:start+len(tail)]+=tail[:len(mix)-start]
    mix=mix[:end]
    subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar',str(sr),'-ac','2','-i','-','-c:a','flac',
                    '-sample_fmt','s32','-bits_per_raw_sample','24',str(out)],input=mix.astype('f4').tobytes(),check=True)
    return dict(stop=stop,end=end/sr,coda_peak_dbfs=round(20*math.log10(np.abs(tail).max()),1))


def frames(score,size,fps,offset,length,report,checkpoints=None,preroll=14.):
    r=MortRenderer(score,size,fps);wall=last=time.monotonic();count=math.ceil(length*fps)
    if checkpoints:checkpoints.mkdir(parents=True,exist_ok=True)
    try:
        start=max(0.,offset-preroll)
        for i in range(math.floor((offset-start)*fps)):r.draw(start+i/fps,read=False)
        for i in range(count):
            t=offset+i/fps;frame=r.draw(t)
            if checkpoints and (i%round(fps*20)==0):Image.fromarray(frame).save(checkpoints/f'{t:07.2f}.jpg',quality=93)
            yield frame
            now=time.monotonic()
            if now-last>15:
                print(f'{t:.1f}/{offset+length:.1f}s | {(i+1)/(now-wall):.1f} fps | phrase {score.at(t)["phrase"]}',flush=True);last=now
        report['render_fps']=count/(time.monotonic()-wall)
    finally:r.release()


def stills(score,size,fps,times,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for t in times:
        report={};frame=None
        for frame in frames(score,size,fps,t,1/fps,report):pass
        Image.fromarray(frame).save(folder/f'still-{t:07.2f}.png');print(folder/f'still-{t:07.2f}.png',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio',type=Path);p.add_argument('-o','--out',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--size',default='1080x1350')
    p.add_argument('--fps',type=float,default=30.);p.add_argument('--offset',type=float,default=0.)
    p.add_argument('--seconds',type=float);p.add_argument('--bitrate',default='20M')
    p.add_argument('--encoder',default='software');p.add_argument('--checkpoints',type=Path)
    p.add_argument('--stills',help='comma-separated times: write PNG stills instead of a video')
    a=p.parse_args()
    try:size=tuple(map(int,a.size.lower().split('x')))
    except ValueError:p.error('size must be WIDTHxHEIGHT')
    if len(size)!=2 or min(size)<96 or any(x%2 for x in size):p.error('use even dimensions >=96')
    data=analyse(a.audio,a.cache or a.out.with_suffix('.analysis.npz'))
    stop=find_stop(a.audio);score=MortScore(data,stop)
    if a.stills:
        stills(score,size,a.fps,[float(x) for x in a.stills.split(',')],a.out.parent);return
    offset=round(a.offset*a.fps)/a.fps;length=min(a.seconds or score.end,score.end-offset)
    if not 0<=offset<score.end:p.error('offset is beyond the video')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(track=a.audio.name,sha256=str(data['sha256']),tempo=float(data['tempo']),stop=stop,end=score.end,
                offset=offset,seconds=length,size=size,fps=a.fps,phrase0_beat=PHRASE0,phrase_beats=PHRASE_BEATS,
                acts={k:score.time_at(PHRASE0+PHRASE_BEATS*v) for k,v in ACTS.items()},
                phrases=[dict(index=k,start=score.time_at(PHRASE0+PHRASE_BEATS*k),tier=score.params[k]['tier'],
                              ratio=score.params[k]['ratio'],density=round(float(score.density[k]),3))
                         for k in range(score.count)])
    print(f'{score.end:.3f}s | stop at {stop:.3f}s | {data["tempo"]:.2f} BPM | {score.count} phrases',flush=True)
    with tempfile.TemporaryDirectory(prefix='bella-mort-',dir=a.out.parent) as tmp:
        tmp=Path(tmp);report['mix']=build_mix(a.audio,tmp/'mix.flac',stop)
        H.save_mp4(frames(score,size,a.fps,offset,length,report,a.checkpoints),tmp/'video.mp4',fps=a.fps,
                   bitrate=a.bitrate,encoder=a.encoder,on_encoder=lambda e:print('Encoder:',e.name,flush=True))
        subprocess.run(['ffmpeg','-v','error','-i',str(tmp/'video.mp4'),'-ss',str(offset),'-t',str(length),
                        '-i',str(tmp/'mix.flac'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','320k',
                        '-movflags','+faststart','-metadata','title=Bella Mort','-metadata','artist=Anchor Rite',
                        str(tmp/'complete.mp4')],check=True)
        (tmp/'complete.mp4').replace(a.out)
    a.out.with_suffix('.cues.json').write_text(json.dumps(report,indent=2)+'\n');print(a.out,flush=True)


if __name__=='__main__':main()
