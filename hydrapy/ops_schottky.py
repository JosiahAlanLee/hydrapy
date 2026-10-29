"""Bounded Schottky-style circle-reflection orbit field.

A two-generator anti-conformal inversion system. This is an orbit
visualization, not a mathematically complete limit-set solver.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_schottkyfield(st, separation, radius, iterations, env):
    aspect=env.res[0]/env.res[1]
    x=(st[...,0]-.5)*aspect*2
    y=(st[...,1]-.5)*2
    sep=max(float(separation),.01)
    rad=max(float(radius),.001)
    count=np.zeros(x.shape,np.float32)
    for _ in range(max(1,min(int(iterations),32))):
        left=(x+sep)**2+y*y
        right=(x-sep)**2+y*y
        choose=left<right
        dist=np.where(choose,left,right)
        cx=np.where(choose,-sep,sep)
        inside=dist<rad*rad
        factor=rad*rad/np.maximum(dist,1e-8)
        x=np.where(inside,cx+(x-cx)*factor,x)
        y=np.where(inside,y*factor,y)
        count+=inside.astype(np.float32)
    v=count/max(1,min(int(iterations),32))
    return np.stack([v,.35*v,.8*(1-v),np.ones_like(v)],axis=-1).astype(np.float32)

register(Op("schottkyfield","src",
    [F("separation",.45,scalar=True),F("radius",.35,scalar=True),F("iterations",16,scalar=True)],
    np=_np_schottkyfield,
    doc="Experimental circle-reflection orbit field. NOT a complete Kleinian limit-set algorithm; centers ±separation, radius and iterations control bounded inversion.",
    glsl="""float aspect=resolution.x/resolution.y;
  vec2 p=(_st-.5)*vec2(aspect,1.0)*2.0;
  float sep=max(separation,.01),rad=max(radius,.001);
  float count=0.0;
  int n=int(clamp(iterations,1.0,32.0));
  for(int i=0;i<32;i++){
    if(i>=n)break;
    float left=dot(p+vec2(sep,0),p+vec2(sep,0));
    float right=dot(p-vec2(sep,0),p-vec2(sep,0));
    float cx=left<right ? -sep:sep;
    vec2 d=p-vec2(cx,0);
    float d2=dot(d,d);
    if(d2<rad*rad){p=vec2(cx,0)+d*(rad*rad/max(d2,1e-8));count+=1.0;}
  }
  float v=count/float(n);
  return vec4(v,.35*v,.8*(1.0-v),1.0);"""))
