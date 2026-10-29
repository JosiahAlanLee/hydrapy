"""Experimental two-generator Möbius IFS orbit visualization.

Deterministic inverse iteration on two loxodromic generators. This is an
orbit-density field, not a certified Kleinian limit set. No stochastic state.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_kleinianorbit(st, scale, twist, iterations, env):
    asp=env.res[0]/env.res[1]
    x=(st[...,0]-.5)*asp*2
    y=(st[...,1]-.5)*2
    z=x+1j*y
    # Alternating fractional-linear maps; determinant nonzero.
    n=max(1,min(int(iterations),32))
    energy=np.zeros(z.shape,np.float32)
    for k in range(n):
        a=np.exp(1j*float(twist)*(1 if k%2 else -1))
        b=.32 if k%2 else -.32
        d=1.0+0j
        denom=.18*z+d
        z=(a*z+b)/np.where(np.abs(denom)>1e-7,denom,1e-7+0j)
        z=np.where(np.isfinite(z),z,0j)
        energy+=np.exp(-np.minimum(np.abs(z),100)*max(float(scale),.001)).astype(np.float32)
    v=np.clip(energy/n,0,1)
    return np.stack([v,v*v,.8*(1-v),np.ones_like(v)],axis=-1).astype(np.float32)

register(Op("kleinianorbit","src",
    [F("scale",2,scalar=True),F("twist",.4,scalar=True),F("iterations",16,scalar=True)],
    np=_np_kleinianorbit,
    doc="Experimental deterministic alternating Möbius orbit field. Not a rigorous Kleinian limit-set solver.",
    glsl="""vec2 z=(_st-.5)*vec2(resolution.x/resolution.y,1.0)*2.0;
  float energy=0.0;
  int n=int(clamp(iterations,1.0,32.0));
  for(int k=0;k<32;k++){
    if(k>=n)break;
    float angle=twist*(mod(float(k),2.0)<.5 ? -1.0:1.0);
    vec2 a=vec2(cos(angle),sin(angle));
    vec2 az=vec2(a.x*z.x-a.y*z.y,a.x*z.y+a.y*z.x);
    float b=mod(float(k),2.0)<.5 ? -.32:.32;
    vec2 numerator=az+vec2(b,0.0);
    vec2 den=vec2(1.0+.18*z.x,.18*z.y);
    float d=max(dot(den,den),1e-14);
    z=vec2(dot(numerator,den),numerator.y*den.x-numerator.x*den.y)/d;
    z=clamp(z,vec2(-1e6),vec2(1e6));
    energy+=exp(-min(length(z),100.0)*max(scale,.001));
  }
  float v=clamp(energy/float(n),0.0,1.0);
  return vec4(v,v*v,.8*(1.0-v),1.0);"""))
