"""Ramanujan third-order mock theta f(q), finite q-series.

f(q) = sum_{n>=0} q^(n*n) / prod_{k=1..n}(1+q^k)^2.
Domain |q|<=0.8, bounded terms; finite approximants are not modular
completions and do not characterize radial limits at roots of unity.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def mock_f(q, terms=16):
    q=np.asarray(q,dtype=np.complex128)
    nmax=max(1,min(int(terms),24))
    total=np.ones(q.shape,np.complex128)
    denom=np.ones(q.shape,np.complex128)
    for n in range(1,nmax):
        denom=denom*(1+q**n)**2
        total=total+q**(n*n)/np.where(np.abs(denom)>1e-12,denom,1e-12+0j)
    return total

def _np_mocktheta(st, radius, phase, zoom, terms, env):
    aspect=env.res[0]/env.res[1]
    q=((st[...,0]-.5)*aspect+1j*(st[...,1]-.5))*bc(radius,st)/np.maximum(bc(zoom,st),1e-5)
    q=q*np.exp(1j*bc(phase,st))
    q*=np.minimum(1,.8/np.maximum(np.abs(q),1e-10))
    v=mock_f(q,terms)
    hue=np.angle(v)/(2*np.pi)
    rgb=.5+.5*np.cos(2*np.pi*(hue[...,None]-np.array([0,1/3,2/3])))
    rgb*=np.clip(np.abs(v)[...,None]/3,.15,1)
    return np.concatenate([rgb,np.ones((*q.shape,1))],axis=-1).astype(np.float32)

register(Op("mocktheta","src",
    [F("radius",.7),F("phase",0),F("zoom",1),F("terms",12,scalar=True)],
    np=_np_mocktheta,
    doc="Ramanujan third-order mock theta f(q), finite q-series; |q|<=0.8, terms 1..24. Phase color and magnitude brightness.",
    glsl="""float aspect=resolution.x/resolution.y;
  vec2 q=(_st-.5)*vec2(aspect,1.0)*(radius/max(zoom,1e-5));
  float c=cos(phase),s=sin(phase);
  q=vec2(q.x*c-q.y*s,q.x*s+q.y*c);
  q*=min(1.0,.8/max(length(q),1e-10));
  vec2 total=vec2(1.0,0.0),denom=vec2(1.0,0.0);
  vec2 qpower=vec2(1.0,0.0);
  int nmax=int(clamp(terms,1.0,24.0));
  for(int n=1;n<24;n++){
    if(n>=nmax)break;
    // q^n incrementally, then q^(n²) from q^n via bounded multiply.
    qpower=vec2(qpower.x*q.x-qpower.y*q.y,qpower.x*q.y+qpower.y*q.x);
    vec2 factor=vec2(1.0,0.0)+qpower;
    vec2 sq=vec2(factor.x*factor.x-factor.y*factor.y,2.0*factor.x*factor.y);
    denom=vec2(denom.x*sq.x-denom.y*sq.y,denom.x*sq.y+denom.y*sq.x);
    vec2 num=vec2(1.0,0.0);
    for(int j=0;j<576;j++){
      if(j>=n*n)break;
      num=vec2(num.x*q.x-num.y*q.y,num.x*q.y+num.y*q.x);
    }
    float d=max(dot(denom,denom),1e-12);
    total+=vec2(num.x*denom.x+num.y*denom.y,num.y*denom.x-num.x*denom.y)/d;
  }
  float hue=atan(total.y,total.x)/6.28318530718;
  vec3 rgb=.5+.5*cos(6.28318530718*(vec3(hue)-vec3(0.0,.333333333,.666666667)));
  rgb*=clamp(length(total)/3.0,.15,1.0);
  return vec4(rgb,1.0);"""))
