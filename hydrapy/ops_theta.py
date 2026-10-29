"""Namagiri theta-series source: Jacobi theta_3(z,q), |q|<1.

theta_3(z,q)=1+2 sum_{n>=1} q^(n*n) cos(2*n*z).
Finite truncation, bounded to |q|<=0.9 for predictable shader cost.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_thetafield(st, qreal, qimag, scale, phase, terms, env):
    aspect=env.res[0]/env.res[1]
    z=((st[...,0]-.5)*aspect+1j*(st[...,1]-.5))*bc(scale,st)+bc(phase,st)
    q=complex(float(qreal),float(qimag))
    if abs(q)>.9: q*=.9/abs(q)
    result=np.ones(z.shape,np.complex128)
    for n in range(1,max(1,min(int(terms),24))+1):
        result+=2*q**(n*n)*np.cos(2*n*z)
    angle=np.angle(result)
    hue=angle/(2*np.pi)
    rgb=.5+.5*np.cos(2*np.pi*(hue[...,None]-np.array([0,1/3,2/3])))
    rgb*=np.clip(np.abs(result)[...,None]/3,.15,1)
    return np.concatenate([rgb,np.ones((*z.shape,1))],axis=-1).astype(np.float32)

register(Op("thetafield","src",
    [F("qreal",.3,scalar=True),F("qimag",0,scalar=True),
     F("scale",2),F("phase",0),F("terms",12,scalar=True)],
    np=_np_thetafield,
    doc="Finite Jacobi theta_3 series with complex q and z. |q| clamped to 0.9, terms 1..24. Phase color and magnitude brightness.",
    glsl="""float aspect=resolution.x/resolution.y;
  vec2 z=(_st-.5)*vec2(aspect,1.0)*scale+vec2(phase,0.0);
  vec2 q=vec2(qreal,qimag);
  q*=min(1.0,.9/max(length(q),1e-9));
  vec2 power=vec2(1.0,0.0);
  vec2 result=vec2(1.0,0.0);
  int nmax=int(clamp(terms,1.0,24.0));
  for(int n=1;n<=24;n++){
    if(n>nmax)break;
    // q^(n²) via recurrence: q^((n-1)²) * q^(2n-1)
    for(int j=0;j<47;j++){
      if(j>=2*n-1)break;
      power=vec2(power.x*q.x-power.y*q.y,power.x*q.y+power.y*q.x);
    }
    float a=2.0*float(n)*z.x,b=2.0*float(n)*z.y;
    b=clamp(b,-20.0,20.0);
    vec2 cosine=vec2(cos(a)*cosh(b),-sin(a)*sinh(b));
    result+=2.0*vec2(power.x*cosine.x-power.y*cosine.y,power.x*cosine.y+power.y*cosine.x);
  }
  float hue=atan(result.y,result.x)/6.28318530718;
  vec3 rgb=.5+.5*cos(6.28318530718*(vec3(hue)-vec3(0.0,.333333333,.666666667)));
  rgb*=clamp(length(result)/3.0,.15,1.0);
  return vec4(rgb,1.0);"""))
