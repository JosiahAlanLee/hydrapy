"""Finite projection-wave quasicrystal source (aperiodic for 5/7-fold axes)."""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_quasifield(st, order, scale, phase, threshold, env):
    n = max(3,min(int(order),12))
    asp = env.res[0]/env.res[1]
    x = (st[...,0]-.5)*asp*bc(scale,st)
    y = (st[...,1]-.5)*bc(scale,st)
    field = np.zeros(st.shape[:2],np.float32)
    for k in range(n):
        a=np.pi*k/n
        field += np.cos(2*np.pi*(x*np.cos(a)+y*np.sin(a))+bc(phase,st))
    value = field/n
    width=.035
    intensity=np.clip((value-bc(threshold,st)+width)/(2*width),0,1)
    return np.stack([intensity,intensity,intensity,np.ones_like(intensity)],axis=-1).astype(np.float32)

register(Op("quasifield","src",
    [F("order",5,scalar=True),F("scale",8),F("phase",0),F("threshold",0)],
    np=_np_quasifield,
    doc="Finite superposition of equally angled plane waves. A quasiperiodic interference field, not a discrete Penrose tiling; order 3..12.",
    glsl="""float asp=resolution.x/resolution.y;
  vec2 p=(_st-.5)*vec2(asp,1.0)*scale;
  float sum=0.0;
  int n=int(clamp(order,3.0,12.0));
  for(int k=0;k<12;k++){
    if(k>=n)break;
    float a=3.141592653589793*float(k)/float(n);
    sum+=cos(6.28318530718*dot(p,vec2(cos(a),sin(a)))+phase);
  }
  float value=sum/float(n);
  float intensity=clamp((value-threshold+.035)/.07,0.0,1.0);
  return vec4(vec3(intensity),1.0);"""))
