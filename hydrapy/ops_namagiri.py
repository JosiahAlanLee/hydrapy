"""Namagiri: bounded Rogers–Ramanujan continued fraction.

R(q) = q^(1/5)/(1 + q/(1 + q²/(1 + q³/...))).
The operator evaluates finite convergents in the disk |q|<1, avoiding the
unit-circle convergence subtleties. Intermediate depth is a musical/visual
control, not a guarantee of an exact infinite continued fraction.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

PALETTE = np.array([[.95,.28,.17],[.17,.75,.65],[.34,.39,.91]],np.float32)

def convergent(q, depth):
    """Finite convergent evaluated bottom-up; complex ndarray input."""
    n = max(1,min(int(depth),32))
    tail = np.ones_like(q,dtype=np.complex128)
    for k in range(n,0,-1):
        numerator = np.power(q,k)
        denominator = np.where(np.abs(tail)>1e-10,tail,1e-10+0j)
        tail = 1 + numerator/denominator
    return np.exp(np.log(np.where(np.abs(q)>1e-12,q,1e-12+0j))/5)/tail

def _np_ramanujancf(st, radius, phase, depth, zoom, centerX, centerY, env):
    aspect = env.res[0]/env.res[1]
    z = ((st[...,0]-bc(centerX,st))*aspect + 1j*(st[...,1]-bc(centerY,st)))
    q = z*(bc(radius,st)/np.maximum(bc(zoom,st),1e-5))*np.exp(1j*bc(phase,st))
    # Explicitly constrain to the open unit disk for stable bounded evaluation.
    q = q*np.minimum(1,.97/np.maximum(np.abs(q),1e-10))
    val = convergent(q,depth)
    angle = np.angle(val)
    hue = (angle/(2*np.pi)+1)%1
    # Simple continuous trichromatic phase palette.
    rgb = .5+.5*np.cos(2*np.pi*(hue[...,None]-np.array([0,.333333,.666667])))
    rgb *= (.3+.7*np.clip(np.abs(val),0,1))[...,None]
    return np.concatenate([rgb,np.ones((*rgb.shape[:2],1))],axis=-1).astype(np.float32)

register(Op("ramanujancf","src",
    [F("radius",.8),F("phase",0),F("depth",12,scalar=True),
     F("zoom",1),F("centerX",.5),F("centerY",.5)],
    np=_np_ramanujancf,
    doc="Finite Rogers–Ramanujan continued fraction, phase-colored. Domain constrained to |q|<=.97; depth 1..32. Named for Ramanujan's mathematics, not a claim about the origin of any particular identity.",
    glsl="""float aspect=resolution.x/resolution.y;
  vec2 q=(_st-vec2(centerX,centerY))*vec2(aspect,1.0)*(radius/max(zoom,.00001));
  q=mat2(cos(phase),sin(phase),-sin(phase),cos(phase))*q;
  q*=min(1.0,.97/max(length(q),1e-10));
  vec2 tail=vec2(1.0,0.0);
  int n=int(clamp(depth,1.0,32.0));
  for(int i=32;i>=1;i--) {
    if(i>n) continue;
    vec2 power=vec2(1.0,0.0);
    for(int j=0;j<32;j++) {
      if(j>=i) break;
      power=vec2(power.x*q.x-power.y*q.y,power.x*q.y+power.y*q.x);
    }
    float den=max(dot(tail,tail),1e-10);
    vec2 quotient=vec2(power.x*tail.x+power.y*tail.y,power.y*tail.x-power.x*tail.y)/den;
    tail=vec2(1.0,0.0)+quotient;
  }
  float angle=atan(q.y,q.x)/5.0;
  float magnitude=pow(max(length(q),1e-12),.2);
  vec2 numerator=magnitude*vec2(cos(angle),sin(angle));
  float den=max(dot(tail,tail),1e-10);
  vec2 value=vec2(numerator.x*tail.x+numerator.y*tail.y,numerator.y*tail.x-numerator.x*tail.y)/den;
  float hue=fract(atan(value.y,value.x)/6.28318530718+1.0);
  vec3 rgb=.5+.5*cos(6.28318530718*(vec3(hue)-vec3(0.0,.333333,.666667)));
  rgb*=.3+.7*clamp(length(value),0.0,1.0);
  return vec4(rgb,1.0);"""))
