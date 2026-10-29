"""Newton basins for z^3 - c, with bounded iteration and CPU/GLSL twins.

The color identifies the attracting root; brightness reflects convergence speed.
Use a nonzero complex parameter c to separate the three roots.
"""
import numpy as np
from .core import Op, register
from .ops import F, bc

def _np_newtonbasin(st, real, imag, zoom, centerX, centerY, iterations, env):
    aspect = env.res[0] / env.res[1]
    z = ((st[..., 0] - bc(centerX, st))*aspect +
         1j*(st[..., 1] - bc(centerY, st))) * (4.0/np.maximum(bc(zoom, st), 1e-4))
    c = bc(real, st) + 1j*bc(imag, st)
    count = np.zeros(st.shape[:2], dtype=np.float32)
    active = np.ones(st.shape[:2], dtype=bool)
    for _ in range(max(1, min(int(iterations), 64))):
        f = z*z*z-c
        done = np.abs(f) < 1e-4
        count += (active & ~done).astype(np.float32)
        active &= ~done
        denom = 3*z*z
        safe = np.where(np.abs(denom)>1e-8, denom, 1e-8+0j)
        step = f/safe
        z = np.where(active, z - np.clip(np.abs(step),0,10)*np.exp(1j*np.angle(step)), z)
        z = np.where(np.isfinite(z), z, 0j)
    # Cubic roots are separated by 120 degrees; classify via nearest root.
    theta = np.angle(c)/3
    radius = np.abs(c)**(1/3)
    roots = np.stack([radius*np.exp(1j*(theta+2*np.pi*k/3)) for k in range(3)])
    nearest = np.argmin(np.abs(z[None,...]-roots),axis=0)
    hue = nearest.astype(np.float32)/3
    v = np.clip(1-count/max(1,min(int(iterations),64)),0.08,1)
    # Three distinct root colors; brightness encodes iteration count.
    palette = np.array([[0.92,0.29,0.19],[0.20,0.79,0.63],[0.34,0.42,0.94]],dtype=np.float32)
    rgb = palette[nearest]*v[...,None]
    return np.concatenate([rgb,np.ones((*rgb.shape[:2],1),np.float32)],axis=-1).astype(np.float32)

register(Op("newtonbasin", "src",
    [F("real", 1), F("imag", 0), F("zoom", 1),
     F("centerX", 0.5), F("centerY", 0.5),
     F("iterations", 24, scalar=True)],
    np=_np_newtonbasin,
    doc="Newton attraction basins for complex polynomial z^3-(real+i*imag). Three root colors; brightness measures convergence. Limit iterations to 64.",
    glsl="""float aspect = resolution.x/resolution.y;
  vec2 z = (_st-vec2(centerX,centerY))*vec2(aspect,1.0)*(4.0/max(zoom,0.0001));
  vec2 c = vec2(real,imag);
  float count = 0.0;
  int limit = int(clamp(iterations,1.0,64.0));
  for(int i=0;i<64;i++) {
    if(i>=limit) break;
    vec2 z2 = vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y);
    vec2 z3 = vec2(z2.x*z.x-z2.y*z.y,z2.x*z.y+z2.y*z.x);
    vec2 f = z3-c;
    if(dot(f,f)<1e-8) break;
    vec2 d = 3.0*z2;
    float den = max(dot(d,d),1e-8);
    vec2 step = vec2(f.x*d.x+f.y*d.y,f.y*d.x-f.x*d.y)/den;
    float len = length(step);
    z -= step * min(1.0,10.0/max(len,1e-8));
    count += 1.0;
  }
  float rootAngle = atan(c.y,c.x)/3.0;
  float rootRadius = pow(length(c),1.0/3.0);
  float best = 1e30;
  int idx = 0;
  for(int k=0;k<3;k++) {
    float a = rootAngle+6.28318530718*float(k)/3.0;
    vec2 root = rootRadius*vec2(cos(a),sin(a));
    float dist = dot(z-root,z-root);
    if(dist<best) {best=dist;idx=k;}
  }
  vec3 color = idx==0 ? vec3(.92,.29,.19) : (idx==1 ? vec3(.20,.79,.63) : vec3(.34,.42,.94));
  float value = clamp(1.0-count/float(limit),.08,1.0);
  return vec4(color*value,1.0);"""))
