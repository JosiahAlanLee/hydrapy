"""Original demoscene scenes for Anchor Rite's Tensor Junglist.

Uses HydraPy's public thi.ng cellular adapter and rotating-Voronoi adaptation;
does not reproduce C-SCAPE or De/Frag edition assets. See NOTICE.
The full track edit is rendered by docs/tensor_junglist_video.py.
"""
import numpy as np
import hydrapy as H
from hydrapy.core import Input, Op


SCENES = ["CELL ARRAY", "FOLDED CORE", "ORBIT FIELD", "VECTOR CAGE",
          "TUNNEL", "DISTANCE BANDS", "GLYPH MEMORY", "FEEDBACK", "CHROME KNOT"]


TUNNEL = Op("tensorTunnel", "src", [Input("drive"), Input("hit"), Input("turn")], glsl="""
  vec2 p = (_st - .5) * vec2(resolution.x/resolution.y, 1.);
  float angle = atan(p.y,p.x) / 6.2831853 + turn;
  float r = max(length(p), .025);
  float depth = .25/r;
  float z = depth + time*(.5 + drive*.22);
  vec2 uv = vec2(angle*14., z*3.);
  vec2 edge = abs(fract(uv)-.5);
  float line = 1.-smoothstep(.012, .04, min(edge.x, edge.y));
  float check = mod(floor(uv.x)+floor(uv.y),2.);
  float dots = 1.-smoothstep(.04,.07,length(fract(uv*2.)-.5));
  float shade = smoothstep(.035,.18,r) * (.35+.65/(1.+depth*.35));
  vec3 c = mix(vec3(.018,.028,.035),vec3(.08,.13,.16),check);
  c += line * vec3(.4,.85,.7)*(.6+hit*.6) + dots*vec3(.9,.23,.32)*.5;
  return vec4(c*shade,1.);
""")


def make_scene(index, controls, size=(1920, 1080)):
    """Mutable scalar controls are updated once per video frame by the editor."""
    p = H.Patch(size=size)
    signal = lambda name: H.Fn(lambda ctx, name=name: controls[name])
    bass, mid, high = (signal(k) for k in ("bass", "mid", "high"))
    hit, beat, turn = (signal(k) for k in ("hit", "beat", "turn"))
    t = H.Time()
    dark = H.solid(.009, .015, .024)
    if index == 0:
        cells = H.Cellular(width=384, height=216, rate=74, seed=61, stripes=32,
                           perturb=.0015, density=.21)
        x = H.src(cells).color(.42, .95, .7).contrast(1.35)
        x = x.scale(1.02 + beat*.05).scrollY(t*.012)
        x.out(p.o0, patch=p)
    elif index == 1:
        field = H.SDF3.box(.41, .41, .41).fold(3, scale=2.1, x=.7, y=.6, z=.65,
                                              angle=.16 + mid*.14).scale(.94)
        normals = H.raymarch(field, rotX=.35+t*.09, rotY=t*.37, rotZ=t*.06,
                             distance=2.8-bass*.25, steps=112, epsilon=.0015, mode=1)
        lit = normals.relight(azimuth=t*.4, elevation=.7, ambient=.2,
                              diffuse=.85, specular=.7, shininess=28).color(.75,.95,.86)
        dark.layer(lit).out(p.o0, patch=p)
    elif index == 2:
        cells = H.orbitVoronoi(points=7, rings=3, radius=.52, speed=.23,
                               phase=turn*.12, mode=2)
        x = cells.sdfBands(24+bass*7,.16+beat*.04,t*.17)
        x.color(.55,.92,.83).rotate(t*.035).scale(1.15).out(p.o0, patch=p)
    elif index == 3:
        # A retained toroidal lattice; projection and transparent Lines use the
        # repository's demoscene wire pipeline, at the requested output size.
        a = np.linspace(0, 2*np.pi, 97)
        base = []
        for v in np.linspace(0, 2*np.pi, 14, endpoint=False):
            radius = .62 + .23*np.cos(v + a*3)
            base.append(np.stack([radius*np.cos(a),radius*np.sin(a),.23*np.sin(v+a*3)],axis=1))
        for u in np.linspace(0,2*np.pi,28,endpoint=False):
            v=np.linspace(0,2*np.pi,25)
            base.append(np.stack([(.62+.23*np.cos(v))*np.cos(u)*np.ones_like(v),
                                  (.62+.23*np.cos(v))*np.sin(u)*np.ones_like(v),.23*np.sin(v)],axis=1))
        def draw(seconds):
            paths = H.project_wire(base,rotX=.6+seconds*.17,rotY=seconds*.24,
                                   rotZ=seconds*.07,distance=2.5-controls["bass"]*.15,
                                   fov=46,aspect=size[0]/size[1])
            return H.Lines(paths,canvas=size,fit=False,supersample=1,
                           stroke=max(1,size[1]/700),color=(.6,.98,.8)).render()
        dark.layer(H.src(H.FrameSource(draw))).out(p.o0, patch=p)
    elif index == 4:
        H.Chain([H.Transform(TUNNEL,[bass,beat,t*.012+turn*.05])]).out(p.o0,patch=p)
    elif index == 5:
        field=H.sdfMetaballs(5,.15+bass*.035,.34,.1,speed=.41,seed=26)
        hole=H.sdfBox(.16,.7,.04).rotate(t*.19)
        x=field.sdfSubtract(hole).sdfBands(29,.12+hit*.05,t*.39)
        x.color(.94,.48,.78).rotate(t*.05).out(p.o0,patch=p)
    elif index == 6:
        chars=H.Cellular(width=96,height=54,rate=55,seed=35,stripes=12,
                         perturb=.002,chars=" .:;/+x#%@",cell=(10,14))
        H.src(chars).color(.72,.93,.47).scale(1.01).out(p.o0,patch=p)
    elif index == 7:
        fresh=H.osc(16+bass*3,.15,.6).rotate(t*.04).kaleid(5)
        fresh=fresh.color(.42,.8,.9).mult(H.shape(5,.33,.4))
        H.src(p.o0).scale(1.017).rotate(.008).blend(fresh,.21+beat*.08).out(p.o0,patch=p)
    else:
        field=H.SDF3.torus(.6,.2).rotate(.4,0,.3).union(
            H.SDF3.torus(.45,.15).rotate(1.57,0,0),.11)
        normals=H.raymarch(field,rotX=t*.16,rotY=t*.29,distance=2.65-bass*.2,
                          steps=96,epsilon=.0015,mode=1)
        lit=normals.relight(azimuth=t*.2,elevation=.7,ambient=.12,diffuse=.8,
                            specular=.8,shininess=50).color(.7,.82,1.)
        dark.layer(lit).out(p.o0,patch=p)
    return p
