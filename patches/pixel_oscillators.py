"""Independent clocks: run in Studio, or import make_patch for offline export."""
import hydrapy as H


PRESETS = {
    "saddle": dict(rate=.035, spread=1.6, density=90, pixel=4, geometry=0),
    "radial": dict(rate=.045, spread=1.2, density=110, pixel=4, geometry=1),
    "folded": dict(rate=.025, spread=2.0, density=70, pixel=4, geometry=2),
    "lattice": dict(rate=.04, spread=1.4, density=85, pixel=4, geometry=3),
}
BACKGROUND = (17/255, 20/255, 18/255)
FOREGROUND = (216/255, 220/255, 204/255)


def make_patch(size=(1920,1080), preset="saddle", **overrides):
    p=H.Patch(size=size)
    values=dict(PRESETS[preset],phase=.137,threshold=0)
    unknown=set(overrides)-set(values)
    if unknown: raise ValueError(f"unknown controls: {sorted(unknown)}")
    values.update(overrides)
    ranges={
        "rate": (0,2,.005,"Hz","Shared base rate; keep constant for free-running clocks."),
        "spread": (0,6,.01,"Hz","Strength of the fixed spatial frequency differences."),
        "density": (1,300,.25,"cycles","Spatial phase density; high values produce aliasing."),
        "pixel": (1,12,1,"px","Integer cell pitch in output pixels."),
        "geometry": (0,3,1,"","0 saddle; 1 radial; 2 folded; 3 lattice."),
        "phase": (0,1,.001,"cycles","Shared initial phase offset."),
        "threshold": (-.95,.95,.01,"","Sine threshold controlling the two-color balance."),
    }
    controls={}
    for name,(lo,hi,step,unit,description) in ranges.items():
        controls[name]=H.Param(name,values[name],min=lo,max=hi,step=step,unit=unit,
                              description=description,group="Pixel clocks",patch=p)
    mask=H.pixelOsc(**controls)
    delta=tuple(f-b for f,b in zip(FOREGROUND,BACKGROUND))
    mask.color(*delta).add(H.solid(*BACKGROUND,0)).out(p.o0,patch=p)
    return p


patch=make_patch()
