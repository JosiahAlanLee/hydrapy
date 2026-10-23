"""Text and line drawings as dye layers and as masks.

Swap in your own: Text("...") takes any font file or installed font name; Lines("drawing.svg")
takes a plotter SVG (paths, polylines, transforms), or a list of polylines in 0..1 coordinates.
The plants here come from hydrapy.drawing.meadow(), a stand-in for real drawings.
"""
from hydrapy import *
from hydrapy.recipes import film

title = Text("hydrapy", font="DejaVuSerif-Bold", size=0.26, color=(0.85, 0.1, 0.05), y=0.62)
sub = Text("a video synth, on paper", font="DejaVuSerif", size=0.07, color=(0.1, 0.1, 0.1), y=0.36)
plants = Bank([Lines(meadow(seed=s, n=9), stroke=1.6, color=(0.05, 0.08, 0.05)) for s in range(6)], rate=12, hold=2)

# o0: ink on paper. Text and drawing are dye layers; the paper is a slightly warm leader.
paper = solid(0.98, 0.96, 0.9).stain(fbm(6, 4, 0, 0.5).alpha(solid(0.15, 0.15, 0.15)), 0.3)
film(paper.stain(src(plants), 1.0).stain(src(title).bleed(0.12, 0.15, 0.3, iterations=1), 1.0).stain(src(sub), 0.9),
     grain=0.05, scratch=0.15, specks=0.1, bloom=0.15, fps=24).out(o0)

# o1: the drawing as a mask over a synth texture; the text cut out of a wavefolded oscillator
field = osc(12, 0.1, 1.5).modulate(noise(3, 0.2), 0.3).fold(Lfo(0.07, "tri", lo=1, hi=3)).palette(0, 0.3, 0.6, Lfo(0.03))
field.mult(src(plants).a().invert()).layer(osc(40, 0.05, 2).colorama(0.02).alpha(src(title).a())).out(o1)
render(o0)
