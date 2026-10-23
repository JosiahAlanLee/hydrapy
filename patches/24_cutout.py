"""The flat-graphic animation of the sixties East — Zagreb (Vukotić, Mimica), Soyuzmultfilm's
Khitruk years, Pannónia, the Polish poster school: a few flat colours, paper, redrawn by hand every
frame, printed a little off register, on twos.

o0  shapes — a disc, bars, a diagonal — in a muted palette with one red, through recipes.cutout
o1  the same through a halftone screen
o2  a title card: Cyrillic set in red and black on paper (constructivist leftovers)
o3  cut paper: the painted-film drops as flat, torn-edged paper shapes

set_frame_rate(12) is set by cutout(): the whole patch runs on twos.
"""
from hydrapy import *
from hydrapy.recipes import cutout

paper = solid(0.93, 0.88, 0.78)
red, ink, ochre, teal = (0.75, 0.15, 0.1), (0.1, 0.1, 0.12), (0.85, 0.6, 0.2), (0.2, 0.45, 0.45)

disc = solid(*red).alpha(shape(64, 0.22, 0.004)).scroll(Lfo(0.07, "tri", lo=-0.22, hi=0.22), Lfo(0.11, "sine", lo=-0.08, hi=0.08))
bars = solid(*ink).alpha(shape(4, 0.3, 0.003).scale(1, 3.2, 0.18).repeat(1, 3, 0, Lfo(0.04))).rotate(0.35)
diag = solid(*ochre).alpha(shape(4, 0.3, 0.003).scale(1, 4, 0.08)).rotate(-0.8).scroll(0, Lfo(0.05, "tri", lo=-0.3, hi=0.3))
dots = solid(*teal).alpha(shape(64, 0.06, 0.004).repeat(6, 3, 0, 0).scroll(Lfo(0.03), 0))
pic = paper.layer(dots).layer(diag).layer(bars).layer(disc)

cutout(pic).out(o0)
cutout(pic, screen=5, paper=0.3).out(o1)

title = paper.layer(solid(*ink).alpha(shape(4, 0.5, 0.003).scale(1, 2.6, 0.03).rotate(0.0).scroll(0, 0.28))) \
             .layer(src(Text("КИНО", font="DejaVuSans-Bold", size=0.34, y=0.56, color=red))) \
             .layer(src(Text("глаз", font="DejaVuSerif-Italic", size=0.14, x=0.5, y=0.3, color=ink)))
cutout(title, boil=0.002, levels=4).out(o2)

torn = paper.stain(drops(3, 0.6, 0.5, 1.6, 0.0, 0.002, 0.3, 0.0, SH(2, 0, 99), hue=0.03, hueSpread=0.4, sat=0.55), 1.4)
cutout(torn, levels=3, paper=0.6, misregister=0.004).out(o3)

render(o0)
