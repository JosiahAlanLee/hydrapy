"""Slow marbled bands with controls for the bath, comb and colour."""
from hydrapy import *
from hydrapy.recipes import bands

STUDIO = {'title': 'Marbling bath', 'tags': ['Paint & texture'], 'preview': 'docs/studio/marbling.jpg'}
amount = Param('Comb depth', .15, min=0, max=.5, step=.005)
rate = Param('Motion', .08, min=.01, max=.5, scale='log', unit='Hz')
spacing = Param('Comb spacing', .13, min=.03, max=.4, step=.005)
colour = Param('Colour', .04, min=0, max=1)
water = osc(18, .03, .4).color(.6,.85,.7)
water.tine(.6, 0, (Time()*rate*6.283185).sin()*amount, .03, spacing).swirl(.5,.5, amount,.3,.06).hue(colour).out(o0)
src(o0).edge().out(o1)
render(o0)
