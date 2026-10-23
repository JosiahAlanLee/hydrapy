"""A playable feedback instrument with five exposed controls.

Low-frequency motion bends the source; persistence controls the image's memory.
"""
from hydrapy import *

STUDIO = {'title': 'Feedback instrument', 'tags': ['Feedback', 'Sound & rhythm'],
          'preview': 'docs/studio/feedback.jpg'}
frequency = Param('Frequency', 12, min=2, max=60, step=.1, unit='cycles')
motion = Param('Motion', .12, min=.01, max=2, step=.01, unit='Hz', scale='log')
distortion = Param('Distortion', .18, min=0, max=.6, step=.005)
persistence = Param('Persistence', .72, min=0, max=.96, step=.01)
colour = Param('Colour', .1, min=0, max=1, step=.01)
set_bpm(110)
osc(frequency, .08, .7).modulate(noise(3).rotate((Time()*motion*6.283185).sin()), distortion).kaleid(5).hue(colour).out(o0)
src(o1).scale(1.003).rotate(.002).blend(src(o0), 1-persistence).out(o1)
src(o1).edge().invert().out(o2)
src(o1).color(.7, .9, 1).out(o3)
render(o1)
