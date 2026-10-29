"""Scene transitions: a scene grows out of the one before it, instead of a crossfade.

`scene_transition('luma')` (before the scenes are declared) swaps the plain blend
between this output's scenes for a keyed crossfade: the incoming scene appears in
order of its own luminance, brightest first — day to night here means the stars
and moon come out first, then the dark floods in. Try 'shadow' (the incoming
floods in through the outgoing picture's darkest areas — a dusk fall) or 'grain'
(a soft, textured noise front; `scale` is its noise scale). The tracker's SCENE
lane still chooses scenes and their fades; the style is chosen here, in the patch.

To preview a transition offline (no score, weights are driven by hand):

    p = default_patch()
    p.scene_weights = {'day': 1 - w, 'dusk': 0, 'night': w}   # w = 0..1
    save_png(render_frames(seconds=.1, fps=1)[0], 'transition.png')

Amount 0 shows exactly the scene before and amount 1 exactly the scene after, so
a fade of 8 bars is one picture turning into another, never a double exposure.
"""
from hydrapy import *

STUDIO = {'title': 'Scene transitions', 'tags': ['Tracker', 'Scenes']}
set_bpm(90)

roll = Param('Roll', 0, min=-45, max=45, step=1, unit='degrees')

scene_transition('luma', softness=0.2)          # per output: scene_transition('grain', output=o1, ...)


def sky(c):
    # gradient's r channel is 1 - y after this rotation: bright at the top
    ramp = gradient(0).rotate(1.5708).r()
    return solid(*c).mult(ramp, 1)


def ground(c):
    # r channel is y: a soft-edged mask below the horizon (y ~ .4), composited by alpha
    field = gradient(0).rotate(-1.5708).r().thresh(0.4, 0.05)
    return solid(*c).alpha(field)


def disc(c, radius, at):
    return solid(*c).alpha(shape(60, radius, 0.02).scroll(0.0, at))


@scene('day')
def _():
    return sky((0.55, 0.9, 1.1)).layer(disc((1.15, 1.05, 0.7), 0.2, -0.3)).layer(ground((0.5, 0.9, 0.45))).rotate(roll * .01745)


@scene('dusk')
def _():
    return sky((1.0, 0.55, 0.25)).layer(disc((1.1, 0.45, 0.08), 0.22, -0.1)).layer(ground((0.32, 0.16, 0.09))).rotate(roll * .01745)


@scene('night')
def _():
    stars = white(1, 7).thresh(0.985, 0.005).color(0.9, 0.95, 1.1)
    return (sky((0.05, 0.12, 0.35)).layer(ground((0.02, 0.07, 0.04)))
            .layer(solid(1, 1, 1).alpha(stars)).layer(disc((0.95, 0.95, 1.1), 0.16, -0.28)).rotate(roll * .01745))


render(o0)
