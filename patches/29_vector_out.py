"""Vectors out: the picture as a beam on an oscilloscope (or a laser), and the beam as sound.

o0  a picture: a slow oscillator through a kaleidoscope, bent by noise
o1  what an XY scope shows when o0's isolines are traced on it (Contours -> XY), phosphor preview
o2  a damped harmonograph drawn by a second beam, its ratio stepping through a scale
o3  both previews together, on paper-coloured light

Out of the ES-3: channels 1/2 = beam A x/y, 3 = beam A blanking (z), 4/5 = beam B x/y. Into a
scope in XY mode they draw; into a mixer they are oscilloscope music (x left, y right). Without an
interface the five channels go to vector_out.wav to play from a DAW into the ES-3.

    hydrapy render patches/29_vector_out.py --gpu --seconds 12 --out vector.mp4     # beam B as the soundtrack
"""
import atexit

from hydrapy import *

osc(5, 0.04, 0).kaleid(5).modulate(noise(1.5, 0.1), 0.25).out(o0)

traced = XY(Contours(o0, levels=3, max_paths=24), rate=60, gain=0.6)          # the picture, scanned into vectors
ratio = Seq([1.5, 1.333, 1.25, 2.0, 1.5, 1.2]).fast(0.25)                     # just intervals: the figure is the chord
beam = XY(Harmonograph(3, 3 * ratio, 3.01, 0, 1.5708, 0.4, 4, 0.9, 0.05), rate=Seq([55, 55 * 1.5]).fast(0.5), gain=0.5)

src(traced).out(o1)
src(beam).out(o2)
solid(0.06, 0.05, 0.04).add(src(traced).color(1.0, 0.85, 0.6)).add(src(beam).color(0.6, 0.9, 1.0)).out(o3)

try:
    out = CvOut(channels=(0, 1, 2, 3, 4))
except Exception:
    out = CvOut(file="vector_out.wav", channels=(0, 1, 2, 3, 4), fps=60, samplerate=48000)
    atexit.register(out.close)
out.send(0, traced.x).send(1, traced.y).send(2, traced.z).send(3, beam.x).send(4, beam.y)

render(o3)
