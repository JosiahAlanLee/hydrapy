"""Ornament sound: the patch draws its own soundtrack.

In 1932 Fischinger photographed drawn ornaments onto the optical track of the film and played
them back; the same year in Leningrad Arseny Avraamov, Evgeny Sholpo (the Variophone) and Boris
Yankovsky were drawing sound by hand and by machine. Variable-density sound-on-film: the brightness
along the track is the waveform, so a pattern repeating n times per frame height sounds at
n × 24 Hz, and the *shape* of the pattern is the timbre.

Here the Soundtrack reads the middle column of o0, top to bottom, every frame: what you see is
what you hear. Pitch comes from how many shapes stack in the frame, timbre from the shape (a
triangle, a square, a disc), and the whole picture is the score.

    python -c "import runpy; runpy.run_path('patches/25_ornament_sound.py')" is silent;
    hydrapy render patches/25_ornament_sound.py --gpu --seconds 12 --out ornament.mp4   mixes the track in,
    hydrapy html patches/25_ornament_sound.py  plays it live in the browser (click to start).
"""
from hydrapy import *

set_bpm(120)
notes = Seq([10, 12, 16, 20, 16, 12, 10, 8]).fast(1)      # shapes per frame height -> 240, 288, 384, 480 ... Hz at 24 fps
sides = Seq([3, 4, 64, 3]).fast(0.25)                     # the waveform's shape: triangle, square, round
width = Seq([0.9, 0.6, 0.35]).fast(0.5)                   # duty: how much of each cycle is bright

# the ornament: a column of shapes, mirrored into a frieze across the frame
glyph = shape(sides, 0.25 * width, 0.01).repeat(1, notes)
frieze = glyph.repeat(7, 1).modulate(osc(3, 0.1, 0).rotate(1.5708), 0.015).color(0.95, 0.85, 0.6)
frieze.out(o0)

track = Soundtrack(o0, x=0.5, width=0.004, fps=24, gain=0.8)      # read the middle column

# o1: the same column shown as the film strip it is: the track at the right edge, the picture behind it
strip = gate(0.012, 1.0, 0.0, 0.001).scroll(-0.485, 0)
src(o0).mult(solid(0.45, 0.45, 0.45)).layer(src(o0).scroll(-0.485, 0).mult(strip).color(1.4, 1.4, 1.2)).out(o1)

render(o0)
