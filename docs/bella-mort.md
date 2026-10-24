# Bella Mort — one thread, spun, measured and cut

A full-track video for "Bella Mort", the opening track of Anchor Rite's *Pinned*.
The whole picture is one continuous line: a damped harmonograph whose pendulums
are locked to the measured beat. The video is 7:49 at 1080×1350 (4:5 portrait,
for phones) and 30 fps.

The idea comes from the plant. *Atropa belladonna* is named after Atropos, the
Fate who cuts the thread of life. "Bella donna" refers to its use to dilate
pupils. A dilated pupil that does not respond is also one of the signs of death.
The three acts follow the three Fates, and each is a different way of drawing
the same thread.

| Act | Time | What the thread does |
| --- | --- | --- |
| Title | 0–6.9 s | The title is on screen from frame 0. A pen tip spins out of the centre. |
| Clotho, spinning | 6.9–144.0 s | One bone-white harmonograph line. Its figures get more complex from phrase to phrase, from an ellipse to 2:3 and 3:4 knots. |
| Lachesis, measuring | 144.0–322.3 s | The line repeats five times round the centre, with mirror images, and fans out into a belladonna flower. The flower opens wider in brighter phrases. |
| Atropos, cutting | 322.3–445.8 s | The flower becomes an iris of radial fibres. The pupil dilates from 0.075 to 0.345 frame widths. |
| The cut | 445.8–465.7 s | The eye closes: the iris flattens into one taut, vibrating thread. On the recording's last hit the thread is cut, and the two halves recoil for 3.6 s. |

## Timing

The track runs at **139.94 BPM**. Phrases are **32 beats (13.72 s)** long, and the
first one starts on tracked beat 15 (6.89 s). Measured section changes fall on
this grid (for example 157.8, 171.5, 308.6, 322.3 and 336.0 s), so the acts
change on phrase starts. The pendulum frequencies are counted in beats, not
seconds: the base swing is half a cycle per beat, and the ratios are small
integers with a slight detune.

Each phrase strikes the pendulums again. The old and new motions are cross-faded
over two beats, so the pen swings out into the new figure without lifting. The
line is unbroken from the first frame to the cut. In Act I the damping is 0.06
per beat, so each phrase spirals down to about 15% of its starting amplitude.
In Act II it is 0.028 per beat, which keeps the petals open and lacy.

## What follows the music

- **Beat:** pendulum periods, the phrase strikes, the flower's rotation, and the
  iris fibres (1.25–2.5 radial cycles per beat, chosen per phrase).
- **Upper register (2–10 kHz):** how much ink the pen lays down. The same signal
  drives the pupil's light reflex: the pupil narrows by up to 7% when the band is
  loud, with a 0.6 s lag. The pupil also has a slow natural tremble (hippus).
- **Each phrase's brightness rank:** how wide the flower opens. Bright phrases
  open fully and dark phrases close into a bud with a green throat.
- **Onset density per phrase:** how complex the figure is (which tier of
  frequency ratios it draws from).
- **Kicks:** a bead of light at the pen tip, and a 0.6% swell of the whole picture.

## Drawing

Ink is laid as Gaussian splats spaced 0.55 px apart along the pen's path, drawn
additively into a float canvas that fades slowly. The fade time constant is 5 s
for the single thread and 6.5 s for the flower, and it drops to 1.2 s as the eye
closes. Each splat's weight combines path length (62%) and
the time the pen spent there (38%). The result is an even line that glows where
the pen turns, like a long-exposure photograph of a light pen.

The composite applies the tone map, the pupil mask with its catchlight, the
title, a vignette and grain. The tone map mostly preserves hue, so dense parts
of the flower stay violet instead of clipping to white. The pen-tip bead is
drawn on a separate layer that is cleared every frame, so it leaves no trail.

## Audio

The recording is not changed. Its loudness and its stop stay exactly as
released. One sound is added: the cut.

`find_stop` finds where the sound ends: the end of the last 5 ms block above
−45 dBFS. If a hit starts within the 120 ms before that, the cut goes on the hit
instead. Here the last hit is at **465.682 s** and it decays to silence by
465.73 s, so the snip and the final strike fuse into one sound. The added sound
has three parts:

- a dry click of shears, with a short metallic ring that is gone within 0.2 s;
- two slack halves of a C string (130.8 Hz) whose pitch drops 12% as their
  tension goes, one panned to each side, 11 ms apart;
- a short reverb on the string only.

It peaks at −9 dBFS and is silent (below −50 dBFS) by the last half-second. The
video is 3.6 s longer than the music.

## Reproduce

```bash
python docs/bella_mort_video.py "Anchor Rite - Pinned - 01 Bella Mort.flac" \
  --cache bella-analysis.npz -o Bella_Mort_1080x1350.mp4 --checkpoints checkpoints
```

The defaults are 1080×1350, 30 fps, and x264 at 20 Mbit/s with 320 kbit/s AAC.
The render runs at about 99 fps on an RTX 4090. Use `--offset` and `--seconds`
for an excerpt; the 14 s of history before the excerpt start is drawn first,
without being saved, so an excerpt matches the full render. `--stills 30,200,400`
writes PNG stills instead of a video.

For Bluesky (10 minutes and 300 MB at most), re-encode the master in two passes.
The recording's true peak is +0.4 dBTP (−13.8 LUFS). The upload's audio is
lowered 1.5 dB and encoded at 256 kbit/s, which gives −0.7 dBTP after AAC and
leaves headroom for Bluesky's own re-encode. A limiter was worse: its fast gain
changes made AAC overshoot to +2.1 dBTP. The master keeps the recording's level.
The result is 274 MB.

```bash
ffmpeg -i Bella_Mort_1080x1350.mp4 -c:v libx264 -preset slow -b:v 4400k -maxrate 8M \
  -bufsize 12M -pix_fmt yuv420p -pass 1 -an -f null /dev/null
ffmpeg -i Bella_Mort_1080x1350.mp4 -c:v libx264 -preset slow -b:v 4400k -maxrate 8M \
  -bufsize 12M -pix_fmt yuv420p -pass 2 -af volume=-1.5dB -c:a aac -b:a 256k -movflags +faststart \
  Bella_Mort_Bluesky.mp4
```

## Checks

```bash
python -m pytest -q tests/test_bella_mort.py
```

- **Phrases and acts:** the phrase grid and act weights.
- **Continuity:** the whole timeline sampled at 600 Hz. No step is more than 2.5×
  its neighbours, which a lifted pen would be.
- **Layout:** one thread in Act I and ten copies after it; the iris lies between
  the pupil and its rim; the closed eye is a line wider than the frame.
- **The cut and the coda:** `find_stop` on a sustained tone and on a final hit;
  the mixed length; the coda's peak level and silent tail.
- **Renderer:** a GPU smoke test (skipped when there is no OpenGL context).
