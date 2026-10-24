# Bell Riots Pt 2 — counterpoint mix and fourth edit

The fourth pass uses the same closing **6:01.093**, starting at **7:14.462** in
the original recording. It adds the user's `bell_hats_v2` Pure Data score,
balances the new material against the original mix, and gives the upper voices
separate visual responses. The recording, Pd score and Generative_Art drawings
are external inputs.

## Audio

The supplied ZIP contains a Pd patch and score arrays rather than a rendered
stem. `bell_riots_mix.py` copies those inputs into a working cache and exposes
four stereo buses before the patch's hard clips: hats, cowbell, stabs with delay,
and the shared percussion room. Voice synthesis, score patterns, seeded
velocities, gates and the dotted-eighth ping-pong delay are retained.

The supplied `rev3~` objects both had output level zero, which mutes their returns.
The rendering copy sets quiet return levels of 82/80 on Pd's 0–100 dB scale;
the shared percussion return is then reduced by a further 6 dB. The original
uploaded files are unchanged.

- Original mix: 24 Hz high-pass and a **+1.8 dB low shelf at 105 Hz**.
- Hats: **−1 dB**, preserving the attacks.
- Cowbell: **−3.5 dB**, with 45% parallel soft saturation.
- Stabs: **−3 dB**, with 40% parallel soft saturation, including their echoes.
- Saturation: 4× oversampling and RMS compensation before the level reductions.
- Master: constant gain toward −16 LUFS, followed by a delay-compensated,
  oversampled limiter at −1.25 dB. Gain is capped to avoid forcing loudness
  through heavy peak reduction. Export is 48 kHz, 24-bit FLAC.

For the delivered pass, measured loudness is **−16.00 LUFS / −1.25 dBTP**,
with no clipped samples. The original closing edit measured −18.3 LUFS.
The new 30–100 Hz band is 2.24 dB stronger in the delivered master, including
the overall gain change. Bass is adjusted within the existing stereo mix;
this is not an isolated bass-instrument stem or source separation.

```bash
python docs/bell_riots_mix.py /path/to/original.flac \
  --stem-root /path/to/unzipped/bell_hats \
  --pd /path/to/pd --cache /tmp/bell-counterpoint-audio \
  -o Bell_Riots_Counterpoint_Mix.flac
```

Dependencies are NumPy, SciPy, FFmpeg and Pd vanilla with `rev3~`. If Pd's
extra directory is outside its normal search path, supply `--pd-extra`.
The delivered mix used upstream [Pure Data](https://github.com/pure-data/pure-data)
at `75381de0ff127e9bfb36cacb589c1a0f6df4397d`, built for offline rendering.

The score's own 137 BPM grid is preserved: beat zero is −8.301 ms relative to
the edit, and the first scheduled beat is at 429.655 ms. Neither input is time
stretched. The master is trimmed to the source edit's sample duration, with
short 8 ms / 350 ms edge fades. The cache includes floating-point source buses,
the premaster and measured visual-control envelopes. The `.mix.json` sidecar
records input hashes, bus levels, gain and loudness measurements.

## Fourth video

The 76-shot score establishes a distant signal, passes through network and
register fields, dissolves into pigment and interference, becomes more cut up,
then returns along the tunnel toward the same signal. The ending lets that
small white/black center recede into darkness.

There are four pairings, with **at most two visible layers**:

| Pair | Upper-register response |
| --- | --- |
| Perspective tunnel + monochrome Unfolding center | Hats open wall lines and light moving segments; kick pulses the distant center. |
| One painting + pixel-clock moiré | Stabs perturb the moiré phase and pigment edges; hats brighten its fine pattern. |
| Physarum + mirrored thi.ng glyphs | Hats open and tear glyph lanes; cowbell changes network exposure. |
| Bit registers + one Driftless drawing | Cowbell shifts the integer register; hats and stabs change the two materials' emphasis. |

Full-screen Unfolding appears only in short interstitials totaling **7.45
seconds**, about **2.1%** of the edit. Titles consume the second layer while
visible. Cuts switch pairings directly; there is no four-layer crossfade.
The paint's denser regions are selected using per-texture luminance quantiles,
leaving the moiré visible behind and between them. A slowly moving nutrient
field keeps the Physarum simulation from coarsening into one broad vein; that
field drives the agents and is not a third visible layer.

The high-register controls are measured from the processed, separated audio
buses at 200 Hz, with a short attack hold for 30 fps presentation. They are not
an invented uniform beat pulse. The original kick analysis still drives the
center. Simulation, material motion and palette clocks continue through cuts.

```bash
python docs/bell_riots_counterpoint.py Bell_Riots_Counterpoint_Mix.flac \
  --controls /tmp/bell-counterpoint-audio/upper-controls.npz \
  --kicks bell-kicks.npz --art-root /path/to/Generative_Art \
  --assets-cache /tmp/bell-organic-assets \
  -o Bell_Riots_Edit_4_1080p.mp4 --checkpoints render-checkpoints
```

Use the kick cache from [the sparse edit](bell-riots-sparse.md), derived from
the original recording and the same trim. `--offset 14 --seconds 32` selects
an excerpt within this edit. Excerpt offsets are rounded to the output frame;
both audio and simulation use that value. The output defaults to 1080p30,
7 Mbit/s H.264 and 320 kbit/s AAC. 4K accepts the same fixed simulation grids
but has not been performance-qualified.

## Validation and provenance

```bash
python -m pytest -q tests/test_bell_riots_counterpoint.py \
  tests/test_bell_riots_sparse.py tests/test_audio_timing.py \
  tests/test_bell_riots_organic.py tests/test_bell_riots_video.py
```

Tests cover the bass shelf's response, saturation gain and alias reduction,
measured transient timing, continuous cue coverage, the layer limit, brief
interstitials and rendering responses to upper controls in all four pairings.
Synthetic vector fixtures keep the user's artwork out of the tests.

Earlier source and influence credits remain in [the first video](bell-riots-video.md#provenance),
[the Unfolding edit](bell-riots-sparse.md), [the drawing manifest](assets/bell-organic/README.md)
and `NOTICE`. The supplied Pd patch remains an external user asset; this repo
contains the rendering/mixing adapter, not a redistributed copy of that patch.
