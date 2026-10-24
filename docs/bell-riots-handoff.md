# Bell Riots — handoff for an alternate edit 5

This handoff is for a Claude Code session on the render machine, which has the GPU, Blender 5.1+ and Pd. The session will work with the user on an alternate version of [edit 5](bell-riots-dancer.md). It comes from a measurement pass over the delivered `Bell_Riots_Edit_5_1080p.mp4`. Numbers come from [`bell_riots_checks.py`](bell_riots_checks.py) unless noted.

## Pipeline

```
original recording ──┐
bell_hats Pd score ──┴─ bell_riots_mix.py ──> mix .flac + upper-controls.npz ──┐
Dancer_Overlay.blend ── render_body_roll.py ── prepare_dancer_matte.py ──> matte ┤
kick cache (bell-kicks.npz, from the Unfolding edit) ───────────────────────────┤
Generative_Art drawings ────────────────────────────────────────────────────────┴─ bell_riots_dancer.py ──> MP4 + .cues.json
```

Commands are in three docs:
- [Counterpoint mix / edit 4](bell-riots-counterpoint.md): the mix and the shot score.
- [Dancer / edit 5](bell-riots-dancer.md): Blender print, matte and render.
- [Unfolding edit](bell-riots-sparse.md): how the kick cache is made.

The picture's upper-register responses are measured from the mix. Any audio change therefore means re-rendering the video.

External inputs are not in the repository. Ask the user where they are:
- the original recording (the edit is its last 6:01.093, from 7:14.462)
- the unzipped `bell_hats_v2` folder
- Pd vanilla with `rev3~`
- a Generative_Art checkout
- `Dancer_Overlay.blend` from the Mesh2Motion CC0 kit
- the kick cache

## Grid

- **Tempo and beats:** 137.000 BPM. Score beat 0 sits at −8.301 ms, and the first heard beat (b = 1) at 429.655 ms. The edit is 826 beats, 361.093 s.
- **Clocks:** at 30 fps one beat is 13.14 frames. The dancer clock runs 15 ticks per beat and 480 ticks per 32-beat cycle (14.0146 s).
- **Shot list:** the 76 cues are `CounterpointScore`'s act table. They depend only on beat numbers.
- **Downbeat:** the track's harmony stays on a D pedal throughout, so pitch can't confirm where bars start. The Pd score and the shot score share one assumed bar frame, with bars starting at b ≡ 0 (mod 4).

## Edit 5 as delivered

| Measure | Value |
| --- | --- |
| Screen time | Tunnel 25.7%, paint/moiré with the dancer 38.1%, glyphs/Physarum 21.7%, registers/Driftless 12.5%, Unfolding bridges 2.1% (score totals) |
| Cuts | 74 of the 75 cue starts are found in the picture, with none extra. All fall on a beat, on the first frame at or after it (median +17.7 ms) |
| Cut phase | 59 of 74 cuts fall on score beats b ≡ 2 (mod 4). Each act opens with a 2-beat bridge on the bar line, so the act's later cuts land on beat 3 of the score's bar |
| Hat rolls | 36 rolls; 5 lead into a cut (chance ≈ 9%) |
| Dancer | Poses repeat with the 480-tick clock (same-tick pose similarity 0.72, against 0.46 for random pairs). Only 3 of 15 motion holds fall within 3 ticks of a half-time beat (chance ≈ 23%) |
| Audio | −16.0 LUFS, −1.2 dBTP, loudness range 0.9 LU. The groove stays at full level to the end. The mix applies 8 ms / 350 ms edge fades |
| Encode | x264 single-pass ABR at 7000 kbit/s (the renderer's default `--bitrate 7M`) |

## Changes for the alternate version

### 1. Re-score the hats to this edit's cuts

`bell_hats_v2`'s per-beat arrays were generated from the organic edit's cuts. Its pattern changes, rolls and stabs therefore belong to that edit, not this one. [`bell_riots_rescore.py`](bell_riots_rescore.py) rebuilds the arrays from the 76 cues, with the same rules and seeds, in a new folder. The patch, voices and bar patterns are copied unchanged.

```bash
python docs/bell_riots_rescore.py --patch-dir /path/to/bell_hats --out /path/to/bell_hats_edit5
python docs/bell_riots_mix.py /path/to/original.flac --stem-root /path/to/bell_hats_edit5 \
  --pd /path/to/pd --cache /tmp/bell-alt-audio -o Bell_Riots_Alt_Mix.flac
```

Then render as in the dancer doc, with the new mix and `/tmp/bell-alt-audio/upper-controls.npz`.

- **Bar phase (ask the user):** `--bar-offset 0` keeps v2's bar frame. `--bar-offset 2` starts the bar patterns on the beat where most cuts land. Pd renders in seconds, so render both and compare by ear.
- **Bridge rolls:** `--roll-into-bridges` forces a roll into every Unfolding bridge. It is off by default.
- **Reverb:** keep mixing through `bell_riots_mix.py`. v2's `rev3~` objects have output level 0, which is muted on Pd's 0–100 scale; the mix script restores 82/80.
- **Done when** `python docs/bell_riots_checks.py rolls ALT.mp4` reports every roll leading into a cut.

### 2. Dancer accents (the user's call)

The half-time loop is locked by design: one Body Roll cycle per 8 bars, anchored to measured kicks. Within the cycle, the movement's holds and surges land at arbitrary points relative to the half-time beats. It reads as slow motion rather than hitting.

If the user wants hits:
1. In `render_body_roll.py`, replace the linear key scaling with a monotonic piecewise-linear time map. The map should move the action's natural holds onto multiples of 30 output frames.
2. Keep 480 frames per cycle so the clock, matte and delay taps still line up.
3. Re-print, rebuild the matte and re-render.
4. Check that `holds_within_3_ticks_of_half_beat` rises well above 0.23.

### 3. Ending (the user's call)

The picture recedes down the tunnel for the last 38 s (from 5:23) and fades out over the final 1.6 s. The recording plays at full level until its own end, with a 350 ms fade.

Options:
- a longer audio fade (the `fade` array in `bell_riots_mix.py`'s `make_mix`)
- a shorter act 5
- leaving it as it is, since this is how the recording ends

### 4. Master encode

On any re-encode, the 7 Mbit/s single-pass ABR smears the fine moiré (about 5 px line spacing) and the bridge frames. For an upload master, either render at a much higher bitrate (`--bitrate 40M`) or expose `hydrapy.save_mp4`'s quality path, which uses CRF when `bitrate=None`. Keep 7M for previews.

### Optional

- **Title card:** the small lines are about 12 px tall at 1080p and vanish on a phone. The fonts are DejaVu Sans and DejaVu Sans Mono, via `find_font`.
- **Framing:** every pairing is centred, and the dancer matte sits mid-frame at a single scale. Adding a per-cue scale and offset on the dancer tap in `DancerRenderer` would move it without re-printing the Blender scene; `delay_patch` already scales it by 1.10.

## Checks

```bash
python -m pytest -q tests/test_bell_riots_rescore.py tests/test_bell_riots_dancer.py \
  tests/test_bell_riots_counterpoint.py
python docs/bell_riots_checks.py all Bell_Riots_Alt_1080p.mp4 --json alt-checks.json
```

Compare the output with the edit 5 table. The thresholds that find the dancer assume the chalk-white figure (`--figure-min`, `--figure-sat`). One full check pass takes about 1.5 minutes on a 2-core CPU.

## Questions for the user before rendering

1. Hat bar phase: 0 or 2?
2. Dancer: keep the float, or warp it so it hits the half-time beats?
3. Ending: leave it, lengthen the audio fade, or shorten the tunnel?
4. Where will the master go: a platform upload, phone viewing or a screening? The answer sets the bitrate and the title size.
