# Tensor Junglist: two-tone study

A short alternative visual direction for Anchor Rite's **Tensor Junglist**:
three sustained patterns, a fixed two-color palette, integer pixel boundaries,
small persistent edits, and two cuts on tracked beats. The 42-second study uses
the recording from 1:52 to 2:34. The music is not time-stretched or rearranged.

```bash
pip install -e '.[gpu,video]'
python docs/two_tone_study.py /path/to/track.flac \
  -o Two_Tone_Study.mp4 --start 112 --seconds 42
```

FFmpeg must be available. Default output is 1920×1080 at 30 fps. `--seed 57`
changes the defects and edit sequence; `--pixel 6` makes larger cells;
`--cache analysis.npz` reuses the recording's existing analysis cache. The cache
is checked against the audio's SHA-256. `--stills contact.png` additionally saves
six frames from the rendered sequence for inspection. A JSON sidecar records
the settings and exact scene boundaries.

| Passage | Starting pattern | Ongoing behavior | Entropy lock |
| --- | --- | --- | --- |
| Register | History of elementary cellular rules | Small cellular updates and row shifts | 0.08 |
| Lattice | XOR, AND, and bit shifts on integer coordinates | Local transport and interference | 0.30 |
| Quadrants | Repeated binary motifs | Local Game of Life updates and recurring edits | 0.65 |

The controls use HydraPy's existing `Fx` entropy locking, after **aebrer**:
each random draw can reset the generator to its seed state, causing parts of the
control sequence to recur. This is distinct from freezing a framebuffer.
The state continues evolving between recurring edits. The independent cellular
clock runs at 12 Hz; the tracked beats and detected onsets trigger local shifts.

The simulation holds a binary field in two R8UI textures (432×240 at the default
settings). Fragment shaders perform integer updates on that small field, then
one display pass maps its cells into two RGB colors with a fixed margin and
stable reflection. There is one RGB readback per delivered video frame. It uses
OpenGL 3.3, without requiring compute shaders or a neural model. Hardware
encoding is tried automatically; software encoding is the fallback.

The user's visual references were **mono / @mono_krome_** (*Energy Ripples*,
*Energy Quadrants*, and *Modulo Art*) and **nicthib** (*56 Modes*, *partition()*).
These are original compositions and update rules informed by that direction;
no edition code, images, or assets from those works were copied. The elementary
automata and Life rules are standard algorithms, and this study does not claim
to reproduce *56 Modes*' specific hybrid system.

- [mono's Energy Quadrants announcement](https://x.com/mono_krome_/status/1518845552611532801)
- [56 Modes on fxhash](https://www.fxhash.xyz/generative/2395)
- [partition() on fxhash](https://www.fxhash.xyz/generative/1844)
- [Existing entropy-locking implementation](../hydrapy/signals.py)

The source track and generated videos stay outside the repository. Validation:
`python -m pytest tests/test_two_tone_study.py -q` checks deterministic binary
initialization, integer texture updates, conservation under local shifts,
unchanged cells outside the edited rows, exact presentation colors, and symmetry.
