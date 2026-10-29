# Per-cell note length in score v3

An audio cell may include `len`, a finite JSON number from **0.25 to 16**, inclusive.
For example, `{"note":"C-4","vel":100,"len":3}` holds its gate for 1440 ticks.
Omitting `len` preserves existing playback, including the Gate length fraction
(default half of a row or `ply` sub-step). Blank Length in the editor removes `len`.

Select an audio cell and press **Alt+L**, type a length, then Tab to apply it.
The inspector also exposes Length in its normal tab order. The grid shows `L3`
for a length of three. Copy/paste, transpose, velocity edits and undo/redo retain
length. Using length upgrades a legacy document to format 3; undo restores it.

Length uses **base rows of 480 ticks**, independent of the pattern's Rate.
At 120 BPM, a base row is 125 ms. With Rate 2, `len:1` spans two pattern rows.
Fractional lengths need not land on whole ticks: `len:0.333` is 159.84 ticks.
Candidate onsets and evaluator interval boundaries remain integer ticks.
`ply` still places sub-hits within the row; each inherits the cell's full length.
Each candidate's duration is `min(len * 480, pattern_end - candidate_at)`.
Length follows the source cell through rotation, reversal, conditions and degrade.
Zero velocity and muted/failed candidates open no gate.

Explicit lengths bypass Gate length in patch lane signals and TrackerCV.
Lanes are monophonic: a later firing hit replaces the earlier note and gate;
an earlier long note never resumes after its replacement ends. If either hit
has an explicit length and the previous span reaches the new onset, the new
hit starts with **1 ms low**, then opens for the remainder of its span.
This also separates notes whose spans meet exactly, including across callbacks.
Transport changes retain their existing 1 ms retrigger when interrupting a high gate.

Seeking or attaching within a held explicit note restores its remaining span,
even when its onset is more than four rows earlier. Explicit notes do not get
an extended late landing pulse; legacy notes keep their current landing behavior.
All notes with explicit length are **clipped at the pattern entry's end**,
even if the following entry plays the same pattern. Ties across pattern
boundaries are not supported. Length controls gates and candidate spans; existing
one-shot audition/drum envelopes keep their own sound duration.

The canonical score-v3 document and JSON Schema live in Gorgon's `spec/`.
The schema addition is an optional cell property:

```json
"len": {"type": "number", "minimum": 0.25, "maximum": 16}
```

Add this to the version-3 cell definition only; keep `note` and `vel` required,
retain existing optional locks and conditions, and reject non-finite values.
No format or package version changes are needed. Gorgon's Rack parser, evaluator,
trace and follower need matching changes before Rack plays these lengths.
