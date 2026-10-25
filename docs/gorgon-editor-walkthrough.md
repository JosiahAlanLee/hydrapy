# Gorgon editor: walkthrough and video capture guide

This is a reproducible recording sequence for the workstation. Record after the
GPU training job has finished. It covers the editor, a second output display,
and the Gorgon bridge in VCV Rack. The hardware checks at the end require the
actual interface and modular; container tests do not replace them.

## Before recording

1. Use the HydraPy checkout containing the editor workflow changes. Record
   `git rev-parse --short HEAD` for both HydraPy and Gorgon in the video notes.
2. Close other Studio instances. The Rack bridge discovers one advertised Studio
   through `~/.hydrapy/studio.json`.
3. Launch from the HydraPy repository:

   ```sh
   hydrapy studio patches/gorgon_corridor.py --tracker --backend gl --size 1280x720 --fps 60
   ```

   For a CPU rehearsal, use `--backend cpu --size 320x180 --fps 20` instead.
4. Open the printed local URL, including its session fragment. Do not show or
   copy that fragment into the tutorial notes. Allow output pop-ups for this site.
5. Capture the controller at 1920×1080 if available. Use **View → Text size →
   Large** if the cell text is too small in the recording. **Instruments** folds
   the left palette; **View → Automation curve** folds the lower curve panel.
6. Browser Sound starts off. Route browser audio and Rack's audio deliberately
   in the screen recorder. Keep playback levels down for the first audition.
7. Save the starting score with **Save session**. Keep its downloaded JSON so the
   same starting state can be restored between takes.

The initial score has four fixed patterns, each 64 sixteenth-note rows. The row
labels are hexadecimal: `0F` precedes `10`, and `3F` is row 63. Arrangement cards
show actual note activity in green and explicit automation keys in amber; they
are not audio waveforms.

## 1. Explain playback and editing separately

1. Press **Space** to start; show the musical position and moving green row.
2. Press Space again to pause. Click a different cell, including a visual cell.
   The amber edit cursor moves; the transport position does not.
3. Select another arrangement card. This changes the pattern being edited.
4. Click **Seek to row**. This explicitly moves playback to that pattern and row.
   It preserves the renderer's simulation time and feedback history.
5. Click the small **▶** on an arrangement card. This launches that pattern and
   starts automation. **Stop** pauses and returns to the selected pattern start,
   or the active loop's start.
6. Turn **Follow** on to follow playback. Manually selecting cells or a pattern
   turns Follow off so playback cannot move an editing selection underneath you.
7. **Loop** loops the selected pattern. While it loops, selecting another card
   only changes the editor. Launch or Seek explicitly switches the loop target.

Keep the shot on both the transport and the edit cursor when demonstrating this.
A seek does not rewind an audio file or clear the patch's feedback.

## 2. Enter a pitched phrase

1. Pause, select pattern 1 and an empty **FM hits** cell, such as row `01`.
2. Show the selected-cell editor immediately beneath the grid. Type `E4` in
   **Note**, then Tab. It displays as `E-4`; both spellings mean E4.
3. Set **Velocity** to `100`, then Tab. **Audition** turns browser Sound on and
   previews this event. Explain that FM and Sub are pitched browser demo voices.
4. Set **Octave** to `4` and **Advance** to `4 rows`.
5. Enable **Piano keys**; it focuses the grid. Type `Z`, `X`, `C`, `V` slowly.
   They enter C4, D4, E4, F4, advancing four rows after each note.
6. Show `S` for C♯4 and `Q` for C5. The complete mapping is in **? Help**.
7. Turn Piano keys off. Press **Enter** twice on an empty cell: the first press
   adds a hit and the second removes it. Focus stays in the grid.
8. Use **Undo** and **Redo** to restore and remove the last edit.

With Piano keys on, `B` enters G in the audio grid. Use the visible **BLACK OUT**
button while entering notes. Outside the audio grid, B remains the blackout
shortcut. Holding a note key does not insert an auto-repeat run of notes.

The key selector transposes the D-based score. Leave it at D minor for this
sequence so entered note names and heard pitches match. Notes currently use
12-tone equal temperament; there is no microtuning or pitch glide control.

## 3. Build a phrase with block operations

1. Select the first phrase cell. Hold **Shift** and click the last cell, or use
   Shift+arrows. A rectangle and a rows×columns count show the selected block.
   Mouse dragging also selects a block; Shift+arrows works across scrolled rows.
2. Click **Copy**, select an empty destination, then **Paste**. Empty cells in
   the copied rectangle overwrite destination cells too.
3. Demonstrate **Cut**, then Undo. **Lane** selects every row in the current
   lane; **All** or Ctrl/Cmd+A in the grid selects the whole pattern.
4. Select the pasted notes and press **+12** to move up an octave. **−1** and
   **+1** move by semitones. Alt+Up/Down are the keyboard equivalents; add Shift
   for octaves. AIR events are not transposed. A range-limit failure changes none
   of the selected notes.
5. Set Velocity, then **Apply velocity to block** to update existing notes in
   the selection. A velocity of zero makes an existing note silent.
6. Demonstrate Ctrl/Cmd+Z, then Ctrl/Cmd+Shift+Z. History holds the last 100 edits
   in this controller, including pattern operations and imports. Reload starts
   a fresh history; edits arriving from another controller clear local history.

Toolbar Copy/Paste uses a clipboard inside this controller. Native Ctrl/Cmd+C,
X and V while the grid has focus can also transfer validated Gorgon blocks through
text clipboard data. Audio can paste between audio lanes. Automation keys paste
only into the same lane type, so roll degrees cannot become feedback amounts.
Blocks crop at the pattern edge and report skipped cells.

## 4. Shape visual automation

1. Select a **Dolly Z** cell. The bottom editor changes to an exact numeric value
   and **Write key**. Enter `0.65`, then Tab; playback stays where it was.
2. Click **Seek to row** and resume playback to see that value applied.
3. Add another key using the curve. A curve click writes at the edit cursor; it
   does not seek. Select **Hold**, **Linear**, and **Smooth** in turn and show the
   difference. Interpolation applies to that lane throughout the selected pattern.
4. Copy a few visual keys to later rows of the same lane. Undo the paste.
5. Show **Patch bindings**. A missing control is marked; controls already bound
   to another lane are unavailable. Lane values keep their native units rather
   than being normalized to the bound control's slider range.
6. Explain that this patch maps Dolly Z to 2D scale, Roll to rotation and Feedback
   to recursive image memory. It is not depth reprojection.

Automation can be enabled while rendering is paused. The picture remains frozen
until rendering resumes. Turning automation off leaves the last base values in
place and lets Studio or incoming Rack CV control the parameters again.

## 5. Name and copy patterns

1. Open **Pattern tools**. Rename the pattern to `FIRST PHRASE`.
2. Choose pattern 2 under **Copy to**, then **Copy pattern to destination**.
   This replaces the destination's contents, including its name and interpolation.
3. Select pattern 2, open Pattern tools, and **Clear this pattern**. It clears
   notes and keys while retaining the pattern name. Undo restores the contents.
4. Change pattern 2's phrase and launch it with its ▶ button. Pattern order and
   lengths remain fixed; the editor does not yet provide a variable song chain.

## 6. Use separate output displays

1. Open **Program**. Move that window to display 2; press **F** there.
2. Open **Auxiliary** on display 3 if available. Select separate `o0`–`o3` routes
   in **Output routing**. Auxiliary does not change Studio's program output.
3. Show **BLACK OUT** and restore. It affects browser monitors, not the rendering
   engine, recordings, snapshots, or an external video device.
4. After saving, close the controller briefly. The output windows and Studio's
   automation keep running. The controller's browser synth stops with its window.
5. Reopen the controller using Studio's URL. Do not promise identical physical
   audio/video latency without measuring the workstation.

## 7. Connect Rack

1. Start Rack with the current GorgonBridge installed. Add **Gorgon**. Its display
   should show **Linked** and Studio's tempo without a manual address or token.
2. Patch CLOCK and RESET to a sequencer; patch RUN where the destination supports
   a run signal. Gorgon/Studio remains the master clock.
3. For pitched voices, split the polyphonic V/OCT, GATES and VEL outputs. Their
   four channels correspond to Breaks, FM hits, Sub and Texture in that order.
   Start with channel 2 (FM hits) or channel 3 (Sub).
4. Demonstrate an octave change and mute. In Rack, C4 is 0 V by default; the context
   menu also offers C3 as 0 V. AIR keeps held pitch; a zero-velocity event sends no
   gate. Browser Sound does not control the hardware or Rack gates.
5. Patch a slow CV into CV 1 and choose a named free patch control under **CV 1
   drives**. 0–10 V spans that control's range. A control owned by an enabled
   tracker lane ignores incoming CV; choose an unbound control or turn automation
   off before demonstrating it.
6. Demonstrate PLAY and RESET triggers. They command Studio rather than creating
   a second independent transport.
7. Explain that the present module continues its cached score if Studio
   disconnects. Show link status before relying on its outputs.

Use a loopback click-and-flash recording to set OFFSET. Its positive direction
advances Rack outputs relative to the picture. A software phase-error test is
not a measurement of DAC, oscillator or monitor latency. Physical pitch accuracy
and pitch-before-gate settling need separate measurements.

## 8. Save, restore and recover

1. **Save session** or Ctrl/Cmd+S writes the workspace score and downloads the
   same JSON. **File → Restore saved score** reloads that workspace copy.
2. **File → Import score** accepts a saved JSON. The import can be undone in this
   controller. **New empty score** clears all four patterns and can also be undone.
3. **File → Download recovery copy** downloads the local score even if it has not
   reached Studio. A failed edit request displays a recovery banner. **Retry edit**
   first checks the server score, including the case where an acknowledgement was
   lost after the server accepted the edit.
4. A genuine concurrent edit is not overwritten. Download the local recovery,
   then **Use server score** if you choose to discard the local version. Transport
   starts and seeks wait until pending edits are resolved. Pause, Stop and
   Blackout remain available during edit recovery. There is no offline editing
   guarantee or automatic merge between controllers.
5. Browser reload reconnects to the current server score. Restarting Studio still
   requires an explicit Restore or Import to recover a persisted score.
6. Use **File → Patch editor & export** for Studio's recording and export tools.
   Enabled visual automation is included in supported video exports; the browser
   demo synth is not. Record Rack or the modular separately when making a music
   video with those voices.

## Recording finish checks

- Save the final score and include its JSON alongside the video project.
- Confirm the recorded browser audio, Rack audio and video are the intended ones.
- Check an octave jump, a silent event, a pause/resume and a loop boundary.
- Reopen both score and Rack patch before calling the walkthrough repeatable.
- Record any workstation-specific PipeWire routing or latency values as measured
  setup notes, not as defaults for other machines.
