// Score operations have no DOM, audio clock, or transport side effects.
import {noteMidi, canonicalNote} from './notes.js';

export const LANES = ['z', 'roll', 'feedback'];
const LIMITS = [[0, 1], [-45, 45], [0, .9]];
export const COLUMNS = 8;                     // four audio lanes, three automation lanes, SCENE
export const SCENE = 7;
export const SCENE_NAME = /^[A-Za-z0-9_-]{1,24}$/;
// What a column holds, so blocks only paste into columns of the same kind.
export const kindOf = col => col < 4 ? 'audio' : col < SCENE ? LANES[col - 4] : 'scene';
const EDITABLE = ['patterns', 'bpm', 'key', 'mutes', 'bindings'];
const clone = value => structuredClone(value);
const snapshot = score => Object.fromEntries(EDITABLE.map(key => [key, clone(score[key])]));
const apply = (score, data) => EDITABLE.forEach(key => { score[key] = clone(data[key]); });

export class ScoreHistory {
  constructor(limit = 100) { this.limit = limit; this.clear(); }
  clear() { this.past = []; this.future = []; }
  change(score, label, mutate, context, merge = '') {
    const before = snapshot(score);
    mutate();
    const after = snapshot(score);
    if (JSON.stringify(before) === JSON.stringify(after)) return false;
    const last = this.past.at(-1), now = Date.now();
    if (merge && last?.merge === merge && now - last.time < 600 && !this.future.length) {
      last.after = after;
      last.time = now;
    } else {
      this.past.push({before, after, label, context: clone(context), merge, time: now});
      if (this.past.length > this.limit) this.past.shift();
    }
    this.future = [];
    return true;
  }
  undo(score) {
    const entry = this.past.pop();
    if (!entry) return null;
    apply(score, entry.before);
    this.future.push(entry);
    return entry;
  }
  redo(score) {
    const entry = this.future.pop();
    if (!entry) return null;
    apply(score, entry.after);
    this.past.push(entry);
    return entry;
  }
}

export function bounds(cell, anchor = cell) {
  return {r0: Math.min(cell.row, anchor.row), r1: Math.max(cell.row, anchor.row),
          c0: Math.min(cell.col, anchor.col), c1: Math.max(cell.col, anchor.col)};
}

export function getCell(pattern, row, col) {
  if (col === SCENE) return (pattern.scenes ?? []).find(key => key.r === row) ?? null;
  return col < 4 ? pattern.audio[col][row] : pattern.keys[LANES[col - 4]].find(key => key.r === row) ?? null;
}

export function putCell(pattern, row, col, value) {
  if (col === SCENE) {
    pattern.scenes = (pattern.scenes ?? []).filter(key => key.r !== row);
    if (value) pattern.scenes.push({r: row, name: value.name, fade: value.fade ?? 0});
    pattern.scenes.sort((a, b) => a.r - b.r);
  } else if (col < 4) {
    pattern.audio[col][row] = value ? {note: value.note, vel: value.vel} : null;
  } else {
    const lane = LANES[col - 4];
    pattern.keys[lane] = pattern.keys[lane].filter(key => key.r !== row);
    if (value) pattern.keys[lane].push({r: row, v: value.v});
    pattern.keys[lane].sort((a, b) => a.r - b.r);
  }
}

export function copyBlock(pattern, box) {
  const columns = Array.from({length: box.c1 - box.c0 + 1}, (_, i) => kindOf(box.c0 + i));
  const rows = Array.from({length: box.r1 - box.r0 + 1}, (_, r) =>
    columns.map((_, c) => clone(getCell(pattern, box.r0 + r, box.c0 + c))));
  return {format: 'gorgon-block-1', columns, rows};
}

// Clipboard text is untrusted. Validate before either rendering it or changing a score.
export function readBlock(text) {
  if (text.length > 100000) throw new Error('Clipboard block is too large.');
  const block = JSON.parse(text);
  if (block?.format !== 'gorgon-block-1' || !Array.isArray(block.columns) || !block.columns.length ||
      block.columns.length > COLUMNS || block.columns.some(c => c !== 'audio' && c !== 'scene' && !LANES.includes(c)) ||
      !Array.isArray(block.rows) || !block.rows.length || block.rows.length > 64) throw new Error('Copy a Gorgon cell block first.');
  for (const row of block.rows) {
    if (!Array.isArray(row) || row.length !== block.columns.length) throw new Error('Invalid clipboard row.');
    row.forEach((value, c) => {
      if (value === null) return;
      if (!value || typeof value !== 'object') throw new Error('Invalid clipboard cell.');
      if (block.columns[c] === 'audio') {
        value.note = canonicalNote(value.note);
        if (!Number.isInteger(value.vel) || value.vel < 0 || value.vel > 127) throw new Error('Invalid clipboard velocity.');
      } else if (block.columns[c] === 'scene') {
        if (typeof value.name !== 'string' || !SCENE_NAME.test(value.name)) throw new Error('Invalid clipboard scene name.');
        value.fade ??= 0;
        if (!Number.isInteger(value.fade) || value.fade < 0 || value.fade > 64) throw new Error('Invalid clipboard scene fade.');
      } else {
        const [low, high] = LIMITS[LANES.indexOf(block.columns[c])];
        if (!Number.isFinite(value.v) || value.v < low || value.v > high) throw new Error('Invalid clipboard automation value.');
      }
    });
  }
  return block;
}

export function pasteBlock(pattern, start, block, {lastColumn = COLUMNS - 1} = {}) {
  let written = 0, skipped = 0;
  block.rows.forEach((row, r) => row.forEach((value, c) => {
    const dest = start.col + c, y = start.row + r;
    if (dest >= COLUMNS || dest > lastColumn || y > 63) { skipped++; return; }
    const kind = kindOf(dest);
    if (kind !== block.columns[c]) { skipped++; return; }
    putCell(pattern, y, dest, value);
    written++;
  }));
  return {written, skipped};
}

export function clearBlock(pattern, box) {
  for (let r = box.r0; r <= box.r1; r++) for (let c = box.c0; c <= box.c1; c++) putCell(pattern, r, c, null);
}

export function midiNote(midi) {
  if (!Number.isInteger(midi) || midi < 12 || midi > 127) return null;
  return ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-'][midi % 12] + (Math.floor(midi / 12) - 1);
}

export function transposeBlock(pattern, box, semitones) {
  // Preflight the entire selection: never partially transpose a chord at a range limit.
  const changes = [];
  for (let r = box.r0; r <= box.r1; r++) for (let c = box.c0; c <= Math.min(3, box.c1); c++) {
    const hit = pattern.audio[c][r];
    if (!hit || hit.note === 'AIR') continue;
    const note = midiNote(noteMidi(hit.note) + semitones);
    if (!note) throw new Error('Transpose would put a note outside C0–G9.');
    changes.push([hit, note]);
  }
  changes.forEach(([hit, note]) => { hit.note = note; });
}

export function setBlockVelocity(pattern, box, velocity) {
  for (let r = box.r0; r <= box.r1; r++) for (let c = box.c0; c <= Math.min(3, box.c1); c++) {
    if (pattern.audio[c][r]) pattern.audio[c][r].vel = velocity;
  }
}

// Z–M is the selected octave, Q–U the next. Only active in the focused note grid.
const PIANO = {KeyZ:0, KeyS:1, KeyX:2, KeyD:3, KeyC:4, KeyV:5, KeyG:6, KeyB:7, KeyH:8, KeyN:9, KeyJ:10, KeyM:11,
               KeyQ:12, Digit2:13, KeyW:14, Digit3:15, KeyE:16, KeyR:17, Digit5:18, KeyT:19, Digit6:20, KeyY:21, Digit7:22, KeyU:23};
export function pianoNote(code, octave) { return code in PIANO ? midiNote(12 * (octave + 1) + PIANO[code]) : null; }

export function activity(pattern) {
  return Array.from({length:64}, (_, r) => ({
    audio: pattern.audio.reduce((n, lane) => n + (lane[r]?.vel > 0 ? 1 : 0), 0),
    keys: LANES.reduce((n, lane) => n + (pattern.keys[lane].some(k => k.r === r) ? 1 : 0), 0) +
          ((pattern.scenes ?? []).some(k => k.r === r) ? 1 : 0),
  }));
}
