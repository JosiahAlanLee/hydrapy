import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {ScoreHistory, bounds, copyBlock, readBlock, pasteBlock, clearBlock, putCell,
  transposeBlock, setBlockVelocity, pianoNote, activity} from '../hydrapy/tracker/editor.js';
const score = () => JSON.parse(readFileSync(new URL('../hydrapy/tracker/demo.json', import.meta.url)));
const box = {r0:0,r1:3,c0:0,c1:6};

test('mixed blocks preserve sparse automation, clear empty destinations and crop at boundaries', () => {
  const d=score(), p=d.patterns[0];
  clearBlock(p,{r0:0,r1:63,c0:0,c1:6});
  putCell(p,1,1,{note:'C#4',vel:91});putCell(p,2,4,{v:.75});putCell(p,3,5,{v:-12});
  const block=readBlock(JSON.stringify(copyBlock(p,box)));
  putCell(d.patterns[1],11,1,{note:'C-3',vel:110});
  const result=pasteBlock(d.patterns[1],{row:10,col:0},block);
  assert.deepEqual(result,{written:28,skipped:0});
  assert.deepEqual(d.patterns[1].audio[1][11],{note:'C#4',vel:91});
  assert.equal(d.patterns[1].audio[0][10],null);
  assert.deepEqual(d.patterns[1].keys.z.find(k=>k.r===12),{r:12,v:.75});
  assert(!d.patterns[1].keys.z.some(k=>k.r===10));
  assert.deepEqual(pasteBlock(p,{row:63,col:0},block),{written:7,skipped:21});
  assert.deepEqual(pasteBlock(p,{row:20,col:1},block),{written:12,skipped:16});
  p.audio[1][1].vel=2;assert.equal(block.rows[1][1].vel,91,'clipboard is independent');
});

test('untrusted clipboard rejects malformed notes, non-finite values and out-of-range lanes', () => {
  for(const value of [
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'<img src=x onerror=alert(1)>',vel:90}]]},
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'C4',vel:128}]]},
    {format:'gorgon-block-1',columns:['feedback'],rows:[[{v:45}]]},
    {format:'gorgon-block-1',columns:['z'],rows:[[{v:'0.5'}]]},
    {format:'gorgon-block-1',columns:['roll'],rows:[[]]},
    {format:'gorgon-block-1',columns:['audio'],rows:Array(65).fill([null])},
  ]) assert.throws(()=>readBlock(JSON.stringify(value)));
  assert.throws(()=>readBlock('x'.repeat(100001)));
});

test('transpose preflights the whole block and preserves AIR and velocities', () => {
  const p=score().patterns[0];clearBlock(p,{r0:0,r1:63,c0:0,c1:6});
  putCell(p,0,0,{note:'C-4',vel:100});putCell(p,1,0,{note:'AIR',vel:64});putCell(p,2,0,{note:'G-9',vel:80});
  assert.throws(()=>transposeBlock(p,box,1));assert.equal(p.audio[0][0].note,'C-4');
  transposeBlock(p,{r0:0,r1:1,c0:0,c1:3},12);
  assert.equal(p.audio[0][0].note,'C-5');assert.equal(p.audio[0][1].note,'AIR');
  setBlockVelocity(p,box,0);assert.equal(p.audio[0][0].vel,0);assert.equal(p.audio[1][0],null);
});

test('undo/redo restores compositions while preserving current routing and transport preferences', () => {
  const d=score(), h=new ScoreHistory(2),context={pattern:0,selected:{row:3,col:1}};
  const initial=structuredClone(d.patterns);
  h.change(d,'clear',()=>clearBlock(d.patterns[0],box),context);
  d.pattern=3;d.loop=true;d.sound=true;d.modes.program='o3';
  assert.deepEqual(h.undo(d).context,context);assert.deepEqual(d.patterns,initial);
  assert.equal(d.pattern,3);assert(d.loop&&d.sound);assert.equal(d.modes.program,'o3');
  h.redo(d);assert.equal(d.patterns[0].audio[0][0],null);
  h.undo(d);h.change(d,'tempo',()=>{d.bpm=91;},context);assert.equal(h.future.length,0);
  h.change(d,'tempo',()=>{d.bpm=92;},context);h.change(d,'tempo',()=>{d.bpm=93;},context);
  assert.equal(h.past.length,2);h.undo(d);assert.equal(d.bpm,92);
});

test('continuous slider changes coalesce and no-ops do not erase redo', () => {
  const d=score(),h=new ScoreHistory(),ctx={pattern:0,selected:{row:0,col:4}};
  h.change(d,'key',()=>putCell(d.patterns[0],0,4,{v:.4}),ctx,'z0');
  h.change(d,'key',()=>putCell(d.patterns[0],0,4,{v:.7}),ctx,'z0');
  assert.equal(h.past.length,1);h.undo(d);assert.equal(h.future.length,1);
  h.change(d,'no-op',()=>{},ctx);assert.equal(h.future.length,1);
});

test('keyboard octaves use the score note convention; activity follows real events', () => {
  assert.equal(pianoNote('KeyZ',4),'C-4');assert.equal(pianoNote('KeyS',4),'C#4');
  assert.equal(pianoNote('KeyQ',4),'C-5');assert.equal(pianoNote('KeyU',8),null);
  assert.equal(pianoNote('KeyF',4),null);
  assert.deepEqual(bounds({row:3,col:1},{row:1,col:3}),{r0:1,r1:3,c0:1,c1:3});
  const p=score().patterns[0];clearBlock(p,{r0:0,r1:63,c0:0,c1:6});
  putCell(p,3,0,{note:'C-4',vel:0});putCell(p,3,1,{note:'C-5',vel:100});putCell(p,3,4,{v:.5});
  assert.deepEqual(activity(p)[3],{audio:1,keys:1});
});
