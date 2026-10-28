import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {ScoreHistory, bounds, copyBlock, readBlock, pasteBlock, clearBlock, putCell,
  transposeBlock, setBlockVelocity, pianoNote, activity} from '../hydrapy/tracker/editor.js';
import {breaksVoice} from '../hydrapy/tracker/notes.js';
import {incoming, settleFormat, emptyPattern} from '../hydrapy/tracker/score3.js';
// The editor keeps scores as version 3: the demo arrives mapped, marked with its old format.
const score = () => incoming(JSON.parse(readFileSync(new URL('../hydrapy/tracker/demo.json', import.meta.url))));
// Columns: audio lanes 0-5, Dolly Z 6, Roll 7, Feedback 8, SCENE 9.
const box = {r0:0,r1:3,c0:0,c1:8};

test('mixed blocks preserve sparse automation, clear empty destinations and crop at boundaries', () => {
  const d=score(), p=d.patterns[0];
  clearBlock(p,{r0:0,r1:63,c0:0,c1:8});
  putCell(p,1,1,{note:'C#4',vel:91});putCell(p,2,6,{v:.75});putCell(p,3,7,{v:-12});
  const block=readBlock(JSON.stringify(copyBlock(p,box)));
  putCell(d.patterns[1],11,1,{note:'C-3',vel:110});
  const result=pasteBlock(d.patterns[1],{row:10,col:0},block);
  assert.deepEqual(result,{written:36,skipped:0});
  assert.deepEqual(d.patterns[1].lanes[1].cells[11],{note:'C#4',vel:91});
  assert.equal(d.patterns[1].lanes[0].cells[10],null);
  assert.deepEqual(d.patterns[1].keys.z.find(k=>k.r===12),{r:12,v:.75});
  assert(!d.patterns[1].keys.z.some(k=>k.r===10));
  assert.deepEqual(pasteBlock(p,{row:63,col:0},block),{written:9,skipped:27});
  assert.deepEqual(pasteBlock(p,{row:20,col:1},block),{written:20,skipped:16});   // five audio columns still land on audio
  p.lanes[1].cells[1].vel=2;assert.equal(block.rows[1][1].vel,91,'clipboard is independent');
});

test('untrusted clipboard rejects malformed notes, non-finite values, conditions and out-of-range lanes', () => {
  for(const value of [
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'<img src=x onerror=alert(1)>',vel:90}]]},
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'C4',vel:128}]]},
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'C4',vel:90,cond:{kind:'every',a:3,b:2}}]]},
    {format:'gorgon-block-1',columns:['audio'],rows:[[{note:'C4',vel:90,cond:{kind:'pre'}}]]},
    {format:'gorgon-block-1',columns:['feedback'],rows:[[{v:45}]]},
    {format:'gorgon-block-1',columns:['z'],rows:[[{v:'0.5'}]]},
    {format:'gorgon-block-1',columns:['roll'],rows:[[]]},
    {format:'gorgon-block-1',columns:['audio'],rows:Array(129).fill([null])},
  ]) assert.throws(()=>readBlock(JSON.stringify(value)));
  assert.throws(()=>readBlock('x'.repeat(100001)));
  const kept=readBlock(JSON.stringify({format:'gorgon-block-1',columns:['audio'],rows:[[{note:'C-4',vel:90,cond:{kind:'prob',p:30,not:true,x:1}}]]}));
  assert.deepEqual(kept.rows[0][0].cond,{kind:'prob',p:30,not:true});
});

test('transpose preflights the whole block and preserves AIR and velocities', () => {
  const p=score().patterns[0];clearBlock(p,{r0:0,r1:63,c0:0,c1:8});
  putCell(p,0,0,{note:'C-4',vel:100});putCell(p,1,0,{note:'AIR',vel:64});putCell(p,2,0,{note:'G-9',vel:80});
  assert.throws(()=>transposeBlock(p,box,1));assert.equal(p.lanes[0].cells[0].note,'C-4');
  transposeBlock(p,{r0:0,r1:1,c0:0,c1:5},12);
  assert.equal(p.lanes[0].cells[0].note,'C-5');assert.equal(p.lanes[0].cells[1].note,'AIR');
  setBlockVelocity(p,box,0);assert.equal(p.lanes[0].cells[0].vel,0);assert.equal(p.lanes[1].cells[0],null);
});

test('a short lane repeats: its rows past the length show and edit the cell they play, once', () => {
  const p=emptyPattern(0);
  p.lanes[2].len=3;p.lanes[2].cells=[null,null,null];
  putCell(p,7,2,{note:'C-4',vel:90});                   // row 7 plays cell 1
  assert.deepEqual(p.lanes[2].cells[1],{note:'C-4',vel:90});
  assert.deepEqual(activity(p).filter(r=>r.audio).length,21);  // rows 1, 4, 7 … 61
  transposeBlock(p,{r0:0,r1:63,c0:2,c1:2},1);           // 21 rows, one hit: up once
  assert.equal(p.lanes[2].cells[1].note,'C#4');
  putCell(p,4,2,{note:'D-4',vel:90,cond:{kind:'fill',not:true}});
  assert.deepEqual(p.lanes[2].cells[1],{note:'D-4',vel:90,cond:{kind:'fill',not:true}});
});

test('undo/redo restores compositions while preserving current routing and transport preferences', () => {
  const d=score(), h=new ScoreHistory(2),context={pattern:0,selected:{row:3,col:1}};
  const initial=structuredClone(d.patterns);
  h.change(d,'clear',()=>clearBlock(d.patterns[0],box),context);
  d.pattern=3;d.loop=true;d.sound=true;d.modes.program='o3';
  assert.deepEqual(h.undo(d).context,context);assert.deepEqual(d.patterns,initial);
  assert.equal(d.pattern,3);assert(d.loop&&d.sound);assert.equal(d.modes.program,'o3');
  h.redo(d);assert.equal(d.patterns[0].lanes[0].cells[0],null);
  h.undo(d);h.change(d,'tempo',()=>{d.bpm=91;},context);assert.equal(h.future.length,0);
  h.change(d,'tempo',()=>{d.bpm=92;},context);h.change(d,'tempo',()=>{d.bpm=93;},context);
  assert.equal(h.past.length,2);h.undo(d);assert.equal(d.bpm,92);
});

test('an upgrade to version 3 and its undo travel with the edit', () => {
  const d=score(),h=new ScoreHistory();
  assert.equal(d.legacy,'hydrapy-tracker-1');
  h.change(d,'chain',()=>{d.chain=[0,0,1];settleFormat(d);},{});
  assert.equal(d.legacy,undefined);
  h.undo(d);assert.equal(d.legacy,'hydrapy-tracker-1');assert.deepEqual(d.chain,[0,1,2,3]);
  h.redo(d);assert.equal(d.legacy,undefined);
});

test('continuous slider changes coalesce and no-ops do not erase redo', () => {
  const d=score(),h=new ScoreHistory(),ctx={pattern:0,selected:{row:0,col:6}};
  h.change(d,'key',()=>putCell(d.patterns[0],0,6,{v:.4}),ctx,'z0');
  h.change(d,'key',()=>putCell(d.patterns[0],0,6,{v:.7}),ctx,'z0');
  assert.equal(h.past.length,1);h.undo(d);assert.equal(h.future.length,1);
  h.change(d,'no-op',()=>{},ctx);assert.equal(h.future.length,1);
});

test('keyboard octaves use the score note convention; activity follows real events', () => {
  assert.equal(pianoNote('KeyZ',4),'C-4');assert.equal(pianoNote('KeyS',4),'C#4');
  assert.equal(pianoNote('KeyQ',4),'C-5');assert.equal(pianoNote('KeyU',8),null);
  assert.equal(pianoNote('KeyF',4),null);
  assert.deepEqual(bounds({row:3,col:1},{row:1,col:3}),{r0:1,r1:3,c0:1,c1:3});
  const p=score().patterns[0];clearBlock(p,{r0:0,r1:63,c0:0,c1:8});
  putCell(p,3,0,{note:'C-4',vel:0});putCell(p,3,1,{note:'C-5',vel:100});putCell(p,3,5,{note:'C-5',vel:100});putCell(p,3,6,{v:.5});
  assert.deepEqual(activity(p)[3],{audio:2,keys:1});
});

test('SCENE cells copy, paste only into SCENE, validate clipboard names and count as activity', async () => {
  const {getCell, SCENE, kindOf} = await import('../hydrapy/tracker/editor.js');
  const d=score(), p=d.patterns[0];
  assert.equal(SCENE,9);assert.equal(kindOf(SCENE),'scene');assert.equal(kindOf(8),'feedback');assert.equal(kindOf(5),'audio');
  delete p.scenes;                                     // scores saved before the lane existed
  assert.equal(getCell(p,0,SCENE),null);
  putCell(p,8,SCENE,{name:'drift',fade:4});putCell(p,2,SCENE,{name:'glow'});
  assert.deepEqual(p.scenes,[{r:2,name:'glow',fade:0},{r:8,name:'drift',fade:4}]);
  const block=readBlock(JSON.stringify(copyBlock(p,{r0:8,r1:8,c0:8,c1:9})));
  assert.deepEqual(block.columns,['feedback','scene']);
  assert.deepEqual(pasteBlock(d.patterns[1],{row:20,col:8},block),{written:2,skipped:0});
  assert.deepEqual(getCell(d.patterns[1],20,SCENE),{r:20,name:'drift',fade:4});
  assert.deepEqual(pasteBlock(d.patterns[1],{row:20,col:7},block),{written:0,skipped:2});   // kinds don't match
  for (const bad of [{name:'no spaces',fade:0},{name:'x'.repeat(25)},{name:'ok',fade:129},{name:'ok',fade:1.5},{name:7}]) {
    const text=JSON.stringify({format:'gorgon-block-1',columns:['scene'],rows:[[bad]]});
    assert.throws(()=>readBlock(text),/scene/);
  }
  clearBlock(p,{r0:0,r1:63,c0:0,c1:9});
  assert.deepEqual(p.scenes,[]);
  putCell(p,5,SCENE,{name:'drift'});
  assert.deepEqual(activity(p)[5],{audio:0,keys:1});
});

test('a block from a mixed editor cannot paste into hidden page columns', () => {
  const source=score().patterns[0],destination=score().patterns[0];
  putCell(source,0,5,{note:'C-4',vel:100});putCell(source,0,6,{v:.9});
  const before=structuredClone(destination.keys);
  const block=copyBlock(source,{r0:0,r1:0,c0:5,c1:6});
  const result=pasteBlock(destination,{row:8,col:5},block,{lastColumn:5});
  assert.deepEqual(result,{written:1,skipped:1});
  assert.deepEqual(destination.lanes[5].cells[8],{note:'C-4',vel:100});
  assert.deepEqual(destination.keys,before);
});

test('the Breaks kit hears equivalent note spellings alike', () => {
  assert.equal(breaksVoice('C-2'),'kick');assert.equal(breaksVoice('C2'),'kick');
  assert.equal(breaksVoice('D-2'),'snare');assert.equal(breaksVoice('D2'),'snare');
  assert.equal(breaksVoice('F#2'),'hat');assert.equal(breaksVoice('C-3'),'hat');assert.equal(breaksVoice('AIR'),'hat');
});

test('sound locks survive clipboard, pitch/velocity edits, history and the version-1 to 2 move', async () => {
  const {resolveSound}=await import('../hydrapy/tracker/sounds.js');
  const d=score(),h=new ScoreHistory();
  h.change(d,'sound',()=>{
    d.sounds[0]={machine:16,params:{p2:50}};
    putCell(d.patterns[0],0,0,{note:'C-4',vel:90,locks:{machine:48,p1:0,p2:127}});
    putCell(d.patterns[0],1,0,{note:'C-4',vel:90});
    settleFormat(d);
  },{});
  assert.equal(d.legacy,'hydrapy-tracker-2');
  assert.deepEqual(resolveSound(d,0,0,0),{machine:48,params:{p1:0,p2:127}});
  assert.deepEqual(resolveSound(d,0,0,1),{machine:16,params:{p2:50}});
  const block=readBlock(JSON.stringify(copyBlock(d.patterns[0],{r0:0,r1:1,c0:0,c1:0})));
  pasteBlock(d.patterns[1],{row:3,col:0},block);
  transposeBlock(d.patterns[1],{r0:3,r1:3,c0:0,c1:0},1);
  setBlockVelocity(d.patterns[1],{r0:3,r1:3,c0:0,c1:0},12);
  assert.deepEqual(d.patterns[1].lanes[0].cells[3],{note:'C#4',vel:12,locks:{machine:48,p1:0,p2:127}});
  h.undo(d);assert.equal(d.legacy,'hydrapy-tracker-1');assert.deepEqual(d.sounds[0],{machine:null,params:{}});
  h.redo(d);assert.equal(d.legacy,'hydrapy-tracker-2');assert.equal(d.sounds[0].machine,16);
  for(const locks of [null,[],{p1:true},{p1:128},{p1:1.5},{machine:192},{wat:1}]){
    const bad=structuredClone(block);bad.rows[0][0].locks=locks;
    assert.throws(()=>readBlock(JSON.stringify(bad)));
  }
});

test('a refused edit leaves the score as it was, with no undo step (#6)', () => {
  const d=score(),h=new ScoreHistory();
  // app.js's reshape: change the shape, then refuse more than 128 rows.
  const reshape=change=>h.change(d,'bars',()=>{
    const p=d.patterns[0];change(p);
    if(p.bar*p.bars>128)throw new Error('A pattern has at most 128 rows (bar × bars).');
  },{});
  assert.equal(reshape(p=>{p.bar=20;}),true);
  const accepted=structuredClone(d);
  assert.throws(()=>reshape(p=>{p.bars=8;}),/128 rows/);
  assert.deepEqual(d,accepted);assert.equal(d.patterns[0].bar*d.patterns[0].bars,80);
  assert.equal(h.past.length,1);assert.equal(h.future.length,0);
  // An empty pattern written in by the refused edit goes back to null too.
  assert.throws(()=>h.change(d,'bars',()=>{(d.patterns[20]??=emptyPattern(20)).bars=9;throw new Error('no');},{}),/no/);
  assert.equal(d.patterns[20],null);
  h.undo(d);assert.equal(d.patterns[0].bar,16);
});

test('a pasted scene fade is cut to the destination pattern\'s rows (#7)', async () => {
  const {SCENE} = await import('../hydrapy/tracker/editor.js');
  const d=score(),big=emptyPattern(5);
  big.bars=8;big.lanes.forEach(l=>{l.len=128;l.cells=Array(128).fill(null);});
  big.scenes=[{r:0,name:'wash',fade:100}];
  const block=readBlock(JSON.stringify(copyBlock(big,{r0:0,r1:0,c0:SCENE,c1:SCENE})));
  const small=emptyPattern(6);small.bars=1;small.lanes.forEach(l=>{l.len=16;l.cells=Array(16).fill(null);});
  assert.deepEqual(pasteBlock(d.patterns[0],{row:3,col:SCENE},block),{written:1,skipped:0});
  assert.deepEqual(d.patterns[0].scenes.find(k=>k.r===3),{r:3,name:'wash',fade:64});
  assert.deepEqual(pasteBlock(small,{row:0,col:SCENE},block),{written:1,skipped:0});
  assert.deepEqual(small.scenes,[{r:0,name:'wash',fade:16}]);
  assert.equal(block.rows[0][0].fade,100,'the clipboard keeps its fade');
});

test('an edit that leaves an empty pattern empty writes nothing and keeps the old format (#24)', async () => {
  const {keepEmptyNull} = await import('../hydrapy/tracker/editor.js');
  assert.equal(typeof keepEmptyNull,'function');
  const d=score(),h=new ScoreHistory();
  assert.equal(d.legacy,'hydrapy-tracker-1');assert.equal(d.patterns[16],null);
  // app.js's changeScore and editable() for pattern B01.
  const edit=(label,mutate)=>h.change(d,label,()=>{keepEmptyNull(d,mutate);settleFormat(d);},{});
  const editable=()=>d.patterns[16]??=emptyPattern(16);
  assert.equal(edit('clear cells',()=>clearBlock(editable(),{r0:0,r1:63,c0:0,c1:9})),false);
  assert.equal(edit('interpolation',()=>{editable().easing.z='smooth';}),false);
  assert.equal(d.patterns[16],null);assert.equal(d.legacy,'hydrapy-tracker-1');assert.equal(h.past.length,0);
  // A real edit still writes the pattern and upgrades, in one undo step.
  assert.equal(edit('toggle cell',()=>putCell(editable(),0,0,{note:'C-4',vel:100})),true);
  assert.equal(d.legacy,undefined);assert.equal(d.patterns[16].lanes[0].cells[0].note,'C-4');
  h.undo(d);assert.equal(d.patterns[16],null);assert.equal(d.legacy,'hydrapy-tracker-1');
  // A pattern that already exists keeps being written, even when it is empty.
  d.patterns[0]=emptyPattern(0);
  assert.equal(edit('interpolation',()=>{d.patterns[0].easing.z='hold';}),true);
  assert.equal(edit('interpolation',()=>{d.patterns[0].easing.z='smooth';}),true);
  assert.deepEqual(d.patterns[0],emptyPattern(0));
});
