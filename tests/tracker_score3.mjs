import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {upgrade, downgrade, incoming, outgoing, fits, settleFormat, emptyPattern, code, lowbias32, chance,
  evaluate, parseCond, condCode, parseTransforms, transformsText} from '../hydrapy/tracker/score3.js';

const demo = () => JSON.parse(readFileSync(new URL('../hydrapy/tracker/demo.json', import.meta.url)));

test('chance matches the spec check values', () => {
  assert.equal(lowbias32(1), 1753845952);
  assert.equal(lowbias32(0xffffffff), 1734902346);
  assert.deepEqual([0, 1, 2, 3, 4, 5].map(bar => chance(0x1234, 0x70726f62, 0, 1, 12, bar, 0)),
    [1936593078, 1284988505, 1732901356, 3891962147, 2080199420, 1655901407]);
});

test('versions 1 and 2 round-trip through the editor unchanged', () => {
  for (const format of ['hydrapy-tracker-1', 'hydrapy-tracker-2']) {
    const old = demo();
    old.format = format;
    if (format === 'hydrapy-tracker-2') {
      old.sounds = Array.from({length: 4}, (_, k) => ({machine: k, params: {p1: k}}));
      old.patterns[1].audio[2][5] = {note: 'D-4', vel: 80, locks: {p3: 4}};
    }
    const kept = incoming(old);
    assert.equal(kept.legacy, format);
    assert.equal(kept.patterns.length, 128);
    assert.deepEqual(kept.lanes.map(l => l.name), ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4', 'Lane 5', 'Lane 6']);
    assert.deepEqual(outgoing(kept), old);
    assert(fits(kept, format));
  }
});

test('old-format metadata named rate, bar, bars or lanes survives the editor (#31)', () => {
  // Unknown pattern keys in a version-1 or 2 score are metadata; version 3 gives these names
  // meanings, so the editor keeps the originals aside and puts them back on the way out.
  for (const format of ['hydrapy-tracker-1', 'hydrapy-tracker-2']) {
    const old = demo();
    old.format = format;
    if (format === 'hydrapy-tracker-2') old.sounds = Array.from({length: 4}, () => ({machine: null, params: {}}));
    old.patterns[0].rate = 2;
    old.patterns[1].bar = 8;
    old.patterns[1].bars = 'two';
    old.patterns[2].lanes = [{note: 'metadata'}];
    old.patterns[3].legacyMeta = {mine: true};        // even the editor's own private name
    const kept = incoming(old);
    assert.deepEqual([kept.patterns[0].rate, kept.patterns[1].bar, kept.patterns[1].bars], [1, 16, 4]);
    assert.equal(kept.patterns[2].lanes.length, 6);
    assert.deepEqual(outgoing(kept), old);
    // An edit the old format holds keeps them too.
    kept.patterns[0].lanes[0].cells[3] = {note: 'C-4', vel: 90};
    assert.equal(settleFormat(kept), false);
    const saved = outgoing(kept);
    assert.equal(saved.patterns[0].rate, 2);
    assert.equal(saved.patterns[1].bar, 8);
    assert.deepEqual(saved.patterns[2].lanes, [{note: 'metadata'}]);
    assert.deepEqual(saved.patterns[3].legacyMeta, {mine: true});
    assert.deepEqual(incoming(saved).patterns[0].lanes[0].cells[3], {note: 'C-4', vel: 90});
    // An upgrade gives the names their version-3 meanings and drops the private key, in the
    // same undo step as the edit (settleFormat runs inside it), so no version-3 score carries it.
    const s = incoming(old);
    s.seed = 7;
    assert.equal(settleFormat(s), true);
    const v3 = outgoing(s);
    assert.equal(v3.format, 'hydrapy-tracker-3');
    assert(v3.patterns.every(p => p === null || !('legacyMeta' in p)));
    assert.deepEqual([v3.patterns[0].rate, v3.patterns[1].bar, v3.patterns[1].bars], [1, 16, 4]);
  }
  // A native version-3 score keeps every unknown key, the private name included.
  const native = upgrade(demo());
  native.seed = 3;
  native.patterns[0].legacyMeta = {theirs: 1};
  assert.deepEqual(outgoing(incoming(native)), native);
});

test('an edit a version-2 score can’t hold upgrades it; one it can keeps the format', () => {
  const score = incoming(demo());
  score.patterns[0].lanes[0].cells[3] = {note: 'C-4', vel: 90, locks: {p1: 2}};
  assert.equal(settleFormat(score), false);
  assert.equal(score.legacy, 'hydrapy-tracker-2');               // locks move version 1 to 2, as before
  for (const change of [
    s => { s.patterns[0].lanes[4].cells[0] = {note: 'C-4', vel: 90}; },
    s => { s.patterns[9] = emptyPattern(9); },
    s => { s.patterns[1].rate = 2; },
    s => { s.patterns[2].lanes[1].len = 7; s.patterns[2].lanes[1].cells.length = 7; },
    s => { s.patterns[0].lanes[0].cells[0] = {note: 'C-4', vel: 90, cond: {kind: 'fill'}}; },
    s => { s.patterns[0].lanes[3].transforms = [{kind: 'rev'}]; },
    s => { s.chain = [0, 0, 1]; },
    s => { s.seed = 7; },
    s => { s.lanes[2].name = 'Bass'; },
    s => { s.mutes[5] = true; },
  ]) {
    const s = incoming(demo());
    change(s);
    assert.equal(settleFormat(s), true);
    assert.equal(s.legacy, undefined);
    assert.equal(outgoing(s).format, 'hydrapy-tracker-3');
  }
});

test('codes and the empty pattern', () => {
  assert.deepEqual([0, 17, 127].map(code), ['A01', 'B02', 'H16']);
  assert.deepEqual(Object.keys(emptyPattern(5)), ['name', 'rate', 'bar', 'bars', 'lanes', 'keys', 'easing', 'scenes']);
});

test('conditions as text', () => {
  for (const text of ['1:3', '%50', 'F', '1ST', '!F', '!2:4', '%1', '%99'])
    assert.equal(condCode(parseCond(text)), text);
  assert.equal(parseCond(''), null);
  assert.deepEqual(parseCond(' !1st '), {kind: 'first', not: true});
  for (const bad of ['3:2', '0:4', '1:9', '%0', '%100', 'X', '!', 'F1'])
    assert.throws(() => parseCond(bad), bad);
});

test('transforms as text', () => {
  const text = 'rot -3, every 2:2 (rev), every 1:4 (every 1:2 (ply 2)), degrade 50';
  const nodes = parseTransforms(text);
  assert.equal(transformsText(nodes), text);
  assert.deepEqual(parseTransforms(''), []);
  assert.deepEqual(parseTransforms('ROT 1,REV'), [{kind: 'rot', k: 1}, {kind: 'rev'}]);
  for (const bad of ['ply 5', 'ply 2, ply 3', 'rot 200', 'degrade 0', 'every 3:2 (rev)', 'every 1:2 rev',
                     'every 1:2 (every 1:2 (every 1:2 (rev)))', 'swing 3', 'rev rev',
                     Array(9).fill('rev').join(', ')])
    assert.throws(() => parseTransforms(bad), bad);
});

test('evaluate() matches HydraPy, candidate by candidate', {skip: !process.env.GORGON_V3_CASES}, () => {
  const cases = JSON.parse(readFileSync(process.env.GORGON_V3_CASES));
  let compared = 0;
  for (const {score, transport, u0, u1, candidates} of cases) {
    assert.deepEqual(evaluate(score, transport, u0, u1), candidates);
    compared += candidates.length;
  }
  assert(compared > 5000, `only ${compared} candidates compared`);
});
