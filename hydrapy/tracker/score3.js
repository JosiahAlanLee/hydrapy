// Score format 3 in the editor: the version-1/2 mapping, the format the editor keeps, the
// text forms of conditions and transforms, and evaluate() for the browser's audition synth.
// No DOM. Gorgon's spec is the contract (score-v3.md, evaluator.md, transforms.md, bus-io.md);
// HydraPy's tracker_v3.py is the reference, and the tests compare this with it.
//
// The editor keeps every score as version 3. A version-1 or 2 score is mapped onto version 3
// when it arrives and marked `legacy`, and goes back to Studio in its own format while it still
// fits there: opening or saving never upgrades a score. The first edit it can't hold (a fifth
// lane, a pattern past A04, a rate, a condition…) drops `legacy`, so the score upgrades, and
// undo restores it. An old pattern's own keys named like version 3's (`rate`, `bar`, `bars`,
// `lanes`) are unknown metadata there; while the score is `legacy` each pattern keeps them
// under the private key LEGACY_META, which downgrade puts back and an upgrade drops.
export const FORMAT = 'hydrapy-tracker-3', TICKS = 1920, LANE_COUNT = 6, PATTERN_COUNT = 128;
// Keep these format constants aligned with tracker_v3.py.
export const ROW_TICKS = TICKS / 4, MIN_NOTE_LENGTH = .25, MAX_NOTE_LENGTH = 16;
export const HOLD_TICKS = MAX_NOTE_LENGTH * ROW_TICKS;
export const RATES = [.25, .5, 1, 2, 4, 8];
const LEGACY_META = 'legacyMeta', COLLIDING = ['rate', 'bar', 'bars', 'lanes', LEGACY_META];
const TOP_COLLIDING = ['chain', 'seed', 'lanes', 'legacy', 'legacyTop'];
const PROB = 0x70726f62, DEGRADE = 0x64656772;       // "prob" and "degr", chance's salts
const clone = value => structuredClone(value);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const quiet = () => ({machine: null, params: {}});

export const code = index => 'ABCDEFGH'[Math.floor(index / 16)] + String(index % 16 + 1).padStart(2, '0');

// What a null pattern plays as, and exactly what the editor writes on its first edit.
export function emptyPattern(index) {
  return {name: code(index), rate: 1, bar: 16, bars: 4,
          lanes: Array.from({length: LANE_COUNT}, () => ({len: 64, cells: Array(64).fill(null), transforms: []})),
          keys: {z: [], roll: [], feedback: []}, easing: {z: 'smooth', roll: 'smooth', feedback: 'smooth'}, scenes: []};
}

// {w, rows, bar, bars}: ticks per row and the shape of a pattern; null is the empty pattern.
export function geometry(pattern) {
  return pattern ? {w: TICKS / (4 * pattern.rate), rows: pattern.bar * pattern.bars, bar: pattern.bar, bars: pattern.bars}
    : {w: 480, rows: 64, bar: 16, bars: 4};
}

// score-v3.md's mapping of a valid version-1 or 2 score.
export function upgrade(document) {
  const d = clone(document);
  const sounds = d.sounds ?? Array.from({length: 4}, quiet);
  Object.assign(d, {format: FORMAT, chain: [0, 1, 2, 3], seed: 0, mutes: [...d.mutes, false, false],
                    lanes: Array.from({length: LANE_COUNT}, (_, n) => ({name: `Lane ${n + 1}`})),
                    sounds: [...sounds, quiet(), quiet()]});
  for (const p of d.patterns) {
    const cells = [...p.audio, Array(64).fill(null), Array(64).fill(null)];
    delete p.audio;
    Object.assign(p, {rate: 1, bar: 16, bars: 4, lanes: cells.map(c => ({len: 64, cells: c, transforms: []}))});
  }
  d.patterns = [...d.patterns, ...Array(PATTERN_COUNT - 4).fill(null)];
  return d;
}

// Whether a version-3 score still says nothing a `format` score can't.
export function fits(score, format) {
  if (score.pattern > 3 || !same(score.chain, [0, 1, 2, 3]) || score.seed !== 0 || score.mutes[4] || score.mutes[5]) return false;
  if (score.lanes.some((lane, n) => !same(lane, {name: `Lane ${n + 1}`}))) return false;
  if (score.patterns.slice(4).some(p => p !== null) || !same(score.sounds.slice(4), [quiet(), quiet()])) return false;
  for (const p of score.patterns.slice(0, 4)) {
    if (!p) continue;
    if (p.rate !== 1 || p.bar !== 16 || p.bars !== 4) return false;
    for (const [n, lane] of p.lanes.entries())
      if (lane.len !== 64 || lane.transforms.length || Object.keys(lane).length !== 3 ||
          lane.cells.some(c => c && (n >= 4 || 'cond' in c || 'len' in c))) return false;
  }
  if (format === 'hydrapy-tracker-1')
    return score.sounds.every(s => same(s, quiet())) && !score.patterns.some(p => p?.lanes.some(l => l.cells.some(c => c?.locks)));
  return true;
}

// The version-1 or 2 score a version-3 score that fits that format maps back to.
export function downgrade(score, format) {
  const d = {};
  for (const [key, value] of Object.entries(score))
    if (!TOP_COLLIDING.includes(key)) d[key] = clone(value);
  d.format = format;
  d.mutes = d.mutes.slice(0, 4);
  if (format === 'hydrapy-tracker-2') d.sounds = d.sounds.slice(0, 4); else delete d.sounds;
  d.patterns = d.patterns.slice(0, 4).map((p, i) => {
    p = p ?? emptyPattern(i);
    const out = {};
    for (const [key, value] of Object.entries(p)) if (!COLLIDING.includes(key)) out[key] = value;
    out.audio = p.lanes.slice(0, 4).map(lane => lane.cells);
    return Object.assign(out, p[LEGACY_META]);
  });
  return Object.assign(d, clone(score.legacyTop ?? {}));
}

// A score from Studio, as the editor keeps it.
export function incoming(document) {
  if (document.format === FORMAT) return clone(document);
  const d = {...upgrade(document), legacy: document.format};
  d.legacyTop = Object.fromEntries(TOP_COLLIDING.filter(key => key in document)
    .map(key => [key, clone(document[key])]));
  // Recovery belongs only to the editor's legacy representation. Public upgrade() is
  // the same document as Python's spec mapping, including unknown legacyMeta keys.
  d.patterns.slice(0, 4).forEach((p, i) => {
    const old = document.patterns[i];
    const meta = Object.fromEntries(COLLIDING.filter(key => key in old).map(key => [key, clone(old[key])]));
    delete p[LEGACY_META];
    if (Object.keys(meta).length) p[LEGACY_META] = meta;
  });
  return d;
}

// The score to send Studio: in its old format while it fits there.
export function outgoing(score) {
  if (score.legacy) return downgrade(score, score.legacy);
  const d = clone(score);
  delete d.legacy;
  return d;
}

// After an edit: keep the old format if the score still fits it, moving a version-1 score to
// version 2 for sound locks, as the editor always has; else upgrade. Returns true on upgrade.
// An upgrade drops the kept old-format metadata, as score-v3.md's mapping does, so no version-3
// score carries LEGACY_META; undo brings it back with the old format.
export function settleFormat(score) {
  if (!score.legacy || fits(score, score.legacy)) return false;
  if (score.legacy === 'hydrapy-tracker-1' && fits(score, 'hydrapy-tracker-2')) {
    score.legacy = 'hydrapy-tracker-2';
    return false;
  }
  delete score.legacy;
  delete score.legacyTop;
  for (const p of score.patterns) if (p) delete p[LEGACY_META];
  return true;
}

// -- conditions and transforms as text -------------------------------------------------------

export function condCode(cond) {
  if (!cond) return '';
  const text = {every: () => `${cond.a}:${cond.b}`, prob: () => `%${cond.p}`, fill: () => 'F', first: () => '1ST'}[cond.kind]();
  return (cond.not ? '!' : '') + text;
}

// "1:3", "%50", "F", "1ST", each optionally "!"; "" is no condition (null).
export function parseCond(text) {
  const t = String(text).trim().toUpperCase().replace(/\s+/g, '');
  if (!t) return null;
  const not = t.startsWith('!'), body = not ? t.slice(1) : t;
  let cond, m;
  if ((m = /^(\d):(\d)$/.exec(body))) {
    const a = Number(m[1]), b = Number(m[2]);
    if (a < 1 || b > 8 || a > b) throw new Error('A bar condition is a:b with 1 ≤ a ≤ b ≤ 8.');
    cond = {kind: 'every', a, b};
  } else if ((m = /^%(\d{1,2})$/.exec(body))) {
    const p = Number(m[1]);
    if (p < 1) throw new Error('A probability is %1–%99.');
    cond = {kind: 'prob', p};
  } else if (body === 'F') cond = {kind: 'fill'};
  else if (body === '1ST') cond = {kind: 'first'};
  else throw new Error('A condition is a:b, %p, F or 1ST, with ! to negate.');
  return not ? {...cond, not: true} : cond;
}

const nodeText = node => node.kind === 'every' ? `every ${node.a}:${node.b} (${nodeText(node.inner)})`
  : node.kind === 'rev' ? 'rev' : `${node.kind} ${node.kind === 'rot' ? node.k : node.kind === 'ply' ? node.n : node.p}`;
export const transformsText = transforms => transforms.map(nodeText).join(', ');

// "rot 1, every 2:2 (rev), ply 2" into transform nodes, checked as transforms.md says.
export function parseTransforms(text) {
  const tokens = String(text).toLowerCase().match(/-?\d+|[a-z]+|[(),:]/g) ?? [];
  let at = 0;
  const peek = () => tokens[at], take = () => tokens[at++];
  const integer = (low, high, what) => {
    const t = take();
    if (!/^-?\d+$/.test(t ?? '') || Number(t) < low || Number(t) > high) throw new Error(`${what} takes a whole number from ${low} to ${high}.`);
    return Number(t);
  };
  const node = depth => {
    const kind = take();
    if (kind === 'rot') return {kind, k: integer(-127, 127, 'rot')};
    if (kind === 'rev') return {kind};
    if (kind === 'ply') return {kind, n: integer(2, 4, 'ply')};
    if (kind === 'degrade') return {kind, p: integer(1, 99, 'degrade')};
    if (kind === 'every') {
      if (depth === 1) throw new Error('every nests at most three deep.');
      const a = integer(1, 8, 'every');
      if (take() !== ':') throw new Error('every takes a:b, like every 1:4 (rev).');
      const b = integer(1, 8, 'every');
      if (a > b) throw new Error('every a:b needs a ≤ b.');
      if (take() !== '(') throw new Error('every holds one transform in brackets: every 2:2 (rev).');
      const inner = node(depth - 1);
      if (take() !== ')') throw new Error('Close every’s bracket.');
      return {kind, a, b, inner};
    }
    throw new Error('Transforms are rot k, rev, ply n, degrade p and every a:b (…), separated by commas.');
  };
  const nodes = [];
  while (at < tokens.length) {
    nodes.push(node(3));
    if (at < tokens.length && take() !== ',') throw new Error('Separate transforms with commas.');
  }
  if (nodes.length > 8) throw new Error('A lane has at most eight transforms.');
  const product = list => list.reduce((n, x) => n * (x.kind === 'ply' ? x.n : x.kind === 'every' ? product([x.inner]) : 1), 1);
  if (product(nodes) > 4) throw new Error('The ply counts in a lane multiply to at most 4.');
  if (peek() !== undefined) throw new Error('Unexpected text after the transforms.');
  return nodes;
}

// -- evaluate() ------------------------------------------------------------------------------

export function lowbias32(x) {
  x >>>= 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15; x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

export function chance(seed, salt, pattern, lane, row, bar, sub) {
  let h = lowbias32((seed ^ salt) >>> 0);
  for (const v of [pattern, lane, row, bar % 4294967296, sub]) h = lowbias32((h ^ v) >>> 0);
  return h;
}

// The cycle in ticks: entries, their starts and the length (score-v3.md#the-timeline).
export function timeline(score) {
  const entries = score.loop ? [score.pattern] : [...score.chain], starts = [0];
  for (const p of entries) { const g = geometry(score.patterns[p]); starts.push(starts.at(-1) + g.w * g.rows); }
  const n = entries.length, oneVisit = entries.every(p => p === entries[0]);
  // Entries just before each one that play its pattern too, going back round the cycle.
  const runs = entries.map((p, i) => { let r = 0; while (!oneVisit && entries[((i - r - 1) % n + n) % n] === p) r++; return r; });
  const entryAt = tick => { let i = 0; while (i + 1 < starts.length - 1 && starts[i + 1] <= tick) i++; return i; };
  return {loop: score.loop, entries, starts, cycle: starts.at(-1), oneVisit, runs, entryAt};
}

// An [{at, on}…] list at tick u: the last transition at or before u; the anchor's state before it.
export function stateAt(transitions, u) {
  if (!transitions?.length) return false;
  u = Math.max(u, 0);
  let on = transitions[0].on;
  for (const t of transitions) { if (t.at > u) break; on = t.on; }
  return on;
}

function judge(cond, seed, pattern, lane, row, bar, fill) {
  let holds, why;
  if (cond.kind === 'every') { holds = bar % cond.b === cond.a - 1; why = `bar ${bar} mod ${cond.b} = ${bar % cond.b}`; }
  else if (cond.kind === 'prob') { const h = chance(seed, PROB, pattern, lane, row, bar, 0); holds = h % 100 < cond.p; why = `chance ${h} mod 100 = ${h % 100}`; }
  else if (cond.kind === 'fill') { holds = fill; why = `fill ${fill ? 'on' : 'off'}`; }
  else { holds = bar === 0; why = `bar ${bar}`; }
  return [holds !== !!cond.not, why];
}

// The active leaves for a bar, in order, each with its node number and JSON Pointer.
function leaves(transforms, bar) {
  const out = [];
  let count = 0;
  const walk = (node, pointer, active) => {
    const number = count++;
    if (node.kind === 'every') walk(node.inner, pointer + '/inner', active && bar % node.b === node.a - 1);
    else if (active) out.push([node, number, pointer]);
  };
  transforms.forEach((node, k) => walk(node, `/${k}`, true));
  return out;
}

function row(out, score, t, u0, u1, play, entry, p, r, rowAt, w, bar, fill) {
  score.patterns[p].lanes.forEach((lane, laneIndex) => {
    const active = leaves(lane.transforms, bar);
    const laneRow = r % lane.len;
    let cell = laneRow;
    for (const [node] of [...active].reverse()) {
      if (node.kind === 'rot') cell = ((cell - node.k) % lane.len + lane.len) % lane.len;
      else if (node.kind === 'rev') cell = lane.len - 1 - cell;
    }
    const hit = lane.cells[cell];
    if (!hit || hit.vel <= 0) return;
    let events = [[0, 1, null]];
    for (const [node, number, pointer] of active) {
      if (node.kind === 'ply') events = events.flatMap(([s, m, dropped]) => Array.from({length: node.n}, (_, q) => [s * node.n + q, m * node.n, dropped]));
      else if (node.kind === 'degrade') events = events.map(([s, m, dropped]) => {
        if (dropped === null) {
          const h = chance(score.seed, DEGRADE + number, p, laneIndex, r, bar, s);
          if (h % 100 >= node.p) dropped = `degrade ${pointer}: chance ${h} mod 100 = ${h % 100}`;
        }
        return [s, m, dropped];
      });
    }
    const cond = hit.cond;
    const [holds, condWhy] = cond ? judge(cond, score.seed, p, laneIndex, r, bar, fill) : [true, null];
    const via = active.map(([, , pointer]) => pointer);
    for (const [s, m, dropped] of events) {
      const at = rowAt + s * w / m;
      const pattern = score.patterns[p];
      const duration = 'len' in hit ? Math.min(hit.len * ROW_TICKS, rowAt + (pattern.bar * pattern.bars - r) * w - at) : w / m;
      if (at < u0 || at >= u1 || at + duration <= 0) continue;
      let fires = false, why;
      if (score.mutes[laneIndex]) why = 'muted (score)';
      else if (stateAt(t.bus?.mutes?.[laneIndex], at)) why = 'muted (bus)';
      else if (!holds) why = condWhy;
      else if (dropped) why = dropped;
      else { fires = true; why = condWhy; }
      out.push({id: `${play}.${laneIndex}.${r}` + (m > 1 ? `.${s}` : ''), at, lane: laneIndex, duration, ...('len' in hit ? {len: hit.len} : {}),
                note: hit.note, vel: hit.vel, locks: hit.locks ? clone(hit.locks) : null, fires, why,
                origin: {play, entry, pattern: p, row: r, lane_row: laneRow, cell, sub: s, of: m, bar,
                         cond: cond ? condCode(cond) : null, via: [...via]}});
    }
  });
}

// Every candidate with u0 ≤ at < u1 that ends after the anchor, in order of at, then lane.
// `transport` is {tick, fill, bus: {fill, mutes}}; missing lists are off.
export function evaluate(score, transport, u0, u1, line = timeline(score)) {
  const {entries, starts, cycle} = line, n = entries.length, a = transport.tick;
  if (!(a >= 0 && a < cycle)) throw new Error('The anchor is a cycle tick.');
  const fillAt = u => stateAt(transport.fill, u) || stateAt(transport.bus?.fill, u);
  const first = line.entryAt(a), from = Math.max(u0, 0);
  let cycles = Math.floor((a + from) / cycle), i = line.entryAt((a + from) % cycle);
  let play = cycles * n + i - first, playStart = cycles * cycle + starts[i] - a;
  let r = Math.max(0, Math.floor((Math.max(u0, -HOLD_TICKS) - playStart) / geometry(score.patterns[entries[i]]).w));
  const out = [];
  while (playStart < u1) {
    const p = entries[i], g = geometry(score.patterns[p]);
    const visit = line.oneVisit ? play : Math.min(play, line.runs[i]);
    for (; r < g.rows; r++) {
      const rowAt = playStart + r * g.w;
      if (rowAt >= u1) break;
      if (score.patterns[p]) row(out, score, transport, u0, u1, play, line.loop ? null : i, p, r, rowAt, g.w,
                                 visit * g.bars + Math.floor(r / g.bar), fillAt(rowAt));
    }
    play++;
    if (++i === n) { i = 0; cycles++; }
    playStart = cycles * cycle + starts[i] - a;
    r = 0;
  }
  return out.sort((x, y) => x.at - y.at || x.lane - y.lane);
}
