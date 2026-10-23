"""Per-mint randomness: a patch whose numbers are drawn in the browser from the page's hash.

``Fx(hash)`` gives plain Python numbers, fixed when the patch is built, so an exported page shows
one edition.  A ``Mint`` records its draws instead: each one is a signal whose value in Python is
the draw for the preview hash, and whose value in the exported page is the same draw made again,
in the same order, from the page's hash (``$fx.hash`` on fxhash, ``?fxhash=`` in the URL, or the
preview hash).  The generator is ``Fx``'s, ported to JavaScript, so a hash gives the same piece in
Python, in ``to_html`` and on fxhash.

    fx = Mint()                                   # FXHASH env var, or a fixed development hash
    sides = fx.rnd_int(3, 8)
    mood = fx.pick(["calm", "restless"])          # strings: the draw is the index; the label is a feature
    base = fx.pick([osc(fx.rnd_btw(5, 40)), noise(fx.rnd_btw(2, 8)), voronoi(6)])   # chains: switched per mint
    base.kaleid(sides).modulate(noise(2), mood * 0.3).out()
    fx.features({"Sides": sides, "Mood": mood})
    density = fx.number("density", default=0.5, min=0, max=1, update="sync")       # fx(params)

A draw can go anywhere a signal can (op arguments, signal arithmetic, NCA/physarum parameters,
``EntropyLock(fx, …)``).  It can't steer Python control flow (``if``, ``range``, ``float()``):
that would freeze it, so it raises; ``.preview`` is the value for this hash if freezing is what
you want.  ``pick``/``switch`` over chains compile every branch and crossfade with 0/1 weights,
so the cost is all branches together.
"""
from __future__ import annotations

import json
import math
import os

from .signals import Fx, Signal, _num, as_signal

DEV_HASH = "ooMASi45ub7Qe4ZE36UT5G6cU4ud8Fhhe4deS4F3cw9KTAb8dLc"


class _Deferred(Signal):
    """A value only the page knows: refuses to become a Python number or a branch."""

    describe = "this value"

    def _frozen(self, what):
        raise TypeError(f"{self.describe} is drawn per mint in the browser, so Python can't {what} it. "
                        "Use it as an op argument or in signal arithmetic, or take .preview to fix this hash's value.")

    def __bool__(self): self._frozen("branch on")
    def __float__(self): self._frozen("turn into a float")
    def __int__(self): self._frozen("turn into an int")
    def __index__(self): self._frozen("count with")
    def __lt__(self, o): self._frozen("compare")
    def __le__(self, o): self._frozen("compare")
    def __gt__(self, o): self._frozen("compare")
    def __ge__(self, o): self._frozen("compare")


class FxDraw(_Deferred):
    """One draw of a ``Mint``: ``preview`` in Python, ``__fxv[index]`` in the page."""

    def __init__(self, mint: "Mint", index: int, preview: float, label, describe: str):
        self.mint, self.index, self.preview, self.label, self.describe = mint, index, float(preview), label, describe

    def value(self, ctx):
        return self.preview

    def js(self):
        return f"__fxval({self.index},{_num(self.preview)})"


class FxParam(_Deferred):
    """An fx(params) value: the default (or an override) in Python, ``$fx.getParam`` in the page.

    Numbers are numbers, booleans 0/1, selects the option's index; a colour is four of these, one per
    channel (0..1)."""

    def __init__(self, mint: "Mint", pid: str, kind: str, preview: float, label, component=None):
        self.mint, self.pid, self.kind, self.preview, self.label, self.component = (
            mint, pid, kind, float(preview), label, component)
        self.describe = f"fx param {pid!r}"

    def value(self, ctx):
        return self.preview

    def js(self):
        k = "" if self.component is None else f",{int(self.component)}"
        return f"__fxparam({json.dumps(self.pid)},{_num(self.preview)}{k})"


class FxColor:
    """A colour param: ``.r .g .b .a`` signals (0..1); unpacks as r, g, b for ``.color(*c)``."""

    def __init__(self, mint, pid, hex8):
        self.mint, self.pid, self.hex = mint, pid, "#" + hex8
        self.r, self.g, self.b, self.a = (FxParam(mint, pid, "color", int(hex8[2 * k:2 * k + 2], 16) / 255.0, self.hex, k)
                                          for k in range(4))

    def __iter__(self):
        return iter((self.r, self.g, self.b))


def _hex8(c) -> str:
    """The SDK's colour normalisation: strip '#', 6 digits get 'ff', 3 digits double up."""
    s = str(c).replace("#", "")
    if len(s) == 6:
        s += "ff"
    if len(s) == 3:
        s = s[0] * 2 + s[1] * 2 + s[2] * 2 + "ff"
    int(s, 16)
    return s.lower()


def switch(index, *chains):
    """Show ``chains[index]``: every chain is compiled and crossfaded with weights clip(index - k + 1),
    so an integer index (a Mint draw, a select param) picks exactly and a moving one morphs between
    neighbours."""
    if not chains:
        raise ValueError("switch needs at least one chain")
    idx = as_signal(index)
    out = chains[0]
    for k, c in enumerate(chains[1:], start=1):
        out = out.blend(c, (idx - (k - 1)).clip(0, 1))
    return out


class Mint:
    """Draws, params and features made per mint in the exported page (see the module docstring).

    ``hash``: the preview hash; the ``FXHASH`` environment variable (``hydrapy … --fxhash``) overrides
    it, and without either a fixed development hash is used.  ``params``: preview values for
    fx(params) (``FXPARAMS``, a JSON object, overrides).  ``lock`` and ``legacy`` are ``Fx``'s.
    """

    def __init__(self, hash=None, lock: float = 0.0, params: dict | None = None, legacy: bool = False):
        self.hash = os.environ.get("FXHASH") or (hash if hash is not None else DEV_HASH)
        self.lock, self.legacy = float(lock), bool(legacy)
        self._fx = Fx(self.hash, lock=self.lock, legacy=self.legacy)
        self.overrides = dict(params or {})
        if os.environ.get("FXPARAMS"):
            self.overrides.update(json.loads(os.environ["FXPARAMS"]))
        self.tape: list = []
        self.draws: list[FxDraw] = []
        self.tables: list[dict] = []
        self.param_defs: list[dict] = []
        self.param_preview: dict = {}
        self._features: dict = {}
        self._slots: list = []

    # -- plumbing ----------------------------------------------------------------------------
    def _arg(self, x):
        if isinstance(x, FxDraw):
            if x.mint is not self:
                raise ValueError("a draw from another Mint")
            return x.preview, {"ref": x.index}
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"draw bounds must be numbers or earlier draws of this Mint, not {x!r}")
        return float(x), float(x)

    def _draw(self, op, value, label=None, describe="a draw") -> FxDraw:
        self.tape.append(op)
        d = FxDraw(self, len(self.draws), value, value if label is None else label, describe)
        self.draws.append(d)
        return d

    @property
    def locks(self) -> int:
        return self._fx.locks

    # -- draws (Fx's names) -------------------------------------------------------------------
    def rand(self) -> FxDraw:
        return self._draw(["r"], self._fx.rand(), describe="fx.rand()")

    fxrand = rand

    def rnd_btw(self, lo, hi) -> FxDraw:
        (a, ja), (b, jb) = self._arg(lo), self._arg(hi)
        return self._draw(["b", ja, jb], self._fx.rnd_btw(a, b), describe="fx.rnd_btw(...)")

    def rnd_btwexp(self, lo, hi) -> FxDraw:
        (a, ja), (b, jb) = self._arg(lo), self._arg(hi)
        return self._draw(["e", ja, jb], self._fx.rnd_btwexp(a, b), describe="fx.rnd_btwexp(...)")

    def rnd_int(self, lo, hi) -> FxDraw:
        (a, ja), (b, jb) = self._arg(lo), self._arg(hi)
        return self._draw(["i", ja, jb], self._fx.rnd_int(a, b), describe="fx.rnd_int(...)")

    def pick(self, seq):
        """A choice per mint.  Numbers: the draw is the number.  Strings/booleans: the draw is the
        index and the choice is its label (for ``features``).  Chains: a chain switched per mint."""
        seq = list(seq)
        if not seq:
            raise ValueError("pick from an empty sequence")
        from .core import Chain
        if all(isinstance(c, Chain) for c in seq):
            return switch(self.rnd_int(0, len(seq) - 1), *seq)
        if not all(isinstance(c, (int, float, str, bool)) for c in seq):
            raise TypeError("Mint.pick takes numbers, strings/booleans (labels) or chains")
        i = self._fx.rnd_int(0, len(seq) - 1)
        c = seq[i]
        numeric = isinstance(c, (int, float)) and not isinstance(c, bool)
        return self._draw(["p", seq], float(c) if numeric else float(i), label=c, describe="fx.pick(...)")

    def table(self, n: int, lo=0.0, hi=1.0) -> list[FxDraw]:
        return [self.rnd_btw(lo, hi) for _ in range(int(n))]

    def reseed(self):
        self.tape.append(["z"])
        self._fx.reseed()

    def save(self) -> int:
        """A bottleneck: ``restore(slot)`` later returns the stream here (in Python and in the page)."""
        self._slots.append(self._fx.save())
        self.tape.append(["s", len(self._slots) - 1])
        return len(self._slots) - 1

    def restore(self, slot: int):
        self.tape.append(["l", int(slot)])
        self._fx.restore(self._slots[int(slot)])

    switch = staticmethod(switch)

    def _entropy_table(self, lock, steps, lo, hi):
        """For ``EntropyLock(mint, …)``: a locked table drawn from a fresh generator on the page's hash."""
        (lk, jl), (a, ja), (b, jb) = self._arg(lock), self._arg(lo), self._arg(hi)
        self.tables.append({"lock": jl, "steps": int(steps), "lo": ja, "hi": jb})
        return len(self.tables) - 1, Fx(self.hash, lock=lk, legacy=self.legacy).table(int(steps), a, b)

    # -- fx(params) -----------------------------------------------------------------------------
    def _param(self, pid, kind, default, options, name, update):
        if any(d["id"] == pid for d in self.param_defs):
            raise ValueError(f"fx param {pid!r} defined twice")
        d = {"id": pid, "name": name or pid, "type": kind}
        if default is not None:
            d["default"] = default
        if update and update != "page-reload":
            d["update"] = update
        if options is not None:
            d["options"] = options
        self.param_defs.append(d)
        return self.overrides.get(pid, default)

    def number(self, pid: str, default=None, min: float = 0.0, max: float = 1.0, step=None, name=None,
               update: str = "page-reload") -> FxParam:
        """A number param (``update="sync"`` follows fxlens' sliders live)."""
        opts = {"min": min, "max": max}
        if step is not None:
            opts["step"] = step
        v = self._param(pid, "number", default, opts, name, update)
        v = (min + max) / 2 if v is None else float(v)
        self.param_preview[pid] = v
        return FxParam(self, pid, "number", v, v)

    def boolean(self, pid: str, default=None, name=None, update: str = "page-reload") -> FxParam:
        v = bool(self._param(pid, "boolean", default, None, name, update))
        self.param_preview[pid] = v
        return FxParam(self, pid, "boolean", 1.0 if v else 0.0, v)

    def select(self, pid: str, options, default=None, name=None, update: str = "page-reload") -> FxParam:
        """A select param: its value is the option's index (use with ``switch`` or arithmetic)."""
        options = [str(o) for o in options]
        v = self._param(pid, "select", default, {"options": options}, name, update)
        v = options[0] if v is None else str(v)
        self.param_preview[pid] = v
        return FxParam(self, pid, "select", float(options.index(v)), v)

    def color(self, pid: str, default=None, name=None, update: str = "page-reload") -> FxColor:
        v = self._param(pid, "color", None if default is None else _hex8(default), None, name, update)
        h = _hex8("808080ff" if v is None else v)
        self.param_preview[pid] = "#" + h
        return FxColor(self, pid, h)

    # -- features -------------------------------------------------------------------------------
    def features(self, mapping: dict) -> "Mint":
        """fxhash features, set per mint in the page: constants, draws (a pick's label, a number's
        value) or params (the value as fxhash shows it)."""
        for k, v in mapping.items():
            if not isinstance(v, (str, int, float, bool, FxDraw, FxParam, FxColor)):
                raise TypeError(f"feature {k!r}: use a constant, a draw or a param")
            if isinstance(v, (FxDraw, FxParam, FxColor)) and v.mint is not self:
                raise ValueError(f"feature {k!r} comes from another Mint")
        self._features.update(mapping)
        return self

    def feature_values(self) -> dict:
        """The features for the preview hash, as the page will report them."""
        out = {}
        for k, v in self._features.items():
            if isinstance(v, FxDraw):
                out[k] = v.label
            elif isinstance(v, FxColor):
                out[k] = v.hex
            elif isinstance(v, FxParam):
                out[k] = v.label
            else:
                out[k] = v
        return out

    # -- export -----------------------------------------------------------------------------------
    def spec(self) -> dict:
        feats = []
        for k, v in self._features.items():
            if isinstance(v, FxDraw):
                feats.append({"name": k, "draw": v.index})
            elif isinstance(v, (FxParam, FxColor)):
                feats.append({"name": k, "param": v.pid})
            else:
                feats.append({"name": k, "const": v})
        return {"hash": self.hash, "legacy": self.legacy, "lock": self.lock, "tape": self.tape,
                "tables": self.tables, "params": self.param_defs, "preview": self.param_preview,
                "features": feats, "draws": len(self.draws)}

    def __repr__(self):
        return f"<Mint {self.hash!r}: {len(self.draws)} draws, {len(self.param_defs)} params>"


def find_mints(objs) -> list[Mint]:
    """The Mints reachable from signals (and the signal attributes of textures) in ``objs``."""
    found, seen = [], set()

    def visit(x, depth=0):
        if id(x) in seen or depth > 40:
            return
        seen.add(id(x))
        if isinstance(x, Mint):
            if x not in found:
                found.append(x)
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                visit(y, depth + 1)
            return
        if isinstance(x, dict):
            for y in x.values():
                visit(y, depth + 1)
            return
        if isinstance(x, (Signal, FxColor)):
            for y in getattr(x, "__dict__", {}).values():
                if isinstance(y, (Signal, Mint, FxColor, list, tuple, dict)):
                    visit(y, depth + 1)

    for o in objs:
        visit(o)
        if not isinstance(o, Signal):           # a texture (NCA, Physarum…): look at its signal attributes
            for y in getattr(o, "__dict__", {}).values():
                if isinstance(y, (Signal, Mint)):
                    visit(y)
    return found


# The page side: Fx's generator ported (seeding paths, sfc32, the lock), the tape replayed from the
# page's hash, fx(params) and features handed to $fx when the fxhash SDK is there.
MINT_JS = r"""
var __fxv = null, __fxl = null, __fxt = null, __fxdefs = {};
var hydrapyMint = (function(){
  var M = (typeof SPEC !== 'undefined' && SPEC) ? SPEC.mint : null;
  if (!M) return null;
  var SDKA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  var OLDA = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ";
  function inAlpha(s){ for (var i = 0; i < s.length; i++) if (SDKA.indexOf(s[i]) < 0) return false; return true; }
  function b58int(s){ var r = 0; for (var i = 0; i < s.length; i++) r = (r * 58 + SDKA.indexOf(s[i])) | 0; return r; }
  function chunks(t, skip, dec){
    var s = t.slice(skip), n = (t.length - skip) >> 2, out = [];
    if (n > 0) { for (var i = 0; i + n <= s.length; i += n) out.push(dec(s.slice(i, i + n)) >>> 0); } else out.push(0);
    while (out.length < 4) out.push(0);
    return out.slice(0, 4);
  }
  var CRC = (function(){ var t = []; for (var n = 0; n < 256; n++) { var c = n; for (var k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1); t.push(c >>> 0); } return t; })();
  function crc32(str){
    var bytes = new TextEncoder().encode(str), c = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  function splitmix(x){
    var out = [];
    for (var i = 0; i < 4; i++) {
      x = (x + 0x9E3779B9) >>> 0; var z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85EBCA6B) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xC2B2AE35) >>> 0;
      out.push((z ^ (z >>> 16)) >>> 0);
    }
    return out;
  }
  function seedFrom(h, legacy){
    if (typeof h === 'number') return splitmix(h >>> 0);
    var t = String(h), oo = t.slice(0, 2) === 'oo' && t.length > 10 && inAlpha(t.slice(2));
    if (legacy && oo) {
      var n = Math.floor(t.length / 4), tr = t.slice(2), ch = [];
      for (var i = 0; i < tr.length - tr.length % n; i += n) ch.push(tr.slice(i, i + n));
      ch = ch.slice(0, 4); while (ch.length < 4) ch.push(ch[ch.length - 1]);
      return ch.map(function(c){ var p = 0; for (var j = 0; j < c.length; j++) p = p + OLDA.indexOf(c[j]) * Number(BigInt(58) ** BigInt(c.length - j - 1)); return p >>> 0; });
    }
    if (/^(0x)?[A-Fa-f0-9]{64}$/.test(t) || /^(0x)?[0-9a-fA-F]{40}$/.test(t))
      return chunks(t, 2, function(r){ return Number(BigInt('0x' + r) % BigInt(4294967295)); });
    if (t.length === 36 && /^(tz|KT)[1-4]/.test(t) && inAlpha(t)) return chunks(t, 3, b58int);
    if (oo) return chunks(t, 2, b58int);
    return splitmix(crc32(t));
  }
  function gen(seed, lock){
    var s = seed.slice(), g = {locks: 0};
    g.raw = function(){
      var a = s[0], b = s[1], c = s[2], d = s[3];
      var t = (a + b + d) >>> 0; d = (d + 1) >>> 0;
      a = (b ^ (b >>> 9)) >>> 0; b = (c + (c << 3)) >>> 0;
      c = ((c << 21) | (c >>> 11)) >>> 0; c = (c + t) >>> 0;
      s = [a, b, c, d]; return t / 4294967296;
    };
    g.reseed = function(){ s = seed.slice(); g.locks++; };
    g.rand = function(){ var v = g.raw(); if (lock > 0 && g.raw() > 1.0 - lock) g.reseed(); return v; };
    g.save = function(){ return s.slice(); };
    g.restore = function(x){ s = x.slice(); };
    return g;
  }
  var hasFx = typeof $fx !== 'undefined' && !!$fx && typeof $fx.hash === 'string';
  var q = new URLSearchParams(location.search);
  var hash = hasFx ? $fx.hash : (q.get('fxhash') || M.hash);
  var seed = seedFrom(hash, M.legacy), g = gen(seed, M.lock);
  var vals = [], labels = [], slots = [];
  function arg(a){ return (a !== null && typeof a === 'object') ? vals[a.ref] : a; }
  M.tape.forEach(function(op){
    var k = op[0], v, lo, hi, r;
    if (k === 'r') { v = g.rand(); vals.push(v); labels.push(v); }
    else if (k === 'b') { lo = arg(op[1]); hi = arg(op[2]); v = g.rand() * (hi - lo) + lo; vals.push(v); labels.push(v); }
    else if (k === 'e') { lo = arg(op[1]); hi = arg(op[2]); r = g.rand(); v = r * r * (hi - lo) + lo; vals.push(v); labels.push(v); }
    else if (k === 'i') { lo = Math.ceil(arg(op[1])); hi = Math.floor(arg(op[2])); v = Math.floor(g.rand() * (hi - lo + 1)) + lo; vals.push(v); labels.push(v); }
    else if (k === 'p') { var ch = op[1], i = Math.floor(g.rand() * ch.length); vals.push(typeof ch[i] === 'number' ? ch[i] : i); labels.push(ch[i]); }
    else if (k === 'z') g.reseed();
    else if (k === 's') slots[op[1]] = g.save();
    else if (k === 'l') g.restore(slots[op[1]]);
  });
  __fxv = vals; __fxl = labels;
  __fxt = M.tables.map(function(tb){
    var g2 = gen(seed, arg(tb.lock)), lo = arg(tb.lo), hi = arg(tb.hi), out = new Array(tb.steps);
    for (var i = 0; i < tb.steps; i++) out[i] = g2.rand() * (hi - lo) + lo;
    return out;
  });
  M.params.forEach(function(d){ __fxdefs[d.id] = d; });
  if (hasFx && M.params.length && typeof $fx.params === 'function') $fx.params(M.params);
  var feats = {};
  M.features.forEach(function(f){
    if ('const' in f) feats[f.name] = f.const;
    else if ('draw' in f) feats[f.name] = labels[f.draw];
    else {
      var d = __fxdefs[f.param], v = M.preview[f.param];
      if (hasFx) { try { v = $fx.getParam(f.param); } catch (e) {} }
      if (d && d.type === 'color' && v && v.hex) v = v.hex.rgba;
      if (typeof v === 'bigint') v = Number(v);
      feats[f.name] = v;
    }
  });
  if (hasFx && typeof $fx.features === 'function') $fx.features(feats);
  if (hasFx && M.previewAt !== null && M.previewAt !== undefined && typeof $fx.preview === 'function') {
    var previewed = false;
    window.hydrapyAfterFrame = function(time){ if (!previewed && time >= M.previewAt) { previewed = true; $fx.preview(); } };
  }
  if (!hasFx) {
    var hud = document.getElementById('hud');
    if (hud) hud.innerHTML += ' &middot; n: next mint';
    window.addEventListener('keydown', function(e){
      if (e.key !== 'n') return;
      var h = 'oo'; for (var i = 0; i < 49; i++) h += SDKA[(Math.random() * SDKA.length) | 0];
      q.set('fxhash', h); location.search = q.toString();
    });
  }
  if (typeof console !== 'undefined') console.log('hydrapy mint', hash, feats);
  return {hash: hash, values: vals, labels: labels, tables: __fxt, features: feats, locks: g.locks, seed: seed};
})();
"""

# A stand-in for the fxhash SDK so an fxhash export opens without it (fxlens and uploads need the real
# fxhash.min.js next to index.html: hash and params then come from the platform).
FX_SHIM_JS = r"""
if (typeof $fx === 'undefined') (function(){
  var A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", q = new URLSearchParams(location.search);
  var hash = q.get('fxhash'); if (!hash) { hash = 'oo'; for (var i = 0; i < 49; i++) hash += A[(Math.random() * A.length) | 0]; }
  var defs = [], vals = {}, feats = {};
  function hex8(c){ var s = String(c).replace('#', ''); if (s.length === 6) s += 'ff'; if (s.length === 3) s = s[0]+s[0]+s[1]+s[1]+s[2]+s[2]+'ff'; return s; }
  window.$fx = {
    hash: hash, minter: q.get('fxminter') || '', iteration: Number(q.get('fxiteration')) || 1,
    context: q.get('fxcontext') || 'standalone', isPreview: q.get('preview') === '1', shim: true,
    params: function(d){ defs = d; d.forEach(function(p){
      var v = p.default;
      if (v === undefined) v = p.type === 'number' ? ((p.options && p.options.min !== undefined ? p.options.min : 0) + (p.options && p.options.max !== undefined ? p.options.max : 1)) / 2
        : p.type === 'boolean' ? false : p.type === 'select' ? p.options.options[0] : p.type === 'color' ? '808080ff' : '';
      vals[p.id] = v; }); },
    getDefinitions: function(){ return defs; },
    getParam: function(id){ var d = defs.filter(function(p){ return p.id === id; })[0]; if (!d) throw new Error('No definition found for param ' + id);
      var v = vals[id]; if (d.type !== 'color') return v; var s = hex8(v);
      var r = parseInt(s.slice(0,2),16), g = parseInt(s.slice(2,4),16), b = parseInt(s.slice(4,6),16), a = parseInt(s.slice(6,8),16);
      return {hex: {rgb: '#' + s.slice(0,6), rgba: '#' + s}, obj: {rgb: {r:r,g:g,b:b}, rgba: {r:r,g:g,b:b,a:a}}, arr: {rgb: [r,g,b], rgba: [r,g,b,a]}}; },
    getParams: function(){ var o = {}; defs.forEach(function(p){ o[p.id] = $fx.getParam(p.id); }); return o; },
    getRawParam: function(id){ return vals[id]; },
    features: function(f){ feats = f; }, getFeatures: function(){ return feats; }, getFeature: function(id){ return feats[id]; },
    preview: function(){ window.dispatchEvent(new Event('fxhash-preview')); },
    on: function(){}, emit: function(){}
  };
  if (typeof console !== 'undefined') console.log('hydrapy: fxhash SDK not found, using a stand-in (put fxhash.min.js next to index.html for fxlens and uploads)');
})();
"""
