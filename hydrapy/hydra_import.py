"""Parse Hydra JavaScript sketches into hydrapy chains.

Handles the usual Hydra sketch: chained calls, numbers, nested chains, arrays with
``.fast()/.smooth()/.offset()/.ease()/.fit()``, arrow-function parameters
(``() => Math.sin(time)``, ``({time}) => ...``), ``o0..o3``, ``s0..s3`` with
``initCam/initImage/initVideo``, ``a.fft[n]``, ``bpm = ...``, ``speed = ...``,
``.out(oN)`` and ``render(oN)``.

    patch = from_hydra('''
        osc(10, 0.1, 1.2).rotate(() => time * 0.1).kaleid(5).out(o0)
        render(o0)
    ''')
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .core import REGISTRY, Camera, Chain, Image, Output, Patch, Transform, Video, default_patch, normalize_args
from .signals import Audio, Fft, Fn, Seq


class HydraSyntaxError(ValueError):
    pass


_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")


def js_expr_to_python(src: str) -> str:
    """Translate a JS arithmetic expression (as used in Hydra arrow functions) to Python."""
    s = src.strip()
    s = re.sub(r"\bMath\.PI\b", "math.pi", s)
    s = re.sub(r"\bMath\.E\b", "math.e", s)
    s = re.sub(r"\bMath\.(sin|cos|tan|asin|acos|atan|atan2|exp|log|sqrt|floor|ceil|abs|pow|min|max|hypot)\b", r"math.\1", s)
    s = re.sub(r"\bMath\.round\b", "round", s)
    s = re.sub(r"\bMath\.random\(\)", "_random()", s)
    s = re.sub(r"\bmath\.abs\b", "abs", s)
    s = re.sub(r"\bmath\.min\b", "min", s)
    s = re.sub(r"\bmath\.max\b", "max", s)
    s = re.sub(r"\bmath\.pow\b", "pow", s)
    s = re.sub(r"\ba\.fft\[(\d+)\]", r"_fft(\1)", s)
    s = re.sub(r"\bmouse\.x\b", "(_ctx.mouse[0]*_ctx.width)", s)
    s = re.sub(r"\bmouse\.y\b", "((1-_ctx.mouse[1])*_ctx.height)", s)
    s = re.sub(r"\btime\b", "_ctx.time", s)
    s = re.sub(r"\bbpm\b", "_ctx.bpm", s)
    s = re.sub(r"\bwidth\b", "_ctx.width", s)
    s = re.sub(r"\bheight\b", "_ctx.height", s)
    s = s.replace("&&", " and ").replace("||", " or ").replace("!==", "!=").replace("===", "==")
    # one level of ternary: cond ? a : b  ->  (a if cond else b)
    m = re.match(r"^(.*?)\?(.*?):(.*)$", s)
    if m and "?" not in m.group(2):
        s = f"(({m.group(2).strip()}) if ({m.group(1).strip()}) else ({m.group(3).strip()}))"
    return s


def make_fn(js_body: str, audio: Optional[Audio]) -> Fn:
    py = js_expr_to_python(js_body)
    ns = {"math": math, "abs": abs, "min": min, "max": max, "pow": pow, "round": round,
          "_fft": (lambda i: 0.0) if audio is None else None, "_random": __import__("random").random}
    try:
        code = compile(py, "<hydra>", "eval")
    except SyntaxError as e:
        raise HydraSyntaxError(f"could not translate arrow function `{js_body.strip()}` (became `{py}`): {e}")

    def fn(ctx):
        local = dict(ns)
        local["_ctx"] = ctx
        if audio is not None:
            local["_fft"] = lambda i: audio.band_at(ctx.time, int(i))
        return eval(code, {"__builtins__": {}}, local)

    f = Fn(fn)
    f.js_source = js_body.strip()
    return f


class _Parser:
    def __init__(self, src: str, patch: Patch):
        self.s = src
        self.i = 0
        self.patch = patch
        self.sources: dict[str, object] = {}
        self.audio: Optional[Audio] = None

    # -- scanning helpers ------------------------------------------------
    def ws(self):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t\r\n;":
                self.i += 1
            elif self.s.startswith("//", self.i):
                j = self.s.find("\n", self.i)
                self.i = len(self.s) if j < 0 else j
            elif self.s.startswith("/*", self.i):
                j = self.s.find("*/", self.i + 2)
                self.i = len(self.s) if j < 0 else j + 2
            else:
                break

    def ws_inline(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def peek(self, k: int = 1) -> str:
        return self.s[self.i:self.i + k]

    def expect(self, ch: str):
        self.ws()
        if not self.s.startswith(ch, self.i):
            raise HydraSyntaxError(f"expected {ch!r} at {self.where()}")
        self.i += len(ch)

    def where(self) -> str:
        return f"position {self.i}: ...{self.s[max(0, self.i - 20):self.i + 20]!r}..."

    def ident(self) -> Optional[str]:
        self.ws()
        m = _IDENT.match(self.s, self.i)
        if not m:
            return None
        self.i = m.end()
        return m.group(0)

    def number(self) -> Optional[float]:
        self.ws()
        sign = 1.0
        j = self.i
        if self.s.startswith("-", j):
            sign = -1.0
            j += 1
        m = _NUMBER.match(self.s, j)
        if not m:
            return None
        self.i = m.end()
        return sign * float(m.group(0))

    def matching(self, open_ch: str, close_ch: str) -> int:
        """Index of the bracket matching the one at self.i."""
        depth = 0
        j = self.i
        while j < len(self.s):
            c = self.s[j]
            if c in "\"'":
                k = self.s.find(c, j + 1)
                j = len(self.s) if k < 0 else k
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return j
            j += 1
        raise HydraSyntaxError(f"unbalanced {open_ch!r} at {self.where()}")

    def raw_arg(self) -> str:
        """Raw text of one argument (until a top-level ',' or ')')."""
        depth = 0
        j = self.i
        while j < len(self.s):
            c = self.s[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            j += 1
        text = self.s[self.i:j]
        self.i = j
        return text

    # -- grammar ---------------------------------------------------------
    def program(self):
        while True:
            self.ws()
            if self.i >= len(self.s):
                break
            self.statement()

    def statement(self):
        save = self.i
        name = self.ident()
        if name is None:
            raise HydraSyntaxError(f"unexpected input at {self.where()}")
        self.ws_inline()
        # render(oN)
        if name == "render":
            self.expect("(")
            self.ws()
            target = self.ident() if self.peek() != ")" else None
            self.expect(")")
            self.patch.render(self.output(target) if target else self.patch.o0)
            return
        # bpm = 60 / speed = 2
        if self.peek() == "=" and self.peek(2) != "==" and self.peek(2) != "=>":
            self.i += 1
            v = self.number()
            if name == "bpm" and v is not None:
                self.patch.bpm = v
            elif name == "speed" and v is not None:
                self.patch.speed = v
            else:
                self.raw_arg()  # ignore other assignments
            return
        # s0.initCam() / a.setBins(..)
        if re.fullmatch(r"s[0-3]", name) and self.peek() == ".":
            self.i += 1
            method = self.ident()
            self.expect("(")
            args = self.args_raw()
            self.expect(")")
            self.source_init(name, method, args)
            return
        if name == "a" and self.peek() == ".":
            self.i += 1
            method = self.ident()
            self.expect("(")
            args = self.args_raw()
            self.expect(")")
            a = self.get_audio()
            setter = {"setBins": a.set_bins, "setSmooth": a.set_smooth, "setScale": a.set_scale, "setCutoff": a.set_cutoff}.get(method)
            if setter and args:
                setter(float(args[0]))
            return
        # a chain
        self.i = save
        chain, out = self.chain()
        if out is not None:
            self.patch.assign(out, chain)

    def args_raw(self) -> list[str]:
        out = []
        while True:
            self.ws()
            if self.peek() == ")":
                return out
            out.append(self.raw_arg().strip().strip("'\""))
            self.ws()
            if self.peek() == ",":
                self.i += 1

    def source_init(self, slot: str, method: str, args: list[str]):
        if method == "initCam":
            self.sources[slot] = Camera(int(args[0]) if args else 0)
        elif method == "initImage":
            self.sources[slot] = Image(args[0])
        elif method == "initVideo":
            self.sources[slot] = Video(args[0])
        elif method == "initScreen":
            raise HydraSyntaxError("initScreen() has no equivalent here")
        # clear()/init with other args: ignore

    def get_audio(self) -> Audio:
        if self.audio is None:
            self.audio = Audio()  # microphone in the HTML export, silence offline
        return self.audio

    def output(self, name: str) -> Output:
        m = re.fullmatch(r"o([0-3])", name)
        if not m:
            raise HydraSyntaxError(f"unknown output {name!r}")
        return self.patch.outputs[int(m.group(1))]

    def chain(self) -> tuple[Chain, Optional[Output]]:
        chain = None
        out = None
        while True:
            self.ws()
            name = self.ident()
            if name is None:
                raise HydraSyntaxError(f"expected a function name at {self.where()}")
            if name == "out":
                self.expect("(")
                self.ws()
                target = self.ident() if self.peek() != ")" else None
                self.expect(")")
                out = self.output(target) if target else self.patch.o0
                break
            op = REGISTRY.get(name)
            if op is None:
                raise HydraSyntaxError(f"unknown function {name}() at {self.where()}")
            self.expect("(")
            args, kwargs = self.args(op)
            self.expect(")")
            norm = normalize_args(op, tuple(args), kwargs)
            if chain is None:
                if op.kind != "src":
                    raise HydraSyntaxError(f"chains must start with a source, not {name}()")
                chain = Chain([Transform(op, norm)])
            else:
                chain = chain._with(op, norm)
            self.ws_inline()
            if self.peek() == ".":
                self.i += 1
                continue
            # allow the chain to continue on the next line with a leading dot
            save = self.i
            self.ws()
            if self.peek() == ".":
                self.i += 1
                continue
            self.i = save
            break
        return chain, out

    def args(self, op) -> tuple[list, dict]:
        args = []
        while True:
            self.ws()
            if self.peek() == ")":
                return args, {}
            args.append(self.arg())
            self.ws()
            if self.peek() == ",":
                self.i += 1

    def end_of_arg(self) -> int:
        depth = 0
        j = self.i
        while j < len(self.s):
            c = self.s[j]
            if c in "\"'":
                k = self.s.find(c, j + 1)
                j = len(self.s) if k < 0 else k
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            j += 1
        return j

    def arg(self):
        self.ws()
        raw = self.s[self.i:self.end_of_arg()]
        stripped = raw.strip()
        # array literal with modifiers
        if stripped.startswith("["):
            return self.array()
        # arrow function: ( ... ) => body   |   x => body
        arrow = re.match(r"^\s*(\([^()]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>", raw)
        if arrow:
            self.i += arrow.end()
            return self.arrow_body()
        m = _IDENT.match(self.s, self.i)
        if m:
            name = m.group(0)
            if name in REGISTRY and REGISTRY[name].kind == "src" and self.s[m.end():m.end() + 1] == "(":
                chain, _ = self.chain()
                return chain
            if re.fullmatch(r"o[0-3]", stripped):
                self.i = m.end()
                return self.output(name)
            if re.fullmatch(r"s[0-3]", stripped):
                self.i = m.end()
                if name not in self.sources:
                    raise HydraSyntaxError(f"{name} is used before {name}.initCam()/initImage()/initVideo()")
                return self.sources[name]
            # audio expression without an arrow (Hydra would evaluate it once; we keep it live)
            if "a.fft[" in stripped:
                mm = re.fullmatch(r"a\.fft\[(\d+)\]", stripped)
                self.i += len(raw)
                return Fft(self.get_audio(), int(mm.group(1))) if mm else make_fn(stripped, self.get_audio())
            if re.search(r"\b(time|bpm|mouse|width|height)\b", stripped):
                self.i += len(raw)
                return make_fn(stripped, self.audio)
            return self.numeric_expr()
        start = self.i
        v = self.number()
        if v is not None and self.s[self.i:start + len(raw)].strip() == "":
            return v
        self.i = start
        if stripped:
            return self.numeric_expr()
        raise HydraSyntaxError(f"could not parse argument at {self.where()}")

    def numeric_expr(self) -> float:
        text = self.raw_arg()
        try:
            return float(eval(js_expr_to_python(text), {"__builtins__": {}}, {"math": math, "abs": abs, "min": min, "max": max, "pow": pow}))
        except Exception as e:
            raise HydraSyntaxError(f"could not evaluate `{text.strip()}`: {e}")

    def arrow_body(self):
        self.ws()
        if self.peek() == "{":
            close = self.matching("{", "}")
            body = self.s[self.i + 1:close]
            self.i = close + 1
            m = re.search(r"return\s+(.*)", body, re.S)
            if not m:
                raise HydraSyntaxError("arrow function bodies with braces must `return` an expression")
            text = m.group(1).strip().rstrip(";")
        else:
            text = self.raw_arg()
        audio = self.get_audio() if "fft" in text else self.audio
        return make_fn(text, audio)

    def array(self) -> Seq:
        self.expect("[")
        vals = []
        while True:
            self.ws()
            if self.peek() == "]":
                self.i += 1
                break
            v = self.number()
            if v is None:
                raise HydraSyntaxError(f"arrays may only contain numbers ({self.where()})")
            vals.append(v)
            self.ws()
            if self.peek() == ",":
                self.i += 1
        seq = Seq(vals)
        while True:
            self.ws_inline()
            if self.peek() != ".":
                break
            self.i += 1
            method = self.ident()
            self.expect("(")
            raw = self.args_raw()
            self.expect(")")
            nums = [float(x) for x in raw if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", x)]
            if method == "fast":
                seq = seq.fast(*nums[:1])
            elif method == "smooth":
                seq = seq.smooth(*nums[:1])
            elif method == "offset":
                seq = seq.offset(*nums[:1])
            elif method == "ease":
                seq = seq.ease(raw[0] if raw else "linear")
            elif method == "fit":
                seq = seq.fit(*nums[:2])
            else:
                raise HydraSyntaxError(f"unknown array method .{method}()")
        return seq


def from_hydra(src: str, patch: Optional[Patch] = None) -> Patch:
    """Parse a Hydra sketch into a Patch (the default patch unless one is given)."""
    patch = patch or default_patch()
    p = _Parser(src, patch)
    p.program()
    return patch


def chain_from_hydra(src: str) -> Chain:
    """Parse a single chain expression (``.out()`` optional) and return it."""
    p = _Parser(src, Patch())
    chain, _ = p.chain()
    return chain
