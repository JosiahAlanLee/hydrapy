"""Compile chains to GLSL ES 1.00 fragment shaders (multi-pass where needed).

The nesting mirrors Hydra's ``generateGlsl``: ``osc(10).rotate(0.3).color(1,0,0)``
becomes ``color(osc(rotate(st, 0.3, 0.), 10., 0.1, 0.), 1., 0., 0., 1.)``.

Texture ops (``blur``, ``life``, ``rd`` ...) need their input as a texture, so
the chain before them is split off into its own pass that renders to an
intermediate target ``tmpN``; the op then samples ``tex_tmpN``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .core import Camera, Chain, DelayRef, Image, Op, Output, Patch, Texture, Video, default_patch
from .noise import GLSL_NOISE
from .signals import Const, Expr, Fn, Fn1, Signal, as_signal


@dataclass
class Uniform:
    name: str
    type: str                 # float | sampler2D
    source: object            # Signal (float) or Texture / TmpRef (sampler)


@dataclass
class TmpRef(Texture):
    """A compiler-allocated intermediate render target."""
    name: str


@dataclass
class ComputeStep:
    kernel: str
    sampler: str
    arguments: dict[str, str]   # per-frame scalar expressions (literals or uniforms)


@dataclass
class Pass:
    target: str               # 'o0'..'o3' or 'tmpN'
    expr: str                 # GLSL expression for gl_FragColor (functions/uniforms are shared)
    source: str = ""          # full fragment shader, filled in by Program.finish()
    compute: Optional[ComputeStep] = None
    lets: list = field(default_factory=list)   # "vec4 _sN = ...;": sub-chains used more than once, computed once


@dataclass
class _TextureCall:
    """Retain eligibility until composition/coordinate changes wrap the call."""
    op: Op
    sampler: str
    args: list[Callable[[str], str]]
    iteration: Optional[int]

    def __call__(self, uv):
        rest = "".join(", " + f(uv) for f in self.args)
        last = f", {glsl_float(self.iteration)}" if self.iteration is not None else ""
        return f"{self.op.name}({self.sampler}, {uv}{rest}{last})"

    def compute_step(self):
        inputs = [i for i in self.op.inputs if i.name != "iterations"]
        if self.op.compute and all(i.scalar and i.type == "float" for i in inputs):
            return ComputeStep(self.op.compute, self.sampler,
                               {i.name: f("st") for i, f in zip(inputs, self.args)})
        return None


@dataclass
class Program:
    """All passes needed to render one output for one frame, in order (output pass last)."""
    output: Optional[Output]
    passes: list[Pass] = field(default_factory=list)
    uniforms: list[Uniform] = field(default_factory=list)
    functions: list[Op] = field(default_factory=list)
    precision: str = "highp"
    history: dict = field(default_factory=dict)    # output name -> deepest delay() used

    @property
    def float_uniforms(self) -> list[Uniform]:
        return [u for u in self.uniforms if u.type == "float"]

    @property
    def sampler_uniforms(self) -> list[Uniform]:
        return [u for u in self.uniforms if u.type == "sampler2D"]

    @property
    def tmp_targets(self) -> list[str]:
        return [p.target for p in self.passes if p.target.startswith("tmp")]

    def header(self) -> str:
        lines = [f"precision {self.precision} float;", "uniform float time;", "uniform vec2 resolution;",
                 "uniform vec2 mouse;"]
        for u in self.uniforms:
            lines.append(f"uniform {u.type} {u.name};")
        helpers = list(dict.fromkeys(op.glsl_helpers for op in self.functions if op.glsl_helpers))
        return ("\n".join(lines) + "\n" + GLSL_NOISE + "\n" + "\n".join(helpers)
                + "\n" + "\n".join(function_source(op) for op in self.functions))

    def finish(self):
        head = self.header()
        for p in self.passes:
            lets = "".join(f"  {line}\n" for line in p.lets)
            p.source = (head + "\nvoid main () {\n  vec2 st = gl_FragCoord.xy / resolution.xy;\n" + lets +
                        f"  gl_FragColor = {p.expr};\n}}\n")
        return self

    @property
    def source(self) -> str:
        return self.passes[-1].source


def iterated(op: Op) -> bool:
    """Texture ops with an ``iterations`` input get the 0-based iteration index as a trailing ``_pass`` argument."""
    return op.kind == "texture" and any(i.name == "iterations" for i in op.inputs)


def function_source(op: Op) -> str:
    params = [op.fixed_args]
    for inp in op.inputs:
        if inp.type == "chain" or (inp.name == "iterations" and op.kind == "texture"):
            continue
        t = {"float": "float", "int": "int", "vec2": "vec2", "vec4": "vec4", "sampler2D": "sampler2D"}[inp.type]
        params.append(f"{t} {inp.name}")
    if iterated(op):
        params.append("float _pass")
    return f"{op.returns} {op.name}({', '.join(params)}) {{\n  {op.glsl}\n}}\n"


# Ops that are exactly the identity when one input takes one value: the mixes at amount 0, the
# displacements at amount 0. A renderer can compile a variant of a program without them while that
# input sits there (``gates`` + ``compile_output(skip=...)``), so a layer that is faded out, or a
# section of a piece that isn't playing, costs nothing on the GPU.  User ops can set ``Op.identity``.
IDENTITY_AT = {name: ("amount", 0.0) for name in ("blend", "add", "sub", "mult", "lighten", "darken", "screen", "overlay",
                                                   "modulate", "displace", "invert", "brightness", "swirl", "tine")}
IDENTITY_AT["ripple"] = ("amplitude", 0.0)


def identity_of(op: Op):
    return op.identity or IDENTITY_AT.get(op.name)


def gates(chain: Chain) -> list:
    """``(transform, value, identity)`` for each transform in ``chain`` (and in the chains it takes as
    arguments) that does nothing while ``value`` (a Signal or a number) equals ``identity``."""
    out, seen = [], set()

    def walk(ch):
        if id(ch) in seen:
            return
        seen.add(id(ch))
        for tr in ch.transforms:
            ident = identity_of(tr.op)
            if ident and tr.op.kind != "src":
                names = [i.name for i in tr.op.inputs]
                k = names.index(ident[0]) if ident[0] in names else -1
                if 0 <= k < len(tr.args):
                    a = tr.args[k]
                    if isinstance(a, Signal) or (isinstance(a, (int, float)) and not isinstance(a, bool)):
                        out.append((tr, a, float(ident[1])))
            for a in tr.args:
                if isinstance(a, Chain):
                    walk(a)
    walk(chain)
    return out


SHARE_SUBCHAINS = True      # compute a repeated sub-chain once per pass (glsl._Compiler.shared)


def glsl_float(v: float) -> str:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("non-finite number in chain")
    s = repr(v)
    if "e" in s or "E" in s:
        s = f"{v:.9g}"
        if "." not in s and "e" not in s:
            s += ".0"
    elif "." not in s:
        s += ".0"
    return s


class _Compiler:
    def __init__(self, program: Program, patch: Optional[Patch], skip=frozenset()):
        self.p = program
        self.patch = patch
        self.skip = skip              # ids of transforms to leave out (they are the identity this frame)
        self._tmp = 0
        self._float_ids: dict[int, Uniform] = {}
        self._sampler_names: dict[str, Uniform] = {}
        self._prefix_textures: dict[tuple, str] = {}
        self._mode = None             # sharing: None (off), "count" or "emit", while a pass is generated
        self._counts: dict = {}
        self._names: dict = {}
        self._lets: list = []
        self._chain_keys: dict = {}

    # -- structural keys: equal keys generate equal GLSL -----------------------
    @staticmethod
    def signal_key(sig):
        if isinstance(sig, Const):
            return ("c", sig.v)
        if isinstance(sig, Expr):
            return ("e", sig.op, _Compiler.signal_key(sig.a), _Compiler.signal_key(sig.b))
        if isinstance(sig, Fn1):
            return ("f", sig.fn, _Compiler.signal_key(sig.a), tuple(sig.extra))
        return ("id", id(sig))        # anything else may carry state: one uniform per object

    def arg_key(self, a):
        if isinstance(a, Chain):
            return self.chain_key(a)
        if isinstance(a, Signal):
            return self.signal_key(a)
        if isinstance(a, bool):
            return ("n", float(a))
        if isinstance(a, (int, float)):
            return ("n", float(a))
        if isinstance(a, Texture):
            return ("t", a.name)
        return ("id", id(a))

    def chain_key(self, chain: Chain):
        key = self._chain_keys.get(id(chain))
        if key is None:
            key = tuple(("skip",) if id(tr) in self.skip else (tr.op.name, tuple(self.arg_key(a) for a in tr.args))
                        for tr in chain.transforms)
            self._chain_keys[id(chain)] = key
        return key

    # -- uniforms --------------------------------------------------------
    def float_uniform(self, sig) -> str:
        sig = as_signal(sig)
        key = self.signal_key(sig)    # structurally equal expressions share one uniform
        if key not in self._float_ids:
            u = Uniform(f"u_{len(self._float_ids)}", "float", sig)
            self._float_ids[key] = u
            self.p.uniforms.append(u)
        return self._float_ids[key].name

    def sampler_uniform(self, ref: Texture, name: Optional[str] = None) -> str:
        name = name or f"tex_{ref.name}"
        if name not in self._sampler_names:
            u = Uniform(name, "sampler2D", ref)
            self._sampler_names[name] = u
            self.p.uniforms.append(u)
        return name

    def new_tmp(self) -> str:
        self._tmp += 1
        return f"tmp{self._tmp - 1}"

    def use(self, op: Op):
        if op not in self.p.functions:
            self.p.functions.append(op)

    def add_pass(self, target: str, expr: Callable[[str], str]):
        """Generate one pass. Argument chains that occur more than once at the same coordinates (the same
        chain object, or an equal one) are computed once into a local: count them, then emit."""
        compute = expr.compute_step() if isinstance(expr, _TextureCall) else None
        saved = (self._mode, self._counts, self._names, self._lets)
        self._mode, self._counts = "count", {}
        expr("st")
        self._mode, self._names, self._lets = "emit", {}, []
        code = expr("st")
        lets = self._lets
        self._mode, self._counts, self._names, self._lets = saved
        self.p.passes.append(Pass(target, code, compute=compute, lets=lets))

    def shared(self, key: tuple, frag: Callable[[str], str]) -> Callable[[str], str]:
        """``frag`` (the GLSL of a chain, or of a chain's first k transforms, whose structural key is ``key``),
        hoisted into a local when the same key is generated more than once at the same coordinates."""
        if not SHARE_SUBCHAINS or isinstance(frag, _TextureCall):
            return frag                                       # a _TextureCall keeps its compute eligibility

        def f(uv):
            k = (key, uv)
            if self._mode == "count":
                n = self._counts.get(k, 0)
                self._counts[k] = n + 1
                return frag(uv) if n == 0 else "_"           # nested uses are counted once per distinct site
            if self._mode == "emit" and self._counts.get(k, 0) > 1:
                if k not in self._names:
                    code = frag(uv)
                    self._names[k] = f"_s{len(self._lets)}"
                    self._lets.append(f"vec4 {self._names[k]} = {code};")
                return self._names[k]
            return frag(uv)
        return f

    # -- arguments -------------------------------------------------------
    def arg(self, a, inp) -> Callable[[str], str]:
        """Return a function of the uv expression giving this argument's GLSL."""
        if inp.type == "sampler2D":
            name = self.sampler_uniform(a)
            return lambda uv: name
        if isinstance(a, bool):
            a = float(a)
        if isinstance(a, int) and inp.type == "int":
            s = str(int(a))
            return lambda uv: s
        if isinstance(a, (int, float)):
            s = glsl_float(a) if inp.type != "int" else str(int(a))
            return lambda uv: s
        if isinstance(a, Chain):
            f = self.gen(a)
            if inp.type == "int":
                return lambda uv: f"int(({f(uv)}).r)"
            return lambda uv: f"({f(uv)}).r"
        if isinstance(a, Signal) or callable(a):
            name = self.float_uniform(a)
            if inp.type == "int":
                return lambda uv: f"int({name})"
            return lambda uv: name
        raise TypeError(f"cannot compile argument {a!r} for {inp.name}")

    # -- chain -----------------------------------------------------------
    def gen(self, chain: Chain) -> Callable[[str], str]:
        frag: Optional[Callable[[str], str]] = None
        parts: list = []                  # the structural key of the prefix generated so far
        for index, tr in enumerate(chain.transforms):
            if frag is not None and id(tr) in self.skip:
                parts.append(("skip",))
                continue
            parts.append((tr.op.name, tuple(self.arg_key(a) for a in tr.args)))
            op, args = tr.op, list(tr.args)
            if op.cpu_only:
                raise NotImplementedError(f".{op.name}() is CPU-only and cannot be compiled to GLSL")
            if op.kind != "src" and frag is None:
                raise ValueError(f"a chain must start with a source, not .{op.name}()")
            self.use(op)
            inputs = list(op.inputs)
            if op.name == "delay":
                ref, frames = args[0], args[1]
                if not isinstance(ref, Output):
                    raise TypeError("delay() takes an output (o0..o3) and a literal number of frames")
                if not isinstance(frames, (int, float)):
                    raise TypeError("delay(frames=...) must be a literal integer for the GLSL backend")
                frames = max(int(frames), 0)
                args = [DelayRef(ref, frames), frames]
                self.p.history[ref.name] = max(self.p.history.get(ref.name, 0), frames)
            chain_arg = None
            if op.kind in ("combine", "combineCoord"):
                chain_arg, args, inputs = args[0], args[1:], inputs[1:]
            iterations = 1
            if op.kind == "texture" and any(i.name == "iterations" for i in inputs):
                k = [i.name for i in inputs].index("iterations")
                it = args[k]
                if not isinstance(it, (int, float)):
                    raise TypeError(f"{op.name}(iterations=...) must be a literal integer for the GLSL backend")
                iterations = max(int(it), 1)
                args, inputs = args[:k] + args[k + 1:], inputs[:k] + inputs[k + 1:]
            argfns = [self.arg(a, i) for a, i in zip(args, inputs)]

            def rest(uv, argfns=argfns):
                return "".join(", " + f(uv) for f in argfns)

            f0 = frag
            name = op.name
            if op.kind == "src":
                frag = lambda uv, name=name, rest=rest: f"{name}({uv}{rest(uv)})"
            elif op.kind == "coord":
                frag = lambda uv, f0=f0, name=name, rest=rest: f0(f"{name}({uv}{rest(uv)})")
            elif op.kind == "color":
                frag = lambda uv, f0=f0, name=name, rest=rest: f"{name}({f0(uv)}{rest(uv)})"
            elif op.kind == "combine":
                f1 = self.gen(chain_arg)
                frag = lambda uv, f0=f0, f1=f1, name=name, rest=rest: f"{name}({f0(uv)}, {f1(uv)}{rest(uv)})"
            elif op.kind == "combineCoord":
                f1 = self.gen(chain_arg)
                frag = lambda uv, f0=f0, f1=f1, name=name, rest=rest: f0(f"{name}({uv}, {f1(uv)}{rest(uv)})")
            elif op.kind == "texture":
                # materialise the prefix into its own pass
                key = tuple(id(t) for t in chain.transforms[:index])
                # bake() already materialised this exact prefix. Reuse its
                # half-float target instead of making an identity copy.
                if index and chain.transforms[index - 1].op.name == "bake":
                    key = key[:-1]
                # Longer iterations overwrite their input while ping-ponging.
                # Give them a private target: an earlier branch may still
                # sample a cached prefix in the final composite pass.
                reusable = iterations <= 2
                tmp = self._prefix_textures.get(key) if reusable else None
                if tmp is None:
                    tmp = self.new_tmp()
                    self.add_pass(tmp, f0)
                    if reusable:
                        self._prefix_textures[key] = tmp
                tex = self.sampler_uniform(TmpRef(tmp))
                # iterated ops get their 0-based iteration index as the last argument
                last = iterations - 1 if iterated(op) else None
                # extra iterations ping-pong between this target and one more
                if iterations > 1:
                    names = [tmp, self.new_tmp()]
                    for i in range(1, iterations):
                        target = names[i % 2]
                        self.add_pass(target, _TextureCall(op, tex, argfns, i - 1))
                        tex = self.sampler_uniform(TmpRef(target))
                frag = _TextureCall(op, tex, argfns, last)
            else:
                raise ValueError(op.kind)
            frag = self.shared(tuple(parts), frag)
        if frag is None:
            raise ValueError("empty chain")
        return frag


def compile_output(output: Output, patch: Optional[Patch] = None, precision: str = "highp", skip=frozenset()) -> Program:
    """``skip``: ids of transforms to leave out — ones ``gates`` reports at their identity value."""
    if output.chain is None:
        raise ValueError(f"{output.name} has no chain assigned")
    prog = Program(output=output, precision=precision)
    c = _Compiler(prog, patch, skip)
    c.sampler_uniform(output, "prevBuffer")  # always available for prev()
    c.add_pass(output.name, c.gen(output.chain))
    return prog.finish()


def compile_chain(chain: Chain, precision: str = "highp") -> Program:
    """Compile a bare chain as if it were assigned to o0."""
    o = Output(0)
    o.chain = chain
    return compile_output(o, None, precision=precision)


def compile_patch(patch: Optional[Patch] = None, precision: str = "highp") -> list[Program]:
    patch = patch or default_patch()
    progs = [compile_output(o, patch, precision=precision) for o in patch.active_outputs()]
    if not progs:
        raise ValueError("patch has no outputs assigned; call .out() on a chain first")
    return progs


# Desktop-GL (moderngl) flavour of the same shader
def to_desktop(source: str) -> str:
    body = "\n".join(l for l in source.splitlines() if not l.strip().startswith("precision "))
    return ("#version 330 core\n#define texture2D texture\nout vec4 _fragColor;\n#define gl_FragColor _fragColor\n"
            + body)


VERTEX_ES = "attribute vec2 position;\nvoid main () { gl_Position = vec4(position, 0.0, 1.0); }\n"
VERTEX_DESKTOP = "#version 330 core\nin vec2 position;\nvoid main () { gl_Position = vec4(position, 0.0, 1.0); }\n"
