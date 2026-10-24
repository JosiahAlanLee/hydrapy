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
from .signals import Fn, Signal, as_signal


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
            p.source = (head + "\nvoid main () {\n  vec2 st = gl_FragCoord.xy / resolution.xy;\n"
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
    def __init__(self, program: Program, patch: Optional[Patch]):
        self.p = program
        self.patch = patch
        self._tmp = 0
        self._float_ids: dict[int, Uniform] = {}
        self._sampler_names: dict[str, Uniform] = {}
        self._prefix_textures: dict[tuple, str] = {}

    # -- uniforms --------------------------------------------------------
    def float_uniform(self, sig) -> str:
        sig = as_signal(sig)
        key = id(sig)
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
        compute = expr.compute_step() if isinstance(expr, _TextureCall) else None
        self.p.passes.append(Pass(target, expr("st"), compute=compute))

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
        for index, tr in enumerate(chain.transforms):
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
        if frag is None:
            raise ValueError("empty chain")
        return frag


def compile_output(output: Output, patch: Optional[Patch] = None, precision: str = "highp") -> Program:
    if output.chain is None:
        raise ValueError(f"{output.name} has no chain assigned")
    prog = Program(output=output, precision=precision)
    c = _Compiler(prog, patch)
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
