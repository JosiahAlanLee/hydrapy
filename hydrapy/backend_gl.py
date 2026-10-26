"""GPU renderer (moderngl): headless offline rendering or a live window.

    pip install moderngl            # headless (EGL on Linux, native elsewhere)
    pip install moderngl glfw       # live window:  hydrapy live patch.py

The runtime mirrors the HTML export: one program per pass, double-buffered
outputs, intermediate targets for texture ops, signals evaluated on the CPU
and sent as uniforms.
"""
from __future__ import annotations

import math
import time as _time
import warnings
from typing import Optional

import numpy as np

from .core import Camera, Ctx, DelayRef, FrameSource, Image, Output, Patch, RenderClock, Video, default_patch
from .glsl import VERTEX_DESKTOP, TmpRef, compile_output, compile_patch, gates, to_desktop
from .signals import as_signal


def _create_context(compute=False):
    import moderngl
    error = None
    for version in ((430, 330) if compute else (330,)):
        for options in ({}, {"backend": "egl"}):
            try:
                return moderngl.create_standalone_context(require=version, **options)
            except Exception as exc:
                error = exc
    raise error


class _MipTarget(str):
    """A pass target whose mip levels are built after it is drawn (a mipmaps op samples it)."""


class GLRenderer:
    """Renders a patch with moderngl. ``ctx`` may be an existing context.

    ``compute=True`` opts into experimental rd/bleed kernels on OpenGL 4.3+.
    Unsupported contexts or rejected kernels retain fragment rendering, with a
    warning. ``compute_passes`` reports the number actually selected.
    """

    def __init__(self, patch: Optional[Patch] = None, size=None, ctx=None, fps: float = 60.0,
                 precision: str = "highp", compute: bool = False, fast_noise: Optional[bool] = None):
        import moderngl
        self.gl = moderngl
        self.patch = patch or default_patch()
        self.size = tuple(size or self.patch.size)
        self.fps = float(fps)
        self.ctx = ctx or _create_context(compute=compute)
        self.precision = precision
        self.compute = compute
        self.compute_passes = 0
        self.compute_fallbacks = []
        use_compute = compute and self.ctx.version_code >= 430
        if compute and not use_compute:
            self.compute_fallbacks.append("OpenGL 4.3 is required; using fragment shaders")
        self._use_compute, self._failed_kernels = use_compute, set()
        self.frame = 0
        self.clock = RenderClock(self.fps)
        self._reduce_targets = {}
        self._reduce_program = self._reduce_vao = None
        self.mouse = (0.5, 0.5)
        self.fast_noise = self.patch.fast_noise if fast_noise is None else bool(fast_noise)
        self._noise_tex = None
        if self.fast_noise:
            from .noise import noise_texture
            self._noise_tex = self.ctx.texture((256, 256), 4, noise_texture().tobytes())
            self._noise_tex.filter = (self.gl.NEAREST, self.gl.NEAREST)   # corners are blended in the shader
            self._noise_tex.repeat_x = self._noise_tex.repeat_y = True
        self.programs = compile_patch(self.patch, precision=precision)
        self.vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        # Each output is drawn by a variant of its program compiled without the transforms that are the
        # identity this frame (an amount at 0, a layer faded out): see glsl.gates. Variants are cached by
        # that set; the full program is built now so shader errors surface at once.
        self._gates = {prog.output.name: gates(prog.output.chain) for prog in self.programs}
        self._variants: dict = {}
        self.specialise = True
        self.settle = 0                      # frames a new idle set must hold before its variant compiles (live uses 12)
        self._pending, self._active = {}, {}
        for prog in self.programs:
            self._variants[(prog.output.name, None)] = self._build(prog)
            self.compute_passes += sum(1 for _, _, vao, _, _ in self._variants[(prog.output.name, None)] if vao is None)
        for reason in self.compute_fallbacks:
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
        self._ready = True
        self._copy = self.ctx.program(
            vertex_shader=VERTEX_DESKTOP,
            fragment_shader="#version 330 core\nuniform sampler2D tex; uniform vec2 resolution; uniform float compositeAlpha; out vec4 f;\n"
                            "void main(){ vec4 c = texture(tex, gl_FragCoord.xy / resolution); f = vec4(c.rgb * mix(1.0, clamp(c.a, 0.0, 1.0), compositeAlpha), 1.0); }")
        self._copy_vao = self.ctx.vertex_array(self._copy, [(self.vbo, "2f", "position")])
        self._alloc()
        self.media: dict[str, object] = {}
        from .nca import NCA
        self._ncas: dict = {}
        self._nca_refs, self._sims = [], []
        for prog in self.programs:
            for u in prog.sampler_uniforms:
                if isinstance(u.source, NCA) and u.source not in self._nca_refs:
                    self._nca_refs.append(u.source)
                elif hasattr(u.source, "gl_step") and u.source not in self._sims:
                    self._sims.append(u.source)
        self._blank = self.ctx.texture((1, 1), 4, data=bytes([0, 0, 0, 255]))

    @property
    def _passes(self):
        """The full (unspecialised) passes in draw order: (output, target, program, vao, floats, samplers)."""
        return [(prog.output, *p) for prog in self.programs for p in self._variants[(prog.output.name, None)]]

    def _build(self, prog):
        passes = []
        for p in prog.passes:
            if self._use_compute and p.compute and p.compute.kernel not in self._failed_kernels:
                from .compute import source
                try:
                    kernel = self.ctx.compute_shader(source(p.compute, prog.float_uniforms))
                except Exception as e:
                    self._failed_kernels.add(p.compute.kernel)
                    reason = f"{p.compute.kernel} compute compilation failed; using fragment shaders: {e}"
                    self.compute_fallbacks.append(reason)
                    if getattr(self, "_ready", False):      # a later variant; __init__ reports its own at the end
                        warnings.warn(reason, RuntimeWarning, stacklevel=3)
                else:
                    passes.append((p.target, kernel, None, prog.float_uniforms,
                                   [u for u in prog.sampler_uniforms if u.name == p.compute.sampler]))
                    continue
            source = to_desktop(p.source, fast_noise=self.fast_noise)
            try:
                gprog = self.ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=source)
            except Exception as e:
                numbered = "\n".join(f"{i + 1:4d}: {l}" for i, l in enumerate(source.splitlines()))
                raise RuntimeError(f"shader compile error in {prog.output.name}/{p.target}:\n{e}\n{numbered}")
            vao = self.ctx.vertex_array(gprog, [(self.vbo, "2f", "position")])
            passes.append((_MipTarget(p.target) if p.mipmap else p.target, gprog, vao, prog.float_uniforms,
                           prog.sampler_uniforms))
            key = (prog.output.name, p.target)
            if p.target.startswith("tmp") and hasattr(self, "tmps") and key not in self.tmps:
                self.tmps[key] = self._fbo(self._osize(prog.output.name))   # a variant compiled after _alloc()
        return passes

    def _idle(self, name, value):
        return frozenset(id(tr) for tr, v, ident in self._gates[name]
                         if (float(v) if isinstance(v, (int, float)) else value(v)) == ident) or None

    def _compiled(self, prog, key):
        if (prog.output.name, key) not in self._variants:
            self._variants[(prog.output.name, key)] = self._build(
                compile_output(prog.output, self.patch, precision=self.precision, skip=key))
        return self._variants[(prog.output.name, key)]

    def _variant(self, prog, value):
        """The passes for ``prog`` this frame: compiled without its transforms at their identity value.

        With ``settle`` > 0 a set of idle transforms that hasn't been seen before must hold for that many
        frames before its variant is compiled; meanwhile the last variant that is still valid (one that
        skips a subset of what is idle now), or the full program, draws. A value that touches zero for a
        frame never costs a compile, and a live set never stalls on a passing combination."""
        name = prog.output.name
        key = self._idle(name, value) if self.specialise else None
        if key is not None and (name, key) not in self._variants and self.settle > 0:
            seen, count = self._pending.get(name, (None, 0))
            count = count + 1 if seen == key else 1
            self._pending[name] = (key, count)
            if count < self.settle:
                # Any compiled variant skipping only transforms that are idle now draws the same
                # image: prefer the one that skips most (a precompiled scene, say) over the full
                # program, which would draw every scene at once.
                usable = max((k for n, k in self._variants if n == name and k is not None and k <= key),
                             key=len, default=None)
                self._active[name] = usable
                return self._variants[(name, usable)]
        self._pending.pop(name, None)
        self._active[name] = key
        return self._compiled(prog, key)

    def prewarm_scenes(self, pairs=()) -> int:
        """Compile each scene's variant (its output with the other scenes' blends compiled out),
        and one for each (scene, scene) pair in ``pairs``: the crossfades the score contains.

        Changing scene then switches between compiled shaders. Returns the number compiled.
        """
        from .scenes import scene_transforms
        transforms = scene_transforms(self.patch)
        if not transforms or not self.specialise:
            return 0
        before = len(self._variants)
        ctx = Ctx(time=float(getattr(self, 'ctx_time', 0.) or 0.), width=self.size[0], height=self.size[1],
                  bpm=self.patch.bpm, dt=1.0 / self.fps, fps=self.fps)
        for prog in self.programs:
            name = prog.output.name
            gated = {id(tr) for tr, _, _ in self._gates[name]}
            mine = [id(tr) for tr in transforms.values() if id(tr) in gated]
            wanted = [{id(transforms[n])} for n in transforms if id(transforms[n]) in mine]
            wanted += [{id(transforms[a]), id(transforms[b])} for a, b in pairs
                       if a in transforms and b in transforms and {id(transforms[a]), id(transforms[b])} <= set(mine)]
            scene_of = {id(tr): sig for tr, sig, _ in self._gates[name] if id(tr) in mine}
            for keep in wanted:
                # The scene alone, and the scene with the output's other gates as they are now
                # (a control at 0, say): the variant live playback will actually ask for.
                idle = frozenset(t for t in mine if t not in keep)
                forced = {id(sig): (1.0 if t in keep else 0.0) for t, sig in scene_of.items()}

                def value(sig, forced=forced):
                    if id(sig) in forced:
                        return forced[id(sig)]
                    try:
                        v = float(as_signal(sig)(ctx))
                    except Exception:
                        return float('nan')              # unknown now: not idle
                    return v if np.isfinite(v) else 0.0
                for key in {idle or None, self._idle(name, value)}:
                    if key:
                        self._compiled(prog, key)
        return len(self._variants) - before

    def prewarm(self, times) -> int:
        """Compile, now, the variants the score will need at ``times`` (seconds), so a render or a set
        never compiles mid-way. Only gates driven by signals that can be evaluated at any time without
        side effects are predicted (audio analysis and live inputs are not); returns the number compiled."""
        from .controls import Param
        from .signals import Fft, Signal

        def pure(sig, depth=0):
            # predictable: hydrapy's own time-driven signals (and Fn, which exports already sample at
            # arbitrary times); not audio analysis, controls, live inputs, probes or unknown subclasses
            if not isinstance(sig, Signal):
                return True
            mod = type(sig).__module__
            if (depth > 40 or isinstance(sig, (Fft, Param)) or not mod.startswith("hydrapy.")
                    or mod in ("hydrapy.live", "hydrapy.scenes")):   # scenes follow the tracker, not time
                return False
            for v in vars(sig).values():
                for item in (v if isinstance(v, (list, tuple)) else (v,)):
                    if isinstance(item, Signal) and not pure(item, depth + 1):
                        return False
            return True

        before = len(self._variants)
        for prog in self.programs:
            name = prog.output.name
            if not self.specialise or not self._gates[name]:
                continue
            if not all(isinstance(v, (int, float)) or pure(v) for _, v, _ in self._gates[name]):
                continue
            seen = set()
            for t in times:
                ctx = Ctx(time=float(t), frame=int(round(float(t) * self.fps)), width=self.size[0], height=self.size[1],
                          bpm=self.patch.bpm, dt=1.0 / self.fps, fps=self.fps)
                def value(sig, ctx=ctx):
                    v = float(as_signal(sig)(ctx))
                    return v if np.isfinite(v) else 0.0
                key = self._idle(name, value)
                if key not in seen:
                    seen.add(key)
                    self._compiled(prog, key)
        return len(self._variants) - before

    # -- buffers -----------------------------------------------------------
    def _osize(self, name) -> tuple:
        """The pixel size an output renders at (Chain.out(scale=...))."""
        return self.patch.output_size(name, self.size)

    def _fbo(self, size=None, dtype="f2"):
        tex = self.ctx.texture(size or self.size, 4, dtype=dtype)
        tex.filter = (self.gl.LINEAR, self.gl.LINEAR)
        tex.repeat_x = tex.repeat_y = True
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        return {"tex": tex, "fbo": fbo}

    def release(self):
        """Free every GL object this renderer created (the context itself is kept)."""
        for o in getattr(self, "outs", {}).values():
            for fb in o["ring"]:
                fb["fbo"].release(); fb["tex"].release()
        for fb in getattr(self, "tmps", {}).values():
            fb["fbo"].release(); fb["tex"].release()
        self._present["fbo"].release(); self._present["tex"].release()
        if getattr(self, "_small", None) is not None:
            self._small["fbo"].release(); self._small["tex"].release()
            self._small = None
        for fb in self._reduce_targets.values():
            fb["fbo"].release(); fb["tex"].release()
        self._reduce_targets.clear()
        if self._reduce_program is not None:
            self._reduce_vao.release(); self._reduce_program.release()
        for m in getattr(self, "media", {}).values():
            if m.get("tex") is not None:
                m["tex"].release()
            for t in m.get("texs", {}).values():
                t.release()
        for passes in getattr(self, "_variants", {}).values():
            for _, gprog, vao, _, _ in passes:
                if vao is not None:
                    vao.release()
                gprog.release()
        for sim in getattr(self, "_sims", []):
            sim.gl_release(self)
        for n in getattr(self, "_ncas", {}).values():
            for gp, vao in n["steps"] + [n["view_prog"]]:
                vao.release(); gp.release()
            for fb in n.get("mrt", []):
                fb.release()
            for fb in [f for sset in n["sets"] for f in sset] + [n["view"]]:
                fb["fbo"].release(); fb["tex"].release()
        self._ncas = {}
        self._copy_vao.release(); self._copy.release(); self.vbo.release(); self._blank.release()
        if self._noise_tex is not None:
            self._noise_tex.release(); self._noise_tex = None
        self.outs, self.tmps, self.media, self._variants = {}, {}, {}, {}
        self._pending, self._active = {}, {}

    def _alloc(self):
        # each output keeps a ring of frames: ring[0] is the latest completed, ring[-1] is written next
        if getattr(self, "outs", None):
            for o in self.outs.values():
                for fb in o["ring"]:
                    fb["fbo"].release(); fb["tex"].release()
            for fb in self.tmps.values():
                fb["fbo"].release(); fb["tex"].release()
            self._present["fbo"].release(); self._present["tex"].release()
        depth = {}
        for prog in self.programs:
            for k, v in prog.history.items():
                depth[k] = max(depth.get(k, 0), v)
        self.outs = {f"o{i}": {"ring": [self._fbo(self._osize(f"o{i}")) for _ in range(max(2, depth.get(f"o{i}", 0) + 2))]}
                     for i in range(4)}
        self.tmps = {}                      # (output name, tmpN) -> target at that output's size
        self._present = self._fbo(dtype="f1")
        for prog in self.programs:
            for p in prog.passes:
                key = (prog.output.name, p.target)
                if p.target.startswith("tmp") and key not in self.tmps:
                    self.tmps[key] = self._fbo(self._osize(prog.output.name))

    def resize(self, size):
        if tuple(size) == self.size:
            return
        self.size = tuple(size)
        self.clock.last_key = None
        for fb in self._reduce_targets.values():
            fb["fbo"].release(); fb["tex"].release()
        self._reduce_targets.clear()
        self._alloc()

    # -- media -------------------------------------------------------------
    def _media_texture(self, ref):
        if hasattr(ref, "dynamic_array"):           # computed in Python every frame (vector.XY phosphor)
            m = self.media.setdefault(ref.name, {"tex": None, "ref": ref, "cap": None, "frame": -1})
            if m["frame"] != self.frame:
                a = np.ascontiguousarray(ref.dynamic_array(self.ctx_time), dtype=np.float32)
                if m["tex"] is None or m["tex"].size != (a.shape[1], a.shape[0]):
                    if m["tex"] is not None:
                        m["tex"].release()
                    m["tex"] = self.ctx.texture((a.shape[1], a.shape[0]), 4, dtype="f4")
                    m["tex"].filter = (self.gl.LINEAR, self.gl.LINEAR)
                m["tex"].write(a)
                m["frame"] = self.frame
            return m["tex"]
        if ref.name in self.media:
            m = self.media[ref.name]
        else:
            m = {"tex": None, "ref": ref, "cap": None}
            self.media[ref.name] = m
            if isinstance(ref, Image) or (hasattr(ref, "array") and not hasattr(ref, "array_at")):   # Image, Text, Lines
                a = (np.clip(ref.array(), 0, 1) * 255).astype(np.uint8)
                m["tex"] = self.ctx.texture((a.shape[1], a.shape[0]), 4, data=np.ascontiguousarray(a).tobytes())
            elif isinstance(ref, Camera):
                try:
                    import cv2
                    m["cap"] = cv2.VideoCapture(ref.device)
                except ImportError:
                    raise RuntimeError("Camera sources in the GL backend need opencv-python (pip install opencv-python)")
        if isinstance(ref, FrameSource):
            if m.get("time") != self.ctx_time:
                frame = ref.frame_at(self.ctx_time)
                m["time"] = self.ctx_time
                if m.get("version") != ref.version:        # upload only frames that changed
                    frame = np.ascontiguousarray(frame[::-1])
                    size = (frame.shape[1], frame.shape[0])
                    if m["tex"] is None or m["tex"].size != size:
                        if m["tex"] is not None:
                            m["tex"].release()
                        m["tex"] = self.ctx.texture(size, 4)
                    m["tex"].write(frame)
                    m["version"] = ref.version
            return m["tex"]
        if hasattr(ref, "array_at"):          # Bank, Quilt: one texture per frame, bound by index
            i = ref.index_at(self.ctx_time)
            texs = m.setdefault("texs", {})
            if i not in texs:
                f = np.ascontiguousarray(ref.frame(i)[::-1])
                rgba = f if f.shape[-1] == 4 else np.concatenate([f, np.full(f.shape[:2] + (1,), 255, np.uint8)], axis=-1)
                texs[i] = self.ctx.texture((rgba.shape[1], rgba.shape[0]), 4, data=np.ascontiguousarray(rgba).tobytes())
            return texs[i]
        if isinstance(ref, Video):
            f = (np.clip(ref.frame_at(self.ctx_time), 0, 1) * 255).astype(np.uint8)
            if m["tex"] is None or m["tex"].size != (f.shape[1], f.shape[0]):
                m["tex"] = self.ctx.texture((f.shape[1], f.shape[0]), 4)
            m["tex"].write(np.ascontiguousarray(f).tobytes())
        elif isinstance(ref, Camera) and m["cap"] is not None:
            ok, frame = m["cap"].read()
            if ok:
                frame = np.ascontiguousarray(frame[::-1, :, ::-1])
                rgba = np.concatenate([frame, np.full(frame.shape[:2] + (1,), 255, np.uint8)], axis=-1)
                if m["tex"] is None or m["tex"].size != (rgba.shape[1], rgba.shape[0]):
                    m["tex"] = self.ctx.texture((rgba.shape[1], rgba.shape[0]), 4)
                m["tex"].write(np.ascontiguousarray(rgba).tobytes())
        return m["tex"] or self._blank

    def _bind(self, ref, current: Output):
        if isinstance(ref, DelayRef):
            ring = self.outs[ref.output.name]["ring"]
            return ring[min(ref.frames, len(ring) - 2)]["tex"]
        if isinstance(ref, Output):
            return self.outs[ref.name]["ring"][0]["tex"]
        if isinstance(ref, TmpRef):
            return self.tmps[(current.name, ref.name)]["tex"]
        from .nca import NCA
        if isinstance(ref, NCA):
            return self._nca(ref)["view"]["tex"]
        if hasattr(ref, "gl_texture"):              # simulations that manage their own GPU state (Physarum)
            return ref.gl_texture(self)
        return self._media_texture(ref)

    # -- neural cellular automata --------------------------------------------
    def _nca(self, ref):
        if ref.name in self._ncas:
            return self._ncas[ref.name]
        from .nca import VIEW_GLSL, shader_header, step_glsl, step_mrt_glsl
        head = shader_header(self.precision)

        def prog(src, mrt=False):
            try:
                source = to_desktop(head + src)
                if mrt:
                    source = source.replace("out vec4 _fragColor;", "")
                p = self.ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=source)
            except Exception as e:
                raise RuntimeError(f"NCA shader compile error: {e}")
            return p, self.ctx.vertex_array(p, [(self.vbo, "2f", "position")])

        def target(filt):
            t = self.ctx.texture(ref.size, 4, dtype="f4")
            t.filter = (filt, filt)
            t.repeat_x = t.repeat_y = True
            fb = self.ctx.framebuffer(color_attachments=[t])
            fb.use(); self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            return {"tex": t, "fbo": fb}

        T = ref.model.textures
        view_filter = self.gl.NEAREST if ref.filter == "nearest" else self.gl.LINEAR
        limit = min(self.ctx.info.get("GL_MAX_DRAW_BUFFERS", 1), self.ctx.info.get("GL_MAX_COLOR_ATTACHMENTS", 1))
        mrt = 1 < T <= limit
        n = {"steps": [prog(step_mrt_glsl(ref.model), mrt=True)] if mrt else [prog(step_glsl(ref.model, k)) for k in range(T)], "view_prog": prog(VIEW_GLSL),
             "sets": [[target(self.gl.NEAREST) for _ in range(T)] for _ in range(2)], "cur": 0,
             "view": target(view_filter)}
        n["mrt"] = [self.ctx.framebuffer(color_attachments=[fb["tex"] for fb in sset]) for sset in n["sets"]] if mrt else []
        self._ncas[ref.name] = n
        return n

    def _step_ncas(self, ctx):
        for ref in self._nca_refs:
            n = self._nca(ref)
            steps = ref.steps_due(ctx.time)
            if ref.gpu_reset:
                for sset in n["sets"]:
                    for fb in sset:
                        fb["fbo"].use(); self.ctx.clear(0.0, 0.0, 0.0, 0.0)
                ref.gpu_reset = False
            T = ref.model.textures
            dmg = self.outs[ref.damage.name]["ring"][0]["tex"] if ref.damage is not None else self._blank
            pnt = self.outs[ref.paint.name]["ring"][0]["tex"] if ref.paint is not None else self._blank
            rate, perturb, amount = float(ref.rate(ctx)), float(ref.perturb(ctx)), float(ref.paint_amount(ctx))
            for _ in range(steps):
                cur, nxt = n["sets"][n["cur"]], n["sets"][1 - n["cur"]]
                for k, (gp, vao) in enumerate(n["steps"]):
                    (n["mrt"][1 - n["cur"]] if n["mrt"] else nxt[k]["fbo"]).use()
                    for t in range(T):
                        cur[t]["tex"].use(location=t)
                        self._set(gp, f"s{t}", t)
                    dmg.use(location=T); self._set(gp, "damage", T)
                    pnt.use(location=T + 1); self._set(gp, "paint", T + 1)
                    self._set(gp, "res", (float(ref.size[0]), float(ref.size[1])))
                    self._set(gp, "stepIndex", float(ref.step_index))
                    self._set(gp, "rate", rate); self._set(gp, "perturb", perturb); self._set(gp, "seed", ref.seed)
                    self._set(gp, "paintAmount", amount)
                    self._set(gp, "damageOn", 1.0 if ref.damage is not None else 0.0)
                    self._set(gp, "paintOn", 1.0 if ref.paint is not None else 0.0)
                    vao.render(self.gl.TRIANGLES)
                n["cur"] = 1 - n["cur"]
                ref.step_index += 1
            vp, vvao = n["view_prog"]
            n["view"]["fbo"].use()
            n["sets"][n["cur"]][0]["tex"].use(location=0)
            self._set(vp, "s0", 0)
            self._set(vp, "res", (float(ref.size[0]), float(ref.size[1])))
            vvao.render(self.gl.TRIANGLES)

    def nca_state(self, ref) -> np.ndarray:
        """The GPU copy of an NCA's state as (h, w, channels) float32, GL-oriented (row 0 = bottom)."""
        n = self._nca(ref)
        parts = [np.frombuffer(fb["tex"].read(), np.float32).reshape(ref.size[1], ref.size[0], 4) for fb in n["sets"][n["cur"]]]
        return np.concatenate(parts, axis=-1)

    # -- frames --------------------------------------------------------------
    def step(self, t: Optional[float] = None):
        ctx, draw, wall_time = self.clock.next(self.patch, self.size, self.mouse, t)
        self.ctx_time = ctx.time
        self.frame += 1
        if not draw:
            self._update_probes(wall_time, changed=False)
            from .live import run_hooks
            run_hooks(ctx, self)
            return
        if self._nca_refs:
            self._step_ncas(ctx)
        for sim in self._sims:
            sim.gl_step(self, ctx)
        values = {}

        def value(source):                   # each signal once per frame, shared by every pass that uses it
            k = id(source)
            if k not in values:
                v = float(as_signal(source)(ctx))
                values[k] = v if np.isfinite(v) else 0.0
            return values[k]

        for prog in self.programs:
            output = prog.output
            w, h = self._osize(output.name)
            for target, gprog, vao, floats, samplers in self._variant(prog, value):
                dest = self.tmps[(output.name, target)] if target.startswith("tmp") else self.outs[output.name]["ring"][-1]
                if vao is not None:
                    dest["fbo"].use()
                    self.ctx.viewport = (0, 0, w, h)
                self._set(gprog, "time", ctx.time)
                self._set(gprog, "resolution", (float(w), float(h)))
                self._set(gprog, "mouse", (self.mouse[0] * w, self.mouse[1] * h))
                for u in floats:
                    self._set(gprog, u.name, value(u.source))
                for i, u in enumerate(samplers):
                    tex = self._bind(u.source, output)
                    tex.use(location=i)
                    self._set(gprog, u.name, i)
                if self._noise_tex is not None and vao is not None:
                    self._noise_tex.use(location=len(samplers))
                    self._set(gprog, "_noiseTex", len(samplers))
                if vao is None:
                    from .compute import GROUP_SIZE
                    dest["tex"].bind_to_image(0, read=False, write=True)
                    gprog.run((w + GROUP_SIZE - 1) // GROUP_SIZE, (h + GROUP_SIZE - 1) // GROUP_SIZE)
                    # Publish image writes to later samplers, image writes, FBO
                    # operations (including readback), and texture read/copy APIs.
                    self.ctx.memory_barrier(self.gl.TEXTURE_FETCH_BARRIER_BIT |
                                            self.gl.SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                            self.gl.FRAMEBUFFER_BARRIER_BIT |
                                            self.gl.TEXTURE_UPDATE_BARRIER_BIT)
                else:
                    vao.render(self.gl.TRIANGLES)
                if isinstance(target, _MipTarget):
                    dest["tex"].build_mipmaps()
                    dest["tex"].filter = (self.gl.LINEAR_MIPMAP_LINEAR, self.gl.LINEAR)
                    dest["mip"] = True
                elif dest.get("mip"):                # the same tmp in another variant: plain sampling again
                    dest["tex"].filter = (self.gl.LINEAR, self.gl.LINEAR)
                    dest["mip"] = False
            self._swap(output)
        self._update_probes(wall_time)
        from .live import run_hooks
        run_hooks(ctx, self)

    def _probe_mean(self, texture, region):
        """Exact tiled sum in float32; only one RGBA pixel crosses to the CPU."""
        if self._reduce_program is None:
            self._reduce_program = self.ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader="""#version 330 core
uniform sampler2D source; uniform ivec2 origin; uniform ivec2 extent; uniform bool visible; out vec4 result;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * 8;
    vec4 total = vec4(0.0);
    for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
        ivec2 p = base + ivec2(x, y);
        if (p.x < extent.x && p.y < extent.y) {
            vec4 c = texelFetch(source, origin + p, 0);
            if (visible) c.rgb *= clamp(c.a, 0.0, 1.0);
            total += c;
        }
    }
    result = total;
}""")
            self._reduce_vao = self.ctx.vertex_array(self._reduce_program, [(self.vbo, "2f", "position")])
        x, y, w, h = region
        count = w * h
        level = 0
        while True:
            size = ((w + 7) // 8, (h + 7) // 8)
            key = (level, size)
            if key not in self._reduce_targets:
                self._reduce_targets[key] = self._fbo(size, "f4")
            dest = self._reduce_targets[key]
            dest["fbo"].use()
            self.ctx.viewport = (0, 0, *size)
            texture.use(location=0)
            self._reduce_program["source"].value = 0
            self._reduce_program["origin"].value = (x, y)
            self._reduce_program["extent"].value = (w, h)
            self._reduce_program["visible"].value = level == 0 and self.patch.composite_alpha
            self._reduce_vao.render(self.gl.TRIANGLES)
            if size == (1, 1):
                data = dest["fbo"].read(components=4, dtype="f4")
                self.ctx.viewport = (0, 0, *self.size)
                return np.frombuffer(data, np.float32).reshape(1, 1, 4) / count
            texture, (w, h), x, y = dest["tex"], size, 0, 0
            level += 1

    def _update_probes(self, wall_time, changed=True):
        from .live import Probe
        for pr in self.patch.probes:
            audio = hasattr(pr, "advance")
            if not changed:
                if not audio or pr.advance(wall_time, 1.0 / self.fps):
                    continue
            if audio and not pr.needs_update(wall_time, 1.0 / self.fps):
                continue
            x0, y0, w, h = pr.region(*self._osize(pr.output.name))
            source = self.outs[pr.output.name]["ring"][0]
            if type(pr) is Probe:
                rgba = self._probe_mean(source["tex"], (x0, y0, w, h))
                pr.update_mean(rgba, self.ctx_time)
                continue
            else:
                data = source["fbo"].read(viewport=(x0, y0, w, h), components=4, dtype="f4")
                rgba = np.frombuffer(data, dtype=np.float32).reshape(h, w, 4)
            if audio:
                pr.update(rgba, wall_time, 1.0 / self.fps)
            else:
                pr.update(rgba, self.ctx_time)

    def _swap(self, output: Output):
        ring = self.outs[output.name]["ring"]
        ring.insert(0, ring.pop())          # the frame just written becomes ring[0]

    def _set(self, prog, name, value):
        u = prog.get(name, None)
        if u is not None:
            u.value = value

    def displayed_texture(self):
        return self.outs[self.patch.display.name]["ring"][0]["tex"]

    def read(self) -> np.ndarray:
        """The displayed output over black as RGB, row 0 = top."""
        self._present_frame()
        data = self._present["fbo"].read(components=3, dtype="f1")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)[::-1].copy()

    def read_scaled(self, size) -> np.ndarray:
        """``read()`` at another size, scaled on the GPU first: a small preview costs a small readback."""
        size = tuple(size)
        if size == tuple(self.size):
            return self.read()
        small = getattr(self, "_small", None)
        if small is None or small["tex"].size != size:
            if small is not None:
                small["fbo"].release(); small["tex"].release()
            self._small = small = self._fbo(size, dtype="f1")
        small["fbo"].use()
        self.ctx.viewport = (0, 0, *size)
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", tuple(map(float, size)))
        self._set(self._copy, "compositeAlpha", float(self.patch.composite_alpha))
        self._copy_vao.render(self.gl.TRIANGLES)
        data = small["fbo"].read(components=3, dtype="f1")
        return np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 3)[::-1].copy()

    def read_rgba(self) -> np.ndarray:
        """Uncomposited straight RGBA, for analysis and transparent-layer tests."""
        fbo = self.outs[self.patch.display.name]["ring"][0]["fbo"]
        w, h = self._osize(self.patch.display.name)    # a scaled display is read at its own size
        return np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(h, w, 4)[::-1].copy()

    def blit_to_screen(self, screen_size):
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, screen_size[0], screen_size[1])
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", (float(screen_size[0]), float(screen_size[1])))
        self._set(self._copy, "compositeAlpha", float(self.patch.composite_alpha))
        self._copy_vao.render(self.gl.TRIANGLES)

    def _present_frame(self):
        self._present["fbo"].use()
        self.ctx.viewport = (0, 0, *self.size)
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", tuple(map(float, self.size)))
        self._set(self._copy, "compositeAlpha", float(self.patch.composite_alpha))
        self._copy_vao.render(self.gl.TRIANGLES)

    def frames(self, n: int, start: float = 0.0, pipelined: bool = True):
        """Yield n RGB frames (row 0 = top). ``pipelined``: each frame is read back through a pixel buffer
        while the GPU renders the next one, instead of stalling on every frame (same frames, same order)."""
        if not pipelined:
            for i in range(n):
                self.step(start + i / self.fps)
                yield self.read()
            return
        w, h = self.size
        pbos = [self.ctx.buffer(reserve=w * h * 3) for _ in range(2)]
        try:
            pending = None
            for i in range(n):
                self.step(start + i / self.fps)
                self._present_frame()
                self._present["fbo"].read_into(pbos[i % 2], components=3, alignment=1)
                if pending is not None:
                    yield np.frombuffer(pending.read(), np.uint8).reshape(h, w, 3)[::-1].copy()
                pending = pbos[i % 2]
            if pending is not None:
                yield np.frombuffer(pending.read(), np.uint8).reshape(h, w, 3)[::-1].copy()
        finally:
            for b in pbos:
                b.release()


def iter_frames_gl(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                   start: float = 0.0, warmup: int = 0, compute: bool = False):
    r = GLRenderer(patch, size=size, fps=fps, compute=compute)
    try:
        r.prewarm([start + i / fps for i in range(-warmup, int(round(seconds * fps)))])
        for i in range(warmup):
            r.step(start - (warmup - i) / fps)
        from .live import clear_recordings
        clear_recordings()
        for pr in r.patch.probes:
            if hasattr(pr, "clear"):
                pr.clear()
        yield from r.frames(int(round(seconds * fps)), start=start)
    finally:
        r.release()


def render_frames_gl(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                     start: float = 0.0, warmup: int = 0, compute: bool = False) -> list[np.ndarray]:
    return list(iter_frames_gl(patch, seconds, fps, size, start, warmup, compute))


def live(patch: Optional[Patch] = None, size=(1280, 720), fps: float = 60.0, title: str = "hydrapy",
         fullscreen: bool = False, watch: Optional[str] = None, loader=None):
    """Open a window and run the patch in real time (needs glfw). Esc quits, S saves a PNG, space pauses.

    ``watch``: a file to watch; when it changes, ``loader()`` builds the new patch and it replaces the
    running one without stopping the clock, the outputs' last frames carried over (feedback
    continues), devices and ports closed and reopened. A patch that fails to load leaves the old one
    running and prints the error — Hydra's ctrl-enter, on save."""
    import glfw
    import moderngl
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    monitor = glfw.get_primary_monitor() if fullscreen else None
    if fullscreen:
        mode = glfw.get_video_mode(monitor)
        size = (mode.size.width, mode.size.height)
    window = glfw.create_window(size[0], size[1], title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create a window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()
    fb = glfw.get_framebuffer_size(window)
    r = GLRenderer(patch, size=fb, ctx=ctx, fps=fps)
    r.settle = LIVE_SETTLE
    transport = r.patch.transport
    state = {"paused": False, "t0": glfw.get_time(), "paused_at": 0.0}

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            if state["paused"]:
                state["paused_at"] = glfw.get_time()
            else:
                state["t0"] += glfw.get_time() - state["paused_at"]
        elif glfw.KEY_1 <= key <= glfw.KEY_4:
            r.patch.render(r.patch.outputs[key - glfw.KEY_1])
        elif key == glfw.KEY_F:
            if glfw.get_window_monitor(win):
                glfw.set_window_monitor(win, None, 80, 80, size[0], size[1], 0)
            else:
                monitor = glfw.get_primary_monitor()
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(win, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        elif key == glfw.KEY_S:
            from .export import save_png
            name = f"hydrapy_{int(_time.time())}.png"
            save_png(r.read(), name)
            print("saved", name)

    glfw.set_key_callback(window, on_key)
    import os
    last_status, measured_fps, measured_frames = 0., 0., 0
    mtime = os.path.getmtime(watch) if watch else None
    last_check = glfw.get_time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if watch and glfw.get_time() - last_check > 0.25:
            last_check = glfw.get_time()
            try:
                m = os.path.getmtime(watch)
            except OSError:
                m = mtime
            if m != mtime:
                mtime = m
                r = _reload(r, loader, ctx, fps)
                transport = r.patch.transport
        fb = glfw.get_framebuffer_size(window)
        if tuple(fb) != r.size:
            r.resize(fb)
        mx, my = glfw.get_cursor_pos(window)
        ws = glfw.get_window_size(window)
        r.mouse = (mx / max(ws[0], 1), 1.0 - my / max(ws[1], 1))
        if not state["paused"]:
            if transport is not None:
                r.patch.bpm = transport.bpm
                r.step(transport.now())
            else:
                r.step(glfw.get_time() - state["t0"])
        r.blit_to_screen(fb)
        glfw.swap_buffers(window)
        measured_frames += 1
        now = glfw.get_time()
        if now - last_status >= .5:
            measured_fps = measured_frames / max(now - last_status, .001)
            measured_frames, last_status = 0, now
            from .live import resource_status
            devices = resource_status()
            failed = sum(d['state'] == 'error' for d in devices)
            state_name = 'PAUSED' if state['paused'] else 'PLAYING'
            clock_name = type(transport).__name__ if transport is not None else 'Internal'
            problem = ' | EDIT REJECTED: ' + getattr(r, 'reload_error', '')[:120] if getattr(r, 'reload_error', '') else ''
            glfw.set_window_title(window, f'{title} | {state_name} | {r.patch.display.name} | {measured_fps:.1f} fps | '
                                  f'{r.size[0]}x{r.size[1]} | {clock_name} {r.patch.bpm:g} BPM | '
                                  f'{len(devices)} connections, {failed} errors{problem} | 1-4 output, Space pause, S save, F fullscreen')
    from .live import close_resources
    close_resources()
    r.release()
    glfw.terminate()
    if transport is not None:
        transport.close()


# live sets: an idle set must hold this many frames before its variant compiles, so a value crossing
# zero never stalls the show on a shader compile
LIVE_SETTLE = 12


def _reload(old: "GLRenderer", loader, ctx, fps):
    """Build the edited patch; on success swap renderers, carrying each output's last frame over."""
    import traceback
    from .live import ResourceStage
    from .controls import carry_controls
    new = None
    try:
        with ResourceStage() as stage:
            patch = loader()
            carry_controls(old.patch, patch)
            new = GLRenderer(patch, size=old.size, ctx=ctx, fps=fps, compute=old.compute)
            new.settle = old.settle
            for name, o in old.outs.items():
                if name in new.outs:
                    for dest, source in zip(new.outs[name]["ring"], o["ring"]):
                        if dest["tex"].size == source["tex"].size:     # an edit may change an output's scale
                            ctx.copy_framebuffer(dest["fbo"], source["fbo"])
            new.clock.calls, new.clock.updates = old.clock.calls, old.clock.updates
            new.clock.last_time, new.clock.last_key = old.clock.last_time, None
            new.frame, new.mouse = old.frame, old.mouse
            stage.commit()
    except Exception as exc:
        if new is not None:
            new.release()
        old.reload_error = str(exc)
        traceback.print_exc()
        print("hydrapy: edit rejected; previous patch retained. Check device status if reconnection failed.")
        return old
    old.release()
    new.reload_error = ""
    print("hydrapy: reloaded")
    return new
