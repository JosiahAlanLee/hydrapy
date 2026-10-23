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
from typing import Optional

import numpy as np

from .core import Camera, Ctx, DelayRef, Image, Output, Patch, RenderClock, Video, default_patch
from .glsl import VERTEX_DESKTOP, TmpRef, compile_patch, to_desktop
from .signals import as_signal


def _create_context():
    import moderngl
    try:
        return moderngl.create_standalone_context()
    except Exception:
        return moderngl.create_standalone_context(backend="egl")


class GLRenderer:
    """Renders a patch with moderngl. ``ctx`` may be an existing context (e.g. from a window)."""

    def __init__(self, patch: Optional[Patch] = None, size=None, ctx=None, fps: float = 60.0, precision: str = "highp"):
        import moderngl
        self.gl = moderngl
        self.patch = patch or default_patch()
        self.size = tuple(size or self.patch.size)
        self.fps = float(fps)
        self.ctx = ctx or _create_context()
        self.precision = precision
        self.frame = 0
        self.clock = RenderClock(self.fps)
        self._reduce_targets = {}
        self._reduce_program = self._reduce_vao = None
        self.mouse = (0.5, 0.5)
        self.programs = compile_patch(self.patch, precision=precision)
        self.vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self._passes = []     # list of (output_name, target, program, vao, float_uniforms, sampler_uniforms)
        for prog in self.programs:
            for p in prog.passes:
                try:
                    gprog = self.ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=to_desktop(p.source))
                except Exception as e:
                    numbered = "\n".join(f"{i + 1:4d}: {l}" for i, l in enumerate(to_desktop(p.source).splitlines()))
                    raise RuntimeError(f"shader compile error in {prog.output.name}/{p.target}:\n{e}\n{numbered}")
                vao = self.ctx.vertex_array(gprog, [(self.vbo, "2f", "position")])
                self._passes.append((prog.output, p.target, gprog, vao, prog.float_uniforms, prog.sampler_uniforms))
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

    # -- buffers -----------------------------------------------------------
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
        for _, _, gprog, vao, _, _ in self._passes:
            vao.release(); gprog.release()
        for sim in getattr(self, "_sims", []):
            sim.gl_release(self)
        for n in getattr(self, "_ncas", {}).values():
            for gp, vao in n["steps"] + [n["view_prog"]]:
                vao.release(); gp.release()
            for fb in [f for sset in n["sets"] for f in sset] + [n["view"]]:
                fb["fbo"].release(); fb["tex"].release()
        self._ncas = {}
        self._copy_vao.release(); self._copy.release(); self.vbo.release(); self._blank.release()
        self.outs, self.tmps, self.media, self._passes = {}, {}, {}, []

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
        self.outs = {f"o{i}": {"ring": [self._fbo() for _ in range(max(2, depth.get(f"o{i}", 0) + 2))]} for i in range(4)}
        self.tmps = {}
        self._present = self._fbo(dtype="f1")
        for prog in self.programs:
            for p in prog.passes:
                if p.target.startswith("tmp") and p.target not in self.tmps:
                    self.tmps[p.target] = self._fbo()

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
                m["tex"].write(a.tobytes())
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
            return self.tmps[ref.name]["tex"]
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
        from .nca import VIEW_GLSL, shader_header, step_glsl
        head = shader_header(self.precision)

        def prog(src):
            try:
                p = self.ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=to_desktop(head + src))
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
        n = {"steps": [prog(step_glsl(ref.model, k)) for k in range(T)], "view_prog": prog(VIEW_GLSL),
             "sets": [[target(self.gl.NEAREST) for _ in range(T)] for _ in range(2)], "cur": 0,
             "view": target(view_filter)}
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
                    nxt[k]["fbo"].use()
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
        current = None
        for output, target, gprog, vao, floats, samplers in self._passes:
            if output is not current and current is not None:
                self._swap(current)
            current = output
            dest = self.tmps[target] if target.startswith("tmp") else self.outs[output.name]["ring"][-1]
            dest["fbo"].use()
            self._set(gprog, "time", ctx.time)
            self._set(gprog, "resolution", (float(self.size[0]), float(self.size[1])))
            self._set(gprog, "mouse", (self.mouse[0] * self.size[0], self.mouse[1] * self.size[1]))
            for u in floats:
                v = as_signal(u.source)(ctx)
                self._set(gprog, u.name, float(v) if np.isfinite(v) else 0.0)
            for i, u in enumerate(samplers):
                tex = self._bind(u.source, output)
                tex.use(location=i)
                self._set(gprog, u.name, i)
            vao.render(self.gl.TRIANGLES)
        if current is not None:
            self._swap(current)
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
            x0, y0, w, h = pr.region(self.size[0], self.size[1])
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
        self._present["fbo"].use()
        self.ctx.viewport = (0, 0, *self.size)
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", tuple(map(float, self.size)))
        self._set(self._copy, "compositeAlpha", float(self.patch.composite_alpha))
        self._copy_vao.render(self.gl.TRIANGLES)
        data = self._present["fbo"].read(components=3, dtype="f1")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)[::-1].copy()

    def read_rgba(self) -> np.ndarray:
        """Uncomposited straight RGBA, for analysis and transparent-layer tests."""
        fbo = self.outs[self.patch.display.name]["ring"][0]["fbo"]
        return np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(self.size[1], self.size[0], 4)[::-1].copy()

    def blit_to_screen(self, screen_size):
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, screen_size[0], screen_size[1])
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", (float(screen_size[0]), float(screen_size[1])))
        self._set(self._copy, "compositeAlpha", float(self.patch.composite_alpha))
        self._copy_vao.render(self.gl.TRIANGLES)

    def frames(self, n: int, start: float = 0.0):
        for i in range(n):
            self.step(start + i / self.fps)
            yield self.read()


def iter_frames_gl(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                   start: float = 0.0, warmup: int = 0):
    r = GLRenderer(patch, size=size, fps=fps)
    try:
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
                     start: float = 0.0, warmup: int = 0) -> list[np.ndarray]:
    return list(iter_frames_gl(patch, seconds, fps, size, start, warmup))


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
        elif key == glfw.KEY_S:
            from .export import save_png
            name = f"hydrapy_{int(_time.time())}.png"
            save_png(r.read(), name)
            print("saved", name)

    glfw.set_key_callback(window, on_key)
    import os
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
    glfw.terminate()
    if transport is not None:
        transport.close()


def _reload(old: "GLRenderer", loader, ctx, fps):
    """Build the edited patch; on success swap renderers, carrying each output's last frame over."""
    import traceback
    from .live import close_resources
    close_resources()
    try:
        patch = loader()
        new = GLRenderer(patch, size=old.size, ctx=ctx, fps=fps)
    except Exception:
        traceback.print_exc()
        print("hydrapy: the edited patch failed; still running the previous one (its devices were closed)")
        return old
    for name, o in old.outs.items():
        if name in new.outs:
            ctx.copy_framebuffer(new.outs[name]["ring"][0]["fbo"], o["ring"][0]["fbo"])
    new.clock.calls, new.frame, new.mouse = old.clock.calls, old.frame, old.mouse
    old.release()
    print("hydrapy: reloaded")
    return new
