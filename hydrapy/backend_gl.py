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

from .core import Camera, Ctx, DelayRef, Image, Output, Patch, Video, default_patch
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
        self.frame = 0
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
            fragment_shader="#version 330 core\nuniform sampler2D tex; uniform vec2 resolution; out vec4 f;\n"
                            "void main(){ f = texture(tex, gl_FragCoord.xy / resolution); }")
        self._copy_vao = self.ctx.vertex_array(self._copy, [(self.vbo, "2f", "position")])
        self._alloc()
        self.media: dict[str, object] = {}
        self._blank = self.ctx.texture((1, 1), 4, data=bytes([0, 0, 0, 255]))

    # -- buffers -----------------------------------------------------------
    def _fbo(self):
        tex = self.ctx.texture(self.size, 4, dtype="f2")
        tex.filter = (self.gl.LINEAR, self.gl.LINEAR)
        tex.repeat_x = tex.repeat_y = True
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        return {"tex": tex, "fbo": fbo}

    def release(self):
        """Free every GL object this renderer created (the context itself is kept)."""
        for o in getattr(self, "outs", {}).values():
            for fb in o["ring"]:
                fb["fbo"].release(); fb["tex"].release()
        for fb in getattr(self, "tmps", {}).values():
            fb["fbo"].release(); fb["tex"].release()
        for m in getattr(self, "media", {}).values():
            if m.get("tex") is not None:
                m["tex"].release()
            for t in m.get("texs", {}).values():
                t.release()
        for _, _, gprog, vao, _, _ in self._passes:
            vao.release(); gprog.release()
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
        depth = {}
        for prog in self.programs:
            for k, v in prog.history.items():
                depth[k] = max(depth.get(k, 0), v)
        self.outs = {f"o{i}": {"ring": [self._fbo() for _ in range(max(2, depth.get(f"o{i}", 0) + 2))]} for i in range(4)}
        self.tmps = {}
        for prog in self.programs:
            for p in prog.passes:
                if p.target.startswith("tmp") and p.target not in self.tmps:
                    self.tmps[p.target] = self._fbo()

    def resize(self, size):
        if tuple(size) == self.size:
            return
        self.size = tuple(size)
        self._alloc()

    # -- media -------------------------------------------------------------
    def _media_texture(self, ref):
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
        return self._media_texture(ref)

    # -- frames --------------------------------------------------------------
    def step(self, t: Optional[float] = None):
        ctx = Ctx(width=self.size[0], height=self.size[1], bpm=self.patch.bpm, mouse=self.mouse, fps=self.fps)
        ctx.time = (self.frame / self.fps if t is None else t) * self.patch.speed
        ctx.frame = self.frame
        self.ctx_time = ctx.time
        if self.patch.frame_rate:
            ctx.time = math.floor(ctx.time * self.patch.frame_rate) / self.patch.frame_rate
            self.ctx_time = ctx.time
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
        self.frame += 1
        self._update_probes()
        from .live import run_hooks
        run_hooks(ctx, self)

    def _update_probes(self):
        for pr in self.patch.probes:
            x0, y0, w, h = pr.region(self.size[0], self.size[1])
            fbo = self.outs[pr.output.name]["ring"][0]["fbo"]
            data = fbo.read(viewport=(x0, y0, w, h), components=4, dtype="f4")
            pr.update(np.frombuffer(data, dtype=np.float32).reshape(h, w, 4), self.ctx_time)

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
        """The displayed output as uint8 RGB (H, W, 3), row 0 = top."""
        fbo = self.outs[self.patch.display.name]["ring"][0]["fbo"]
        data = fbo.read(components=3, dtype="f1")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)[::-1].copy()

    def blit_to_screen(self, screen_size):
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, screen_size[0], screen_size[1])
        self.displayed_texture().use(location=0)
        self._set(self._copy, "tex", 0)
        self._set(self._copy, "resolution", (float(screen_size[0]), float(screen_size[1])))
        self._copy_vao.render(self.gl.TRIANGLES)

    def frames(self, n: int, start: float = 0.0):
        for i in range(n):
            self.step(start + i / self.fps)
            yield self.read()


def render_frames_gl(patch: Optional[Patch] = None, seconds: float = 4.0, fps: float = 30.0, size=None,
                     start: float = 0.0, warmup: int = 0) -> list[np.ndarray]:
    r = GLRenderer(patch, size=size, fps=fps)
    for i in range(warmup):
        r.step(start - (warmup - i) / fps)
    for pr in r.patch.probes:
        if hasattr(pr, "clear"):
            pr.clear()                      # a Soundtrack starts at `start`, not at the warm-up
    return list(r.frames(int(round(seconds * fps)), start=start))


def live(patch: Optional[Patch] = None, size=(1280, 720), fps: float = 60.0, title: str = "hydrapy",
         fullscreen: bool = False):
    """Open a window and run the patch in real time (needs glfw). Esc quits, S saves a PNG, space pauses."""
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
    while not glfw.window_should_close(window):
        glfw.poll_events()
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
