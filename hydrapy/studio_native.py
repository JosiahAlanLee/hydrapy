"""Studio's native Program window: the frame shown straight from the GPU.

The browser's monitors read each frame back, encode a JPEG, fetch it over HTTP and
decode it. This window draws the texture the frame was rendered into, at the
display's refresh, with nothing in between. Studio's render thread owns the window
and the GL context; GLFW allows that on X11 (macOS would need the main thread).

Pacing is late-latched: after each flip the next frame starts as late as the
recent render times allow, so a cue or edit that arrives up to that moment is on
screen at the next refresh.
"""
from __future__ import annotations

import time
from collections import deque


class NativeOutput:
    """A fullscreen (or windowed) Program output on one monitor.

    ``where``: a monitor name (``DP-5``) or index (``1``) for fullscreen there, or
    ``window`` for a plain window on the desktop.
    """

    def __init__(self, where='window', size=(1280, 720)):
        import glfw
        import moderngl
        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError('GLFW could not start; a native output needs an X display')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)  # stay up when focus goes to the editor
        monitor = None if where == 'window' else self.monitor(where)
        if monitor is not None:
            mode = glfw.get_video_mode(monitor)
            size = (mode.size.width, mode.size.height)
            glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
        self.window = glfw.create_window(size[0], size[1], 'HydraPy Program', monitor, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not open the native output window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context(require=330)
        mode = glfw.get_video_mode(monitor or glfw.get_primary_monitor())
        self.refresh = float(mode.refresh_rate or 60)
        self.where = where
        self.size = tuple(glfw.get_framebuffer_size(self.window))
        self.flips = deque(maxlen=121)
        self.missed = 0
        self._texture = None

    def monitor(self, where):
        glfw = self.glfw
        monitors = glfw.get_monitors()
        names = [glfw.get_monitor_name(m).decode(errors='replace') for m in monitors]
        if str(where).isdigit() and int(where) < len(monitors):
            return monitors[int(where)]
        if where in names:
            return monitors[names.index(where)]
        raise ValueError(f'No monitor {where!r}; choose one of {names} (or an index, or "window")')

    @property
    def period(self):
        """Seconds per refresh: measured from recent flips once there are enough, else the mode's rate."""
        if len(self.flips) > 30:
            gaps = sorted(b - a for a, b in zip(self.flips, list(self.flips)[1:]))
            nominal = 1 / self.refresh
            steady = [g for g in gaps if .5 * nominal < g < 1.5 * nominal]
            if len(steady) > 20:
                return steady[len(steady) // 2]
        return 1 / self.refresh

    def present(self, renderer=None, frame=None, *, blackout=False):
        """Draw the displayed output (or black), swap, and wait for the flip.

        A GL renderer's texture is drawn directly; a CPU renderer's ``frame`` (RGB,
        row 0 = top) is uploaded first. Returns (ready, flipped, missed): when the GPU had
        finished the frame, when it flipped (monotonic), and whether a refresh was skipped.
        """
        glfw, ctx = self.glfw, self.ctx
        w, h = glfw.get_framebuffer_size(self.window)
        self.size = (w, h)
        if blackout or (renderer is None and frame is None):
            ctx.screen.use()
            ctx.viewport = (0, 0, w, h)
            ctx.clear(0., 0., 0., 1.)
        elif frame is None:
            renderer.blit_to_screen((w, h))
        else:
            self._draw_array(frame, (w, h))
        # Finish before the swap so the pacer sees the GPU's share of the frame, not only the CPU's.
        ctx.finish()
        ready = time.monotonic()
        glfw.swap_buffers(self.window)
        # With vsync, finishing after the swap returns once the buffer has flipped.
        ctx.finish()
        flipped = time.monotonic()
        missed = bool(self.flips) and flipped - self.flips[-1] > 1.5 / self.refresh
        self.missed += missed
        self.flips.append(flipped)
        glfw.poll_events()
        if glfw.window_should_close(self.window):
            glfw.set_window_should_close(self.window, False)  # Studio decides when outputs close
        return ready, flipped, missed

    def _draw_array(self, frame, size):
        """A CPU renderer's RGB frame, uploaded and scaled to the window."""
        import numpy as np
        ctx = self.ctx
        h, w = frame.shape[:2]
        if self._texture is None or self._texture.size != (w, h):
            if self._texture is not None:
                self._texture.release()
            self._texture = ctx.texture((w, h), 3)
        if not hasattr(self, '_vao'):
            self._prog = ctx.program(
                vertex_shader='#version 330\nin vec2 pos; out vec2 uv;\n'
                              'void main(){ uv = pos * .5 + .5; gl_Position = vec4(pos, 0., 1.); }',
                fragment_shader='#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 color;\n'
                                'void main(){ color = vec4(texture(tex, vec2(uv.x, 1. - uv.y)).rgb, 1.); }')
            self._vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
            self._vao = ctx.vertex_array(self._prog, [(self._vbo, '2f', 'pos')])
        self._texture.write(np.ascontiguousarray(frame).tobytes())
        ctx.screen.use()
        ctx.viewport = (0, 0, *size)
        self._texture.use(location=0)
        self._vao.render(ctx.TRIANGLE_STRIP)

    def describe(self):
        period = self.period
        return dict(where=self.where, size=self.size, refresh=self.refresh,
                    period_ms=period * 1000, missed=self.missed)

    def close(self):
        if self._texture is not None:
            self._texture.release()
        if hasattr(self, '_vao'):
            self._vao.release(); self._vbo.release(); self._prog.release()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


class Pacer:
    """When to start the next frame so it is finished just before the flip after it.

    The budget is the 99th percentile of recent frames' work, from the planned start to
    the GPU finishing (commands that arrived at the last moment, render, blit), plus a
    margin, kept between 2 ms and one refresh less a millisecond. A percentile, not the
    maximum: one stall (a garbage collection) would otherwise add its length to every
    frame's latency for seconds. A skipped refresh widens the margin by a millisecond; a
    long run without one narrows it again, slowly.
    """

    def __init__(self, margin=.0015, window=240, floor=.0015, quantile=.99):
        self.quantile = quantile
        self.margin, self.floor = margin, floor
        self.work = deque(maxlen=window)
        self.hits = 0

    def record(self, seconds, missed=False):
        self.work.append(seconds)
        if missed:
            self.margin, self.hits = min(self.margin + .001, .008), 0
        else:
            self.hits += 1
            if self.hits >= 300 and self.margin > self.floor:
                self.margin, self.hits = max(self.floor, self.margin - .00025), 0

    def budget(self, period):
        if len(self.work) < 30:
            work = max(self.work, default=.004)
        else:
            ordered = sorted(self.work)
            work = ordered[min(len(ordered) - 1, int(self.quantile * len(ordered)))]
        return min(max(work + self.margin, .002), max(period - .001, .002))

    def start_at(self, flipped, period):
        return flipped + period - self.budget(period)
