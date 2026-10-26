"""Studio's native Program output: the frame shown straight from the GPU.

The browser's monitors read each frame back, encode a JPEG, fetch it over HTTP and
decode it. This window draws the texture the frame was rendered into, at the
display's refresh, with nothing in between.

Studio renders in this window's GL context from the start, with the window hidden,
so showing it fullscreen (the tracker's Display menu) moves nothing between
contexts: feedback and compiled shaders carry on. Studio's render thread owns the
window and the context; GLFW allows that on X11, so this is Linux-only (macOS would
need the main thread).

Pacing is late-latched: after each flip the next frame starts as late as the
recent render times allow, so a cue or edit that arrives up to that moment is on
screen at the next refresh.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque


def unavailable():
    """Why a native output cannot run here, or None if it can."""
    if not sys.platform.startswith('linux'):
        return 'Fullscreen output is Linux-only for now.'
    if not os.environ.get('DISPLAY'):
        return 'Fullscreen output needs an X display; Studio was started without DISPLAY.'
    try:
        import glfw  # noqa: F401
    except ImportError:
        return 'Fullscreen output needs glfw: pip install glfw'
    return None


class NativeOutput:
    """Studio's GL context, in a window that can be shown fullscreen on a monitor.

    Created hidden. ``show(where)``: a monitor name (``DP-5``) or index (``1``) for
    fullscreen there, or ``window`` for a plain window on the desktop; ``hide()``.
    """

    def __init__(self):
        import glfw
        import moderngl
        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError('GLFW could not start; a native output needs an X display')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, False)
        # The editor keeps the keyboard: showing the output must not take focus from the browser,
        # and the output stays up when focus is elsewhere.
        glfw.window_hint(glfw.FOCUSED, False)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)
        self.window = glfw.create_window(640, 360, 'HydraPy Program', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not create the native output window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context(require=330)
        self._no_keyboard_focus()
        self.where = None
        self.refresh = 60.
        self.size = tuple(glfw.get_framebuffer_size(self.window))
        self.flips = deque(maxlen=121)
        self.missed = 0
        self._texture = None

    def _no_keyboard_focus(self):
        """Tell the window manager this window takes no keyboard input (ICCCM WM_HINTS input=False).

        Going fullscreen otherwise moves focus to it (GNOME does; GLFW's FOCUS_ON_SHOW hint does not
        stop it), and the next keystroke meant for the editor would land on a window that ignores keys.
        Set while the window is unmapped, so the window manager reads it when it is shown.
        """
        import ctypes
        import ctypes.util

        class XWMHints(ctypes.Structure):
            _fields_ = [('flags', ctypes.c_long), ('input', ctypes.c_int), ('initial_state', ctypes.c_int),
                        ('icon_pixmap', ctypes.c_ulong), ('icon_window', ctypes.c_ulong), ('icon_x', ctypes.c_int),
                        ('icon_y', ctypes.c_int), ('icon_mask', ctypes.c_ulong), ('window_group', ctypes.c_ulong)]
        try:
            x11 = ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11') or 'libX11.so.6')
            display = ctypes.c_void_p(self.glfw.get_x11_display())
            window = ctypes.c_ulong(self.glfw.get_x11_window(self.window))
            input_hint, state_hint, normal_state = 1, 2, 1
            hints = XWMHints(flags=input_hint | state_hint, input=0, initial_state=normal_state)
            x11.XSetWMHints.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XWMHints)]
            x11.XSetWMHints(display, window, ctypes.byref(hints))
            x11.XFlush.argtypes = [ctypes.c_void_p]
            x11.XFlush(display)
        except Exception:
            pass  # not X11 (or no libX11): the output may take focus when shown

    @property
    def active(self):
        return self.where is not None

    def monitors(self, fresh=False):
        """Connected monitors, re-read at most every two seconds (status asks every frame)."""
        now = time.monotonic()
        if fresh or not hasattr(self, '_monitors') or now - self._monitors[0] > 2.:
            glfw = self.glfw
            primary = glfw.get_monitor_name(glfw.get_primary_monitor())
            out = []
            for i, m in enumerate(glfw.get_monitors()):
                mode = glfw.get_video_mode(m)
                out.append(dict(name=glfw.get_monitor_name(m).decode(errors='replace'), index=i,
                                width=mode.size.width, height=mode.size.height, refresh=mode.refresh_rate,
                                primary=glfw.get_monitor_name(m) == primary))
            self._monitors = (now, out)
        return self._monitors[1]

    def default_monitor(self):
        """Another screen than the primary (the editor's), else a window."""
        others = [m['name'] for m in self.monitors(fresh=True) if not m['primary']]
        return others[0] if others else 'window'

    def monitor(self, where):
        glfw = self.glfw
        monitors = glfw.get_monitors()
        names = [glfw.get_monitor_name(m).decode(errors='replace') for m in monitors]
        if str(where).isdigit() and int(where) < len(monitors):
            return monitors[int(where)]
        if where in names:
            return monitors[names.index(where)]
        raise ValueError(f'No monitor {where!r}; choose one of {names} (or an index, or "window")')

    def target_size(self, where):
        """The frame size that fills ``where``: the monitor's mode, or 1280x720 for a window."""
        if where == 'window':
            return (1280, 720)
        mode = self.glfw.get_video_mode(self.monitor(where))
        return (mode.size.width, mode.size.height)

    def show(self, where):
        glfw = self.glfw
        if where == 'window':
            glfw.set_window_monitor(self.window, None, 80, 80, 1280, 720, 0)
            mode = glfw.get_video_mode(glfw.get_primary_monitor())
        else:
            monitor = self.monitor(where)
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        glfw.show_window(self.window)
        self.where, self.refresh = where, float(mode.refresh_rate or 60)
        # The window manager resizes it asynchronously; report the size it settles at.
        want = (1280, 720) if where == 'window' else (mode.size.width, mode.size.height)
        deadline = time.monotonic() + .5
        while tuple(glfw.get_framebuffer_size(self.window)) != want and time.monotonic() < deadline:
            glfw.wait_events_timeout(.01)
        self.size = tuple(glfw.get_framebuffer_size(self.window))
        self.flips.clear()
        self.missed = 0

    def hide(self):
        glfw = self.glfw
        glfw.set_window_monitor(self.window, None, 80, 80, 640, 360, 0)
        glfw.hide_window(self.window)
        glfw.poll_events()
        self.where = None

    def poll(self):
        """Handle window events; True if the user closed the output's window."""
        glfw = self.glfw
        glfw.poll_events()
        if glfw.window_should_close(self.window):
            glfw.set_window_should_close(self.window, False)  # Studio decides when the context goes
            return True
        return False

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
        monitors = self.monitors()
        others = [m['name'] for m in monitors if not m['primary']]
        return dict(available=True, supported=True, active=self.active, where=self.where, size=self.size,
                    refresh=self.refresh, period_ms=self.period * 1000, missed=self.missed, monitors=monitors,
                    suggested=others[0] if others else 'window')

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
