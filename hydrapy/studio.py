"""Local, code-first workspace. All rendering and patch execution stay in one thread."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .controls import apply_controls, carry_controls
from .inspect import catalog, capabilities
from .studio_preview import Preview
from .studio_tracker import Tracker, number


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def project_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Choose a file inside the workspace root')
    return path


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


class Recorder:
    """Record wall-clock time even when preview rendering drops frames."""
    def __init__(self, path, fps):
        self.path, self.fps, self.start = path, fps, time.monotonic()
        self.frames, self.error = 0, ''
        self.q = queue.Queue(180)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        from .export import save_mp4
        def frames():
            while True:
                frame = self.q.get()
                if frame is None:
                    return
                yield frame
        try:
            save_mp4(frames(), str(self.path), fps=self.fps, encoder='auto')
        except Exception as e:
            self.error = str(e)

    def push(self, frame):
        if self.error:
            raise RuntimeError(self.error)
        target = int((time.monotonic() - self.start) * self.fps) + 1
        for _ in range(target - self.frames):
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                raise RuntimeError('Recording encoder fell behind. Stop recording and use offline export.')
            self.frames += 1

    def close(self):
        if self.thread.is_alive():
            self.q.put(None, timeout=10)
            self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Recording is still finishing')
        if self.error:
            raise RuntimeError(self.error)


class Session:
    def __init__(self, root, patch=None, backend='auto', size=(640, 360), fps=30., cv=None):
        self.root, self.backend, self.size, self.fps = Path(root).resolve(), backend, tuple(size), float(fps)
        if not self.root.is_dir() or backend not in ('auto', 'cpu', 'gl'):
            raise ValueError('Choose an existing root and auto/cpu/gl backend')
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 120:
            raise ValueError('Preview fps must be between 1 and 120')
        if len(self.size) != 2 or any(not isinstance(x, int) or not 16 <= x <= 4096 for x in self.size):
            raise ValueError('Preview dimensions must be 16–4096 pixels')
        self.commands, self.lock = queue.Queue(), threading.Lock()
        self.running, self.paused, self.armed = True, False, False
        self.renderer, self.path, self.source = None, None, ''
        self.error, self.notice, self.backend_note = '', 'Choose a patch to begin.', ''
        self.clock_time, self.actual_fps = 0., 0.
        self.last_wall, self.last_watch, self.stamp = time.monotonic(), 0., None
        self.state = {}
        self.preview = Preview()
        self.aux_preview = Preview()
        self.aux_preview.reset('o1')
        self.tracker = Tracker()
        self.tracker_state = {}
        self.tracker_blackout = False
        # Optional tracker CV output (a TrackerCV); opened by the caller so a
        # device error is reported before Studio starts.
        self.cv = cv
        self.preview_profile, self.gpu_query = False, None
        self.render_metrics, self.preview_error = {}, ''
        self.preview_dirty = True
        self.recorder, self.record_path, self.last_frame = None, None, None
        self.jobs, self.job_processes = {}, {}
        self.recent = []
        self.thread = threading.Thread(target=self._run, daemon=True, name='hydrapy-render')
        self.thread.start()
        if patch:
            try:
                self.call('load', path=patch)
            except Exception:
                self.close()
                raise

    def call(self, action, **data):
        if not self.thread.is_alive():
            raise RuntimeError('The preview thread has stopped; restart Studio')
        result = queue.Queue(1)
        self.commands.put((action, data, result))
        response = result.get(timeout=90)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        if self.running:
            try:
                self.call('close')
            finally:
                self.running = False
                self.thread.join(timeout=10)
        if self.cv is not None:
            self.cv.close()
        for proc in self.job_processes.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def _make_renderer(self, patch):
        self.backend_note = ''
        if self.backend in ('auto', 'gl'):
            from .backend_gl import LIVE_SETTLE, GLRenderer
            ctx = getattr(self.renderer, 'ctx', None)
            if self.backend == 'auto' and any(x['status'] == 'unavailable' for x in capabilities(patch, 'gl')['findings']):
                self.backend_note = 'This patch contains CPU-only operations.'
            else:
                # Only context creation failure selects CPU. Shader errors must remain visible.
                if ctx is None:
                    from .backend_gl import _create_context
                    try:
                        ctx = _create_context()
                    except Exception as exc:
                        if self.backend == 'gl':
                            raise
                        self.backend_note = f'GL unavailable; using CPU: {exc}'
                if ctx is not None:
                    r = GLRenderer(patch, size=self.size, fps=self.fps, ctx=ctx)
                    r.settle = LIVE_SETTLE
                    return r
        from .backend_np import Renderer
        return Renderer(patch, size=self.size, fps=self.fps)

    def _load(self, name, *, preserve=True, armed=None):
        if self.recorder:
            raise ValueError('Stop recording before loading or reconnecting a patch.')
        from .cli import load_patch
        from .live import ResourceStage
        path = project_path(self.root, name)
        if path.suffix not in ('.py', '.js') or not path.is_file():
            raise ValueError('Choose a Python patch or Hydra sketch')
        source = path.read_text(encoding='utf-8')
        self.stamp = (path, path.stat().st_mtime_ns)
        old, new = self.renderer, None
        same = preserve and path == self.path
        requested_arm = self.armed if armed is None else bool(armed)
        try:
            with ResourceStage() as stage:
                patch = load_patch(str(path), strict=False)
                if self.tracker.enabled and requested_arm and patch.transport:
                    raise ValueError('Disable tracker automation before enabling an external clock')
                if same and old:
                    carry_controls(old.patch, patch)
                new = self._make_renderer(patch)
                if hasattr(new, '_compile'):
                    new._compile()
                if old and same:
                    if hasattr(old, 'outs') and hasattr(new, 'outs'):
                        for name, output in old.outs.items():
                            for dst, src in zip(new.outs[name]['ring'], output['ring']):
                                if dst['tex'].size == src['tex'].size:
                                    new.ctx.copy_framebuffer(dst['fbo'], src['fbo'])
                    elif hasattr(old, 'env') and hasattr(new, 'env'):
                        new.env.buffers = dict(old.env.buffers)
                        new.env.rings = {k: list(v) for k, v in old.env.rings.items()}
                    new.clock.calls, new.clock.updates = old.clock.calls, old.clock.updates
                    new.clock.last_time = old.clock.last_time
                    new.frame = old.frame
                stage.commit(armed=requested_arm)
        except Exception:
            if new is not None and hasattr(new, 'release'):
                new.release()
            raise
        if old and hasattr(old, 'release'):
            old.release()
        self.renderer, self.path, self.source, self.armed = new, path, source, requested_arm
        # A patch's declared display remains authoritative when it is loaded.
        self.tracker.sync_program(patch.display.name, self.clock_time)
        self.gpu_query = None
        self.preview.reset(patch.display.name)
        self.aux_preview.reset(self.tracker.document['modes']['aux'])
        self.preview_dirty = True
        self.last_frame = None
        if not same:
            self.clock_time = 0.
            self.tracker.enabled = False
            self.tracker.seek(0., self.clock_time)
        self.last_wall = time.monotonic()
        self.error, self.notice = '', 'Loaded. Connections enabled.' if requested_arm else 'Loaded. Connections are disconnected.'
        relative = str(path.relative_to(self.root))
        self.recent = [relative] + [p for p in self.recent if p != relative][:9]
        return dict(path=relative, source=source, version=digest(source))

    def _handle(self, action, data):
        from .live import resource_status
        if action == 'close':
            self.running = False
            return {}
        if action == 'load':
            return self._load(data['path'])
        if action == 'source':
            p = project_path(self.root, data['path'])
            if p.suffix not in ('.py', '.js'):
                raise ValueError('Only patch source can be opened in the editor')
            text = p.read_text(encoding='utf-8')
            return dict(source=text, version=digest(text), path=str(p.relative_to(self.root)))
        if action == 'save':
            p = project_path(self.root, data['path'])
            if p.suffix != '.py':
                raise ValueError('Save Python source to a .py file')
            if p.exists() and digest(p.read_text(encoding='utf-8')) != data.get('version'):
                raise ValueError('The file changed outside this editor. Reopen it before saving.')
            atomic_text(p, data['source'])
            self.stamp = (p, p.stat().st_mtime_ns)
            if data.get('run', True):
                return self._load(str(p.relative_to(self.root)))
            return dict(path=str(p.relative_to(self.root)), source=data['source'], version=digest(data['source']))
        if action == 'devices':
            info = {}
            try:
                import sounddevice as sd
                info['audio'] = [dict(index=i, name=d['name'], inputs=d['max_input_channels'], outputs=d['max_output_channels'])
                                 for i, d in enumerate(sd.query_devices())]
            except Exception as e:
                info['audio_error'] = str(e)
            try:
                import mido
                info['midi'] = mido.get_input_names()
            except Exception as e:
                info['midi_error'] = str(e)
            return info
        if not self.renderer:
            raise ValueError('Load a patch first')
        patch = self.renderer.patch
        if action == 'tracker_score':
            self.tracker.replace(data['document'], self.clock_time, data['revision'])
            modes = self.tracker.document['modes']
            if patch.display.name != modes['program']:
                patch.render(patch.outputs[int(modes['program'][1])])
                self.preview.reset(patch.display.name)
                self.preview_dirty = True
            if self.aux_preview.output != modes['aux']:
                self.aux_preview.reset(modes['aux'])
            return self.tracker.snapshot(self.clock_time, not self.paused)
        elif action == 'tracker_transport':
            enabled = data.get('enabled', self.tracker.enabled)
            playing = data.get('playing', not self.paused)
            beat = number(data.get('beat', self.tracker.beat(self.clock_time)), 0, 64, 'Beat')
            blackout = data.get('blackout', self.tracker_blackout)
            if any(type(v) is not bool for v in (enabled, playing, blackout)):
                raise ValueError('Transport states must be boolean')
            if enabled and self.armed and patch.transport:
                raise ValueError('Disconnect the external clock before enabling tracker automation')
            self.tracker.seek(beat, self.clock_time)
            self.tracker.enabled = enabled
            self.tracker_blackout = blackout
            if self.paused != (not playing):
                self.last_wall = time.monotonic()
            self.paused = not playing
            return dict(self.tracker.snapshot(self.clock_time, not self.paused), blackout=self.tracker_blackout)
        elif action == 'tracker_save':
            revision = data.get('revision')
            if type(revision) is not int or not self.tracker.edit_revision <= revision <= self.tracker.revision:
                raise ValueError('Score changed in another window. Reload before saving.')
            path = self.root / '.hydrapy' / 'tracker.json'
            atomic_text(path, json.dumps(self.tracker.document, indent=2))
            return dict(file=str(path.relative_to(self.root)))
        elif action == 'pause':
            self.paused = bool(data.get('paused', not self.paused))
            self.last_wall = time.monotonic()
        elif action == 'output':
            index = int(data['index'])
            if not 0 <= index < 4:
                raise ValueError('output must be 0–3')
            patch.render(patch.outputs[index])
            self.tracker.sync_program(patch.display.name, self.clock_time)
            self.preview.reset(patch.display.name)
            self.preview_dirty = True
            self._pictures(force=True)
        elif action == 'preview_profile':
            self.preview_profile = bool(data.get('enabled', False))
            self.gpu_query = None
        elif action == 'mouse':
            x, y = float(data['x']), float(data['y'])
            if not all(math.isfinite(v) for v in (x, y)):
                raise ValueError('mouse coordinates must be finite')
            xy = (min(1., max(0., x)), min(1., max(0., y)))
            if hasattr(self.renderer, 'env'):
                self.renderer.env.ctx.mouse = xy
            else:
                self.renderer.mouse = xy
        elif action == 'controls':
            apply_controls(patch, data['values'])
        elif action == 'reset_controls':
            for p in patch.controls.values():
                p.reset()
        elif action == 'restart':
            self._load(str(self.path.relative_to(self.root)), preserve=False)
        elif action == 'connections':
            if data['armed'] and self.tracker.enabled and patch.transport:
                raise ValueError('Disable tracker automation before enabling the external clock')
            self._load(str(self.path.relative_to(self.root)), armed=bool(data['armed']))
        elif action == 'check':
            return capabilities(patch, data.get('target', 'html'))
        elif action in ('preset_save', 'preset_load', 'presets'):
            preset_dir = self.root / '.hydrapy' / 'presets' / digest(str(self.path))[:12]
            if action == 'presets':
                return dict(names=sorted(p.stem for p in preset_dir.glob('*.json')))
            name = data.get('name', '').strip()
            if not name or len(name) > 80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -' for c in name):
                raise ValueError('Preset names may contain letters, numbers, spaces, hyphens and underscores')
            file = preset_dir / (name + '.json')
            if action == 'preset_save':
                atomic_text(file, json.dumps({k: p.snapshot() for k, p in patch.controls.items()}, indent=2))
            else:
                apply_controls(patch, json.loads(file.read_text()))
        elif action == 'snapshot':
            # A snapshot is a fresh lossless read, independent of JPEG delivery.
            self.last_frame = self._read_frame()
            from .export import save_png
            path = self._new_output('png')
            save_png(self.last_frame, str(path))
            return dict(file=str(path.relative_to(self.root)))
        elif action == 'record':
            if self.recorder:
                recorder, self.recorder = self.recorder, None
                recorder.close()
                self.notice = 'Video recording saved.'
                return dict(file=str(recorder.path.relative_to(self.root)))
            self.record_path = self._new_output('mp4')
            self.recorder = Recorder(self.record_path, self.fps)
            self.notice = 'Recording video at wall-clock speed.'
        elif action == 'export':
            return self._export(data)
        else:
            raise ValueError(f'Unknown action: {action}')
        return {'ok': True}

    def _new_output(self, extension):
        folder = self.root / '.hydrapy' / 'exports'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (self.path.stem + '-' + time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2) + '.' + extension)

    def _export(self, data):
        fmt = data.get('format', 'mp4')
        if fmt not in ('mp4', 'gif', 'png', 'html'):
            raise ValueError('format must be mp4/gif/png/html')
        if self.tracker.enabled and fmt == 'html':
            raise ValueError('Disable tracker automation for a static HTML export; use video for the score')
        backend = data.get('backend', 'gl' if hasattr(self.renderer, 'ctx') else 'cpu')
        if backend not in ('cpu', 'gl'):
            raise ValueError('backend must be cpu or gl')
        report = capabilities(self.renderer.patch, 'html' if fmt == 'html' else backend)
        if not report['ready']:
            raise ValueError('Export needs changes: ' + '; '.join(x['path'] + ': ' + x['message'] for x in report['findings'] if x['status'] in ('unavailable', 'requires baking')))
        seconds, fps = float(data.get('seconds', 10)), float(data.get('fps', 30))
        size = tuple(int(x) for x in data.get('size', [1280, 720]))
        if not math.isfinite(seconds) or not .01 <= seconds <= 14400 or not math.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError('Use 0.01–14400 seconds and 1–120 fps')
        if len(size) != 2 or not all(16 <= x <= 4096 for x in size):
            raise ValueError('Dimensions must be 16–4096 pixels')
        if fmt == 'gif' and seconds * fps * size[0] * size[1] > 200_000_000:
            raise ValueError('This GIF would use too much memory; reduce duration/size or export MP4')
        audio = data.get('audio', '').strip()
        if audio:
            audio = str(project_path(self.root, audio))
            if not Path(audio).is_file():
                raise ValueError('Audio file not found')
        job_id = secrets.token_hex(8)
        folder = self.root / '.hydrapy' / 'jobs' / job_id
        folder.mkdir(parents=True)
        output = self._new_output(fmt)
        spec = dict(root=str(self.root), patch=str(self.path), source=self.source,
                    params={k: p.snapshot() for k, p in self.renderer.patch.controls.items()},
                    display=self.renderer.patch.display.index, output=str(output), format=fmt,
                    backend=backend, seconds=seconds, fps=fps, size=size, audio=audio,
                    time=self.clock_time, encoder=data.get('encoder', 'auto'))
        if self.tracker.enabled:
            spec['tracker'] = self.tracker.snapshot(self.clock_time, not self.paused)
        atomic_text(folder / 'job.json', json.dumps(spec))
        atomic_text(folder / 'status.json', json.dumps(dict(state='queued', progress=0)))
        with (folder / 'log.txt').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'hydrapy.studio_export', str(folder / 'job.json')],
                                    cwd=self.root, stdout=log, stderr=log)
        self.jobs[job_id] = dict(id=job_id, file=str(output.relative_to(self.root)), format=fmt, warnings=report['findings'])
        self.job_processes[job_id] = proc
        return self.jobs[job_id]

    def _read_frame(self):
        from .backend_np import to_uint8
        r = self.renderer
        return r.read() if hasattr(r, 'read') else to_uint8(r.displayed(), r.patch.composite_alpha)

    def _pictures(self, *, force=False, recorded=None, query=None):
        r, now = self.renderer, time.monotonic()
        main, thumbnail = self.preview.plan(now, changed=not self.paused or self.preview_dirty)
        chosen = r.patch.display
        frame = recorded[0] if recorded is not None else None
        main_frame = frame
        metrics = dict(self.render_metrics)
        if main or force:
            started = time.monotonic()
            if frame is None:
                frame = self._read_frame()
            metrics.update(readback_ms=recorded[1] if recorded is not None else (time.monotonic() - started) * 1000,
                           captured=time.monotonic())
            # read() has already completed the preceding GPU commands. Do not
            # query results before readback or add a finish() to the render loop.
            if query is not None:
                metrics['gpu_ms'] = query.elapsed / 1_000_000
            self.last_frame = frame
            main_frame = frame
            cache = None
            if self.path and self.clock_time > 1:
                cache = self.root / '.hydrapy' / 'previews' / (digest(str(self.path.relative_to(self.root)))[:16] + '.jpg')
            self.preview.submit(chosen.name, frame, metrics, cache=cache)
            self.preview_dirty = False
        if thumbnail is not None:
            # Restore selection even when readback fails. Thumbnail delivery must
            # never change the patch's recording/export/display output.
            try:
                if thumbnail != chosen.name or frame is None:
                    r.patch.display = r.patch.outputs[int(thumbnail[1])]
                    frame = self._read_frame()
                self.preview.submit(thumbnail, frame, {'captured': time.monotonic()}, thumbnail=True)
            finally:
                r.patch.display = chosen
        aux, _ = self.aux_preview.plan(now, changed=not self.paused or self.aux_preview.latest is None)
        if aux:
            try:
                output = self.aux_preview.output
                r.patch.display = r.patch.outputs[int(output[1])]
                started = time.monotonic()
                frame = main_frame if output == chosen.name and main_frame is not None else self._read_frame()
                self.aux_preview.submit(output, frame, dict(self.render_metrics,
                    readback_ms=(time.monotonic() - started) * 1000, captured=time.monotonic()))
            finally:
                r.patch.display = chosen

    def _publish(self):
        from .live import resource_status
        p = self.renderer.patch if self.renderer else None
        jobs = []
        for key, item in self.jobs.items():
            try:
                status = json.loads((self.root / '.hydrapy' / 'jobs' / key / 'status.json').read_text())
                proc = self.job_processes[key]
                if status.get('state') in ('queued', 'running') and proc.poll() is not None:
                    status = dict(state='error', error=f'Export process exited ({proc.returncode}); see .hydrapy/jobs/{key}/log.txt')
                jobs.append(dict(item, **status))
            except (OSError, ValueError):
                jobs.append(dict(item, state='running', progress=0))
        preview = self.preview.snapshot()
        state = dict(path=str(self.path.relative_to(self.root)) if self.path else None,
                     version=digest(self.source), playing=not self.paused, armed=self.armed,
                     backend='GL' if hasattr(self.renderer, 'ctx') else 'CPU', backend_note=self.backend_note,
                     size=self.size, fps=self.actual_fps, target_fps=self.fps,
                     time=self.clock_time, bpm=p.bpm if p else 30,
                     clock=type(p.transport).__name__ if p and p.transport and self.armed else 'Internal',
                     simulation_fps=p.frame_rate if p else None,
                     output=p.display.name if p else 'o0', outputs=[o.name for o in p.active_outputs()] if p else [],
                     controls=[x.describe() for x in p.controls.values()] if p else [],
                     connections=resource_status(), error=self.error, notice=self.notice,
                     recording=self.recorder is not None, jobs=jobs, recent=self.recent,
                     thumbnails=preview.pop('thumbnails'), preview=preview,
                     preview_generation=self.preview.generation, preview_profile=self.preview_profile,
                     preview_error=self.preview_error)
        state['tracker_enabled'] = self.tracker.enabled
        state['aux_preview'] = self.aux_preview.snapshot()
        with self.lock:
            self.state = state

    def _run(self):
        sample_start, frames = time.monotonic(), 0
        try:
            while self.running:
                started = time.monotonic()
                try:
                    action, data, result = self.commands.get(timeout=.001)
                except queue.Empty:
                    pass
                else:
                    try:
                        value = self._handle(action, data)
                        if action not in ('source', 'check', 'presets', 'devices', 'mouse'):
                            self.error = ''
                        result.put(value)
                    except Exception as exc:
                        self.error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
                        if action in ('load', 'save', 'restart', 'connections'):
                            self.notice = 'Edit rejected. Previous patch retained.'
                        result.put(ValueError(self.error))
                    self._publish()
                if not self.running:
                    break
                now = time.monotonic()
                if self.renderer:
                    query = None
                    if not self.paused:
                        self.clock_time += now - self.last_wall
                        try:
                            transport = self.renderer.patch.transport if self.armed else None
                            if transport:
                                self.renderer.patch.bpm = transport.bpm
                                self.clock_time = transport.now()
                            if self.preview_profile and hasattr(self.renderer, 'ctx'):
                                if self.gpu_query is None:
                                    try:
                                        self.gpu_query = self.renderer.ctx.query(time=True)
                                    except Exception:
                                        self.preview_profile = False
                                query = self.gpu_query
                            render_started = time.monotonic()
                            self.tracker.apply(self.renderer.patch, self.clock_time)
                            with query if query is not None else nullcontext():
                                self.renderer.step(self.clock_time)
                            self.render_metrics = dict(render_cpu_ms=(time.monotonic() - render_started) * 1000,
                                                       frame_started=render_started, gpu_ms=None)
                            frames += 1
                        except Exception as exc:
                            self.error, self.paused = f'Render paused: {type(exc).__name__}: {exc}', True
                    recorded = None
                    if self.recorder:
                        try:
                            # Recording uses every synthesis frame, independently of
                            # the browser's lower-rate JPEG thumbnails.
                            read_started = time.monotonic()
                            self.last_frame = self._read_frame()
                            recorded = (self.last_frame, (time.monotonic() - read_started) * 1000)
                            self.recorder.push(self.last_frame)
                        except Exception as exc:
                            self.error = str(exc)
                            recorder, self.recorder = self.recorder, None
                            try:
                                recorder.close()
                            except Exception:
                                pass
                    try:
                        self._pictures(recorded=recorded, query=query)
                        self.preview_error = ''
                    except Exception as exc:
                        # A transport failure must not pause synthesis or live hooks.
                        self.preview_error = f'{type(exc).__name__}: {exc}'
                self.last_wall = now
                if self.cv is not None:
                    self.cv.publish(self.tracker, self.clock_time, now, not self.paused and self.renderer is not None)
                with self.lock:
                    self.tracker_state = self.tracker.snapshot(self.clock_time, not self.paused)
                    self.tracker_state.update(sampled=time.monotonic(), blackout=self.tracker_blackout,
                                              error=self.error or self.preview_error, fps=self.actual_fps, size=self.size,
                                              backend='GL' if hasattr(self.renderer, 'ctx') else 'CPU',
                                              output=self.renderer.patch.display.name if self.renderer else 'o0',
                                              generation=self.preview.generation, aux_generation=self.aux_preview.generation,
                                              path=str(self.path.relative_to(self.root)) if self.path else None,
                                              controls=[x.describe() for x in self.renderer.patch.controls.values()] if self.renderer else [],
                                              cv=None if self.cv is None else dict(map={str(k): v for k, v in self.cv.mapping.items()},
                                                                                  samplerate=self.cv.samplerate, error=self.cv.error))
                if now - sample_start >= .5:
                    self.actual_fps = frames / (now - sample_start)
                    sample_start, frames = now, 0
                    self._publish()
                if self.path and now - self.last_watch > .5 and not self.recorder:
                    self.last_watch = now
                    try:
                        stamp = (self.path, self.path.stat().st_mtime_ns)
                        if stamp != self.stamp:
                            self._load(str(self.path.relative_to(self.root)))
                    except Exception as exc:
                        self.error, self.notice = str(exc), 'Edit rejected. Previous patch retained.'
                time.sleep(max(.001, 1 / self.fps - (time.monotonic() - started)))
        finally:
            self.running = False
            self.gpu_query = None  # dispose timer on the GL context's thread
            self.preview.close()
            self.aux_preview.close()
            if self.recorder:
                try:
                    self.recorder.close()
                except Exception as exc:
                    self.error = str(exc)
            from .live import close_resources
            close_resources()
            if hasattr(self.renderer, 'release'):
                self.renderer.release()


def make_server(session, port=8765):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            # Headers and JPEG bodies are separate writes. Nagle + delayed ACKs
            # otherwise introduce ~40 ms stalls on persistent loopback requests.
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, fmt, *args):
            pass

        def _authorized(self):
            host = self.headers.get('Host', '')
            allowed = (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')
            origin = self.headers.get('Origin')
            return (host in allowed and (origin is None or origin in tuple('http://' + h for h in allowed))
                    and secrets.compare_digest(self.headers.get('X-Hydrapy-Token', ''), token))

        def _send(self, status, value, content_type='application/json', headers=None):
            payload = json.dumps(value).encode() if content_type == 'application/json' else value
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status >= 400:
                # Rejected POST bodies may be unread; do not parse those bytes
                # as another request on a persistent connection.
                self.close_connection = True
                self.send_header('Connection', 'close')
            for name, val in (headers or {}).items():
                self.send_header(name, str(val))
            self.end_headers()
            try:
                self.connection.settimeout(5)
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                return self._send(200, Path(__file__).with_name('studio.html').read_bytes(), 'text/html; charset=utf-8')
            assets = {'/tracker/': ('index.html', 'text/html; charset=utf-8'),
                      '/tracker/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/tracker/backend.js': ('backend.js', 'text/javascript; charset=utf-8'),
                      '/tracker/style.css': ('style.css', 'text/css; charset=utf-8')}
            if parsed.path in assets:
                name, mime = assets[parsed.path]
                return self._send(200, Path(__file__).with_name('tracker').joinpath(name).read_bytes(), mime)
            if not self._authorized():
                return self._send(403, {'error': 'Open the local URL printed by hydrapy studio.'})
            args = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/tracker':
                    with session.lock:
                        state = dict(session.tracker_state)
                    state['server_now'] = time.monotonic()
                    return self._send(200, state)
                if parsed.path == '/api/tracker_saved':
                    path = session.root / '.hydrapy' / 'tracker.json'
                    return self._send(200, {'document': json.loads(path.read_text()) if path.exists() else None})
                if parsed.path == '/api/state':
                    with session.lock:
                        state = dict(session.state)
                    return self._send(200, state)
                if parsed.path == '/api/catalog':
                    return self._send(200, {'patches': catalog(session.root)})
                if parsed.path == '/api/frame':
                    output = args.get('output', ['o0'])[0]
                    if output not in ('o0', 'o1', 'o2', 'o3'):
                        raise ValueError('output must be o0–o3')
                    after = int(args.get('after', ['-1'])[0])
                    started = time.monotonic()
                    bus = args.get('bus', ['program'])[0]
                    if bus not in ('program', 'aux'):
                        raise ValueError('bus must be program or aux')
                    preview = session.aux_preview if bus == 'aux' else session.preview
                    frame = preview.frame(output, after)
                    if frame is None:
                        return self._send(204, b'', 'image/jpeg')
                    sent = time.monotonic()
                    metrics = frame['metrics']
                    timing = {k: v for k, v in metrics.items() if k.endswith('_ms') and v is not None}
                    timing['wait_ms'] = (sent - started) * 1000
                    return self._send(200, frame['data'], 'image/jpeg', {
                        'X-Preview-Sequence': frame['sequence'],
                        'X-Preview-Age-Ms': max(0., (sent - metrics.get('frame_started', metrics['captured'])) * 1000),
                        'Server-Timing': ', '.join(f'{k};dur={v:.3f}' for k, v in timing.items())})
                if parsed.path == '/api/file':
                    path = project_path(session.root, args['path'][0])
                    if not path.is_relative_to(session.root / '.hydrapy' / 'exports'):
                        raise ValueError('Only completed workspace exports can be downloaded')
                    import mimetypes
                    return self._send(200, path.read_bytes(), mimetypes.guess_type(path)[0] or 'application/octet-stream')
                if parsed.path == '/api/preview':
                    path = project_path(session.root, args['path'][0])
                    if path.suffix.lower() not in ('.jpg', '.png', '.webp'):
                        raise ValueError('Preview must be an image')
                    import mimetypes
                    return self._send(200, path.read_bytes(), mimetypes.guess_type(path)[0])
                return self._send(404, {'error': 'Unknown endpoint'})
            except Exception as exc:
                self._send(400, {'error': str(exc)})

        def do_POST(self):
            if not self._authorized() or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self._send(403, {'error': 'Local session token required'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                action = self.path.removeprefix('/api/')
                result = session.call(action, **body)
                self._send(200, result)
            except Exception as exc:
                self._send(400, {'error': str(exc)})
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    return server, token


def serve(root='.', patch=None, backend='auto', size=(640, 360), fps=30, port=8765, open_browser=True, tracker=False, cv=None):
    root = Path(root).resolve()
    # Match CLI patch execution: relative asset paths belong to the project.
    os.chdir(root)
    try:
        session = Session(root, patch, backend, size, fps, cv=cv)
    except BaseException:
        if cv is not None:
            cv.close()
        raise
    server = None
    try:
        server, token = make_server(session, port)
        url = f'http://127.0.0.1:{server.server_port}/' + ('tracker/' if tracker else '') + f'#{token}'
        print(f'HydraPy Studio: {url}', flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        session.close()
