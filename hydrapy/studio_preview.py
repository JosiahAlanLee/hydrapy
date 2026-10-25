"""Lossy, bounded Studio preview transport. No renderer or GL calls in this worker."""
from __future__ import annotations

import base64
import io
import threading
import time


def jpeg(frame, size=None):
    from PIL import Image
    im = Image.fromarray(frame)
    if size:
        im.thumbnail(size)
    out = io.BytesIO()
    im.save(out, 'JPEG', quality=85)
    return out.getvalue()


class Preview:
    """One pending main frame and at most four thumbnail frames; newest replaces old.

    Locks protect references only. Encoding, disk writes, HTTP waits and socket writes
    never hold up the render thread. A request asks for a fresh capture, not a queue.
    """
    def __init__(self):
        self.condition = threading.Condition()
        self.running = True
        self.generation = self.sequence = 0
        self.output = 'o0'
        self.pending = self.latest = None
        self.thumbnail_pending, self.thumbnails = {}, {}
        self.requested, self.active_until = True, 0.
        self.next_thumbnail, self.thumbnail_index = 0., 0
        self.replaced = self.encoded = 0
        self.error = ''
        self.thread = threading.Thread(target=self._run, name='hydrapy-jpeg', daemon=True)
        self.thread.start()

    def reset(self, output):
        with self.condition:
            self.generation += 1
            self.output = output
            self.pending = self.latest = None
            self.thumbnail_pending, self.thumbnails = {}, {}
            self.requested = True
            self.next_thumbnail = 0.
            self.error = ''
            self.condition.notify_all()

    def plan(self, now, *, changed=True):
        with self.condition:
            main = self.requested and changed
            if main:
                self.requested = False
            thumbnail = None
            # Spread four thumbnails across half a second instead of reading all
            # outputs on one frame. Stop preview-only work when nobody is watching.
            if now < self.active_until and now >= self.next_thumbnail:
                thumbnail = 'o' + str(self.thumbnail_index)
                self.thumbnail_index = (self.thumbnail_index + 1) % 4
                self.next_thumbnail = now + .125
            return main, thumbnail

    def submit(self, output, frame, metrics, *, thumbnail=False, cache=None):
        with self.condition:
            if not self.running:
                return
            self.sequence += 1
            item = (self.generation, self.sequence, output, frame, dict(metrics), cache)
            if thumbnail:
                self.thumbnail_pending[output] = item
            else:
                if self.pending is not None:
                    self.replaced += 1
                self.pending = item
            self.condition.notify_all()

    def frame(self, output, after=-1, timeout=1., *, thumbnails=True):
        deadline = time.monotonic() + timeout
        with self.condition:
            if output != self.output:
                return None
            self.requested = True
            if thumbnails:
                self.active_until = time.monotonic() + 2.
            while self.running and output == self.output:
                if self.latest and self.latest['sequence'] > after:
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            return None

    def snapshot(self):
        with self.condition:
            return dict(thumbnails=dict(self.thumbnails), encoded=self.encoded,
                        replaced=self.replaced, pending=int(self.pending is not None),
                        thumbnail_pending=len(self.thumbnail_pending), error=self.error)

    def close(self):
        with self.condition:
            self.running = False
            self.pending = None
            self.thumbnail_pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout=2)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: not self.running or self.pending is not None or self.thumbnail_pending)
                if not self.running:
                    return
                thumbnail = self.pending is None
                if thumbnail:
                    _, item = self.thumbnail_pending.popitem()
                else:
                    item, self.pending = self.pending, None
            generation, sequence, output, frame, metrics, cache = item
            try:
                started = time.monotonic()
                metrics['encode_wait_ms'] = (started - metrics['captured']) * 1000
                data = jpeg(frame, (160, 90) if thumbnail else None)
                metrics['jpeg_ms'] = (time.monotonic() - started) * 1000
                with self.condition:
                    if not self.running or generation != self.generation:
                        continue
                    self.error = ''
                    if thumbnail:
                        self.thumbnails[output] = base64.b64encode(data).decode()
                    else:
                        self.latest = dict(sequence=sequence, output=output, data=data, metrics=metrics)
                        self.encoded += 1
                        self.condition.notify_all()
                if cache:
                    try:
                        if not cache.exists():
                            cache.parent.mkdir(parents=True, exist_ok=True)
                            cache.write_bytes(jpeg(frame, (320, 180)))
                    except OSError:
                        pass  # Read-only workspaces still support preview.
            except Exception as exc:
                with self.condition:
                    if self.running and generation == self.generation:
                        self.error = f'{type(exc).__name__}: {exc}'
