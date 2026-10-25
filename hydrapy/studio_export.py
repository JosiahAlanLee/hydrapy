"""One export per process, isolated from the live preview and its hardware hooks."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys
import time


def run(job_path):
    from .cli import load_patch
    from .controls import apply_controls
    from .live import ResourceStage
    from .inspect import capabilities
    from .studio import atomic_text
    from .export import save_mp4, save_gif, save_png, to_html
    job_path = Path(job_path)
    spec = json.loads(job_path.read_text())
    status = job_path.with_name('status.json')
    def report(**value):
        atomic_text(status, json.dumps(value))
    try:
        report(state='running', progress=0)
        with ResourceStage():
            patch = load_patch(spec['patch'], strict=False, source=spec['source'])
            apply_controls(patch, spec['params'])
            if spec.get('scene_weights') is not None:
                patch.scene_weights = {name: float(w) for name, w in spec['scene_weights'].items() if name in patch.scenes}
            patch.render(patch.outputs[spec['display']])
            target = 'html' if spec['format'] == 'html' else spec['backend']
            check = capabilities(patch, target)
            if not check['ready']:
                raise ValueError('; '.join(f"{f['path']}: {f['message']}" for f in check['findings'] if f['status'] in ('unavailable', 'requires baking')))
            if spec['format'] == 'html':
                output = Path(spec['output'])
                temporary = output.with_suffix('.tmp.html')
                try:
                    to_html(patch, str(temporary), title=Path(spec['patch']).stem, size=tuple(spec['size']))
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                if spec['backend'] == 'gl':
                    from .backend_gl import iter_frames_gl as render
                else:
                    from .backend_np import iter_frames as render
                seconds = 1 / spec['fps'] if spec['format'] == 'png' else spec['seconds']
                start = spec['time'] if spec['format'] == 'png' else 0
                iterator = render(patch, seconds=seconds, fps=spec['fps'], size=tuple(spec['size']), start=start)
                if spec.get('tracker'):
                    from .studio_tracker import Tracker
                    score = spec['tracker']
                    tracker = Tracker(score['document'])
                    tracker.enabled = True
                    beat = score['beat'] if spec['format'] == 'png' else 0.
                    iterator = tracker.frames(patch, seconds=seconds, fps=spec['fps'], size=tuple(spec['size']),
                                              backend=spec['backend'], start=start, beat=beat)
                total = max(1, round(seconds * spec['fps']))
                selected = ''
                def encoder(e):
                    nonlocal selected
                    selected = e.name
                def frames():
                    last_report = 0
                    try:
                        for i, frame in enumerate(iterator):
                            now = time.monotonic()
                            if now - last_report >= .5:
                                report(state='running', progress=i / total, encoder=selected)
                                last_report = now
                            yield frame
                    finally:
                        iterator.close()
                if spec['format'] == 'mp4':
                    track = next((p for p in patch.probes if hasattr(p, 'chunks')), None)
                    if track is not None and not spec['audio']:
                        track.record, track.keep = True, max(track.keep, seconds + 1)
                    save_mp4(frames(), spec['output'], fps=spec['fps'], encoder=spec['encoder'],
                             audio=spec['audio'] or track, on_encoder=encoder)
                elif spec['format'] == 'gif':
                    save_gif(frames(), spec['output'], fps=spec['fps'])
                else:
                    stream = frames()
                    try:
                        save_png(next(stream), spec['output'])
                    finally:
                        stream.close()
        report(state='done', progress=1, encoder=selected if spec['format'] != 'html' else '')
    except Exception as exc:
        report(state='error', progress=0, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    run(sys.argv[1])
