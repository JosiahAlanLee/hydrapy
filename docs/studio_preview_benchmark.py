"""Measure the actual Studio page with synthetic patches and headless Chromium.

python docs/studio_preview_benchmark.py --backend gl --size 640x360 --json preview.json
Requires optional playwright + `python -m playwright install chromium`.
--checkout /path/to/base runs the SAME harness against an older checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time


PATCHES = {
    'single': 'from hydrapy import *\nosc(12, .2, .5).rotate(0, .1).out(o0)\n',
    'multi': '''from hydrapy import *
noise(4, .2).modulate(noise(7, .1), .2).blur(3, 3).out(o0)
src(o1).scale(1.01).blend(osc(18, .1, .5).modulate(noise(3), .2), .15).blur(2, 2).out(o1)
voronoi(7, .3).modulate(noise(5), .2).blur(2, 3).out(o2)
src(o0).blend(src(o1), .4).add(src(o2), .2).blur(2, 2).out(o3)
render(o3)
''',
}


def summarize(samples):
    result = {}
    for key in sorted({k for row in samples for k in row if k.endswith('_ms')}):
        values = sorted(row[key] for row in samples if row.get(key) is not None)
        if values:
            result[key] = dict(median=statistics.median(values),
                               p95=values[min(len(values) - 1, int(.95 * len(values)))])
    return result


def measure(playwright, args, name):
    from hydrapy.studio import Session, make_server
    with tempfile.TemporaryDirectory(prefix='hydrapy-preview-') as folder:
        Path(folder, 'patch.py').write_text(PATCHES[name])
        session = Session(folder, 'patch.py', backend=args.backend, size=args.size, fps=args.fps)
        server, token = make_server(session, 0)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        browser = None
        try:
            if args.profile and hasattr(session, 'preview_profile'):
                session.call('preview_profile', enabled=True)
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            # Legacy Studio replaces src; the new page reports decoded presentations.
            page.add_init_script('''window.legacyFrames = 0;
                document.addEventListener('load', e => {
                    if (e.target.id === 'preview') window.legacyFrames++;
                }, true);''')
            page.goto(f'http://127.0.0.1:{server.server_port}/#{token}')
            count = 'window.hydrapyPreview ? window.hydrapyPreview.frames : window.legacyFrames'
            page.wait_for_function('(' + count + ') >= 3', timeout=30000)
            page.wait_for_timeout(args.warmup * 1000)
            page.evaluate('if (window.hydrapyPreview) window.hydrapyPreview.samples.length = 0')
            first, render_first, started = page.evaluate(count), session.renderer.frame, time.monotonic()
            page.wait_for_timeout(args.seconds * 1000)
            duration = time.monotonic() - started
            shown = page.evaluate(count) - first
            stats = page.evaluate('window.hydrapyPreview || null')
            report = dict(case=name, display_fps=shown / duration,
                          render_fps=(session.renderer.frame - render_first) / duration,
                          seconds=duration, displayed=shown, browser=browser.version,
                          timings=summarize(stats['samples']) if stats else {},
                          errors=errors + [x for x in (session.error,
                              getattr(session, 'preview_error', ''), (stats or {}).get('error', '')) if x])
            if hasattr(session, 'preview'):
                report['queue'] = {k: v for k, v in session.preview.snapshot().items() if k != 'thumbnails'}
            if report['errors']:
                raise RuntimeError(json.dumps(report))
            return report
        finally:
            if browser:
                browser.close()
            server.shutdown(); server.server_close(); session.close()
            server_thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'gl'], default='gl')
    parser.add_argument('--size', default='640x360')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--warmup', type=float, default=2)
    parser.add_argument('--case', action='append', choices=list(PATCHES))
    parser.add_argument('--profile', action='store_true', help='opt in to GPU timer queries')
    parser.add_argument('--checkout', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    try:
        args.size = tuple(int(n) for n in args.size.lower().split('x'))
        if len(args.size) != 2 or min(args.size) < 16 or max(args.size) > 4096:
            raise ValueError
        if not 1 <= args.seconds <= 60 or not 0 <= args.warmup <= 30 or not 1 <= args.fps <= 120:
            raise ValueError
    except ValueError:
        parser.error('Use WIDTHxHEIGHT (16–4096), 1–60 seconds, 0–30 warmup seconds, 1–120 fps')
    sys.path.insert(0, str(args.checkout.resolve()))
    from playwright.sync_api import sync_playwright
    import hydrapy
    report = dict(backend=args.backend, size=args.size, target_fps=args.fps, gpu_profile=args.profile,
                  python=platform.python_version(), platform=platform.platform(),
                  package=str(Path(hydrapy.__file__).resolve()),
                  commit=subprocess.check_output(['git', '-C', str(args.checkout), 'rev-parse', 'HEAD'], text=True).strip(),
                  dirty=bool(subprocess.check_output(['git', '-C', str(args.checkout), 'status', '--porcelain'], text=True)),
                  cases=[])
    if args.backend == 'gl':
        # Context info must be queried on its owning thread, never from the
        # running Studio renderer on this browser-driving thread.
        from hydrapy.backend_gl import _create_context
        context = _create_context()
        try:
            report['renderer'] = context.info['GL_RENDERER']
            report['vendor'] = context.info['GL_VENDOR']
        finally:
            context.release()
    with sync_playwright() as playwright:
        for name in args.case or PATCHES:
            result = measure(playwright, args, name)
            report['cases'].append(result)
            print(json.dumps(result), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
