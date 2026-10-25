"""Measure the actual Studio page with synthetic patches and headless Chromium.

python docs/studio_preview_benchmark.py --backend gl --size 640x360 --json preview.json
Requires optional playwright + `python -m playwright install chromium`.
--checkout /path/to/base runs the SAME harness against an older checkout.
--view outputs opens Gorgon, program and auxiliary windows; --case gorgon uses
the four-output demo patch. No screen/video capture runs during measurement.
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
    'gorgon': '''from hydrapy import *
z = Param('Dolly Z', .35, min=0, max=1)
roll = Param('Roll', 0, min=-45, max=45)
feedback = Param('Feedback', .25, min=0, max=.9)
hue = Param('Hue', .2, min=0, max=1)
set_bpm(138)
osc(12,.035,.6).kaleid(6).modulate(noise(2.4,.08),.12).scale(1+z*.7).rotate(
    roll*.01745329252).color(.48,.82,.68).hue(hue).contrast(1.3).out(o1)
src(o3).scale(1.008).rotate(.003).blend(src(o1),1-feedback).out(o3)
src(o1).blend(src(o3),.6).saturate(1.3).out(o0)
src(o3).edge().color(.7,.9,1).out(o2)
render(o0)
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
            errors = []
            routes = {'studio': '/'} if args.view == 'studio' else {'tracker': '/tracker/'}
            if args.view == 'outputs':
                routes.update(program='/tracker/?output=program', aux='/tracker/?output=aux')
            stats_expr = 'window.hydrapyPreview' if args.view == 'studio' else 'window.gorgonPreview'
            count = f'({stats_expr} ? {stats_expr}.frames : window.legacyFrames)'
            pages, clients = {}, {}
            for label, route in routes.items():
                # Separate contexts keep all three surfaces visible in headless Chromium.
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.add_init_script('''window.legacyFrames = 0; window.previewPaints = 0;
                    document.addEventListener('load', e => {
                        if (e.target.id === 'preview') window.legacyFrames++;
                    }, true);
                    const draw = CanvasRenderingContext2D.prototype.drawImage;
                    CanvasRenderingContext2D.prototype.drawImage = function(...args) {
                        if (['previewCanvas','sceneThumb','outputCanvas'].includes(this.canvas.id)) window.previewPaints++;
                        return draw.apply(this, args);
                    };''')
                page.goto(f'http://127.0.0.1:{server.server_port}{route}#{token}')
                if label in ('program', 'aux'):
                    page.locator('#outputDismiss').click()
                page.wait_for_function(count + ' >= 3', timeout=30000)
                clients[label] = page.context.new_cdp_session(page)
                clients[label].send('Performance.enable')
                pages[label] = page
            if args.view != 'studio':
                session.call('tracker_transport', enabled=True, playing=True)
            def metrics(client):
                return {m['name']: m['value'] for m in client.send('Performance.getMetrics')['metrics']}
            page = next(iter(pages.values()))
            page.wait_for_timeout(args.warmup * 1000)
            first, before = {}, {}
            for label, view in pages.items():
                view.evaluate(f'if ({stats_expr}) {stats_expr}.samples.length = 0; window.previewPaints = 0')
                first[label], before[label] = view.evaluate(count), metrics(clients[label])
            readbacks = []
            read_frame = session._read_frame
            def measured_read():
                start = time.perf_counter()
                try:
                    return read_frame()
                finally:
                    readbacks.append((time.perf_counter() - start) * 1000)
            session._read_frame = measured_read
            render_first, cpu_start, started = session.renderer.frame, time.process_time(), time.monotonic()
            page.wait_for_timeout(args.seconds * 1000)
            duration = time.monotonic() - started
            report = dict(case=name, view=args.view, seconds=duration, browser=browser.version,
                          render_fps=(session.renderer.frame-render_first)/duration,
                          server_cpu_seconds_per_second=(time.process_time()-cpu_start)/duration,
                          readbacks_per_second=len(readbacks)/duration,
                          readback_timings=summarize([{'readback_ms': v} for v in readbacks]), views={})
            session._read_frame = read_frame
            for label, view in pages.items():
                stats = view.evaluate(stats_expr + ' || null')
                after = metrics(clients[label])
                report['views'][label] = dict(display_fps=(view.evaluate(count)-first[label])/duration,
                    preview_paints_per_second=view.evaluate('window.previewPaints')/duration,
                    timings=summarize(stats['samples']) if stats else {},
                    browser_seconds_per_second={k: (after[k]-before[label][k])/duration
                        for k in ('TaskDuration', 'ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration')})
                if stats and stats.get('error'):
                    errors.append(stats['error'])
            primary = next(iter(report['views'].values()))
            report.update(display_fps=primary['display_fps'], timings=primary['timings'],
                          errors=errors+[x for x in (session.error, getattr(session, 'preview_error', '')) if x])
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
    parser.add_argument('--view', choices=['studio', 'gorgon', 'outputs'], default='studio')
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
    report = dict(backend=args.backend, size=args.size, target_fps=args.fps, view=args.view, gpu_profile=args.profile,
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
