"""Compare identical Studio + MP4 + optional JACK runs, diagnostics off/on.

python docs/studio_diagnostics_benchmark.py --output /tmp/diagnostic-check --seconds 600 --audio
Requires a running JACK server when --audio is selected; never starts/reconfigures one.
Run without other benchmarks/tests competing for CPU/GPU. This is not a Rack or hardware test.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from hydrapy.studio import Session
from hydrapy.studio_diagnostics import distribution, environment


class MeasuredSession(Session):
    def __init__(self, *args, **kwargs):
        self.measured_starts = []
        self.measuring = False
        self.measured_graphics = None
        super().__init__(*args, **kwargs)

    def _load(self, *args, **kwargs):
        result = super()._load(*args, **kwargs)
        # GL information must be queried on the thread owning the context.
        ctx = getattr(self.renderer, 'ctx', None)
        self.measured_graphics = ctx.info.get('GL_RENDERER') if ctx is not None else None
        return result

    def _step(self, now):
        result = super()._step(now)
        if self.measuring and result[1]:
            self.measured_starts.append(now)
        return result


def run(folder, args, enabled):
    folder.mkdir()
    source = Path(__file__).resolve().parents[1] / 'patches/gorgon_scenes.py'
    (folder / 'patch.py').write_text(source.read_text() + '\nSoundtrack(o0)\n')
    session = MeasuredSession(folder, 'patch.py', backend=args.backend, size=args.size, fps=30,
                              diagnostics='diagnostics.jsonl' if enabled else None)
    result = dict(diagnostics=enabled, backend=args.backend, size=args.size, seconds=args.seconds)
    try:
        score = copy.deepcopy(session.tracker.document)
        score['bpm'] = 138
        for pattern in score['patterns']:
            pattern['scenes'] = [dict(r=0, name='drift', fade=4), dict(r=16, name='fracture', fade=4),
                                 dict(r=32, name='bloom', fade=4), dict(r=48, name='drift', fade=4)]
            for r in range(0, 64, 4):
                pattern['audio'][0][r] = dict(note='C-2', vel=100)
        session.call('tracker_score', document=score, revision=session.tracker.revision)
        session.call('tracker_transport', enabled=True, playing=True, beat=0)
        if args.audio:
            session.call('jack_audio', enabled=True, lead_blocks=4)
        session.call('record')
        time.sleep(args.warmup)
        before = session.audio.describe()
        began, cpu = time.monotonic(), time.process_time()
        session.measuring = True
        deadline = began + args.seconds
        while time.monotonic() < deadline:
            if session.error or session.audio.error or not session.thread.is_alive() or not session.recorder:
                raise RuntimeError(session.error or session.audio.error or 'render/recorder stopped')
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        session.measuring = False
        elapsed = time.monotonic() - began
        starts = list(session.measured_starts)
        after = session.audio.describe()
        result.update(wall_s=elapsed, cpu_s=time.process_time() - cpu, frames=len(starts),
                      fps=len(starts) / elapsed, interval_ms=distribution([(b - a) * 1000 for a, b in zip(starts, starts[1:])]),
                      audio={k: after[k] - before[k] for k in ('xruns', 'underruns', 'late_packets', 'suspensions')},
                      audio_rate=after['samplerate'], audio_period=after['blocksize'],
                      repeated_frames=session.recorder.repeated_frames,
                      graphics=session.measured_graphics)
        session.call('record')
    finally:
        session.close()
    if enabled:
        diag = session.diagnostics
        result.update(diagnostic_bytes=diag.path.stat().st_size, diagnostic_error=diag.error,
                      dropped_frames=diag.dropped_frames, dropped_events=diag.dropped_events)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--backend', choices=('cpu', 'gl'), default='gl')
    parser.add_argument('--size', default='320x180')
    parser.add_argument('--audio', action='store_true', help='require real JACK audio')
    parser.add_argument('--order', choices=('off,on', 'on,off', 'on'), default='off,on')
    args = parser.parse_args()
    args.size = tuple(map(int, args.size.lower().split('x')))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(environment=environment(), runs=[])
    for name in args.order.split(','):
        result = run(args.output / name, args, name == 'on')
        report['runs'].append(result)
        (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
