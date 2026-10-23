"""Patch-specific capability reports and static example discovery."""
from __future__ import annotations

import ast
from pathlib import Path
from .core import Chain, Texture, Camera, FrameSource, External
from .signals import Signal, Fn


def walk_patch(patch):
    seen = set()
    def walk(obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        yield path, obj
        if isinstance(obj, Chain):
            for tr in obj.transforms:
                op_path = f'{path}.{tr.op.name}'
                yield op_path, tr.op
                for inp, arg in zip(tr.op.inputs, tr.args):
                    yield from walk(arg, f'{op_path}({inp.name})')
        elif isinstance(obj, (Signal, Texture)):
            for key, value in vars(obj).items():
                if key.startswith('_') or key in ('chain', 'output'):
                    continue
                yield from walk(value, f'{path}.{key}')
        elif isinstance(obj, dict):
            for key, value in obj.items():
                yield from walk(value, f'{path}.{key}')
        elif isinstance(obj, (tuple, list)):
            for i, value in enumerate(obj):
                yield from walk(value, f'{path}[{i}]')
    for output in patch.active_outputs():
        yield from walk(output.chain, output.name)


def capabilities(patch, target='html'):
    from .core import Op
    from .live import LiveValue, TrigPhase
    from .controls import Param
    if target not in ('cpu', 'gl', 'html', 'hydra'):
        raise ValueError('target must be cpu, gl, html or hydra')
    findings = []
    def note(path, status, message):
        entry = dict(path=path, status=status, message=message)
        if entry not in findings:
            findings.append(entry)
    for path, obj in walk_patch(patch):
        if isinstance(obj, Op):
            if obj.cpu_only and target in ('gl', 'html'):
                note(path, 'unavailable', 'CPU-only operator; choose CPU rendering or replace it.')
            if target == 'hydra' and not obj.hydra:
                note(path, 'unavailable', 'Not part of Hydra JavaScript; use standalone HTML or video.')
        if target == 'cpu' and isinstance(obj, Camera):
            note(path, 'unavailable', 'Live camera input requires desktop GL or the browser.')
        if target in ('html', 'hydra'):
            if isinstance(obj, FrameSource) or type(obj).__name__ in ('Flow', 'XY'):
                note(path, 'requires baking', 'Python-computed texture; bake a finite sequence into a Bank first.')
            if isinstance(obj, (LiveValue, TrigPhase)):
                note(path, 'fixed value', 'Live hardware input becomes a fixed value in this export.')
            if isinstance(obj, Fn):
                note(path, 'baked loop', f'Python function sampled at {obj.bake_rate:g} Hz over {obj.bake_seconds:g}s, then looped.')
            elif callable(obj) and not isinstance(obj, (Signal, Chain, Texture, Op)):
                note(path, 'baked loop', 'Python callable becomes a sampled, looping table.')
            if isinstance(obj, Param):
                note(path, 'snapshot', 'Base control value is captured; declared signal modulation is retained.')
        if target == 'html' and type(obj).__name__ in ('NCA', 'Physarum'):
            note(path, 'requirement', 'Requires WebGL2 with floating-point render targets.')
        if isinstance(obj, External):
            note(path, 'unavailable', f'Unresolved external source: {obj.note or obj.name}.')
    if target in ('html', 'hydra'):
        from .live import resource_status
        for r in resource_status():
            if r['kind'] in ('CvIn', 'CvOut', 'OscIn', 'OscOut', 'Midi', 'Link', 'VideoOut'):
                note(r['kind'], 'not included', 'Desktop hardware or stream connection is not carried into this export.')
    return dict(target=target, ready=not any(x['status'] in ('unavailable', 'requires baking') for x in findings),
                findings=findings)


TASKS = {
    'Feedback': ('rd', 'life', 'prev', 'delay', 'Decay'),
    'Paint & texture': ('handpainted', 'charcoal', 'stain', 'film', 'NCA', 'Physarum', 'inkdrop', 'studie'),
    'Sound & rhythm': ('Audio', 'Beats', 'Soundtrack', 'Clock', 'Env'),
    'Geometry': ('Cellular', 'SDF3', 'raymarch', 'Lines', 'Harmonograph', 'Rose', 'sdfCircle', 'orbitVoronoi'),
    'Connections': ('CvIn', 'CvOut', 'Midi', 'OscIn', 'OscOut', 'Link', 'VideoOut', 'XY'),
    'Plotting': ('to_plotter_svg', 'streamlines', 'chain_contours', 'to_svg'),
}


def catalog(root):
    """Read example metadata without executing the patch or opening its devices."""
    root = Path(root).resolve()
    search = root / 'patches' if (root / 'patches').is_dir() else root
    entries = []
    for path in sorted(search.rglob('*.py')):
        if any(p.startswith('.') or p in ('__pycache__', 'node_modules') for p in path.relative_to(root).parts):
            continue
        try:
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            description = ast.get_docstring(tree) or ''
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            tags = [k for k, terms in TASKS.items() if names.intersection(terms)] or ['Patterns']
            meta = {}
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STUDIO' for t in n.targets):
                    meta = ast.literal_eval(n.value)
                    if not isinstance(meta, dict):
                        raise ValueError('STUDIO metadata must be a dictionary')
            if not isinstance(meta.get('tags', []), (list, tuple)) or not all(isinstance(t, str) for t in meta.get('tags', [])):
                raise ValueError('STUDIO tags must be a list of strings')
            if any(k in meta and not isinstance(meta[k], str) for k in ('title', 'description', 'preview')):
                raise ValueError('STUDIO text metadata must be strings')
            from .studio import digest
            cached = Path('.hydrapy/previews') / (digest(str(path.relative_to(root)))[:16] + '.jpg')
            if not meta.get('preview') and (root / cached).is_file():
                meta['preview'] = str(cached)
            entries.append(dict(path=str(path.relative_to(root)), title=meta.get('title', path.stem.replace('_', ' ')),
                                description=meta.get('description', description.split('\n\n')[0]),
                                tags=meta.get('tags', tags), controls='Param' in names,
                                preview=meta.get('preview')))
        except (OSError, SyntaxError, ValueError):
            entries.append(dict(path=str(path.relative_to(root)), title=path.stem, description='Patch needs attention',
                                tags=['Needs attention'], controls=False, preview=None))
    return entries
