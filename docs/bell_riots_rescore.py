"""Re-score the external bell_hats_v2 Pd arrays to the edit 4/5 shot list.

bell_hats_v2's per-beat arrays were generated from the organic edit's cut list,
so its pattern changes, 32nd-note rolls and stabs fall on that edit's cuts. This
rebuilds the six per-beat arrays from CounterpointScore's 76 cues with the same
rules: a bar pattern per shot chosen by shot length, a roll on the beat before
some cuts (likelier before short shots), stabs before some cuts or in bar two of
long shots, cowbell per shot, a sparse opening and a thinning tail.

The patch, its voices and its bar patterns are copied unchanged into a new folder,
so bell_riots_mix.py accepts the result as ``--stem-root``. The supplied folder is
never modified.

    python docs/bell_riots_rescore.py --patch-dir /path/to/bell_hats \\
      --out /path/to/bell_hats_edit5 [--bar-offset 2] [--roll-into-bridges]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PERIOD=60/137.
BEAT0=-.008301162003440554          # the score's beat 0, relative to the edit
DURATION=361.093
BRIDGE=4                            # CounterpointScore mode for Unfolding bridges
BY_LENGTH={2:(1,4),4:(2,4,1),8:(1,2,0),16:(0,3,2)}
CHORDS=(0,3,0,2,1,0,3,2)            # fmstab~ voicings: Dm7, Bbmaj7, Am7, F add9
SCORE_ARRAYS=('hv','fill','stab','chord','gain','cb')
BAR_PATTERNS=(('closed','hclosed'),('open','hopen'),('cowpat','cowpat'))


def beat_of(t):
    return int(round((t-BEAT0)/PERIOD))


def score_cues(duration=DURATION):
    """(start beat, end beat, mode) for every cue, from the renderer's own score code."""
    sys.path[:0]=[str(ROOT),str(ROOT/'docs'),str(ROOT/'patches')]
    from bell_riots_counterpoint import CounterpointScore
    kicks=BEAT0+np.arange(1,826)*PERIOD   # the shot list depends only on beat numbers
    s=CounterpointScore(dict(duration=duration),kicks,np.ones_like(kicks))
    return [(beat_of(c['start']),beat_of(c['end']),int(c['mode'])) for c in s.cues]


def file_cues(path):
    """Cues from a render's .cues.json sidecar."""
    cues=json.loads(Path(path).read_text())['cues']
    return [(beat_of(c['start']),beat_of(c['end']),int(c['mode'])) for c in cues]


def score(cues,n=826,roll_into_bridges=False):
    """Per-beat arrays for bell_hats.pd; beat b starts at BEAT0+b*PERIOD, b=1 is the first heard.

    The random draws follow bell_hats_v2's generator (seeds 137 and 808, same order),
    so only the cut list differs from the supplied score.
    """
    starts=sorted({a for a,_,_ in cues if 1<a<n})
    bounds=[1]+starts+[n]
    bridges={a for a,_,m in cues if m==BRIDGE}
    shots=list(zip(bounds[:-1],bounds[1:]))
    rng=np.random.default_rng(137);crng=np.random.default_rng(808)
    hv,fill,stab,chord,cb=(np.zeros(n) for _ in range(5));gain=np.ones(n)
    nstab=0
    for i,(a,b) in enumerate(shots):
        length=b-a
        choices=BY_LENGTH.get(length,(2,0) if length>8 else (1,4))
        hv[a:b]=choices[(i*7+i//3)%len(choices)]
        nxt=shots[i+1][1]-shots[i+1][0] if i+1<len(shots) else 16
        if b<n:
            draw=rng.random()<(.55 if nxt<=4 else .25)
            if draw or (roll_into_bridges and b in bridges):fill[b-1]=1
        if b<n and rng.random()<.3:
            stab[b-1]=2 if rng.random()<.6 else 3
            chord[b-1]=CHORDS[nstab%len(CHORDS)];nstab+=1
        elif length>=16 and rng.random()<.5:
            stab[a+5]=2;chord[a+5]=CHORDS[nstab%len(CHORDS)];nstab+=1
    for a,b in shots:
        if a<33:continue
        busy=hv[a] in (1,2,4);r=crng.random()
        if busy and r<.75:cb[a:b]=1 if crng.random()<.65 else 2
        elif not busy and r<.35:cb[a:b]=2
    hv[:9]=0;gain[:5]=np.linspace(.4,1,5);gain[n-9:]=np.linspace(1,0,9);stab[n-9:]=0;cb[n-9:]=0
    return dict(hv=hv,fill=fill,stab=stab,chord=chord,gain=gain,cb=cb),shots


def rotate(pattern,beats):
    """Shift 16-step bar patterns so their first beat falls on score beats b = beats (mod 4)."""
    p=np.asarray(pattern,dtype=float).reshape(-1,16)
    return np.roll(p,4*beats,axis=1).ravel()


def table_sizes(patch_text):
    return {m[1]:int(m[2]) for m in re.finditer(r'#X obj -?\d+ -?\d+ table (\w+) (\d+);',patch_text)}


def write_array(path,values):
    path.write_text(' '.join(f'{v:g}' for v in values)+'\n')


def rescore(patch_dir,out,cues,bar_offset=0,roll_into_bridges=False):
    patch_dir,out=Path(patch_dir),Path(out)
    if out.resolve()==patch_dir.resolve():raise ValueError('write the re-score to a new folder')
    sizes=table_sizes((patch_dir/'bell_hats.pd').read_text())
    missing=[t for t in SCORE_ARRAYS+tuple(t for _,t in BAR_PATTERNS) if t not in sizes]
    if missing:raise ValueError('not the bell_hats_v2 patch; missing tables: '+', '.join(missing))
    n=sizes['hv']
    if any(sizes[t]!=n for t in SCORE_ARRAYS):raise ValueError('per-beat tables differ in size')
    out.mkdir(parents=True,exist_ok=True)
    for f in patch_dir.iterdir():
        if f.is_file() and f.suffix in ('.pd','.txt'):shutil.copy2(f,out/f.name)
    arrays,shots=score(cues,n,roll_into_bridges)
    for name in SCORE_ARRAYS:write_array(out/f'{name}.txt',arrays[name])
    for name,table in BAR_PATTERNS:
        values=np.array((patch_dir/f'{name}.txt').read_text().split(),dtype=float)
        if len(values)!=sizes[table] or len(values)%16:raise ValueError(f'{name}.txt does not fit table {table}')
        write_array(out/f'{name}.txt',rotate(values,bar_offset))
    rolls=np.where(arrays['fill']>0)[0]
    summary=dict(source=str(patch_dir),beats=n,period=PERIOD,beat0=BEAT0,bar_offset=bar_offset,
                 roll_into_bridges=roll_into_bridges,shots=len(shots),cut_beats=[b for _,b in shots[:-1]],
                 rolls=len(rolls),rolls_into_cuts=int(sum(r+1 in {b for _,b in shots} for r in rolls)),
                 stabs=int((arrays['stab']>0).sum()),cowbell_beats=int((arrays['cb']>0).sum()))
    (out/'rescore.json').write_text(json.dumps(summary,indent=1)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--patch-dir',type=Path,required=True,help='unzipped bell_hats_v2 folder')
    p.add_argument('--out',type=Path,required=True,help='new folder for the re-scored patch')
    p.add_argument('--cues',type=Path,help="a render's .cues.json; default: CounterpointScore's shot list")
    p.add_argument('--bar-offset',type=int,default=0,choices=range(4),
                   help='score beat (mod 4) where bar patterns start; 0 keeps v2, 2 matches most edit 4/5 cuts')
    p.add_argument('--roll-into-bridges',action='store_true',help='always roll into the Unfolding bridges')
    a=p.parse_args()
    cues=file_cues(a.cues) if a.cues else score_cues()
    s=rescore(a.patch_dir,a.out,cues,a.bar_offset,a.roll_into_bridges)
    print(f"{s['shots']} shots | {s['rolls']} rolls, all into cuts: {s['rolls_into_cuts']==s['rolls']} | "
          f"{s['stabs']} stabs | cowbell on {s['cowbell_beats']} beats | bar offset {s['bar_offset']} -> {a.out}")


if __name__=='__main__':main()
