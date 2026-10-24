"""Reduce a 480-frame RGBA Body Roll print to the delay ring's grayscale input."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames',type=Path);p.add_argument('-o','--out',type=Path,required=True)
    p.add_argument('--size',default='960x540')
    a=p.parse_args();size=tuple(map(int,a.size.split('x')))
    paths=[a.frames/f'{i:04}.png' for i in range(1,481)]
    if any(not x.is_file() for x in paths):p.error('Expected frames 0001.png through 0480.png')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    matte=np.lib.format.open_memmap(a.out,mode='w+',dtype=np.uint8,shape=(480,size[1],size[0]))
    coverage=[]
    for i,path in enumerate(paths):
        with Image.open(path) as im:
            if 'A' not in im.getbands():raise ValueError(f'No alpha channel: {path.name}')
            alpha=np.asarray(im.getchannel('A').resize(size,Image.Resampling.LANCZOS))
        if alpha.max()==0 or alpha.min()!=0:raise ValueError(f'Invalid empty/opaque matte: {path.name}')
        matte[i]=alpha;coverage.append(float(np.mean(alpha>127)))
    matte.flush()
    report=dict(shape=list(matte.shape),dtype=str(matte.dtype),coverage_min_max=[min(coverage),max(coverage)],
                full_beats_per_cycle=32,half_bpm=68.5,cycle_seconds=1920/137)
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
