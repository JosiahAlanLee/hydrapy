"""Mix the Bell Riots closing edit with the user's external bell_hats_v2 Pd score.

The supplied score and recording remain local inputs, outside this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt

SR = 48000
START = 434.46189569161


def db(gain):
    return 10 ** (gain / 20)


def shelf(x, sr=SR, frequency=105., gain=1.8):
    """RBJ low shelf, S=1, same coefficients on both channels."""
    a=10**(gain/40);w=2*np.pi*frequency/sr;c=np.cos(w)
    alpha=np.sin(w)/2*np.sqrt(2);term=2*np.sqrt(a)*alpha
    b=a*np.array([(a+1)-(a-1)*c+term,2*((a-1)-(a+1)*c),(a+1)-(a-1)*c-term])
    den=np.array([(a+1)+(a-1)*c+term,-2*((a-1)+(a+1)*c),(a+1)+(a-1)*c-term])
    return lfilter(b/den[0],den/den[0],x,axis=0).astype('f4')


def saturate(x, drive=6., blend=.40, reduction=-3.):
    """4x oversampled parallel soft saturation, RMS compensated before trim."""
    up=resample_poly(x,4,1,axis=0).astype('f4')
    wet=np.tanh(up*drive)
    energy=float(np.mean(up.astype('f8')**2))
    wet*=np.sqrt(energy/max(float(np.mean(wet.astype('f8')**2)),1e-20))
    up=(1-blend)*up+blend*wet
    return (resample_poly(up,1,4,axis=0)[:len(x)]*db(reduction)).astype('f4')


def instrument_patch(source, destination, duration=361.3):
    """Expose four stereo buses before the score's hard clips.

    Only the known top-level patch is adapted. The voices, score arrays,
    timing, seeded velocities, gates and ping-pong delay are preserved.
    """
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.pd','.txt','.json'):
            shutil.copy2(p,destination/p.name)
    s=(source/'bell_hats.pd').read_text()
    head=s[:s.index('#X connect')]
    if sum(line.startswith('#X ') for line in head.splitlines())!=109:
        raise ValueError('Expected the supplied bell_hats_v2 patch; object layout changed')
    for token in ('writesf~ 2','open -bytes 3 bell_hats_stem.wav','rev3~ 0 92 4000 25','rev3~ 0 90 3000 35'):
        if token not in s:raise ValueError('Unrecognized score: '+token)
    # Pd's rev3~ output level is 0..100 dB, with 0 muted. Restore the described
    # rooms at quiet return levels, then control the shared percussion room.
    s=s.replace('rev3~ 0 92 4000 25','rev3~ 82 92 4000 25').replace('rev3~ 0 90 3000 35','rev3~ 80 90 3000 35')
    s=s.replace('writesf~ 2','writesf~ 8').replace('open -bytes 3 bell_hats_stem.wav','open -bytes 4 bell_hats_buses.wav')
    s=s.replace('delay 361300;',f'delay {duration*1000:.6f};')
    s=s.replace('#X connect 95 0 98 0;\n','').replace('#X connect 96 0 98 1;\n','')
    head,tail=s.split('#X connect',1)
    head+=''.join(f'#X obj {20+i*100} 1180 *~ 0.65;\n' for i in range(8))
    s=head+'#X connect'+tail
    for i,nodes in enumerate(((62,),(62,),(64,),(65,),(74,86,91),(75,87,92),(72,),(73,))):
        for n in nodes:s+=f'#X connect {n} 0 {109+i} 0;\n'
        s+=f'#X connect {109+i} 0 98 {i};\n'
    (destination/'bell_hats_buses.pd').write_text(s)
    return destination/'bell_hats_buses.pd'


def control_envelope(x, sr=SR, band=None):
    mono=np.mean(x,axis=1)
    if band is not None:mono=sosfilt(butter(3,band,btype='bandpass',fs=sr,output='sos'),mono)
    hop=sr//200
    rms=np.sqrt(np.maximum(0,uniform_filter1d(mono*mono,round(.008*sr))))[::hop]
    # A measured attack, followed by a short visual hold; suppresses room tails.
    novelty=np.maximum(rms-np.r_[np.zeros(6),rms[:-6]],0)
    reference=max(float(np.percentile(novelty[novelty>1e-7],92)) if np.any(novelty>1e-7) else 1.,1e-6)
    return np.clip(maximum_filter1d(novelty/reference,5,origin=2),0,1.25).astype('f4')


def loudness(path):
    r=subprocess.run(['ffmpeg','-hide_banner','-i',str(path),'-vn','-af',
                      'loudnorm=I=-16:TP=-1.2:LRA=8:print_format=json','-f','null','-'],
                     capture_output=True,text=True,check=True)
    return json.loads(r.stderr[r.stderr.rfind('{'):r.stderr.rfind('}')+1])


def make_mix(audio, stem_root, pd, out, cache, start=START, extra=None):
    audio,stem_root,out,cache=map(Path,(audio,stem_root,out,cache))
    cache.mkdir(parents=True,exist_ok=True);out.parent.mkdir(parents=True,exist_ok=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-of','json',str(audio)]))
    duration=float(info['format']['duration'])-start
    if not 0<duration<=361.31:raise ValueError('This score covers the closing six-minute edit')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stem_root.iterdir() if p.suffix in ('.pd','.txt','.json')}
    fingerprint=dict(audio=hashlib.sha256(audio.read_bytes()).hexdigest(),score=hashes,start=start,sr=SR,version=1)
    patchdir=cache/'pd';raw=patchdir/'bell_hats_buses.wav';stamp=cache/'inputs.json'
    if not raw.exists() or not stamp.exists() or json.loads(stamp.read_text())!=fingerprint:
        patch=instrument_patch(stem_root,patchdir)
        command=[str(pd),'-nogui','-batch','-noprefs','-noaudio','-r',str(SR)]
        if extra:command+=['-path',str(extra)]
        command+=['-send','render bang',patch.name]
        r=subprocess.run(command,cwd=patchdir,capture_output=True,text=True,timeout=180)
        (cache/'pd.log').write_text(r.stdout+r.stderr)
        if r.returncode or 'error:' in r.stderr.lower() or "couldn't create" in r.stderr.lower():
            raise RuntimeError('Pd render failed; see '+str(cache/'pd.log'))
        stamp.write_text(json.dumps(fingerprint,indent=2))
    original=cache/'original.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-t',str(duration),'-i',str(audio),
                    '-vn','-ar',str(SR),'-ac','2','-c:a','pcm_f32le',str(original)],check=True)
    sr,x=wavfile.read(original);bsr,buses=wavfile.read(raw,mmap=True)
    n=len(x)
    if sr!=SR or bsr!=SR or buses.shape[1]!=8 or len(buses)<n:raise ValueError('Unexpected Pd render shape')
    if not np.isfinite(buses[:n]).all() or np.max(abs(buses[:n]))>3:raise ValueError('Invalid Pd samples')
    print(f'{duration:.6f}s; {n} samples; rendering four separate stereo buses',flush=True)
    original_metrics=loudness(original)
    # Gain changes are relative to the supplied patch's bus levels.
    base=shelf(sosfilt(butter(2,24,btype='highpass',fs=SR,output='sos'),x,axis=0).astype('f4'))
    hats=np.array(buses[:n,:2])*db(-1.)
    cow=saturate(np.array(buses[:n,2:4]),drive=7.,blend=.45,reduction=-3.5)
    stabs=saturate(np.array(buses[:n,4:6]),drive=9.,blend=.40,reduction=-3.)
    room=np.array(buses[:n,6:8])*db(-6.)
    controls=dict(hat=control_envelope(hats,band=(3500,16000)),cow=control_envelope(cow),
                  stab=control_envelope(stabs),rate=200.,duration=n/SR,source_start=start)
    y=base+hats+cow+stabs+room
    controls['high']=control_envelope(y,band=(4000,16000))
    np.savez_compressed(cache/'upper-controls.npz',**controls)
    # Short edge fades prevent a cut through a running oscillator or delay tail.
    fade=np.ones(n,dtype='f4');a=round(.008*SR);b=round(.35*SR)
    fade[:a]=np.linspace(0,1,a);fade[-b:]=np.linspace(1,0,b)
    y*=fade[:,None]
    pre=cache/'premaster.wav';wavfile.write(pre,SR,y.astype('f4'))
    measured=loudness(pre)
    # A constant gain target, followed by a compensated true-peak safety stage.
    # Avoid forcing loudness if it would require more than ~1 dB peak reduction.
    gain=min(-16.-float(measured['input_i']),-.2-float(measured['input_tp']))
    filt=f'volume={gain:.8f}dB,aresample=192000,alimiter=limit={db(-1.25):.10f}:attack=5:release=85:level=false:latency=true,aresample=48000'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(pre),'-af',filt,'-c:a','flac','-sample_fmt','s32','-bits_per_raw_sample','24',
                    '-metadata','artist=Anchor Rite','-metadata','title=Bell Riots Pt 2 — Counterpoint Mix',str(out)],check=True)
    final=loudness(out)
    if float(final['input_tp'])>-.9:raise RuntimeError('Master exceeded the true-peak ceiling')
    report=dict(**fingerprint,source_end=start+n/SR,duration=n/SR,output=str(out.name),
                original_loudness=original_metrics,pre_loudness=measured,final_loudness=final,master_gain_db=gain,
                bass_shelf_db=1.8,bass_shelf_hz=105.,hat_db=-1.,cowbell_db=-3.5,stabs_db=-3.,room_db=-6.,saturation_oversample=4,
                bus_peak=[float(np.max(abs(buses[:n,i:i+2]))) for i in (0,2,4,6)],
                controls_sha256=hashlib.sha256((cache/'upper-controls.npz').read_bytes()).hexdigest())
    out.with_suffix('.mix.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('duration','master_gain_db','final_loudness','bus_peak')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio',type=Path);p.add_argument('--stem-root',type=Path,required=True)
    p.add_argument('--pd',default='pd');p.add_argument('--pd-extra',type=Path)
    p.add_argument('-o','--out',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--source-start',type=float,default=START)
    a=p.parse_args();make_mix(a.audio,a.stem_root,a.pd,a.out,a.cache,a.source_start,a.pd_extra)
