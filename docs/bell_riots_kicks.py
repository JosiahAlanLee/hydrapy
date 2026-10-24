"""Offline low-frequency onset extraction for the Bell Riots closing edit."""
from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


def detect_kicks(samples, sr=8000, period=60/137, anchors=None):
    """Estimate kick attacks from 30–155 Hz energy; never invent missing hits.

    Beat anchors restrict the early edge of the attack search, preventing bass
    notes before a kick from moving its trigger. Peaks still require measured
    low-frequency novelty. This is an onset detector, not stem separation.
    """
    if len(samples)<sr//4 or np.max(np.abs(samples),initial=0)<1e-7:
        return np.array([]),np.array([])
    filtered=sosfiltfilt(butter(4,[30,155],fs=sr,btype="bandpass",output="sos"),samples)
    hop=round(sr/200); rate=sr/hop
    env=np.sqrt(np.maximum(0,uniform_filter1d(filtered*filtered,max(1,round(sr*.008)))))[::hop]
    lag=round(rate*.04)
    novelty=np.maximum(env-np.r_[np.full(lag,env[0]),env[:-lag]],0)
    reference=float(np.percentile(novelty,85))
    floor=max(1e-6,float(env.max())*.012)
    peaks,_=find_peaks(novelty,distance=max(1,round(period*.55*rate)),
                       prominence=max(floor,reference*.35),height=floor)
    times=[];strength=[]
    for i in peaks:
        lo=max(0,i-round(.12*rate))
        if anchors is not None and len(anchors):
            beat=float(anchors[np.argmin(abs(anchors-i/rate))])
            if abs(beat-i/rate)<period*.32:
                lo=max(lo,min(i,int(np.ceil((beat-.025)*rate))))
        window=env[lo:i+1]; baseline=float(window.min())
        threshold=baseline+.20*(env[i]-baseline)
        crossed=np.flatnonzero((window>=threshold)&(novelty[lo:i+1]>=novelty[i]*.10))
        onset=(lo+crossed[0])/rate if len(crossed) else i/rate
        times.append(onset);strength.append(novelty[i])
    times=np.asarray(times,dtype=float);strength=np.asarray(strength,dtype=float)
    if len(strength):strength=np.clip(strength/max(np.percentile(strength,90),1e-6),0,1)
    return times,strength


def analyse_kicks(audio, data, start, cache):
    duration=float(data["duration"])-start; digest=str(data["sha256"])
    cache=Path(cache)
    if cache.exists():
        with np.load(cache) as old:
            if (str(old["sha256"])==digest and float(old["start"])==start
                    and float(old["duration"])==duration and int(old["version"])==1):
                return old["times"].copy(),old["strength"].copy()
    raw=subprocess.check_output(["ffmpeg","-v","error","-ss",str(start),"-t",str(duration),
                                 "-i",str(audio),"-ac","1","-ar","8000","-f","f32le","-"])
    anchors=np.asarray(data["beats"])-start
    times,strength=detect_kicks(np.frombuffer(raw,np.float32),period=60/float(data["tempo"]),anchors=anchors)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,sha256=digest,start=start,duration=duration,version=1,times=times,strength=strength)
    return times,strength


class KickClock:
    """Eight cellular generations between kicks; 32/64-step folds stay intact."""
    def __init__(self,times,strength,rate=30,duration=None):
        self.times=np.asarray(times);self.strength=np.asarray(strength);self.rate=rate
        self.fold_ticks=set();self.fold_generations=set();self.drop_generations=set();self.fold_times=[]
        generation_times=[];count=0;folds=0;every=4
        period=float(np.median(np.diff(self.times))) if len(self.times)>1 else .44
        for i,t in enumerate(self.times):
            end=float(self.times[i+1]) if i+1<len(self.times) else t+period
            generation_times.extend(float(t+(end-t)*j/8) for j in range(8))
            if i==0 or count>=every:
                count=0;folds+=1;tick=int(np.ceil(t*rate-1e-9))
                self.fold_ticks.add(tick);self.fold_times.append(float(t))
                self.fold_generations.add(i*8)
                if folds%6==0:self.drop_generations.add(i*8)
                every=(4,4,8,4,8,8,4,8)[int(t/53)%8]
            count+=1
        self.generation_times=np.asarray(generation_times)
        if duration is not None:self.generation_times=self.generation_times[self.generation_times<duration]

    def pulse(self,t):
        i=int(np.searchsorted(self.times,t,side="right")-1)
        if i<0:return 0.
        return float((.74+.26*self.strength[i])*np.exp(-(t-self.times[i])/.105))
