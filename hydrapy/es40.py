"""ES-40 + ESX-8GT/8CV transport, compatible with Expert Sleepers' Rack encoders.

Protocol adapted from src/8GT.cpp, src/8CV.cpp and src/ES40.cpp at
https://github.com/expertsleepersltd/vcvrack-encoders/tree/95496e8a955407889bbab94cf404cf356802bb76
The following upstream license covers this adaptation:

The MIT License (MIT)

Copyright (c) 2019 Expert Sleepers Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from .tracker_cv import SIGNALS, TrackerCV, parse_map

GATE_SIGNALS = ('clock', 'reset', 'run', *(f'gate{i}' for i in range(1, 7)))
POLARITIES = ('normal', 'invert_left', 'invert_right', 'invert_both')


def flip_pcm24(pcm, mode):
    """Polarity compensation AFTER encoding, at 24-bit PCM precision.

    Saturation is explicit: +8388608 cannot be represented by a signed 24-bit
    device. Count this endpoint so a user can distinguish compensation from a
    proven bit-preserving path. Ordinary voltages and audio never enter here.
    """
    if mode not in POLARITIES:
        raise ValueError('Choose normal, invert_left, invert_right or invert_both')
    if mode == 'normal':
        return pcm, 0
    out = pcm.copy()
    clipped = 0
    for column in ((0, 1) if mode == 'invert_both' else (0,) if mode == 'invert_left' else (1,)):
        words = -(pcm[:, column].astype(np.int64) >> 8)
        clipped += int(np.count_nonzero(words == 8388608))
        out[:, column] = (np.clip(words, -8388608, 8388607) << 8).astype(np.int32)
    return out, clipped


class Encoder8GT:
    """One bit per gate: jack 1 is bit 0, jack 8 is bit 7, every sample.

    Match Rack's hysteresis: set at >= 1 V, clear below 0.5 V, otherwise hold.
    """
    def __init__(self):
        self.bits = np.zeros(8, dtype=bool)

    def encode(self, volts):
        if not len(volts):
            return np.empty(0, np.int32)
        volts = np.asarray(volts, np.float32)
        high = volts >= 1.
        changed = high | (volts < .5)
        index = np.maximum.accumulate(np.where(changed, np.arange(len(volts))[:, None], -1), axis=0)
        bits = np.where(index >= 0, high[np.maximum(index, 0), np.arange(8)], self.bits)
        self.bits = bits[-1].copy()
        return np.sum(bits * (1 << np.arange(8)), axis=1, dtype=np.int32)


class Encoder8CV:
    """Three bytes per DAC, eight DACs per cycle; state survives audio blocks.

    Factory calibrated mode: 384 counts/volt, 12-bit offset binary. All eight
    outputs update every 24 audio samples. Sample a voltage only at packet start.
    """
    def __init__(self, *, coherent=False):
        self.phase = 0
        self.value = 2048
        self.coherent = coherent
        self.cycle = np.zeros(8, np.float32)

    def encode(self, volts):
        frames = len(volts)
        if not frames:
            return np.empty(0, np.int32)
        phase = (np.arange(frames) + self.phase) % 24
        volts = np.asarray(volts, np.float32)
        if self.coherent:
            # A note can change between a pitch packet and its gate packet.
            # Latch all eight CVs together so a later-numbered gate jack never
            # opens with the previous cycle's pitch. Wire pitch before gate.
            index = np.maximum.accumulate(np.where(phase == 0, np.arange(frames), -1))
            volts = np.where(index[:, None] >= 0, volts[np.maximum(index, 0)], self.cycle)
            self.cycle = volts[-1].copy()
        stage, dac = phase % 3, phase // 3
        first = stage == 0
        values = np.zeros(frames, np.int32)
        # float32 and truncation match the manufacturer's reference encoder.
        values[first] = 2048 + np.clip(volts[first, dac[first]] * 384,
                                      -2048, 2047).astype(np.int32)
        index = np.maximum.accumulate(np.where(first, np.arange(frames), -1))
        values = np.where(index >= 0, values[np.maximum(index, 0)], self.value)
        self.value = int(values[-1])
        self.phase = int((self.phase + frames) % 24)
        return np.where(stage == 0, 0x80 | (values & 31),
                        np.where(stage == 1, (values >> 5) & 31,
                                 np.where(dac > 3, 0x40, 0x20) | (values >> 10) | ((dac & 3) << 2)))


def pack_es40(headers):
    """Five header bytes → stereo signed int32, 24 significant bits at the top.

    Integer PortAudio output bypasses float-to-PCM device categories entirely.
    No gain, dither, resampling or mixing may follow these data words.
    """
    # A header byte outside 0..255 would spill into its neighbours' bits.
    b = np.clip(np.asarray(headers, dtype=np.int64), 0, 255).astype(np.uint32)
    left = (b[:, 0] << 16) | (b[:, 1] << 8) | (b[:, 4] & 0xf0)
    right = (b[:, 2] << 16) | (b[:, 3] << 8) | ((b[:, 4] << 4) & 0xf0)
    return (np.column_stack((left, right)) << 8).astype(np.int32)


def channel(value):
    if type(value) is not int or not 0 <= value < 256:
        raise ValueError('Audio channels must be zero-based integers from 0 to 255')
    return value


class ES40CV(TrackerCV):
    """One audio device, optionally multiple ES-40 pairs plus direct DC outputs.

    The score evaluator runs once. Each configured ES-40 has one ESX-8GT or
    ESX-8CV with independent encoder state. Use CVGroup for separate devices.
    """
    def __init__(self, units, *, direct=None, device=None, samplerate=48000,
                 blocksize=256, open_stream=True, **options):
        if type(samplerate) is not int or samplerate not in (44100, 48000):
            raise ValueError('ES-40 transport supports 44100 or 48000 Hz; match the hardware clock')
        if not isinstance(units, list) or not units:
            raise ValueError('At least one ES-40 unit is required')
        self.direct = parse_map(direct) if direct else {}
        used = {channel(c) for c in self.direct}
        self.units = []
        signals = list(dict.fromkeys(self.direct.values()))
        for unit in units:
            if not isinstance(unit, dict) or set(unit) - {'name', 'channels', 'header', 'map', 'expander', 'polarity'}:
                raise ValueError('ES-40 entries use name, channels, header, expander, map and optional polarity')
            polarity = unit.get('polarity', 'normal')
            if polarity not in POLARITIES:
                raise ValueError('Invalid ES-40 polarity mode')
            expander = unit.get('expander', '8cv')  # preserve existing CV profiles
            if expander not in ('8gt', '8cv'):
                raise ValueError('ES-40 expander must be 8gt or 8cv')
            pair, header, mapping = unit.get('channels'), unit.get('header'), unit.get('map')
            if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError('Each ES-40 needs two distinct S/PDIF audio channels [left, right]')
            pair = [channel(c) for c in pair]
            if used.intersection(pair):
                raise ValueError('ES-40 and direct outputs cannot share audio channels')
            used.update(pair)
            if type(header) is not int or not 1 <= header <= 5:
                raise ValueError('ES-40 expansion header must be 1–5')
            if not isinstance(mapping, dict) or not mapping or any(str(k) not in tuple(map(str, range(1, 9))) or v not in SIGNALS for k, v in mapping.items()):
                raise ValueError('Expander map uses jack numbers 1–8 and tracker signal names')
            if expander == '8gt' and any(v not in GATE_SIGNALS for v in mapping.values()):
                raise ValueError('ESX-8GT carries gates only: use clock, reset, run or gate1–gate6; route pitch CV to a CV output')
            name = unit.get('name', f'ES-40 {len(self.units) + 1}')
            if not isinstance(name, str) or not 1 <= len(name) <= 80:
                raise ValueError('ES-40 name must contain 1–80 characters')
            self.units.append(dict(name=name, channels=pair, header=header, expander=expander, polarity=polarity,
                                   map={int(k): v for k, v in mapping.items()}))
            signals.extend(v for v in mapping.values() if v not in signals)
        self.encoders = [Encoder8GT() if u['expander'] == '8gt' else Encoder8CV(coherent=True) for u in self.units]
        self.columns = {name: i for i, name in enumerate(signals)}
        self.channels = max(used) + 1
        super().__init__(dict(enumerate(signals)), samplerate=samplerate, open_stream=False, **options)
        self._silence_frames = 48 if any(u['expander'] == '8cv' for u in self.units) else 1
        self.device = device
        self.polarity_clips = [0] * len(self.units)
        if open_stream:
            self._open_stream(device, samplerate, blocksize, channels=self.channels, dtype='int32')

    def _write_volts(self, outdata, volts):
        # NaN would cast to INT_MIN and corrupt the packed words; infinity has
        # no meaning as a voltage either. Send 0 V for both. (The vendor's
        # encoder sends full scale for NaN; zero is the safer failure here.)
        volts = np.where(np.isfinite(volts), volts, 0.)
        for index, (unit, encoder) in enumerate(zip(self.units, self.encoders)):
            cv = np.zeros((len(volts), 8), np.float32)
            for jack, signal in unit['map'].items():
                cv[:, jack - 1] = volts[:, self.columns[signal]]
            headers = np.zeros((len(volts), 5), np.int32)
            headers[:, unit['header'] - 1] = encoder.encode(cv)
            pcm, clipped = flip_pcm24(pack_es40(headers), unit['polarity'])
            self.polarity_clips[index] += clipped
            outdata[:, unit['channels']] = pcm
        for output, signal in self.direct.items():
            value = volts[:, self.columns[signal]] / self.volts_per_fs
            outdata[:, output] = np.clip(value * 2147483648., -2147483648, 2147483647).astype(np.int32)

    def describe(self):
        return dict(device=self.device, es40=[dict(u, polarity_clips=self.polarity_clips[i]) for i, u in enumerate(self.units)],
                    polarity_routes=self.polarity_routes(), direct=self.direct, samplerate=self.samplerate,
                    backend=self.backend, ports=[p.name for p in getattr(self.stream, 'ports', [])],
                    error=self.error, status=self.status, underflows=self.underflows)

    def polarity_routes(self):
        return [dict(id=json.dumps([self.device, *u['channels']], separators=(',', ':')),
                     name=f"{u['name']} · {self.device} · channels {u['channels'][0]}/{u['channels'][1]}",
                     mode=u['polarity'], clipped=self.polarity_clips[i]) for i, u in enumerate(self.units)]

    def set_polarity(self, route, mode):
        if mode not in POLARITIES:
            raise ValueError('Invalid polarity mode')
        for item, unit in zip(self.polarity_routes(), self.units):
            if item['id'] == route:
                old = unit['polarity']
                unit['polarity'] = mode
                return old
        raise ValueError('That ES-40 output pair is not active')


class CVGroup:
    def __init__(self, outputs):
        self.outputs = outputs

    def publish(self, *args):
        for output in self.outputs:
            output.publish(*args)

    def describe(self):
        devices = [output.describe() for output in self.outputs]
        return dict(devices=devices, polarity_routes=self.polarity_routes(), ports=[p for d in devices for p in d['ports']],
                    error='; '.join(d['error'] for d in devices if d['error']),
                    underflows=sum(d['underflows'] for d in devices))

    def polarity_routes(self):
        return [r for output in self.outputs for r in output.polarity_routes()]

    def set_polarity(self, route, mode):
        for output in self.outputs:
            if any(r['id'] == route for r in output.polarity_routes()):
                return output.set_polarity(route, mode)
        raise ValueError('That ES-40 output pair is not active')

    def close(self):
        # Always close all devices, even if one driver fails during shutdown.
        error = None
        for output in self.outputs:
            try:
                output.close()
            except Exception as exc:
                error = exc
        if error:
            raise error


def load_profile(path, *, open_stream=True, polarity_settings=None, **options):
    if polarity_settings is not None and (not isinstance(polarity_settings, dict) or
            any(v not in POLARITIES for v in polarity_settings.values())):
        raise ValueError('Invalid saved polarity settings')
    profile = json.loads(Path(path).read_text())
    if not isinstance(profile, dict) or set(profile) != {'format', 'devices'} or profile['format'] != 'hydrapy-cv-1':
        raise ValueError('Expected a hydrapy-cv-1 profile with devices')
    devices = profile['devices']
    if not isinstance(devices, list) or not 1 <= len(devices) <= 8:
        raise ValueError('Configure 1–8 audio devices')
    outputs, seen = [], set()
    # Validate the complete profile before opening any hardware.
    for config in devices:
        if not isinstance(config, dict) or set(config) - {'device', 'samplerate', 'es40', 'direct', 'backend'}:
            raise ValueError('Device entries use device, samplerate, es40 and optional direct/backend')
        device = config.get('device')
        if not ((type(device) is int and device >= 0) or (isinstance(device, str) and device.strip())):
            raise ValueError('Choose an explicit audio device name or index from hydrapy devices')
        if device in seen:
            raise ValueError('Combine units on the same device in one device entry')
        seen.add(device)
        if 'samplerate' not in config:
            raise ValueError('Set the hardware samplerate explicitly')
        prepared = dict(options, backend=config.get('backend', options.get('backend', 'portaudio')))
        if prepared['backend'] == 'jack' and not isinstance(device, str):
            raise ValueError('For JACK, device is a client name, not a PortAudio index')
        outputs.append(ES40CV(config.get('es40'), device=device, samplerate=config['samplerate'],
                             direct=config.get('direct'), open_stream=False, **prepared))
    group = CVGroup(outputs)
    for route in group.polarity_routes():
        if polarity_settings is not None and route['id'] in polarity_settings:
            group.set_polarity(route['id'], polarity_settings[route['id']])
    try:
        if open_stream:
            for output in outputs:
                output._open_stream(output.device, output.samplerate, 256, channels=output.channels, dtype='int32')
    except BaseException:
        group.close()
        raise
    return group


def diagnostic_wav(path, *, header=1, samplerate=48000, expander='8cv'):
    """18 seconds: off, then jacks 1..8 individually, with off intervals.

    A stereo 24-bit WAV for testing the digital path independently of Studio.
    8GT: one 5 V gate at a time. 8CV: one 1 V output at a time.
    Intended only for a bit-preserving S/PDIF route to the matching expander.
    """
    if type(header) is not int or not 1 <= header <= 5:
        raise ValueError('ES-40 expansion header must be 1–5')
    if samplerate not in (44100, 48000):
        raise ValueError('Use 44100 or 48000 Hz')
    if expander not in ('8gt', '8cv'):
        raise ValueError('Choose expander 8gt or 8cv')
    encoder = Encoder8GT() if expander == '8gt' else Encoder8CV()
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(3)
        wav.setframerate(samplerate)
        for step in range(18):
            cv = np.zeros((samplerate, 8), np.float32)
            if 1 <= step <= 16 and step % 2:
                cv[:, (step - 1) // 2] = 5. if expander == '8gt' else 1.
            headers = np.zeros((samplerate, 5), np.int32)
            headers[:, header - 1] = encoder.encode(cv)
            words = pack_es40(headers).astype('<i4').view(np.uint8).reshape(-1, 4)
            wav.writeframes(words[:, 1:].tobytes())
