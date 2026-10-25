"""JACK transport for TrackerCV, including exact signed-24 PCM ES-40 words.

No gain, mixing, resampling or dither is applied. This reuses TrackerCV's
existing soft-realtime callback evaluator and zero-voltage shutdown sequence.
"""
from types import SimpleNamespace
import time

import numpy as np


class JackCVStream:
    def __init__(self, owner, device, samplerate, channels, dtype):
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError('For JACK, device is a client name, not a PortAudio index')
        try:
            import jack
        except ImportError as exc:
            raise RuntimeError('Install the jack extra: pip install -e ".[jack]"') from exc
        self.owner, self.dtype = owner, dtype
        self.client = jack.Client(device or 'GorgonCV', no_start_server=True)
        self._active = False
        self.buffer = None
        self.ports = []
        try:
            actual = int(self.client.samplerate)
            if samplerate is not None and actual != samplerate:
                raise ValueError(f'JACK runs at {actual} Hz; the CV profile requires {samplerate} Hz (no resampling)')
            owner.samplerate = actual
            self.ports = [self.client.outports.register(f'out_{c:02d}') for c in range(channels)]
            self.client.set_process_callback(self._process)
            self.client.set_xrun_callback(self._xrun)
            self.client.set_shutdown_callback(self._shutdown)
        except BaseException:
            self.client.close()
            raise

    @property
    def active(self):
        return self._active

    @property
    def latency(self):
        # One software period; physical routing latency is calibrated with CV offset.
        return self.client.blocksize / self.owner.samplerate

    def _xrun(self, delayed):
        self.owner.underflows += 1
        self.owner.status = 'JACK xrun'

    def _shutdown(self, status, reason):
        self._active = False
        self.owner.error = 'JACK CV server stopped: ' + str(reason)

    def _process(self, frames):
        owner = self.owner
        arrays = [p.get_array() for p in self.ports]
        for out in arrays:
            out.fill(0)
        try:
            if self.buffer is None or len(self.buffer) != frames:
                self.buffer = np.zeros((frames, len(arrays)), dtype=self.dtype)
            if int(self.client.samplerate) != owner.samplerate:
                owner.error = 'JACK CV sample rate changed; outputs cleared. Restart with a matching profile.'
                owner._silencing = True
            cycle = time.monotonic() - self.client.frames_since_cycle_start / owner.samplerate
            owner._callback(self.buffer, frames, SimpleNamespace(monotonic_start=cycle + self.latency), None)
            for c, out in enumerate(arrays):
                # Every signed 24-bit word shifted left by 8 is exactly
                # representable in float32 after this power-of-two scaling.
                if self.dtype == 'int32':
                    out[:] = self.buffer[:, c].astype(np.float32) * np.float32(1 / 2147483648)
                else:
                    out[:] = self.buffer[:, c]
        except Exception as exc:
            owner.error = f'JACK CV callback failed: {exc}'
            for out in arrays:
                out.fill(0)

    def start(self):
        self.client.activate()
        self._active = True

    def stop(self):
        try:
            if self._active and self.owner._silenced:
                # Let downstream clients consume the final zero-voltage words
                # before withdrawing this client from the processing graph.
                time.sleep(min(.1, 2 * self.latency))
            self.client.deactivate()
        finally:
            self._active = False

    def close(self):
        self._active = False
        self.client.close()
