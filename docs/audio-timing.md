# Optical audio timing regression

The original `0e55d0b` implementation rounded each optical frame independently.
At 24 fps and 44,100 Hz it produced 1,838 samples per frame: 7,200 excess samples
(0.163265 seconds) over ten minutes. Independently rounding those chunks to
48,000 Hz produced 2,001 samples per frame: 14,400 excess samples (0.300 seconds).
These are arithmetic predictions, not measurements of an ES-3 performance.

Commit `1fcecd4` already corrected cumulative sample counts. This follow-up keeps
that correction and addresses waveform timing: fitting every chunk separately
to its integer length still changed the sampling phase at each seam. On the
previous main branch, splitting the same 997 Hz tone into different chunks changed
the resampled waveform by up to 0.054 full scale (44.1 to 48 kHz) and 0.081
(48 to 44.1 kHz) in the deterministic regression fixture. A continuous 317 Hz
optical sine had RMS errors of 0.00738 at 24 fps/44.1 kHz and 0.00903 at
18 fps/48 kHz against a uniformly sampled reference.

## Correction

- Assign optical samples by their centres on the cumulative clock. Carry the
  fractional boundary into pixel interpolation as well as the chunk length.
- Resample on one continuous destination grid, retaining the preceding source
  sample for interpolation across boundaries. Counts use integer cumulative
  nearest-even rounding. Changing chunk sizes no longer changes the waveform.
- Use the same conversion for `CvOut` WAV recording and live callbacks.
- Clearing or rewinding a soundtrack resets queued audio and conversion state,
  including any partially consumed warm-up chunk.

Conversion has a fixed half-input-sample delay (about 11 microseconds at
44.1 kHz), allowing causal interpolation without waiting for the next chunk.
Equal-rate audio passes through unchanged. Linear interpolation remains the
existing quality level; no anti-aliasing filter or external DSP dependency was
added. The stateful high-pass filter, underrun repetition of the last complete
chunk, six-chunk backlog limit, and painted oscillator patch remain in place.

## Deterministic checks

Run `python -m pytest tests/test_audio_timing.py`. NumPy is required; GLSL checks
also need ModernGL/a headless GL context. Encoded synchronization checks need
`ffmpeg` and `ffprobe`; unavailable optional paths are explicitly skipped.

The twelve long simulations cross each optical rate (18, 24, 30 fps) with:

| Source → output | Presentation fps | Simulated duration |
| --- | ---: | ---: |
| 44,100 → 48,000 Hz | 60 | 600 seconds |
| 48,000 → 44,100 Hz | 12 | 1,800 seconds |
| 44,100 → 44,100 Hz | 24 | 1,800 seconds |
| 48,000 → 48,000 Hz | 30 | 600 seconds |

Every optical boundary must stay within half a source sample; resampled bounds
include the source quantization plus half an output sample. Final sample counts
must be exact, with zero accumulated drift. First/last-second FFT peaks must
remain at the drawn 180/240/300 Hz pitch and their waveforms must agree.
These tests stream and discard audio with `record=False`; retaining a complete
long recording requires setting `keep` to its length (default: 1,200 seconds).

Two real ten-minute MP4 exports exercise NumPy at 24 fps/44.1 kHz and GLSL at
30 presentation fps/18 fps frame hold/48 kHz. They check source PCM sample counts,
video frame counts, stream durations and starts, and a visible/audible midpoint
marker after decoding. AAC padding is distinguished from source PCM duration.
Additional tests cover split-versus-whole conversion, a continuous optical sine,
filter history, missing pictures, rewind/clear, callback block boundaries, held
waveforms, and both painted oscillator voices on NumPy and GLSL.

These are software executions with simulated presentation timestamps, including
software OpenGL where available. They do not measure ES-3 hardware-clock drift,
device latency, scheduler jitter, or sustained real-time throughput.
