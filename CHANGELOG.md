# Changelog

## 0.1.1 — 2026-10-28

- Keep the full rendered video and audio duration in Python, CLI and Studio
  exports, including with FFmpeg 4.4.2. Pad short audio with silence and cut long
  audio at the video end instead of relying on `-shortest`.
- Exercise the application mux in A/V prerequisites, with missing tools/codecs
  still skipped and application regressions reported as failures.
