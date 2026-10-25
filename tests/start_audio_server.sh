#!/usr/bin/env bash
# CI-only isolated server. Invoke in a workflow step; cleanup via CI process teardown.
set -euo pipefail
backend=${1:?jack or pipewire}
if [ "$backend" = jack ]; then
  jackd --no-realtime -d dummy -r 48000 -p 256 > /tmp/gorgon-jack.log 2>&1 &
elif [ "$backend" = pipewire ]; then
  : "${XDG_RUNTIME_DIR:?set a private runtime directory}"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
  # The standard daemon configuration includes a dummy driver; no session manager
  # or physical device is needed for manually connected JACK clients.
  pipewire > /tmp/gorgon-pipewire.log 2>&1 &
else
  exit 2
fi
python - <<'PY'
import time
import jack
for attempt in range(60):
    try:
        with jack.Client('gorgon_ready', no_start_server=True):
            break
    except jack.JackError:
        time.sleep(.1)
else:
    raise SystemExit('Audio server did not become ready; inspect /tmp/gorgon-*.log')
PY
