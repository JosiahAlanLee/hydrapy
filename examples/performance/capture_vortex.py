"""Run in an environment with sardine-system and hydrapy installed.

Captures a finite pattern into editable JSON. Does not start Sardine's REPL,
SuperDirt, a clock, audio, or model inference.
"""
from pathlib import Path
import sys

from hydrapy.performance import Binding, Session, capture_vortex


def main():
    try:
        from sardine_core.sequences.tidal_parser import TimeSpan, n
    except ImportError as exc:
        raise SystemExit("Use your Sardine environment to run this optional example.") from exc
    root = Path(__file__).resolve().parent
    session = Session.from_json((root / "session.json").read_text(encoding="utf-8"))
    # These values are octave offsets for the drawn voice, not MIDI note numbers.
    pattern = n("0 0.5833333333 1 1.5833333333").slow(2)
    session.events = capture_vortex(
        pattern.query(TimeSpan(0, 2)),
        [Binding("picture", "/voice/pitch", ("n",))],
        cycles=2,
    )
    # Keep the visible voice's level explicit; no hardware/synthesis host is started.
    from hydrapy.performance import Event
    session.events = (Event(0, "picture", "/voice/level", (.4,)), *session.events)
    session.processes = ()
    # Revalidate the assembled score before saving.
    session = Session.from_json(session.to_json())
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "captured.json"
    with out.open("x", encoding="utf-8") as f:
        f.write(session.to_json())
    print(out)


if __name__ == "__main__":
    main()
