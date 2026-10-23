"""Render a captioned 30-second tour of the new animated generative sources.

    python docs/generative_demo.py --out generative_demo.mp4
    python docs/generative_demo.py --seconds 3 --size 640x360 --out quick.mp4

No audio is included by default; --audio accepts a user-supplied WAV path.
--seconds is per chapter, --encoder defaults to hardware-first auto selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrapy import default_patch, reset_patch, save_mp4  # noqa: E402
from hydrapy.backend_gl import GLRenderer, _create_context  # noqa: E402
from hydrapy.drawing import find_font  # noqa: E402


CHAPTERS = [
    ("28_cellular.py", 0, "01 / CELLULAR", "Interacting rules + age + scrolling history"),
    ("28_cellular.py", 1, "02 / GLYPH FIELD", "The same engine, rendered as moving type"),
    ("29_orbit_voronoi.py", 0, "03 / ORBITING VORONOI", "Opposing rings reshape the cell boundaries"),
    ("30_distance_morph.py", 0, "04 / DISTANCE MORPH", "Smooth unions + moving cutouts + contour bands"),
    ("31_living_curves.py", 0, "05 / LIVING CURVES", "Subdivision + coherent seeded overdraw"),
]


def frames(size, fps, seconds, contact=None):
    ctx = _create_context()
    r = previous = None
    snapshots = []
    font = find_font("DejaVuSans", max(12, round(size[1] * .037)))
    small = find_font("DejaVuSans", max(10, round(size[1] * .026)))
    count = max(1, round(seconds * fps))
    transition = max(1, min(round(.65 * fps), count // 4))
    try:
        for chapter, (filename, output, title, subtitle) in enumerate(CHAPTERS):
            reset_patch()
            runpy.run_path(str(ROOT / "patches" / filename))
            patch = default_patch()
            chosen = patch.outputs[output]
            # Each chapter is independent, so unused outputs need not be evaluated.
            for o in patch.outputs:
                if o is not chosen:
                    o.chain = None
            patch.render(chosen)
            r = GLRenderer(patch, size=size, fps=fps, ctx=ctx)
            print(title, flush=True)
            for i in range(count):
                r.step(i / fps)
                arr = r.read()
                if previous is not None and i < transition:
                    f = (i + 1) / transition
                    f = f * f * (3 - 2 * f)
                    arr = np.round(previous * (1 - f) + arr * f).astype(np.uint8)
                image = Image.fromarray(arr)
                draw = ImageDraw.Draw(image, "RGBA")
                bar = round(size[1] * .15)
                draw.rectangle((0, size[1] - bar, size[0], size[1]), fill=(5, 8, 14, 225))
                x = round(size[0] * .025)
                draw.text((x, size[1] - bar + 8), title, font=font, fill=(233, 240, 245, 255))
                draw.text((x, size[1] - bar + 12 + font.size), subtitle, font=small, fill=(156, 184, 194, 255))
                draw.rectangle((0, size[1] - 3, round(size[0] * ((chapter + (i + 1) / count) / len(CHAPTERS))), size[1]), fill=(105, 225, 191, 255))
                if i == count // 2:
                    snapshots.append(image.copy())
                yield np.asarray(image)
            previous = arr.copy()
            r.release(); r = None
        if contact:
            sheet = Image.new("RGB", (size[0] * 3, size[1] * 2), (5, 8, 14))
            for i, image in enumerate(snapshots):
                sheet.paste(image, ((i % 3) * size[0], (i // 3) * size[1]))
            sheet.save(contact)
    finally:
        if r is not None:
            r.release()
        ctx.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="generative_demo.mp4")
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--audio")
    parser.add_argument("--encoder", default="auto")
    parser.add_argument("--contact", help="optional contact sheet PNG")
    args = parser.parse_args()
    size = tuple(int(x) for x in args.size.lower().split("x"))
    if len(size) != 2 or min(size) < 64 or not np.isfinite([args.fps, args.seconds]).all() or min(args.fps, args.seconds) <= 0:
        parser.error("size must be at least 64x64; fps and seconds must be positive and finite")
    save_mp4(frames(size, args.fps, args.seconds, args.contact), args.out,
             fps=args.fps, encoder=args.encoder, audio=args.audio,
             on_encoder=lambda enc: print("Encoder:", enc.name, flush=True))
    print(args.out)


if __name__ == "__main__":
    main()
