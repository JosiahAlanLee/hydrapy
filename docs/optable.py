"""Regenerate docs/operators.md from the op registry.

    python docs/optable.py            # rewrites docs/operators.md
    python docs/optable.py --print    # prints the reference without writing
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hydrapy import REGISTRY  # noqa: E402

SECTIONS = [
    ("src", "Sources"),
    ("coord", "Coordinate ops"),
    ("color", "Colour ops"),
    ("combine", "Combine ops (take another chain)"),
    ("combineCoord", "Modulate ops (coordinates driven by another chain)"),
    ("texture", "Texture ops (neighbourhood; multi-pass on the GPU)"),
]


def _default(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _args(op) -> str:
    parts = []
    for i in op.inputs:
        if i.type == "chain":
            continue
        if i.type == "sampler2D":
            parts.append(i.name)
        else:
            parts.append(f"{i.name}={_default(i.default)}")
    return ", ".join(parts)


def section() -> str:
    out = ["# Operator reference", "",
           "[Documentation](README.md) · [Project home](../README.md)", "",
           "Generated from the operator registry with `python docs/optable.py`.",
           "Edit operator declarations, then regenerate this page.", "",
           "[Sources](#sources) · [Coordinates](#coordinate-ops) · [Colour](#colour-ops) ·",
           "[Combining](#combine-ops-take-another-chain) · [Modulation](#modulate-ops-coordinates-driven-by-another-chain) ·",
           "[Textures](#texture-ops-neighbourhood-multi-pass-on-the-gpu)", ""]
    for kind, title in SECTIONS:
        out += [f"## {title}", "", "| op | args | origin | notes |", "|---|---|---|---|"]
        for name, op in REGISTRY.items():
            if op.kind != kind:
                continue
            tag = "Hydra" if op.hydra else "**new**"
            if op.cpu_only:
                tag += ", CPU only"
            doc = (op.doc or "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| `{name}` | `{_args(op)}` | {tag} | {doc} |")
        out.append("")
    return "\n".join(out)


def main():
    text = section()
    if "--print" in sys.argv:
        print(text)
        return
    path = os.path.join(os.path.dirname(__file__), "operators.md")
    with open(path, "w", encoding="utf-8") as reference:
        reference.write(text + "\n")
    print(f"docs/operators.md: op reference regenerated ({len(REGISTRY)} ops)")


if __name__ == "__main__":
    main()
