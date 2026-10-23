"""Render the feature tour: every primitive for a second, then the combination patches.

    python docs/tour.py                 # docs/tour.mp4 (+ tour_contact.png, tour.gif)
    python docs/tour.py --size 320x180 --fps 15 --quick

Renders on the GPU backend (moderngl; EGL headless on Linux), captions with PIL.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydrapy import *  # noqa: E402,F401,F403
from hydrapy import default_patch, reset_patch  # noqa: E402
from hydrapy.backend_gl import GLRenderer  # noqa: E402
from hydrapy.backend_np import Renderer  # noqa: E402
from hydrapy.drawing import find_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(HERE, "..", "patches")

base = "osc(8, 0.1, 0.6)"
mod = "noise(3, 0.2)"

# (caption line, expression) — expressions are evaluated with hydrapy's names in scope
PRIMITIVES = [
    ("SOURCES", None),
    ("osc", "osc(10, 0.1, 0.8)"),
    ("noise", "noise(4, 0.2)"),
    ("voronoi", "voronoi(6, 0.4, 0.3)"),
    ("shape", "shape(5, 0.4, 0.02)"),
    ("gradient", "gradient(1)"),
    ("solid", "solid(0.9, 0.3, 0.1)"),
    ("fbm  (new)", "fbm(3, 5, 0.2)"),
    ("rings  (new)", "rings(12, 2)"),
    ("grid  (new)", "grid(10, 0.06)"),
    ("white  (new)", "white(24)"),
    ("drops  (new)", "drops(4, 0.5, 0.6, 1.5, seed=SH(4, 0, 99))"),
    ("crackle  (new)", "crackle(6, 0.04, 0.4, seed=SH(2, 0, 99))"),
    ("veins  (new)", "veins(3, 5, 2.5, 0.2)"),
    ("scratches  (new)", "scratches(1.0, 0.7, 1.5, 1, 24, Drift(0.2, 0, 50))"),
    ("dust  (new)", "dust(0.6, 2, 0.4, seed=SH(24, 0, 99))"),
    ("dropouts  (new)", "osc(6, 0.1, 1.2).kaleid(4).lighten(dropouts(2, 0.4, 30), 0.9)"),
    ("gate  (new)", "osc(6, 0.1, 1.2).kaleid(4).mult(gate(0.9, 0.86, Lfo(0.3, lo=0.01, hi=0.3), 0.01).pan(SH(18, -0.004, 0.004), 0))"),
    ("harmonograph  (new)", "solid().layer(harmonograph(3, 4, 3.01, 4.02, 0, 0.6, 2.5, 0.42, 0.012, 0.4, 0.3))"),
    ("wave  (new)", "solid().layer(wave(3, Lfo(0.4, lo=0.05, hi=0.3), 0.02, 1, 0, 0.9, 1)).layer(wave(5, 0.15, 0.012, -0.7, 0.3, 0.9, 1).rotate(0.3))"),
    ("arc  (new)", "solid().layer(arc(0.3, 0, 2.4, 0.05, 1, 1.2)).layer(arc(0.18, 1.0, 3.5, 0.03, 1, -2.0))"),
    ("dab  (new)", "solid().layer(dab(Lfo(0.3, lo=0.1, hi=0.3), 0.5, Lfo(0.2, lo=0, hi=6.28), 0.8, 0.2, SH(24, 0, 99)).color(0.9, 0.5, 0.3))"),
    ("COORDINATE OPS", None),
    ("rotate", f"{base}.rotate(0.3, 0.4)"),
    ("scale", f"{base}.scale(Lfo(0.5, lo=0.5, hi=2))"),
    ("pixelate", f"{base}.pixelate(Seq([40, 20, 8]).fast(2), 12)"),
    ("repeat", f"{base}.repeat(3, 3, 0.3, 0)"),
    ("repeatX / repeatY", f"{base}.repeatX(4, 0.5).repeatY(2)"),
    ("kaleid", f"{base}.kaleid(5)"),
    ("scroll", f"{base}.scroll(0, 0, 0.1, 0.2)"),
    ("scrollX / scrollY", f"{base}.scrollX(0, 0.3).scrollY(0, -0.1)"),
    ("polar  (new)", f"{base}.polar(1, 2)"),
    ("unpolar  (new)", f"{base}.repeat(2, 3).unpolar(Lfo(0.3, lo=0.6, hi=1.4))"),
    ("twist  (new)", f"{base}.twist(Lfo(0.2, lo=-4, hi=4))"),
    ("fisheye  (new)", f"{base}.kaleid(4).fisheye(Lfo(0.3, 'tri', -0.8, 1.5))"),
    ("mirror  (new)", f"{base}.rotate(0.4).mirror(1, 1)"),
    ("wobble  (new)", f"{base}.wobble(3, 0.03, 2)"),
    ("glitch  (new)", f"{base}.glitch(10, 0.1, 6)"),
    ("pan  (new)", "drops(4, 0.5, 0.6, seed=2).pan(0, 0, 0, 1.5)"),
    ("shear  (new)", f"{base}.kaleid(4).shear(Lfo(0.3, 'tri', -0.6, 0.6), Lfo(0.2, 'tri', -0.3, 0.3))"),
    ("linejitter  (new)", f"{base}.kaleid(4).linejitter(Lfo(0.25, 'tri', 0, 0.03), 30, 0.004)"),
    ("headswitch  (new)", f"{base}.kaleid(4).headswitch(Lfo(0.3, 'tri', 0.05, 0.3), 0.25, 30)"),
    ("boil  (new)", "shape(4, 0.3, 0.004).repeat(3, 2).boil(Lfo(0.3, lo=0.002, hi=0.03), 12, 5)"),
    ("polarRepeat  (new)", "solid().layer(arc(0.3, 0, 1.0, 0.04, 1).polarRepeat(Seq([3, 5, 8]).fast(0.5), Time() * 0.3))"),
    ("COLOUR OPS", None),
    ("posterize", f"{base}.posterize(Seq([2, 3, 5]).fast(2), 0.6)"),
    ("shift", f"{base}.shift(Lfo(0.3), 0.2, 0)"),
    ("invert", f"{base}.invert(Lfo(0.5))"),
    ("contrast", f"{base}.contrast(Lfo(0.4, lo=0.5, hi=3))"),
    ("brightness", f"{base}.brightness(Lfo(0.5, 'tri', -0.5, 0.5))"),
    ("luma", f"{base}.luma(0.5, 0.1)"),
    ("thresh", f"{base}.thresh(Lfo(0.3, lo=0.2, hi=0.8), 0.04)"),
    ("color", f"{base}.color(1, 0.3, -1)"),
    ("saturate", f"{base}.saturate(Lfo(0.4, lo=0, hi=3))"),
    ("hue", f"{base}.hue(Lfo(0.2))"),
    ("colorama", f"{base}.colorama(0.01)"),
    ("sum / r / g / b / a", f"{base}.r(1, 0).add({base}.g().color(0,1,0), 0.5)"),
    ("fold  (new)", f"rings(4, 1).fold(Lfo(0.15, 'tri', lo=1, hi=8))"),
    ("gamma  (new)", f"{base}.gamma(Lfo(0.3, lo=0.3, hi=3))"),
    ("levels  (new)", f"{base}.levels(0.3, 0.7)"),
    ("dither  (new)", f"{base}.dither(2, 3)"),
    ("palette  (new)", f"noise(3, 0.2).palette(0, 0.33, 0.67, Lfo(0.1), 2)"),
    ("grain  (new)", f"{base}.grain(0.3, 24, Seq([1, 2, 3]).fast(1))"),
    ("contours  (new)", f"noise(2.5, 0.2).contours(Lfo(0.2, lo=3, hi=12), 0.2, 0.4)"),
    ("sinefold  (new)", f"{base}.kaleid(3).sinefold(Lfo(0.15, lo=1, hi=4), Lfo(0.3))"),
    ("filmstock  (new)", f"{base}.kaleid(4).filmstock(Seq([0, 1, 2, 3, 4]).fast(0.75))"),
    ("halftone  (new)", f"{base}.kaleid(4).halftone(Lfo(0.3, lo=3, hi=10), 0.78, 0.8)"),
    ("vignette  (new)", f"{base}.vignette(0.9, 0.4)"),
    ("COMBINE OPS", None),
    ("add", f"{base}.add({mod}, Lfo(0.4))"),
    ("sub", f"{base}.sub({mod})"),
    ("layer", f"{base}.layer(shape(4, 0.4).color(0.2, 0.8, 1).luma())"),
    ("blend", f"{base}.blend({mod}, Lfo(0.4))"),
    ("mult", f"{base}.mult({mod})"),
    ("diff", f"{base}.diff({mod}.scrollX(0, 0.1))"),
    ("mask", f"{base}.mask(shape(3, 0.5, 0.2).rotate(0, 0.5))"),
    ("lighten / darken  (new)", f"{base}.lighten(rings(6, 1)).darken(shape(32, 0.9))"),
    ("screen / overlay  (new)", f"{base}.screen({mod}).overlay(gradient())"),
    ("wipe  (new)", f"{base}.wipe({mod}, Lfo(0.3, 'tri'), 0.05, 0.4)"),
    ("bits  (new)", "gradient().scroll(Lfo(0.1)).bits(gradient().rotate(1.5708), Seq([0, 1, 2]).fast(1), 6)"),
    ("stain  (new)", "solid(1,1,1).stain(drops(3, 0.6, 0.5, 1.2, seed=1)).stain(drops(5, 0.5, 0.5, 2.5, seed=2, hue=0.62), 0.8)"),
    ("alpha  (new)", "solid(1,1,1).stain(solid(0.1, 0.3, 1).alpha(shape(5, 0.4).rotate(0, 0.5)), Lfo(0.5, lo=0.2, hi=2))"),
    ("MODULATE OPS", None),
    ("modulate", f"{base}.modulate({mod}, Lfo(0.3, lo=0, hi=0.5))"),
    ("modulateRepeat", f"{base}.modulateRepeat({mod}, 3, 3, 0.5, 0.5)"),
    ("modulateRepeatX / Y", f"{base}.modulateRepeatX({mod}, 3, 0.5).modulateRepeatY({mod}, 2, 0.3)"),
    ("modulateKaleid", f"{base}.modulateKaleid({mod}, 4)"),
    ("modulateScrollX / Y", f"{base}.modulateScrollX({mod}, 0.5, 0.1).modulateScrollY({mod}, 0.2)"),
    ("modulateScale", f"{base}.modulateScale({mod}, Lfo(0.3, lo=0, hi=2), 1)"),
    ("modulatePixelate", f"{base}.modulatePixelate({mod}, 20, 3)"),
    ("modulateRotate", f"{base}.modulateRotate({mod}, Lfo(0.2, lo=0, hi=3))"),
    ("modulateHue", f"{base}.modulateHue({mod}, 40)"),
    ("displace  (new)", f"{base}.displace({mod}, Lfo(0.3, lo=0, hi=0.6))"),
    ("per-pixel CV  (new)", "osc(noise(2, 0.1).r() * 30 + 5, 0.1, 0.8)"),
    ("TEXTURE OPS", None),
    ("blur", f"{base}.kaleid(3).blur(Lfo(0.3, lo=0, hi=6), iterations=3)"),
    ("sharpen", f"{base}.kaleid(3).sharpen(Lfo(0.5, lo=0, hi=8))"),
    ("edge", f"voronoi(6, 0.4).edge(1.5)"),
    ("emboss", f"voronoi(6, 0.4).emboss(3)"),
    ("dilate / erode", f"shape(3, 0.3, 0.01).repeat(3, 2).dilate(Lfo(0.5, lo=0, hi=6), iterations=2)"),
    ("life  (feedback)", "PATCH:life"),
    ("rd  reaction-diffusion", "PATCH:rd"),
    ("chroma  (new)", f"{base}.kaleid(4).chroma(Lfo(0.5, lo=0, hi=0.03), Lfo(0.3, lo=0, hi=6.28))"),
    ("bleed  (new)", "drops(3, 0.5, 0.5, seed=3).bleed(0.1, 0.15, 0.5, iterations=4).a()"),
    ("pixelsort  (CPU)", "CPU:noise(4, 0.3).thresh(0.4, 0.2).mult(osc(20, 0.1, 0.5)).pixelsort(0.3, 0)"),
    ("psort  (GPU, feedback)", "PATCH:psort"),
    ("ruttetra  (new)", "noise(2.5, 0.3).brightness(0.1).contrast(1.6).ruttetra(60, Lfo(0.2, lo=0.1, hi=0.4), 1.4, 0.3).color(0.7, 1, 0.8)"),
    ("yiq  (new)", f"{base}.posterize(4, 1).kaleid(5).yiq(Lfo(0.25, 'tri', 0, 0.08), 2, 0.006)"),
    ("ringing  (new)", f"{base}.posterize(4, 1).kaleid(5).ringing(Lfo(0.3, 'tri', 0, 2), 3)"),
    ("charcoal  (new)", "solid().layer(arc(0.3, 0, 6.2832, 0.05).layer(wave(3, 0.2, 0.03, 1, 0, 0.9, 1)).charcoal(Lfo(0.3, lo=0, hi=1), 2.0, 80))"),
    ("smear  (new)", "shape(5, 0.3, 0.01).color(1, 0.6, 0.2).smear(noise(2, 0.15) * 6.28, Lfo(0.3, lo=0, hi=0.08))"),
    ("plane3d  (new)", f"{base}.kaleid(4).plane3d(1.2, 0, Lfo(0.1, lo=-0.3, hi=0.3), 0.55, 70, 1, 0.4, 0.4)"),
    ("sphere3d  (new)", f"{base}.kaleid(4).sphere3d(0.3, Time() * 0.6, 0, 0.45, 1, 60, 0.6)"),
    ("cylinder3d  (new)", f"{base}.cylinder3d(0.3, Time() * 0.6, 0, 0.4, 0.9, 1, 60, 0, 0.5)"),
    ("SIGNALS", None),
    ("Lfo", "osc(Lfo(0.3, lo=4, hi=40), 0.1, Lfo(0.5, 'tri', 0, 2))"),
    ("Seq  (Hydra arrays)", "osc(Seq([10, 20, 40]).fast(2), 0.1, Seq([0, 1, 2]).fast(4).smooth())"),
    ("Clock + Env", "shape(4, Env(Clock(2), 0.02, 0.5, 2, 0.1, 0.6), 0.02).color(1, 0.5, 0.2)"),
    ("SH + Drift", "osc(SH(4, 5, 40), 0.1, Drift(1, 0, 2)).rotate(Drift(0.5, 0, 6.28))"),
    ("signal arithmetic", "osc(20).rotate((Lfo(0.5) * 2 + SH(3)).sin() * 3).kaleid(Lfo(0.2, lo=2, hi=8).quantize(1))"),
    ("delay  (new)", "PATCH:delay"),
    ("Text / Lines  (new)", "solid(0.98,0.96,0.9).stain(src(Lines(meadow(seed=4, n=8), stroke=1.6, color=(0.05,0.08,0.05)))).stain(src(Text('hydrapy', font='DejaVuSerif-Bold', size=0.28, color=(0.85,0.1,0.05))))"),
]

COMBINATIONS = [
    ("00_showreel.py", "o0", "reaction-diffusion, palette", 4),
    ("00_showreel.py", "o1", "per-pixel CV, wavefolder, chroma, glitch", 4),
    ("00_showreel.py", "o2", "Life as video feedback", 4),
    ("02_cv_everywhere.py", "o0", "signals and chains on every parameter", 3),
    ("05_wavefolder.py", "o0", "wavefolder, palette, dither", 3),
    ("06_multipass.py", "o0", "voronoi, blur, edge, chroma, glitch, grain", 3),
    ("07_polar_tunnel.py", "o1", "polar / unpolar, mirror, fisheye", 3),
    ("10_operators.py", "o0", "a * b + c | d ^ e", 3),
    ("00_painted_showreel.py", "o0", "hand-painted frames, step-printed", 4),
    ("00_painted_showreel.py", "o1", "a painted strip through the gate", 4),
    ("00_painted_showreel.py", "o2", "scratched black leader, dyed", 3),
    ("00_painted_showreel.py", "o3", "mothlight", 4),
    ("13_quilt.py", "o0", "image quilting instead of a GAN", 3),
    ("14_text_and_lines.py", "o0", "text and line drawings as dye", 4),
    ("14_text_and_lines.py", "o1", "text and drawings as masks", 3),
    ("15_canvas_in_3d.py", "o0", "the canvas in 3D: plane and sphere", 4),
    ("15_canvas_in_3d.py", "o1", "text on a cylinder", 3),
    ("16_delay_line.py", "o1", "delay(): echoes", 3),
    ("16_delay_line.py", "o2", "delay(): A/B roll", 3),
    ("19_entropy_lock.py", "o0", "entropy-locked glitch feedback", 4),
    ("20_video_synth.py", "o0", "Rutt-Etra scan processor", 3),
    ("20_video_synth.py", "o1", "ramp XOR ramp, sine staircase", 3),
    ("20_video_synth.py", "o3", "contour keyer over a pixel-sort loop", 3),
    ("21_vhs.py", "o1", "VHS, first generation", 3),
    ("21_vhs.py", "o2", "VHS, fourth-generation dub", 3),
    ("22_super8.py", "o0", "Super 8, Kodachrome, 18 fps", 4),
    ("22_super8.py", "o2", "Super 8, Tri-X", 3),
    ("23_studie.py", "o1", "Fischinger: charcoal oscillators to a beat", 4),
    ("23_studie.py", "o2", "the harmonograph, in charcoal", 3),
    ("23_studie.py", "o3", "Motion Painting: one touch a frame", 4),
    ("24_cutout.py", "o0", "cut paper, on twos", 3),
    ("24_cutout.py", "o2", "a title card (КИНО)", 3),
    ("25_ornament_sound.py", "o1", "ornament sound: the patch draws its soundtrack", 3),
    ("26_painted_oscillator.py", "o1", "a painted oscillator, patched into the modular", 4),
]


def special_patch(kind: str, size):
    p = Patch(size=size, bpm=120)
    if kind == "life":
        src(o0).life().add(noise(40, 0.5).thresh(0.62).color(1, 0, 0), Lfo(0.5, "pulse", width=0.06)).color(1, 0.6, 0.2).out(p.o0, patch=p)
        p.render(p.o0)
        return p, 40
    if kind == "rd":
        seed_ = shape(6, 0.06, 0.01).scroll(Lfo(0.11, lo=-0.3, hi=0.3), Lfo(0.17, "tri", lo=-0.3, hi=0.3)).color(0, 1, 0)
        src(o3).rd(0.037, 0.06, iterations=8).add(seed_, 0.1).out(p.o3, patch=p)
        src(o3).g().palette(0.1, 0.4, 0.7, Lfo(0.02), 1.5).out(p.o0, patch=p)
        p.render(p.o0)
        return p, 60
    if kind == "psort":
        pic = noise(2.5, 0.15).brightness(0.1).contrast(1.6).kaleid(3)
        src(o2).psort(0.3, 0, iterations=8).blend(pic, Lfo(1 / 3, "pulse", width=0.06)).out(p.o2, patch=p)
        p.render(p.o2)
        return p, 2
    if kind == "delay":
        shape(5, 0.18, 0.02).scroll(Lfo(0.3, "tri", -0.3, 0.3), Lfo(0.21, "sine", -0.2, 0.2)).rotate(Time() * 2).color(1, 0.6, 0.2).out(p.o0, patch=p)
        src(o0).add(delay(o0, 4).color(0.8, 0.3, 0.9), 0.7).add(delay(o0, 8).color(0.3, 0.9, 0.8), 0.5).add(delay(o0, 12).color(0.9, 0.9, 0.3), 0.35).out(p.o1, patch=p)
        p.render(p.o1)
        return p, 16
    raise ValueError(kind)


class Captioner:
    def __init__(self, size):
        self.size = size
        self.font = find_font("DejaVuSansMono", max(int(size[1] * 0.045), 10))
        self.font_b = find_font("DejaVuSansMono-Bold", max(int(size[1] * 0.05), 11))
        self.font_big = find_font("DejaVuSerif-Bold", max(int(size[1] * 0.13), 16))
        self.font_mid = find_font("DejaVuSans", max(int(size[1] * 0.05), 11))

    def caption(self, frame, title, code):
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame).convert("RGBA")
        W, H = im.size
        bar_h = int(H * 0.17)
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 170))
        d.text((int(W * 0.02), H - bar_h + int(bar_h * 0.12)), title, font=self.font_b, fill=(255, 255, 255, 255))
        code = code if len(code) < 96 else code[:93] + "..."
        d.text((int(W * 0.02), H - bar_h + int(bar_h * 0.55)), code, font=self.font, fill=(200, 200, 190, 255))
        return np.asarray(Image.alpha_composite(im, overlay).convert("RGB"))

    def card(self, frame, title, sub=""):
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame).convert("RGBA")
        W, H = im.size
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 150))
        d = ImageDraw.Draw(overlay)
        bbox = d.textbbox((0, 0), title, font=self.font_big)
        d.text(((W - bbox[2]) / 2, H * 0.4 - bbox[3] / 2), title, font=self.font_big, fill=(255, 255, 255, 255))
        if sub:
            bbox2 = d.textbbox((0, 0), sub, font=self.font_mid)
            d.text(((W - bbox2[2]) / 2, H * 0.58), sub, font=self.font_mid, fill=(220, 220, 210, 255))
        return np.asarray(Image.alpha_composite(im, overlay).convert("RGB"))


_CTX = None


def render_patch(patch, size, fps, seconds, warmup=0, cpu=False):
    global _CTX
    if cpu:
        r = Renderer(patch, size=size, fps=fps)
    else:
        if _CTX is None:
            from hydrapy.backend_gl import _create_context
            _CTX = _create_context()
        r = GLRenderer(patch, size=size, fps=fps, ctx=_CTX)   # one shared context; released after each clip
    for i in range(warmup):
        r.step(-(warmup - i) / fps)
    frames = []
    for i in range(int(seconds * fps)):
        r.step(i / fps)
        f = r.read() if not cpu else (np.clip(np.nan_to_num(r.displayed()[..., :3]), 0, 1) * 255 + 0.5).astype(np.uint8)
        frames.append(f)
    if not cpu:
        r.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=1.3, help="seconds per primitive")
    ap.add_argument("--quick", action="store_true", help="fewer clips, for testing")
    ap.add_argument("--out", default=os.path.join(HERE, "tour.mp4"))
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    fps = a.fps
    cap = Captioner(size)
    import imageio.v2 as iio
    w = iio.get_writer(a.out, fps=fps, codec="libx264", quality=None, macro_block_size=None, pixelformat="yuv420p",
                       output_params=["-crf", "27", "-preset", "slow", "-movflags", "+faststart"])
    stills = []
    t0 = time.time()
    try:
        # title card on a live background
        bg = Patch(size=size)
        osc(10, 0.1, 1.2).kaleid(5).modulate(noise(3, 0.2), 0.3).colorama(0.01).out(bg.o0, patch=bg)
        for f in render_patch(bg, size, fps, 2.5):
            w.append_data(cap.card(f, "hydrapy", "Hydra's video synth in Python, with additions"))
        prims = PRIMITIVES if not a.quick else PRIMITIVES[:14]
        ns = dict(globals())
        for i, (title, expr) in enumerate(prims):
            if expr is None:
                for f in render_patch(bg, size, fps, 1.4):
                    w.append_data(cap.card(f, title))
                continue
            cpu = expr.startswith("CPU:")
            if expr.startswith("PATCH:"):
                p, warm = special_patch(expr[6:], size)
                code = {"life": "src(o0).life().add(seed, pulse).out(o0)", "rd": "src(o3).rd(0.037, 0.06, iterations=8).add(seed, 0.1).out(o3)",
                        "delay": "src(o0).add(delay(o0, 4)).add(delay(o0, 8)).add(delay(o0, 12))",
                        "psort": "src(o2).psort(0.3, 0, iterations=8).blend(pic, pulse).out(o2)"}[expr[6:]]
            else:
                code = expr[4:] if cpu else expr
                p = Patch(size=size, bpm=120)
                chain = eval(code, ns)
                chain.out(p.o0, patch=p)
                warm = 0
            frames = render_patch(p, size, fps, a.seconds if not cpu else 1.0, warmup=warm, cpu=cpu)
            for f in frames:
                w.append_data(cap.caption(f, title, code))
            stills.append((title, frames[len(frames) // 2]))
            print(f"[{time.time() - t0:5.0f}s] {title}")
        combos = COMBINATIONS if not a.quick else COMBINATIONS[:3]
        for f in render_patch(bg, size, fps, 1.4):
            w.append_data(cap.card(f, "COMBINATIONS", "the example patches"))
        for fname, out, title, secs in combos:
            reset_patch()
            runpy.run_path(os.path.join(PATCHES, fname))
            p = default_patch()
            p.size = size
            p.render(p.outputs[int(out[1])])
            frames = render_patch(p, size, fps, secs, warmup=24)
            code = f"patches/{fname}  ->  {out}"
            for f in frames:
                w.append_data(cap.caption(f, title, code))
            stills.append((title, frames[len(frames) // 2]))
            print(f"[{time.time() - t0:5.0f}s] {fname} {out}")
        for f in render_patch(bg, size, fps, 2.0):
            w.append_data(cap.card(f, "hydrapy", "github.com/JosiahAlanLee/hydrapy"))
    finally:
        w.close()
    print("wrote", a.out)
    # contact sheet of stills
    from PIL import Image
    cols = 6
    tw, th = size[0] // 2, size[1] // 2
    rows = (len(stills) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for i, (title, f) in enumerate(stills):
        sheet.paste(Image.fromarray(cap.caption(f, title, "")).resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(HERE, "tour_contact.png"))
    print("wrote tour_contact.png")


if __name__ == "__main__":
    main()
