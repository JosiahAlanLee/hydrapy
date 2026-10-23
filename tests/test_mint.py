"""Mint: per-mint draws replayed in the page, fxhash export."""
import json
import os
import shutil
import subprocess
import zipfile

import numpy as np
import pytest

from hydrapy import *  # noqa: F403
from hydrapy import Ctx
from hydrapy.export import build_spec
from hydrapy.mint import FX_SHIM_JS, MINT_JS, FxDraw, find_mints
from hydrapy.signals import JS_HELPERS

HASHES = ["ooQ2j7x9KmN3pRtVwYz5AbCdEfGhJkLmNpQrStUvWxYz1234567",
          "0x3f1a5c0e0d6b2a99887766554433221100ffeeddccbbaa998877665544332211",
          "tz1VSUr8wwNhLAzempoch5d6hLRiTh8Cjcjb", "a plain string seed", "ooHydrapyTemplate1"]


def _node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    return node


def _run_page_js(spec_mint, query="", fx=None, extra=""):
    """MINT_JS in node with a stub page: returns hydrapyMint plus a few evaluated helpers."""
    stub = f"""
    var location = {{search: {json.dumps(query)}, href: 'http://x/' + {json.dumps(query)}}};
    var window = {{addEventListener: function(){{}}, dispatchEvent: function(){{}}}};
    var document = {{getElementById: function(){{ return null; }}}};
    var SPEC = {{mint: {json.dumps(spec_mint)}}};
    """
    shim = ""
    if fx == "shim":
        shim = FX_SHIM_JS.replace("window.$fx =", "$fx = window.$fx =").replace("if (typeof $fx === 'undefined')", "var $fx; if (true)")
    js = stub + JS_HELPERS + shim + MINT_JS + extra + "\nconsole.log(JSON.stringify({mint: hydrapyMint, extra: typeof EXTRA === 'undefined' ? null : EXTRA}));"
    out = subprocess.run([_node(), "-e", js], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def _example(h, lock=0.0, legacy=False):
    fx = Mint(h, lock=lock, legacy=legacy)
    a = fx.rnd_btw(2, 9)
    b = fx.rnd_btwexp(0.1, a)           # a bound from an earlier draw
    c = fx.rnd_int(3, a)
    slot = fx.save()
    d = fx.rand()
    fx.restore(slot)
    e = fx.rand()                       # the bottleneck: the same value again
    fx.reseed()
    f = fx.pick([3, 5, 8])
    g = fx.pick(["calm", "restless", "frantic"])
    t = fx.table(20, -1, 1)
    lock_tbl = EntropyLock(fx, rate=12, lock=fx.rnd_btw(0.01, 0.05), lo=0, hi=4, steps=64)
    fx.features({"Sides": c, "Mood": g, "Edition": "first"})
    return fx, [a, b, c, d, e, f, g] + t, lock_tbl


@pytest.mark.parametrize("h", HASHES)
def test_page_replays_python_draws_exactly(h):
    for lock, legacy in ((0.0, False), (0.2, False), (0.0, True)):
        fx, draws, tbl = _example(h, lock, legacy)
        res = _run_page_js(fx.spec())["mint"]
        assert res["values"] == [d.preview for d in fx.draws]
        assert res["labels"][6] == draws[6].label
        assert res["tables"][0] == tbl.values
        assert res["features"] == fx.feature_values()
        assert res["locks"] == fx.locks
    assert draws[3].preview == draws[4].preview


def test_page_follows_the_url_hash_and_int_seeds():
    fx, _, _ = _example(HASHES[0])
    other, _, _ = _example(HASHES[1])
    res = _run_page_js(fx.spec(), query="?fxhash=" + HASHES[1])["mint"]
    assert res["hash"] == HASHES[1] and res["values"] == [d.preview for d in other.draws]
    n, _, _ = _example(12345)
    assert _run_page_js(n.spec())["mint"]["values"] == [d.preview for d in n.draws]


def test_draws_are_signals_that_refuse_python_control_flow():
    fx = Mint(HASHES[0])
    k = fx.rnd_int(3, 8)
    assert isinstance(k, FxDraw) and k(Ctx(time=3.0)) == k.preview == Fx(HASHES[0]).rnd_int(3, 8)
    for bad in (lambda: k > 4, lambda: bool(k), lambda: range(k), lambda: float(k), lambda: Lfo(k)):
        with pytest.raises(TypeError, match="drawn per mint"):
            bad()
    s = (k * 2 + 1).js()
    assert "__fxval(0," in s
    p = osc(k * 3).kaleid(k)
    assert np.isfinite(p.render(0.2, size=(32, 18))).all()
    with pytest.raises(TypeError):
        fx.rnd_btw(0, k * 2)                       # bounds: numbers or plain draws only


def test_pick_of_chains_switches_exactly_and_switch_morphs():
    fx = Mint(HASHES[0])
    chains = [osc(10), noise(3), voronoi(5), shape(4)]
    picked = fx.pick(chains)
    i = int(fx.draws[-1].preview)
    size = (48, 27)
    assert np.allclose(picked.render(0.3, size=size), chains[i].render(0.3, size=size), atol=1e-6)
    half = switch(0.5, osc(10), noise(3)).render(0.3, size=size)
    assert np.allclose(half, 0.5 * osc(10).render(0.3, size=size) + 0.5 * noise(3).render(0.3, size=size), atol=1e-5)


def test_params_and_features_in_python_and_through_the_shim():
    fx = Mint(HASHES[0], params={"density": 0.25})
    dens = fx.number("density", default=0.5, min=0, max=1, update="sync")
    mode = fx.select("mode", ["rings", "grid", "drift"], default="grid")
    on = fx.boolean("invert", default=True)
    ink = fx.color("ink", default="#ff8000")
    fx.features({"Mode": mode, "Ink": ink, "Density": dens})
    assert dens.preview == 0.25 and mode.preview == 1.0 and on.preview == 1.0
    assert [c.preview for c in ink] == [1.0, 128 / 255, 0.0] and ink.a.preview == 1.0
    assert fx.feature_values() == {"Mode": "grid", "Ink": "#ff8000ff", "Density": 0.25}
    with pytest.raises(ValueError):
        fx.number("density")
    spec = fx.spec()
    assert spec["params"][0] == {"id": "density", "name": "density", "type": "number", "default": 0.5,
                                 "update": "sync", "options": {"min": 0, "max": 1}}
    assert spec["params"][3]["default"] == "ff8000ff"
    extra = ("var EXTRA = [" + ",".join(s.js() for s in (dens, mode, on, ink.r, ink.g, ink.b)) + "];")
    res = _run_page_js(spec, query="?fxhash=" + HASHES[2], fx="shim", extra=extra)
    assert res["mint"]["hash"] == HASHES[2]
    assert res["extra"] == [0.5, 1, 1, 1.0, 128 / 255, 0.0]      # the shim serves the defaults
    assert res["mint"]["features"] == {"Mode": "grid", "Ink": "#ff8000ff", "Density": 0.5}
    res = _run_page_js(spec, extra=extra)                          # no $fx at all: Python's preview values
    assert res["extra"] == [0.25, 1, 1, 1.0, 128 / 255, 0.0]


def test_env_overrides_the_preview_hash(monkeypatch):
    monkeypatch.setenv("FXHASH", HASHES[1])
    monkeypatch.setenv("FXPARAMS", json.dumps({"n": 7}))
    fx = Mint(HASHES[0])
    assert fx.hash == HASHES[1] and fx.number("n", default=3, min=0, max=10).preview == 7


def test_build_spec_finds_the_mint_and_rejects_two():
    p = reset_patch()
    fx = Mint(HASHES[0])
    osc(fx.rnd_btw(5, 30)).out(o0)
    spec = build_spec(p, size=(32, 18), preview_at=1.5)
    assert spec["mint"]["tape"] == [["b", 5.0, 30.0]] and spec["mint"]["previewAt"] == 1.5
    assert find_mints([Expr("+", fx.draws[0], 1.0)]) == [fx]
    other = Mint(HASHES[1])
    osc(fx.rnd_btw(5, 30)).blend(noise(other.rand())).out(o0)
    with pytest.raises(ValueError, match="one Mint"):
        build_spec(p, size=(32, 18))
    reset_patch()


def test_hydra_and_python_exports_freeze_the_preview():
    p = reset_patch()
    fx = Mint(HASHES[0])
    k = fx.rnd_int(3, 8)
    osc(10).kaleid(k).out(o0)
    hyd = patch_to_hydra(p)
    assert "__fxval" in hyd and "function __fxval" in hyd       # the helper falls back to the preview value
    from hydrapy.export import patch_to_python
    assert f"kaleid({k.preview:g})" in patch_to_python(p)
    reset_patch()


def test_to_fxhash_writes_a_project(tmp_path):
    p = reset_patch()
    fx = Mint(HASHES[0])
    osc(fx.rnd_btw(5, 30), 0.1, fx.rnd_btw(0, 2)).kaleid(fx.pick([3, 4, 6])).out(o0)
    fx.features({"Sides": fx.draws[-1]})
    sdk = tmp_path / "sdk.js"
    sdk.write_text("/* fake sdk */")
    out = to_fxhash(p, str(tmp_path / "proj"), sdk=str(sdk), preview=1.0)
    html = open(out["index"], encoding="utf-8").read()
    assert '<script src="./fxhash.min.js"></script>' in html and "hydrapyMint" in html and 'id="hud"' not in html
    assert '"previewAt": 1.0' in html
    with zipfile.ZipFile(out["zip"]) as z:
        assert sorted(z.namelist()) == ["fxhash.min.js", "index.html"]
    with pytest.warns(UserWarning, match="stand-in"):
        to_fxhash(p, str(tmp_path / "bare"), zip=False)
    reset_patch()


def test_to_fxhash_runs_with_the_real_sdk_in_node(tmp_path):
    """If the fxhash SDK is available (FXHASH_SDK=path/to/fxhash.min.js), the page's draws match Python."""
    sdk = os.environ.get("FXHASH_SDK")
    if not sdk or not os.path.exists(sdk):
        pytest.skip("set FXHASH_SDK to @fxhash/project-sdk's dist/fxhash.min.js")
    fx, _, _ = _example(HASHES[0])
    dens = fx.number("density", default=0.3, min=0, max=1)
    stub = f"""
    var location = {{search: '?fxhash={HASHES[1]}', href: 'http://x/?fxhash={HASHES[1]}'}};
    var window = {{addEventListener: function(){{}}, dispatchEvent: function(){{}}, location: location}};
    window.parent = window;
    var document = {{getElementById: function(){{ return null; }}}};
    var SPEC = {{mint: {json.dumps(fx.spec())}}};
    """
    src = open(sdk, encoding="utf-8").read()
    js = (stub + "var $fx = (function(window, self){ " + src + "; return window.$fx; })(window, window);\n" + JS_HELPERS + MINT_JS
          + f"\nconsole.log(JSON.stringify({{m: hydrapyMint, d: {dens.js()}, feats: $fx.getFeatures()}}));")
    res = json.loads(subprocess.run([_node(), "-e", js], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    other, _, _ = _example(HASHES[1])
    assert res["m"]["values"] == [d.preview for d in other.draws]
    assert res["d"] == 0.3 and res["feats"]["Mood"] == other.feature_values()["Mood"]
