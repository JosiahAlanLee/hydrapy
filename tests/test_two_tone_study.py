"""Validate integer state preservation and the two-color presentation."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

DOCS=Path(__file__).resolve().parents[1]/"docs"
sys.path.insert(0,str(DOCS))
spec=importlib.util.spec_from_file_location("two_tone_study",DOCS/"two_tone_study.py")
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture
def field():
    pytest.importorskip("moderngl")
    try: ctx=study._create_context()
    except Exception as exc: pytest.skip(f"OpenGL unavailable: {exc}")
    f=study.BitField(ctx,size=(320,180),pixel=2)
    try: yield f
    finally: f.release(); ctx.release()


def test_seeds_are_reproducible_and_binary():
    for variant in range(3):
        a=study.initial_state((144,80),variant,56)
        np.testing.assert_array_equal(a,study.initial_state((144,80),variant,56))
        assert set(np.unique(a))=={0,1}
        assert .15<a.mean()<.85


@pytest.mark.parametrize("variant",range(3))
def test_local_shift_preserves_other_rows_and_cell_count(field,variant):
    field.reset(variant)
    w,h=field.grid
    before=np.frombuffer(field.textures[field.current].read(alignment=1),np.uint8).reshape(h,w).copy()
    field.step(dict(tick=0,rule=90,operation=0,row=12,column=20,shift=-2,extent=2,
                    cellX=0,cellY=0,gate=0),edit=True)
    after=np.frombuffer(field.textures[field.current].read(alignment=1),np.uint8).reshape(h,w)
    outside=np.abs(np.arange(h)-12)>=2
    np.testing.assert_array_equal(before[outside],after[outside])
    np.testing.assert_array_equal(before.sum(axis=1),after.sum(axis=1))
    assert np.any(before!=after)
    assert set(np.unique(after))=={0,1}
    rgb=field.read()
    assert {tuple(v) for v in np.unique(rgb.reshape(-1,3),axis=0)}=={tuple(v) for v in study.PALETTE}
    np.testing.assert_array_equal(rgb,rgb[:,::-1])
    if variant!=1: np.testing.assert_array_equal(rgb,rgb[::-1])
