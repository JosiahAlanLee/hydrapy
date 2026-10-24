import numpy as np
import pytest

import hydrapy as H
from hydrapy.backend_gl import GLRenderer, _create_context
from hydrapy.backend_np import Env
from hydrapy.export import chain_to_python
from hydrapy.ops_phase import _np_pixel_osc


@pytest.fixture(scope="module")
def glctx():
    pytest.importorskip("moderngl")
    try:ctx=_create_context()
    except Exception as exc:pytest.skip(str(exc))
    yield ctx
    ctx.release()


@pytest.mark.parametrize("geometry",range(4))
def test_independent_samples_and_fixed_period(geometry):
    env=Env(120,72);env.ctx.time=1.31
    args=(.25,1.6,90,3,geometry,.137,.13,env)
    full=_np_pixel_osc(env.base_st,*args)
    selected=_np_pixel_osc(env.base_st[11:25,31:54],*args)
    np.testing.assert_array_equal(selected,full[11:25,31:54])
    assert set(np.unique(full[...,:3]))=={0,1}
    assert np.all(full[...,3]==1)
    # With zero spread all clocks advance exactly one cycle in four seconds.
    chain=H.pixelOsc(rate=.25,spread=0,density=17.3,pixel=3,geometry=geometry,phase=.137)
    np.testing.assert_array_equal(chain.render(1.31,(120,72)),chain.render(5.31,(120,72)))


def test_pixel_cells_and_threshold_extremes():
    a=H.pixelOsc(pixel=3).render(.7,(120,72))[...,0]
    cells=a.reshape(24,3,40,3)
    assert np.all(cells==cells[:,0:1,:,0:1])
    assert not H.pixelOsc(threshold=1.01).render(12,(32,24))[...,:3].any()
    assert H.pixelOsc(threshold=-1.01).render(12,(32,24))[...,:3].all()
    np.testing.assert_array_equal(H.pixelOsc(pixel=0).render(.4,(32,24)),
                                  H.pixelOsc(pixel=1).render(.4,(32,24)))


@pytest.mark.parametrize("geometry",range(4))
def test_cpu_gpu_seek_export_and_single_pass(glctx,geometry):
    chain=H.pixelOsc(rate=.037,spread=1.67,density=91.3,pixel=3,geometry=geometry,phase=.137,threshold=.13)
    patch=H.Patch(size=(160,96));chain.out(patch.o0,patch=patch)
    renderer=GLRenderer(patch,ctx=glctx)
    try:
        assert len(renderer.programs)==1
        # The compiler declares prevBuffer for every output; the driver must
        # eliminate it here, since this source has no history dependency.
        assert len(renderer.programs[0].passes)==1
        assert not renderer.programs[0].history
        assert all("prevBuffer" not in compiled for _,_,compiled,_,_,_ in renderer._passes)
        first=None
        for t in (.23,20.07,119.23,.23):
            renderer.step(t);gpu=renderer.read_rgba();cpu=chain.render(t,patch.size)
            # Hard thresholds can flip a few cells on float rounding boundaries.
            assert np.mean(cpu[...,:3]!=gpu[...,:3])<.005
            if first is None:first=gpu.copy()
        np.testing.assert_array_equal(first,gpu)
        rebuilt=eval(chain_to_python(chain),vars(H))
        np.testing.assert_array_equal(chain.render(.23,patch.size),rebuilt.render(.23,patch.size))
        assert "pixelOsc" in H.to_html(patch)
    finally:renderer.release()


def test_spatial_signal_inputs_are_supported(glctx):
    chain=H.pixelOsc(rate=H.gradient(0)*.2,spread=.7,density=H.osc(2,0)*20+50,
                     phase=H.gradient(0),threshold=H.osc(3,0)*.3)
    patch=H.Patch(size=(80,60));chain.out(patch.o0,patch=patch)
    renderer=GLRenderer(patch,ctx=glctx)
    try:
        renderer.step(.6)
        assert np.mean(chain.render(.6,patch.size)!=renderer.read_rgba())<.005
    finally:renderer.release()
