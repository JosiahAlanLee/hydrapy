import numpy as np
import pytest
from hydrapy.namagiri import convergents, convergent_controls

def test_first_convergent():
    q = .2+.1j
    assert np.allclose(convergents(q,1)[0], q**(.2)/(1+q))

def test_zero():
    assert np.allclose(convergents(0,8),0)

def test_depth_and_determinism():
    a=convergents(.3+.2j,12)
    assert a.shape==(12,)
    assert np.array_equal(a,convergents(.3+.2j,12))
    assert np.isfinite(a).all()

def test_controls():
    c=convergent_controls(.25,10)
    assert set(c)=={"real","imag","magnitude","phase","delta"}
    assert all(len(v)==10 for v in c.values())
    assert c["delta"][0]==0

@pytest.mark.parametrize("q",[1,1j,complex(float("nan"),0),2])
def test_invalid_q(q):
    with pytest.raises(ValueError):
        convergents(q)

@pytest.mark.parametrize("depth",[0,65])
def test_invalid_depth(depth):
    with pytest.raises(ValueError):
        convergents(.2,depth)
