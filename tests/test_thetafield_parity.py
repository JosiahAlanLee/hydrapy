"""Reference checks for finite Jacobi theta_3 truncation."""
import numpy as np
from hydrapy.ops_theta import _np_thetafield

class Env:
    res = (8, 8)

def expected(q, z, terms):
    v = 1+0j
    for n in range(1, terms):
        v += 2*q**(n*n)*np.cos(2*n*z)
    hue = np.angle(v)/(2*np.pi)
    rgb = .5+.5*np.cos(2*np.pi*(hue-np.array([0,1/3,2/3])))
    return (rgb*np.clip(abs(v)/3,.15,1)).astype(np.float32)

def test_term_count_one_is_constant():
    st=np.array([[[.5,.5],[.7,.4]]],dtype=np.float32)
    result=_np_thetafield(st,.3,0,2,0,1,Env())
    np.testing.assert_allclose(result[0,0,:3],result[0,1,:3],atol=1e-6)

def test_two_terms_matches_reference():
    st=np.array([[[.5,.5]]],dtype=np.float32)
    actual=_np_thetafield(st,.3,0,2,0,2,Env())[0,0,:3]
    np.testing.assert_allclose(actual,expected(.3,0j,2),atol=1e-6)
