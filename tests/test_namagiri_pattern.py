import pytest
from hydrapy.namagiri_pattern import pattern

def test_deterministic():
    assert pattern(.25+.1j)==pattern(.25+.1j)

def test_bounds():
    events=pattern(.3+.2j,steps=32)
    assert all(0<=e.step<32 and 0<=e.note<=127 and 0<=e.velocity<=1 for e in events)

def test_zero_is_silent():
    assert pattern(0,steps=8,threshold=.001)==[]

@pytest.mark.parametrize("kwargs",[{"steps":65},{"span":0},{"root":128},{"threshold":-1}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        pattern(.2,**kwargs)
