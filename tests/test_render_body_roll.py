"""The Body Roll time map that moves the action's holds onto half-time beats."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs'))
import render_body_roll as r


def test_holds_found_at_speed_minima_including_wrap():
    step=.25;x=np.arange(0,155,step)+step/2
    speed=1.2-np.cos(2*np.pi*x/155*5)          # minima every 31 frames, one at the loop point
    holds=r.find_holds(speed,step,155.,12/(480/155))
    assert len(holds)==5
    assert np.allclose(sorted(h for h,_ in holds),[0,31,62,93,124],atol=step)
    assert all(rel<.7 for _,rel in holds)
    assert r.find_holds(np.ones(620),step,155.,4)==[]


def test_map_lands_holds_on_half_beats_and_stays_monotonic():
    holds=[(37.625,.56),(50.125,.46),(61.875,.44),(71.625,.53),(96.125,.24),(124.625,.21),(137.625,.38),(154.1,.1)]
    xs,ys=r.hold_map(holds,155.)
    assert ys[0]==0 and ys[-1]==480 and xs[-1]==155
    assert np.all(np.diff(xs)>0) and np.all(np.diff(ys)>0)
    assert np.all(ys%30==0)
    assert len(xs)==9                            # the loop-point hold is already on the bar line
    ratio=np.diff(ys)/np.diff(xs)/(480/155)
    assert np.all((ratio>=.5)&(ratio<=2))


def test_map_skips_holds_that_collide_or_overstretch():
    xs,ys=r.hold_map([(30.,.2),(31.,.5),(154.,.3)],155.)
    assert list(ys)==[0,90,480]                   # 31 shares beat 90; 154 would hit the loop point
    xs,ys=r.hold_map([(5.,.2)],155.,max_ratio=1.5)
    assert list(ys)==[0,480]                      # 5 -> 30 frames would be 1.94x the stretch


def test_identity_map_matches_linear_stretch_and_extends_handles():
    xs,ys=np.array([0.,155.]),np.array([0.,480.])
    for x in (-3.,0.,77.5,155.,158.):
        assert np.isclose(r.apply_map(x,xs,ys),x*480/155)
