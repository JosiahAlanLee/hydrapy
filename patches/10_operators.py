"""Chains combine with Python operators: + add, - sub, * mult, | layer, ^ diff."""
from hydrapy import *

a = osc(30, 0.05, 0.3).rotate(0.5)
b = noise(5, 0.2)
(a * b + shape(3, 0.3, 0.1).color(0.2, 0.8, 1).luma() ^ rings(8, 2)).colorama(0.01).out(o0)
