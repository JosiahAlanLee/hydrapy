"""Two interacting rule fields for the Bell Riots video; simulation stays small."""
import hydrapy as H


def make_field(controls, size=(1920, 1080)):
    p = H.Patch(size=size)
    signal = lambda name: H.Fn(lambda ctx, name=name: controls[name])
    once, bass, hit, bank = (signal(k) for k in ("init", "bass", "hit", "bank"))
    coarse = H.src(p.o1).r().pixelate(size[0] // 16, size[1] // 16)
    H.src(p.o1).ca(H.CA_MODES["brians-brain"], .47, .004, .93, 2024).blend(
        H.noise(27, 0).thresh(.55, .001), once).out(p.o1, scale=1/16, patch=p)
    tiles = H.noise(2.2, .025).pixelate(12, 8)
    rule = tiles.mult(H.solid(7, 7, 7)).add(H.solid(bank, bank, bank))
    rule = rule.add(coarse.mult(H.solid(4, 4, 4)))
    disturbance = coarse.mult(H.solid(.012, .012, .012)).add(
        H.solid(.0004 + hit*.003, .0004 + hit*.003, .0004 + hit*.003))
    H.src(p.o0).ca(rule, .62 + bass*.18, disturbance, .91, 47).blend(
        H.noise(82, 0).thresh(.41, .001), once).out(p.o0, scale=1/4, patch=p)
    return p
