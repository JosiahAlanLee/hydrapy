"""Train a texture NCA from one image (needs PyTorch: ``pip install torch``).

    python -m hydrapy.nca_train lichen.jpg --out lichen.npz --iters 2000
    hydrapy nca train lichen.jpg --out lichen.npz

The loss compares *distributions of small patches* between the NCA's output and the photograph,
at three scales, with a sliced Wasserstein distance (random projections, sorted and matched).
That is the statistical core of texture synthesis without a pretrained network: nothing is
downloaded, no one else's images are involved, and it trains on a laptop CPU in tens of minutes
(a GPU or Apple MPS in a few).  The rule learns from a pool of states that it keeps running, so it
learns to keep the texture alive, not only to make it once (Niklasson et al., 2021).

``scale`` decides how big the texture's features are in cells: the exemplar is resized so its
shorter side is that many pixels.  Features about 4-16 cells across work best; shrink the
photograph (smaller ``scale``) if the result is mush, enlarge it if it is noise.
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Callable, Optional

import numpy as np

from .nca import CLAMP, FILTERS, NCAModel


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("training an NCA needs PyTorch: pip install torch") from e


def load_exemplar(image, scale: int = 128) -> np.ndarray:
    """An image path / PIL image / array -> float32 (h, w, 3) in 0..1, shorter side = ``scale`` px."""
    from PIL import Image
    if isinstance(image, np.ndarray):
        a = image.astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        im = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8))
    elif isinstance(image, str):
        im = Image.open(image)
    else:
        im = image
    im = im.convert("RGB")
    w, h = im.size
    k = scale / min(w, h)
    im = im.resize((max(int(round(w * k)), 8), max(int(round(h * k)), 8)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def _build(torch, channels, hidden):
    nn, F = torch.nn, torch.nn.functional
    filt = torch.tensor(FILTERS)[:, None]                       # (4, 1, 3, 3), image-oriented like torch tensors

    class CA(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Conv2d(4 * channels, hidden, 1)
            self.w2 = nn.Conv2d(hidden, channels, 1, bias=False)
            nn.init.zeros_(self.w2.weight)
            self.register_buffer("filt", filt)

        def perceive(self, x):
            b, c, h, w = x.shape
            y = F.pad(x.reshape(b * c, 1, h, w), [1, 1, 1, 1], mode="circular")
            return F.conv2d(y, self.filt).reshape(b, 4 * c, h, w)     # index c * 4 + f

        def forward(self, x, rate=0.5):
            dx = self.w2(torch.relu(self.w1(self.perceive(x))))
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < rate).to(x.dtype)
            return (x + dx * mask).clamp(-CLAMP, CLAMP)

    return CA()


def _patches(torch, img, scale, k):
    """(B, 3, H, W) -> (N, 3 k k) patches of the image downsampled by ``scale``."""
    F = torch.nn.functional
    if scale > 1:
        img = F.avg_pool2d(img, scale)
    p = F.unfold(img, k)                                         # (B, 3kk, L)
    return p.permute(0, 2, 1).reshape(-1, p.shape[1])


class _Loss:
    """Texture statistics of an image batch compared with an exemplar's, without a pretrained net.

    ``gram``: Gram matrices of fixed random ReLU filters at several scales (Ustyuzhaninov et al.
    2017: random multi-scale filters capture texture surprisingly well), plus a sliced-Wasserstein
    match of the colour distribution.  ``sw``: sliced Wasserstein on raw patches at several scales.
    """

    def __init__(self, torch, target, kind="gram", scales=(1, 2, 4, 8), patch=5, features=64, ksize=7,
                 projections=64, seed=0):
        self.torch, self.kind, self.scales, self.patch, self.projections = torch, kind, scales, patch, projections
        g = torch.Generator().manual_seed(seed + 17)
        dev = target.device
        self.filters = []
        for _ in scales:
            w = torch.randn(features, 3, ksize, ksize, generator=g)
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)
            self.filters.append((w / w.flatten(1).norm(dim=1)[:, None, None, None]).to(dev))
        self.mean = target.mean(dim=(0, 2, 3), keepdim=True)
        self.t_grams = [self._grams(target, i) for i in range(len(scales))]
        self.t_colours = target.permute(0, 2, 3, 1).reshape(-1, 3)
        self.t_patches = [self._patches(target, sc) for sc in scales] if kind in ("sw", "both") else []

    def _feat(self, img, i):
        F = self.torch.nn.functional
        sc = self.scales[i]
        x = F.avg_pool2d(img, sc) if sc > 1 else img
        x = F.pad(x - self.mean, [self.filters[i].shape[-1] // 2] * 4, mode="circular")
        return self.torch.relu(F.conv2d(x, self.filters[i]))

    def _grams(self, img, i):
        f = self._feat(img, i)
        b, c, h, w = f.shape
        f = f.reshape(b, c, h * w)
        return f @ f.transpose(1, 2) / (h * w)

    def _patches(self, img, sc):
        F = self.torch.nn.functional
        if sc > 1:
            img = F.avg_pool2d(img, sc)
        p = F.unfold(img, self.patch)
        return p.permute(0, 2, 1).reshape(-1, p.shape[1])

    def _sw(self, a, b):
        torch = self.torch
        proj = torch.randn(a.shape[1], self.projections, device=a.device)
        proj = proj / proj.norm(dim=0, keepdim=True)
        sel = torch.randint(0, b.shape[0], (a.shape[0],), device=a.device)
        return ((torch.sort(a @ proj, dim=0)[0] - torch.sort(b[sel] @ proj, dim=0)[0]) ** 2).mean()

    def __call__(self, rgb):
        loss = 0.0
        if self.kind in ("gram", "both"):
            for i in range(len(self.scales)):
                loss = loss + ((self._grams(rgb, i) - self.t_grams[i]) ** 2).mean() * 10.0
            loss = loss + self._sw(rgb.permute(0, 2, 3, 1).reshape(-1, 3), self.t_colours)
        if self.kind in ("sw", "both"):
            for sc, tp in zip(self.scales[:3], self.t_patches[:3]):
                loss = loss + self._sw(self._patches(rgb, sc), tp)
        return loss


def train(image, out: Optional[str] = None, scale: int = 96, size: int = 64, channels: int = 12, hidden: int = 96,
          iters: int = 2000, batch: int = 4, pool_size: int = 256, steps=(32, 64), lr: float = 1e-3,
          loss: str = "gram", scales=(1, 2, 4, 8), seed: int = 0, device: Optional[str] = None,
          log_every: int = 50, callback: Optional[Callable] = None, threads: Optional[int] = None,
          init=None) -> NCAModel:
    """Train and return an ``NCAModel`` (also saved to ``out`` if given).

    ``init``: an ``NCAModel`` (or a path to one) to start from — fine-tune a rule with longer
    ``steps`` and a lower ``lr`` when it holds its texture for a few hundred steps and then drifts."""
    torch = _torch()
    if threads:
        torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    ex = load_exemplar(image, scale)
    target = torch.tensor(ex).permute(2, 0, 1)[None].to(device)
    scales = tuple(sc for sc in scales if size // sc >= 8 and min(ex.shape[:2]) // sc >= 8) or (1,)   # room for the filters
    texture_loss = _Loss(torch, target, kind=loss, scales=scales, seed=seed)
    if init is not None:
        init = init if isinstance(init, NCAModel) else NCAModel.load(init)
        channels, hidden = init.channels, init.hidden
    ca = _build(torch, channels, hidden)
    if init is not None:
        with torch.no_grad():
            ca.w1.weight[:] = torch.tensor(init.w1)[:, :, None, None]
            ca.w1.bias[:] = torch.tensor(init.b1)
            ca.w2.weight[:] = torch.tensor(init.w2)[:, :, None, None]
    ca = ca.to(device)
    opt = torch.optim.Adam(ca.parameters(), lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.5), int(iters * 0.8)], 0.3)
    pool = torch.zeros(pool_size, channels, size, size, device=device)
    log = []
    t0 = time.time()
    for it in range(iters):
        idx = rng.choice(pool_size, batch, replace=False)
        x = pool[idx]
        if it % 8 == 0:
            x[:1] = 0.0                                          # a fresh seed: it must also grow from nothing
        for _ in range(int(rng.integers(steps[0], steps[1] + 1))):
            x = ca(x)
        tl = texture_loss(x[:, :3] + 0.5)
        overflow = (x - x.clamp(-1.0, 1.0)).abs().sum() / batch    # keep the state in range, hard
        total = tl + overflow
        opt.zero_grad()
        total.backward()
        with torch.no_grad():
            for p in ca.parameters():                            # per-parameter gradient normalisation (Distill)
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        with torch.no_grad():
            x = x.detach()
            blown = x.abs().amax(dim=(1, 2, 3)) > 1.5        # a state that ran away goes back to a seed
            x[blown] = 0.0
            pool[idx] = x
        log.append(float(tl.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"it {it:5d}  loss {np.mean(log[-log_every:]):.5f}  overflow {float(overflow.detach()):.4f}  {time.time() - t0:6.0f}s",
                  flush=True)
        if callback is not None:
            callback(it, log[-1], ca, x.detach())
    w1 = ca.w1.weight.detach().cpu().numpy()[:, :, 0, 0]
    b1 = ca.w1.bias.detach().cpu().numpy()
    w2 = ca.w2.weight.detach().cpu().numpy()[:, :, 0, 0]
    name = image if isinstance(image, str) else "array"
    model = NCAModel(w1, b1, w2, {"source": str(name), "scale": scale, "iters": iters, "loss": float(np.mean(log[-50:]))})
    if out:
        model.save(out)
    return model


def torch_step(model: NCAModel, state_img: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """One step of ``model`` in PyTorch on an *image-oriented* (h, w, C) state (for checking the runtime)."""
    torch = _torch()
    ca = _build(torch, model.channels, model.hidden)
    with torch.no_grad():
        ca.w1.weight[:] = torch.tensor(model.w1)[:, :, None, None]
        ca.w1.bias[:] = torch.tensor(model.b1)
        ca.w2.weight[:] = torch.tensor(model.w2)[:, :, None, None]
        x = torch.tensor(state_img).permute(2, 0, 1)[None]
        y = ca(x, rate=rate)
    return y[0].permute(1, 2, 0).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description="train a texture NCA from one image")
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=int, default=96, help="exemplar's shorter side in cells (feature size)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="gram", choices=["gram", "sw", "both"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, nargs=2, default=(32, 64), metavar=("MIN", "MAX"), help="rollout length range")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None, help="a trained .npz to fine-tune")
    a = ap.parse_args(argv)
    import os
    out = a.out or os.path.splitext(os.path.basename(a.image))[0] + ".npz"
    train(a.image, out, scale=a.scale, size=a.size, channels=a.channels, hidden=a.hidden, iters=a.iters,
          loss=a.loss, seed=a.seed, device=a.device, steps=tuple(a.steps), lr=a.lr, init=a.init)
    print("wrote", out)


if __name__ == "__main__":
    main()
