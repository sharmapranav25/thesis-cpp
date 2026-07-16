"""Refinement-net training data.

Two streams, mixed in the training loop:
  - SyntheticPairDataset → (rough_composite, text_mask, label_text). The refiner
    will preserve text via L1 on the mask region, so it must know where text is.
  - RealCaptchaDataset → unpaired real samples for the discriminator.

All images padded from native (200×50) to (208×64) so 4 stride-2 downsamples
divide cleanly. Pad with edge replication so the boundary doesn't bias the conv.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.render_darkbay_v2 import H, W

if TYPE_CHECKING:
    from src.generator.parametric import ParametricGenerator

PAD_W = 208  # 200 + 8
PAD_H = 64   # 50 + 14
PAD_LEFT = (PAD_W - W) // 2     # 4
PAD_TOP = (PAD_H - H) // 2      # 7


def pad_image(im: Image.Image) -> Image.Image:
    """Pad to (PAD_W, PAD_H) by edge-replication."""
    arr = np.array(im.convert("RGB"))
    padded = np.pad(arr, ((PAD_TOP, PAD_H - H - PAD_TOP),
                          (PAD_LEFT, PAD_W - W - PAD_LEFT),
                          (0, 0)), mode="edge")
    return Image.fromarray(padded)


def crop_image(arr: np.ndarray) -> np.ndarray:
    """Crop a (PAD_H, PAD_W, C) array back to (H, W, C)."""
    return arr[PAD_TOP:PAD_TOP + H, PAD_LEFT:PAD_LEFT + W]


def to_tensor_chw(im: Image.Image) -> torch.Tensor:
    """Image in [0, 255] → float tensor in [-1, 1], CHW."""
    arr = np.array(im).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def mask_from_bboxes(bboxes: list[list[int]], erode: int = 0) -> np.ndarray:
    """Build a binary text mask at native (H, W) from per-char bboxes.
    `erode` shrinks each bbox inward by N pixels per side, leaving more bg
    around glyphs free to change under the L1 constraint.
    """
    m = np.zeros((H, W), dtype=np.float32)
    for x0, y0, x1, y1 in bboxes:
        x0 = max(0, x0 + erode); y0 = max(0, y0 + erode)
        x1 = min(W, x1 - erode); y1 = min(H, y1 - erode)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    return m


def pad_mask(m: np.ndarray) -> np.ndarray:
    return np.pad(m, ((PAD_TOP, PAD_H - H - PAD_TOP),
                      (PAD_LEFT, PAD_W - W - PAD_LEFT)), mode="constant")


def dilate_mask(m: np.ndarray, radius: int) -> np.ndarray:
    """Tiny morphological dilation by max-pool over (2r+1)x(2r+1) window.
    No scipy dependency — keeps this file lightweight."""
    if radius <= 0:
        return m
    h, w = m.shape
    out = m.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(m)
            ys = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, dx), min(w, w + dx))
            ys_src = slice(max(0, -dy), min(h, h - dy))
            xs_src = slice(max(0, -dx), min(w, w - dx))
            shifted[ys, xs] = m[ys_src, xs_src]
            out = np.maximum(out, shifted)
    return out


class SyntheticPairDataset(Dataset):
    """Each item is a freshly-rendered rough composite + its text mask.
    Length is virtual — the dataset is infinite-resampling. Use a fixed
    `length` to define an "epoch."

    `use_alpha_mask=True` uses the exact glyph silhouette (alpha channel of
    the rendered text) as the L1-protection mask — much tighter than a bbox
    mask and prevents the adversarial loss from corrupting glyph identity.
    `mask_dilate` then thickens the mask by that many pixels for safety.
    """
    def __init__(self, generator: "ParametricGenerator", length: int = 2000,
                 t: float = 1.0, mask_erode: int = 0,
                 use_alpha_mask: bool = False, mask_dilate: int = 0,
                 alpha_threshold: float = 0.1):
        self.gen = generator
        self.length = length
        self.t = t
        self.mask_erode = mask_erode
        self.use_alpha_mask = use_alpha_mask
        self.mask_dilate = mask_dilate
        self.alpha_threshold = alpha_threshold

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict:
        if self.use_alpha_mask:
            im, label, _bboxes, alpha = self.gen.render_sample_with_mask(self.t)
            mask = (alpha > self.alpha_threshold).astype(np.float32)
            if self.mask_dilate > 0:
                mask = dilate_mask(mask, self.mask_dilate)
        else:
            im, label, bboxes = self.gen.render_sample(self.t)
            mask = mask_from_bboxes(bboxes, erode=self.mask_erode)
        im_t = to_tensor_chw(pad_image(im))                          # 3xPAD_HxPAD_W
        mask_t = torch.from_numpy(pad_mask(mask)).unsqueeze(0)        # 1xPAD_HxPAD_W
        return {"composite": im_t, "mask": mask_t, "label": label}


class RealCaptchaDataset(Dataset):
    """Unpaired real samples for the discriminator. Output: padded image tensor."""
    def __init__(self, real_dir: Path):
        self.paths = sorted(Path(real_dir).glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"no real captchas in {real_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        im = Image.open(self.paths[idx]).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H))
        return to_tensor_chw(pad_image(im))


if __name__ == "__main__":
    # Sanity-check: build one of each, print shapes.
    gen = ParametricGenerator(seed=0)
    syn = SyntheticPairDataset(gen, length=4)
    real = RealCaptchaDataset(Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled"))
    s = syn[0]
    r = real[0]
    print(f"syn composite: {s['composite'].shape} dtype={s['composite'].dtype} "
          f"min={s['composite'].min():.3f} max={s['composite'].max():.3f}")
    print(f"syn mask:      {s['mask'].shape} sum={s['mask'].sum().item():.0f} "
          f"label={s['label']}")
    print(f"real:          {r.shape} dtype={r.dtype} "
          f"min={r.min():.3f} max={r.max():.3f}")
    print(f"len(real) = {len(real)}")
