"""RefinedGenerator: ParametricGenerator + optional refinement-net pass.

The knob `t` decides how much refinement to apply:
    alpha(t) = max(0, 2t - 1)
At t <= 0.5 the refiner is bypassed entirely (parametric output returned).
At t >= 1.0 alpha = 1 → full refinement.
Between, output is alpha-blended:
    out = alpha * G(composite, mask) + (1 - alpha) * composite

This makes the high-difficulty end of the knob actually realistic, while
keeping the easy/medium range cheap (no GPU pass).

Usage:
    gen = RefinedGenerator(ckpt="checkpoints/darkbay_refiner/latest.pt")
    img, label, bboxes = gen.render_sample(t=0.9)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.render_darkbay_v2 import H, W
from src.generator.parametric import ParametricGenerator
from src.refine.data import (
    PAD_H, PAD_LEFT, PAD_TOP, PAD_W,
    crop_image, mask_from_bboxes, pad_image, pad_mask, to_tensor_chw,
)
from src.refine.model import UNetRefiner

ROOT = Path(__file__).resolve().parents[2]
# v4 step 5000 (alpha-mask, lambda_l1=30) is the best checkpoint we've trained.
# Later steps overtrained — L1 mask drift corrupted text.
DEFAULT_CKPT = ROOT / "checkpoints" / "darkbay_refiner_v4_alpha" / "best.pt"


def _alpha_from_t(t: float) -> float:
    return max(0.0, 2.0 * t - 1.0)


class RefinedGenerator(ParametricGenerator):
    """Drop-in extension of ParametricGenerator that applies a refinement pass
    on the high-t end."""

    def __init__(self, ckpt: Optional[Path] = None,
                 device: Optional[str] = None, residual: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        ckpt = Path(ckpt) if ckpt else DEFAULT_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"refiner checkpoint not found: {ckpt}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = UNetRefiner(residual=residual).to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.refiner.load_state_dict(state["G"])
        for p in self.refiner.parameters():
            p.requires_grad_(False)
        print(f"loaded refiner from {ckpt} (step={state.get('step', '?')})")

    @torch.no_grad()
    def _refine(self, composite: Image.Image, bboxes: list[list[int]]) -> Image.Image:
        x_rgb = to_tensor_chw(pad_image(composite)).unsqueeze(0).to(self.device)
        m = pad_mask(mask_from_bboxes(bboxes))
        mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(self.device)
        x = torch.cat([x_rgb, mask_t], dim=1)
        out = self.refiner(x).clamp(-1.0, 1.0)
        arr = ((out[0].cpu() + 1.0) * 127.5).numpy().astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        arr = crop_image(arr)
        return Image.fromarray(arr)

    def render_sample(self, t: float):
        im, label, bboxes = super().render_sample(t)
        alpha = _alpha_from_t(t)
        if alpha <= 0:
            return im, label, bboxes
        refined = self._refine(im, bboxes)
        if alpha >= 1.0:
            return refined, label, bboxes
        # blend
        a = np.array(im).astype(np.float32)
        b = np.array(refined).astype(np.float32)
        out = (1.0 - alpha) * a + alpha * b
        return Image.fromarray(out.astype(np.uint8)), label, bboxes
