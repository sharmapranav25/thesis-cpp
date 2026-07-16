"""Bg-only supervised data: (LaMa_bg, real_captcha, text_mask) triples.

The bg-only refiner learns f(LaMa_bg) -> real captcha **outside the text
region**. No discriminator, no L1-on-text-region trade-off. Pure supervised
pix2pix-style training — much more stable than the GAN refiners.

Filenames match 1:1 between LaMa output and real captchas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.refine.data import PAD_H, PAD_W, pad_image, pad_mask, to_tensor_chw

PIX_DIFF_THR = 25  # 0..255: pixel changed by more than this between real and LaMa = text
DILATE_ITERS = 1


def text_mask_from_diff(real_rgb: np.ndarray, lama_rgb: np.ndarray) -> np.ndarray:
    """Text mask = where LaMa changed the image substantially.
    More accurate than brightness-thresholding because it adapts to each
    captcha's bg color (dark/light wallpaper alike)."""
    diff = np.abs(real_rgb.astype(np.int16) - lama_rgb.astype(np.int16)).max(axis=-1)
    mask = (diff > PIX_DIFF_THR).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERS)
    return (mask > 0).astype(np.float32)


class BgPairDataset(Dataset):
    """Each item:
        - lama:        3 x PAD_H x PAD_W float [-1, 1] — LaMa-inpainted bg
        - real:        3 x PAD_H x PAD_W float [-1, 1] — original captcha
        - bg_mask:     1 x PAD_H x PAD_W float [0, 1]  — 1 OUTSIDE text (loss applies here)
    """
    def __init__(self, lama_dir: Path, real_dir: Path):
        self.lama_dir = Path(lama_dir)
        self.real_dir = Path(real_dir)
        self.names = sorted(p.name for p in self.lama_dir.glob("*.png"))
        if not self.names:
            raise RuntimeError(f"no LaMa bgs in {lama_dir}")
        # verify pairing on a couple samples
        for n in self.names[:3]:
            if not (self.real_dir / n).exists():
                raise RuntimeError(f"missing real pair: {self.real_dir / n}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        n = self.names[idx]
        lama_pil = Image.open(self.lama_dir / n).convert("RGB")
        real_pil = Image.open(self.real_dir / n).convert("RGB")
        if real_pil.size != lama_pil.size:
            real_pil = real_pil.resize(lama_pil.size)

        text_mask = text_mask_from_diff(np.array(real_pil), np.array(lama_pil))
        bg_mask = 1.0 - text_mask                            # 1 where bg

        lama_t = to_tensor_chw(pad_image(lama_pil))
        real_t = to_tensor_chw(pad_image(real_pil))
        bg_mask_t = torch.from_numpy(pad_mask(bg_mask)).unsqueeze(0)
        return {"lama": lama_t, "real": real_t, "bg_mask": bg_mask_t}


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    ds = BgPairDataset(
        lama_dir=ROOT / "data" / "darkbay_bg_lama",
        real_dir=Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled"),
    )
    print(f"len = {len(ds)}")
    s = ds[0]
    print(f"lama   {tuple(s['lama'].shape)} min={s['lama'].min():.3f} max={s['lama'].max():.3f}")
    print(f"real   {tuple(s['real'].shape)} min={s['real'].min():.3f} max={s['real'].max():.3f}")
    print(f"bg_mask {tuple(s['bg_mask'].shape)} bg_frac={s['bg_mask'].mean().item():.3f}")
