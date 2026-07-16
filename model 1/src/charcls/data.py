"""Single-char crop dataset for the classifier.

Reads `bboxes.jsonl` from a synth dir. Each (image, char) pair becomes one
sample: crop that bbox, pad, jitter (training), resize to fixed square,
output (tensor, label_idx). Labels come from the renderer-truth captcha
string, indexed by bbox position.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.charcls.model import CHAR_TO_IDX, char_to_case


def _to_rgb(im):
    return im.convert("RGB")


def _crop_with_pad(
    img: Image.Image, box: list[float], pad_frac: float, jitter_frac: float = 0.0
) -> Image.Image:
    x0, y0, x1, y1 = box
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    pad_w = w * pad_frac
    pad_h = h * pad_frac
    x0 -= pad_w; x1 += pad_w
    y0 -= pad_h; y1 += pad_h
    if jitter_frac > 0.0:
        jx = w * jitter_frac
        jy = h * jitter_frac
        x0 += random.uniform(-jx, jx)
        x1 += random.uniform(-jx, jx)
        y0 += random.uniform(-jy, jy)
        y1 += random.uniform(-jy, jy)
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(img.width - 1, int(round(x1)))
    y1 = min(img.height - 1, int(round(y1)))
    if x1 <= x0: x1 = x0 + 1
    if y1 <= y0: y1 = y0 + 1
    return img.crop((x0, y0, x1, y1))


class CharCropDataset(Dataset):
    """One captcha row contributes 5 samples (one per char). Indexing is
    (row_idx * 5 + char_idx). Cheap: each access opens one image and crops
    once."""

    def __init__(
        self,
        root: Path,
        rows: list[dict] | None = None,
        crop_size: int = 64,
        pad_frac: float = 0.20,
        jitter_frac: float = 0.0,
        augment: bool = False,
    ):
        self.root = Path(root)
        if rows is None:
            rows = []
            with (self.root / "bboxes.jsonl").open() as f:
                for line in f:
                    rows.append(json.loads(line))
        self.rows = rows
        self.crop_size = crop_size
        self.pad_frac = pad_frac
        self.jitter_frac = jitter_frac

        photometric = []
        if augment:
            photometric += [
                T.ColorJitter(0.30, 0.30, 0.30, 0.10),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
            ]
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize((crop_size, crop_size), antialias=True),
            *photometric,
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

        # erase the post-tensor random erasing if augment
        self.erase = T.RandomErasing(p=0.20, scale=(0.02, 0.10)) if augment else None

    def __len__(self) -> int:
        return len(self.rows) * 5

    def __getitem__(self, i: int):
        row_i, c_i = divmod(i, 5)
        rec = self.rows[row_i]
        img = Image.open(self.root / rec["filename"]).convert("RGB")
        crop = _crop_with_pad(
            img, rec["bboxes"][c_i], self.pad_frac, self.jitter_frac
        )
        x = self.tf(crop)
        if self.erase is not None:
            x = self.erase(x)
        ch = rec["label"][c_i]
        return x, CHAR_TO_IDX[ch], char_to_case(ch)


def split_rows(root: Path, val_frac: float = 0.05, seed: int = 0):
    rows = []
    with (root / "bboxes.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(rows) * val_frac)
    perm = torch.randperm(len(rows), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return train, val
