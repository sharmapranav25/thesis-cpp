"""Synthetic-captcha dataset for TinySTR.

Reuses Model 1's synth_mybb (100k PHP-MyBB renders) and mirrors PARSeq's
augmentation recipe so the only difference between the two trainings is the
model architecture, not the data pipeline. See ../CONTEXT.md.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from model import CHARSET, NUM_POSITIONS

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}


def _to_rgb(im):
    return im.convert("RGB")


def build_train_augment() -> A.Compose:
    """Same albumentations recipe PARSeq used (model 1/src/parseq/train.py)."""
    return A.Compose([
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.4),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(50, 95), p=0.4),
        A.Affine(
            rotate=(-3, 3),
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            scale=(0.95, 1.05),
            shear={"x": (-3, 3)},
            border_mode=0,
            fill=255,
            p=0.6,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(2, 8),
            hole_width_range=(2, 12),
            fill=255,
            p=0.3,
        ),
    ])


def encode_label(label: str) -> torch.Tensor:
    if len(label) != NUM_POSITIONS:
        raise ValueError(f"label {label!r} has {len(label)} chars, expected {NUM_POSITIONS}")
    return torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)


class CaptchaDir(Dataset):
    """Read (filename,label) from `root/labels.csv`, return (image, indices)."""

    def __init__(
        self,
        root: Path | str,
        img_size: tuple[int, int] = (60, 200),  # (H, W)
        rows: Sequence[tuple[str, str]] | None = None,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size
        if rows is None:
            with (self.root / "labels.csv").open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # header
                self.rows = [(fn, lbl) for fn, lbl in reader]
        else:
            self.rows = list(rows)
        self.aug = build_train_augment() if augment else None
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
            img = Image.fromarray(arr)
        img = self.tf(img)
        targets = encode_label(lbl)
        return img, targets


def split_rows(
    root: Path | str,
    val_frac: float = 0.05,
    seed: int = 0,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Same deterministic split PARSeq used (val_frac=0.05, seed=0)."""
    root = Path(root)
    with (root / "labels.csv").open("r", encoding="utf-8") as f:
        rows = [tuple(r) for r in csv.reader(f)][1:]
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(rows) * val_frac)
    perm = torch.randperm(len(rows), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows
