"""Captcha + bbox dataset for the 5-box detector.

Reads `bboxes.jsonl` produced by the PHP renderer with --bboxes. Each line:
    {"filename": "...", "label": "...", "bboxes": [[x0,y0,x1,y1], ... 5x],
     "width": int, "height": int}   # width/height optional for v1 synth

Targets are produced as (cx, cy, w, h) normalised by per-image size.
The image is resized to a canonical (img_h, img_w) for the network input;
bbox normalisation uses the *original* image size from the record.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _to_rgb(im):
    return im.convert("RGB")


class CaptchaBoxDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict] | None = None,
        img_size: tuple[int, int] = (60, 200),  # (H, W) — model input
        augment: bool = False,
    ):
        self.root = Path(root)
        self.net_h, self.net_w = img_size
        if rows is None:
            rows = []
            with (self.root / "bboxes.jsonl").open() as f:
                for line in f:
                    rows.append(json.loads(line))
        self.rows = rows
        self.augment = augment

        # Note: NO geometric augmentation — bbox labels would break.
        # Only photometric.
        photometric = []
        if augment:
            photometric.append(T.ColorJitter(0.25, 0.25, 0.25, 0.08))
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize((self.net_h, self.net_w), antialias=True),
            *photometric,
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        rec = self.rows[i]
        img = Image.open(self.root / rec["filename"])
        # original image size: from JSONL if present (v3 mixed-scale),
        # else fall back to v1/v2 default of 200×60.
        src_w = rec.get("width", 200)
        src_h = rec.get("height", 60)
        x = self.tf(img)
        boxes = []
        for x0, y0, x1, y1 in rec["bboxes"]:
            cx = (x0 + x1) / 2.0 / src_w
            cy = (y0 + y1) / 2.0 / src_h
            w = max(1.0, x1 - x0) / src_w
            h = max(1.0, y1 - y0) / src_h
            boxes.append([cx, cy, w, h])
        # pad/truncate to exactly 5
        while len(boxes) < 5:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        boxes = boxes[:5]
        return x, torch.tensor(boxes, dtype=torch.float32), rec["filename"]


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
