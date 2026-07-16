"""PARSeq training on the CycleGAN-translated synthetic captcha set.

We bypass PARSeq's LMDB+Hydra training entry point and reuse only the model
itself (`strhub.models.parseq.system.PARSeq`). Data is read directly from
data/translated/labels.csv as (image, label_string) pairs."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Anaconda-base venv on Windows: the venv inherits Anaconda's stdlib but the
# native DLLs (sqlite3.dll, etc.) live under Library/bin and aren't on PATH by
# default. Add them so `import nltk` (used by PARSeq's base.py) doesn't blow
# up on its sqlite3 import chain.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

import albumentations as A

from strhub.models.parseq.system import PARSeq  # noqa: E402


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_rgb(im):
    return im.convert("RGB")


def _build_train_augment() -> A.Compose:
    """Diversity #1: heavy training-time augmentation. Each batch sees a
    different perturbation so the model can't memorise pixel patterns."""
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


class CaptchaDir(Dataset):
    """Read (filename,label) from labels.csv in `root`. Resize each image to
    img_size=(H, W). Optionally apply heavy albumentations augmentation
    (training only)."""

    def __init__(
        self, root: Path, img_size: tuple[int, int],
        rows: list[tuple[str, str]] | None = None, augment: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size  # (H, W)
        if rows is None:
            with (root / "labels.csv").open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # header
                self.rows = [(fn, lbl) for fn, lbl in reader]
        else:
            self.rows = rows
        self.aug = _build_train_augment() if augment else None
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
            img = Image.fromarray(arr)
        img = self.tf(img)
        return img, lbl


def captcha_collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)


def split_rows(root: Path, val_frac: float, seed: int) -> tuple[list, list]:
    with (root / "labels.csv").open("r", encoding="utf-8") as f:
        rows = [tuple(r) for r in csv.reader(f)][1:]
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(rows) * val_frac)
    perm = torch.randperm(len(rows), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


def build_parseq(cfg: dict) -> PARSeq:
    return PARSeq(
        charset_train=CHARSET,
        charset_test=CHARSET,
        max_label_length=cfg["max_label_length"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        warmup_pct=cfg["warmup_pct"],
        weight_decay=cfg["weight_decay"],
        img_size=tuple(cfg["img_size"]),
        patch_size=tuple(cfg["patch_size"]),
        embed_dim=cfg["embed_dim"],
        enc_num_heads=cfg["enc_num_heads"],
        enc_mlp_ratio=cfg["enc_mlp_ratio"],
        enc_depth=cfg["enc_depth"],
        dec_num_heads=cfg["dec_num_heads"],
        dec_mlp_ratio=cfg["dec_mlp_ratio"],
        dec_depth=cfg["dec_depth"],
        perm_num=cfg["perm_num"],
        perm_forward=cfg["perm_forward"],
        perm_mirrored=cfg["perm_mirrored"],
        decode_ar=cfg["decode_ar"],
        refine_iters=cfg["refine_iters"],
        dropout=cfg["dropout"],
    )


def main(cfg: dict) -> None:
    root = Path(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])

    train_rows, val_rows = split_rows(root, cfg["val_frac"], cfg["seed"])
    print(f"train={len(train_rows)} val={len(val_rows)}")

    train_ds = CaptchaDir(root, img_size, rows=train_rows, augment=True)
    val_ds = CaptchaDir(root, img_size, rows=val_rows, augment=False)

    # Optional second source: pseudo-labels on real captchas. Repeated
    # to give them weight against the much-larger synth set. Validation
    # stays synth-only — we want the val signal to reflect synth quality.
    if cfg.get("extra_data_dir") and cfg.get("extra_labels_csv"):
        from torch.utils.data import ConcatDataset
        extra_root = Path(cfg["extra_data_dir"])
        extra_rows = []
        with open(cfg["extra_labels_csv"]) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                if not row:
                    continue
                extra_rows.append((row[0], row[1]))
        extra_ds = CaptchaDir(extra_root, img_size, rows=extra_rows, augment=True)
        repeat = int(cfg.get("extra_repeat", 1))
        n_unique = len(extra_rows)
        print(f"extra: {n_unique} pseudo-labels from {extra_root}, repeat={repeat}")
        if repeat > 1:
            extra_ds = ConcatDataset([extra_ds] * repeat)
        train_ds = ConcatDataset([train_ds, extra_ds])
        print(f"combined train: {len(train_ds)} samples "
              f"(synth={len(train_rows)}, pseudo×{repeat}={n_unique * repeat})")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=captcha_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=captcha_collate,
    )

    model = build_parseq(cfg)

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="parseq-{epoch:02d}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=cfg["precision"],
        gradient_clip_val=cfg.get("gradient_clip_val", 20),
        callbacks=[ckpt_cb],
        log_every_n_steps=50,
        default_root_dir=cfg["ckpt_dir"],
    )

    trainer.fit(model, train_loader, val_loader)
    print("best val_accuracy:", ckpt_cb.best_model_score)
    print("best ckpt:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/parseq.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg)
