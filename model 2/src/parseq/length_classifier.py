"""Standalone length-mode classifier.

Input: 32x128 captcha image.
Output: P(is_fixed_length) — single binary.

Trained on synth_curriculum's labels.csv (which has is_fixed_length per image).
At inference time the system uses this to ROUTE images to specialist models:
  is_fixed_prob > 0.5  ->  fixed-N specialist (currently fixed-5 only)
  is_fixed_prob <= 0.5 ->  variable-length specialist (TBD)

Architecture: ResNet-18 from timm with num_classes=1 (binary). Tiny vs PARSeq;
this is a dedicated gating model, not a workhorse.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import timm

# Reuse CaptchaDir + collate from train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.parseq.train import CaptchaDir, captcha_collate, _read_rows


class LengthClassifier(pl.LightningModule):
    """Resnet-18 → 1 logit (sigmoid → P(is_fixed_length))."""

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, model_name: str = "resnet18"):
        super().__init__()
        self.save_hyperparameters()
        # in_chans=3 matches CaptchaDir's RGB output.
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=1, in_chans=3,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)  # [N]

    def training_step(self, batch, batch_idx):
        imgs, _labels, is_fixed = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logit, is_fixed)
        with torch.no_grad():
            pred = (logit > 0).float()
            acc = (pred == is_fixed).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _labels, is_fixed = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logit, is_fixed)
        pred = (logit > 0).float()
        acc = (pred == is_fixed).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt


def split_rows_for_classifier(root: Path, val_frac: float, seed: int):
    rows = _read_rows(root)
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(rows) * val_frac)
    perm = torch.randperm(len(rows), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


def main(cfg: dict) -> None:
    root = Path(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])

    train_rows, val_rows = split_rows_for_classifier(root, cfg["val_frac"], cfg["seed"])
    print(f"train={len(train_rows)} val={len(val_rows)}")

    # Augment less aggressively than the PARSeq trainer — classifier just needs
    # to identify per-image distribution features, not solve the chars.
    train_ds = CaptchaDir(root, img_size, rows=train_rows, augment=False)
    val_ds = CaptchaDir(root, img_size, rows=val_rows, augment=False)

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

    model = LengthClassifier(
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        model_name=cfg.get("model_name", "resnet18"),
    )

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="lenclass-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=cfg.get("precision", "32-true"),
        callbacks=[ckpt_cb],
        log_every_n_steps=20,
        default_root_dir=cfg["ckpt_dir"],
    )

    trainer.fit(model, train_loader, val_loader)
    print("best val_acc:", ckpt_cb.best_model_score)
    print("best ckpt:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/length_classifier.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg)
