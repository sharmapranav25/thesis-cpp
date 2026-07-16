"""Unpaired dataset for CycleGAN: domain A = synthetic, domain B = real captchas."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def _to_rgb(im):
    return im.convert("RGB")


def _build_transforms(train: bool) -> T.Compose:
    return T.Compose([
        T.Lambda(_to_rgb),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _real_aug() -> T.Compose:
    """Heavier augmentation for the small real domain (597 images) to avoid
    discriminator overfit."""
    return T.Compose([
        T.Lambda(_to_rgb),
        T.RandomApply([T.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.6),
        T.RandomAffine(degrees=2, translate=(0.02, 0.05), scale=(0.97, 1.03)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class UnpairedCaptchaDataset(Dataset):
    """Yield random pairs (A=synth, B=real). Both pools are sampled randomly so
    the virtual length can be set independently to control epoch size."""

    def __init__(self, synth_dir: Path, real_dir: Path, train: bool = True,
                 virtual_len: int = 2000, seed: int = 0):
        self.synth_paths = sorted(synth_dir.glob("*.png"))
        self.real_paths = sorted(p for p in real_dir.glob("*.png")
                                 if not p.name.startswith("_"))
        if not self.synth_paths or not self.real_paths:
            raise RuntimeError(f"empty domain: synth={len(self.synth_paths)} real={len(self.real_paths)}")
        self.tf_A = _build_transforms(train)
        self.tf_B = _real_aug() if train else _build_transforms(False)
        self.virtual_len = virtual_len
        self._seed = seed

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, idx: int) -> dict:
        # per-call rng so multiple workers don't desync deterministically
        rng = random.Random((self._seed * 1_000_003) ^ idx)
        a_path = self.synth_paths[rng.randrange(len(self.synth_paths))]
        b_path = self.real_paths[rng.randrange(len(self.real_paths))]
        a = self.tf_A(Image.open(a_path))
        b = self.tf_B(Image.open(b_path))
        return {"A": a, "B": b, "A_path": str(a_path), "B_path": str(b_path)}


def make_loader(synth_dir: Path, real_dir: Path, batch_size: int, num_workers: int = 4,
                train: bool = True, virtual_len: int = 2000,
                seed: int = 0) -> torch.utils.data.DataLoader:
    ds = UnpairedCaptchaDataset(synth_dir, real_dir, train=train,
                                virtual_len=virtual_len, seed=seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=True, drop_last=train, persistent_workers=num_workers > 0,
    )
