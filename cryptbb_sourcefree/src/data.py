"""Datasets for the restyler.

RealDataset    — the UNLABELED scheme pool (GAN target domain Y). We only look at
                 the *look*; filenames are ignored (never used as labels).
SourceDataset  — our styled glyphs composited onto recreated backgrounds
                 (GAN source domain X). Emits (composite_rgb, mask, label). Label
                 is always ours. This is the input the GAN restyles toward real.

Tensors are in [-1,1] at the GAN working size (H,W), divisible by 16 for the
4-down U-Net. CryptBB native 200x60 -> work 64x208; resized back at gen time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

from src.render_source import SourceRenderer
from src.background import procedural_bg, dark_block_bg, LamaBGPool, composite
from src.distort import apply_profile


def to_tensor_pm1(img: Image.Image) -> torch.Tensor:
    t = T.functional.to_tensor(img.convert("RGB"))
    return t * 2.0 - 1.0


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return T.functional.to_tensor(mask.convert("L"))


class RealDataset(Dataset):
    def __init__(self, real_dir: str | Path, work_hw: tuple[int, int]):
        self.paths = sorted(Path(real_dir).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"no pngs in {real_dir}")
        self.H, self.W = work_hw

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        return to_tensor_pm1(img)


class SourceDataset(Dataset):
    """Styled glyphs composited onto recreated backgrounds.

    bg_mode: 'procedural' | 'lama' | 'mix'. 'mix' samples procedural with prob
    p_procedural else a filtered LaMa background.
    """

    def __init__(
        self,
        renderer: SourceRenderer,
        work_hw: tuple[int, int],
        length: int = 10000,
        bg_mode: str = "mix",
        lama_dir: str | None = None,
        p_procedural: float = 0.5,
        seed: int = 0,
        bg_style: str = "pale",     # "pale" (CryptBB tiles) | "dark" (DarkBay blocks)
        distort_profile: str = "none",   # PHP-GD distortion family (src/distort.py)
    ):
        self.r = renderer
        self.H, self.W = work_hw
        self.length = length
        self.bg_mode = bg_mode
        self.p_procedural = p_procedural
        self.bg_style = bg_style
        self.distort_profile = distort_profile
        self.seed = seed
        self._rng = None
        self.lama = None
        if bg_mode in ("lama", "mix") and lama_dir:
            self.lama = LamaBGPool(lama_dir, (self.W, self.H))
            print(f"[SourceDataset] LaMa bg pool: kept {self.lama.kept}, dropped {self.lama.dropped}")
            if len(self.lama) == 0:
                print("[SourceDataset] WARN: no clean LaMa bgs kept; using procedural only")
                self.bg_mode = "procedural"

    def _get_rng(self):
        # per-worker rng so DataLoader workers diverge
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed * 1000 + wid + 1)
            # Re-seed the RENDERER's rng per worker too — otherwise every worker
            # inherits the same seed and renders an IDENTICAL sequence (duplicate
            # images, ~1/num_workers diversity). This diversifies text + styling.
            import random as _random
            self.r.rng = _random.Random(self.seed * 100003 + wid * 7919 + 17)
        return self._rng

    def __len__(self):
        return self.length

    def _proc_bg(self, rng):
        return dark_block_bg(self.W, self.H, rng) if self.bg_style == "dark" \
            else procedural_bg(self.W, self.H, rng)

    def _make_bg(self, rng) -> np.ndarray:
        if self.bg_mode == "procedural" or self.lama is None:
            return self._proc_bg(rng)
        if self.bg_mode == "lama":
            return self.lama.sample(rng)
        # mix
        if rng.random() < self.p_procedural:
            return self._proc_bg(rng)
        return self.lama.sample(rng)

    def __getitem__(self, i):
        rng = self._get_rng()
        glyph_rgba, mask, label = self.r.render_layer(neutral=False)
        glyph_rgba = glyph_rgba.resize((self.W, self.H), Image.BILINEAR)
        mask = mask.resize((self.W, self.H), Image.BILINEAR)
        bg = self._make_bg(rng)
        comp = composite(glyph_rgba, bg)               # (H,W,3) uint8
        comp_pil = Image.fromarray(comp)
        if self.distort_profile != "none":             # PHP-GD wobble + noise
            comp_pil = apply_profile(comp_pil, rng, self.distort_profile)
        x = to_tensor_pm1(comp_pil)
        m = mask_to_tensor(mask)
        return x, m, label


def source_collate(batch):
    xs, ms, labels = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ms, 0), list(labels)
