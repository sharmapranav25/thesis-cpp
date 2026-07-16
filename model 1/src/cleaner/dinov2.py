"""DINOv2 ViT-L/14 wrapper for self-supervised character saliency.

We use timm's DINOv2 weights (Python 3.9 compatible, unlike the official
torch.hub repo). Saliency is derived from the L2 norm of last-layer patch
tokens — a known DINOv2 property where foreground patches have higher norm
than background. This is more uniform across multi-object scenes (like
multi-character captchas) than [CLS]→patches attention, which tends to
concentrate on a single most-salient region."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Anaconda-base venv on Windows: the venv inherits Anaconda's stdlib but
# native DLLs (sqlite3.dll etc.) live under Library/bin. Add to DLL path.
_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn.functional as F
import timm

# DINOv2 patch size and target input dims for our 200x60 captchas.
# Chosen so dims divide cleanly by 14 and roughly preserve the 3.33:1 aspect.
PATCH = 14
TARGET_H = 84   # 6 patches tall
TARGET_W = 252  # 18 patches wide  → 108 patch tokens


class Dinov2Saliency:
    def __init__(self, device: torch.device | str = "cuda", model_name: str = "vit_large_patch14_dinov2.lvd142m"):
        self.device = torch.device(device)
        self.model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation (DINOv2's training norm)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # The model has a CLS token at index 0; some DINOv2 variants also have
        # register tokens. Determine prefix tokens by introspecting model attrs.
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)

    @torch.inference_mode()
    def saliency(self, imgs_uint8: np.ndarray) -> np.ndarray:
        """imgs_uint8: (B, H, W, 3) uint8 — input captchas in original size.
        Returns: (B, H, W) float32 in [0, 1] — saliency upsampled to input size.

        Saliency = first principal component of patch token features per image.
        DINOv2's first PC empirically separates foreground from background on
        most natural images. For each batch element we run PCA independently."""
        if imgs_uint8.ndim == 3:
            imgs_uint8 = imgs_uint8[None]
        B, H, W, _ = imgs_uint8.shape

        # to tensor [B,3,H,W] in [0,1]
        x = torch.from_numpy(imgs_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        feats = self.model.forward_features(x)  # (B, 1+P, D)
        patches = feats[:, self.num_prefix_tokens:, :]  # (B, P, D)

        n_h = TARGET_H // PATCH
        n_w = TARGET_W // PATCH
        n_patches = n_h * n_w
        assert patches.shape[1] == n_patches, f"got {patches.shape[1]} patches, expected {n_patches}"

        # Per-image PCA: first principal component of (P, D) → (P,) saliency
        sal_list = []
        for i in range(B):
            P = patches[i]                       # (P, D)
            P_centered = P - P.mean(dim=0, keepdim=True)
            # SVD; we only need the first right singular vector
            U, S, V = torch.linalg.svd(P_centered, full_matrices=False)
            pc1 = P_centered @ V[0]              # (P,)
            # PC sign is arbitrary; flip so that the more populous side is "low"
            # (assumed background) — use median as the baseline
            if (pc1 > pc1.median()).float().mean() > 0.5:
                pc1 = -pc1
            # Now low pc1 = background, high pc1 = foreground (mostly)
            # Convert to [0,1]: shift so background ~0, foreground ~1
            pc1_min = pc1.min()
            pc1_max = pc1.max()
            sal_i = (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8)
            sal_list.append(sal_i)
        sal = torch.stack(sal_list, dim=0)        # (B, P)

        sal = sal.reshape(B, 1, n_h, n_w)
        sal = F.interpolate(sal, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        return sal.cpu().numpy().astype(np.float32)
