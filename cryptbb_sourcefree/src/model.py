"""Restyler G + PatchGAN D + PatchNCE projection heads.

Reuses `synthetic image generation/src/refine/model.py` (UNetRefiner mask-aware
U-Net + PatchDiscriminator) as ~70% of the stack (PLAN.md). Adds:
  - encoder feature extraction from G (for the CUT PatchNCE loss)
  - PatchSampleF: per-layer MLP projection heads that sample matched patches

CUT (Park ECCV'20): content preservation via PatchNCE instead of cycle. A patch
of the output must be closer to the SAME patch of the input than to other
patches — keeps glyph content in place while style is free to change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# reuse the repo's generator/discriminator
_REFINE = Path(r"E:/pranav/synthetic image generation")
if str(_REFINE) not in sys.path:
    sys.path.insert(0, str(_REFINE))
from src.refine.model import UNetRefiner, PatchDiscriminator  # noqa: E402


class Generator(nn.Module):
    """Mask-conditioned restyler: input (RGB⊕mask)=4ch -> RGB 3ch.

    Wraps UNetRefiner and additionally exposes multi-layer encoder features for
    PatchNCE. Standard (non-residual) mode so it can actually restyle glyphs,
    not just nudge them.
    """

    def __init__(self, base: int = 64, residual: bool = False, residual_scale: float = 0.5):
        super().__init__()
        # residual=True: output = source_RGB + scale*delta (starts at identity =
        # the already-legible composite), a strong content-preservation prior that
        # stops the adversarial loss from smearing glyphs.
        self.net = UNetRefiner(in_ch=4, out_ch=3, base=base,
                               residual=residual, residual_scale=residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def encode_feats(self, x: torch.Tensor, layers=(0, 1, 2, 3)) -> list[torch.Tensor]:
        """Return feature maps at the requested encoder depths.
        layer 0 = raw input (4ch), 1..4 = after d1..d4."""
        feats = {}
        feats[0] = x
        e1 = self.net.d1(x);  feats[1] = e1
        e2 = self.net.d2(e1); feats[2] = e2
        e3 = self.net.d3(e2); feats[3] = e3
        e4 = self.net.d4(e3); feats[4] = e4
        return [feats[i] for i in layers]


class PatchSampleF(nn.Module):
    """CUT's projection: per-feature-layer 2-layer MLP + L2 norm, sampling the
    SAME spatial patch ids from source and target feature stacks."""

    def __init__(self, feat_channels: list[int], nc: int = 256):
        super().__init__()
        self.nc = nc
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(c, nc), nn.ReLU(), nn.Linear(nc, nc))
            for c in feat_channels
        ])

    def forward(self, feats: list[torch.Tensor], num_patches: int = 256,
                patch_ids: list | None = None):
        out, ids = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            f_flat = f.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, HW, C
            if patch_ids is not None:
                pid = patch_ids[i]
            else:
                n = min(num_patches, H * W)
                pid = torch.randperm(H * W, device=f.device)[:n]
            sampled = f_flat[:, pid, :]                          # B, n, C
            sampled = self.mlps[i](sampled.reshape(-1, C))       # (B*n), nc
            sampled = nn.functional.normalize(sampled, dim=1)
            out.append(sampled)
            ids.append(pid)
        return out, ids


def build_models(base: int = 64, nce_layers=(0, 1, 2, 3), nce_dim: int = 256,
                 device="cuda", residual: bool = False, residual_scale: float = 0.5):
    G = Generator(base=base, residual=residual, residual_scale=residual_scale).to(device)
    D = PatchDiscriminator(in_ch=3, base=base).to(device)
    # channel counts at the chosen layers: 0=4(input), 1=base,2=2base,3=4base,4=8base
    ch_by_layer = {0: 4, 1: base, 2: base * 2, 3: base * 4, 4: base * 8}
    F = PatchSampleF([ch_by_layer[l] for l in nce_layers], nc=nce_dim).to(device)
    return G, D, F
