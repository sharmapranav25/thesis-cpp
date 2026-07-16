"""Losses for the recognition-anchored CUT restyler (RESEARCH §7).

  L = adv * L_GAN + nce * L_PatchNCE + mask_l1 * L1_mask + rec * L_rec

L_PatchNCE  — content preservation (CUT): output patch matches its own input
              patch vs. other patches. Replaces CycleGAN's cycle loss.
L1_mask     — soft pixel fidelity where the glyphs are (letters stay put).
L_rec       — recognition consistency (added in M2): a frozen reader must read
              the correct label out of G(x). Implemented in parseq_reader.py.
L_GAN       — LSGAN/BCE adversarial: G(x) looks like a real scheme image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """CUT PatchNCE. Given L2-normalized projected features for source (input)
    and target (output) at matched patch ids, each output patch's positive is
    the same-location input patch; negatives are the other patches in the same
    image. Cross-entropy over similarities / tau."""

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_src: torch.Tensor, feat_tgt: torch.Tensor, batch: int) -> torch.Tensor:
        # feat_*: (B*n, C) L2-normalized. Reshape to per-image (B, n, C).
        Bn, C = feat_src.shape
        n = Bn // batch
        fs = feat_src.reshape(batch, n, C)
        ft = feat_tgt.reshape(batch, n, C)
        # similarity matrix per image: (B, n, n) = ft @ fs^T
        logits = torch.bmm(ft, fs.transpose(1, 2)) / self.tau      # B, n, n
        # positive = diagonal (same patch id)
        targets = torch.arange(n, device=feat_src.device).unsqueeze(0).expand(batch, n)
        return self.ce(logits.reshape(batch * n, n), targets.reshape(-1))


def patchnce_total(G, F, x_in: torch.Tensor, y_out_cat: torch.Tensor,
                   nce_layers, num_patches: int, criterion: PatchNCELoss) -> torch.Tensor:
    """Compute summed PatchNCE over layers.
    x_in       : generator input (B,4,H,W) = source RGB ⊕ mask
    y_out_cat  : G(x) ⊕ (same mask)  (B,4,H,W) so the encoder sees 4ch too
    """
    B = x_in.shape[0]
    feats_src = G.encode_feats(x_in, layers=nce_layers)
    feats_tgt = G.encode_feats(y_out_cat, layers=nce_layers)
    proj_src, ids = F(feats_src, num_patches=num_patches)
    proj_tgt, _ = F(feats_tgt, num_patches=num_patches, patch_ids=ids)
    total = 0.0
    for fs, ft in zip(proj_src, proj_tgt):
        total = total + criterion(fs, ft, B)
    return total / max(1, len(proj_src))


def mask_l1(fake: torch.Tensor, x_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft L1 between output and source RGB, weighted to the glyph region.
    Keeps letter pixels near the rendered glyphs (location + rough shape)."""
    return ((fake - x_rgb).abs() * mask).mean()


def gan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def gan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor, label_smooth: float = 0.0) -> tuple:
    real_t = torch.full_like(d_real, 1.0 - label_smooth)
    fake_t = torch.zeros_like(d_fake)
    lr = F.binary_cross_entropy_with_logits(d_real, real_t)
    lf = F.binary_cross_entropy_with_logits(d_fake, fake_t)
    return 0.5 * (lr + lf), lr, lf
