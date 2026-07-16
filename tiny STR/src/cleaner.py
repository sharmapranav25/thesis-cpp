"""TinyCleaner — small U-Net that pre-processes real captchas before recognition.

Learns a residual mapping (real → real + delta) supervised by a frozen TinySTR
recognizer's confidence on the cleaned image. The cleaner's job is to nudge real
captchas toward a representation the recognizer (trained only on synth) is
confident about — without changing them so much that it hallucinates content.

Trained on real captchas without labels (entropy-minimization loss). Per the
user-specified exception in tiny_str_project.md, this auxiliary module may train
on the eval set; the recognizer cannot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyCleaner(nn.Module):
    """Small U-Net (~0.3M params) producing a bounded residual.

    Input/output shape: [B, 3, 60, 200] in normalized [-1, 1] space.
    """

    def __init__(self, base_ch: int = 32, residual_scale: float = 0.5):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.residual_scale = residual_scale

        self.enc1 = ConvBlock(3, c1)         # 60x200
        self.enc2 = ConvBlock(c1, c2)        # 30x100
        self.enc3 = ConvBlock(c2, c3)        # 15x50
        self.bottleneck = ConvBlock(c3, c3)  # 15x50

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.out_conv = nn.Conv2d(c1, 3, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                       # [B, c1, 60, 200]
        e2 = self.enc2(F.avg_pool2d(e1, 2))    # [B, c2, 30, 100]
        e3 = self.enc3(F.avg_pool2d(e2, 2))    # [B, c3, 15, 50]
        b = self.bottleneck(e3)                # [B, c3, 15, 50]
        u2 = self.up2(b)                        # [B, c2, 30, 100]
        d2 = self.dec2(torch.cat([u2, e2], 1)) # [B, c2, 30, 100]
        u1 = self.up1(d2)                       # [B, c1, 60, 200]
        d1 = self.dec1(torch.cat([u1, e1], 1)) # [B, c1, 60, 200]
        residual = torch.tanh(self.out_conv(d1)) * self.residual_scale
        return x + residual


@torch.no_grad()
def clean_batch(cleaner: TinyCleaner, x: torch.Tensor) -> torch.Tensor:
    cleaner.eval()
    return cleaner(x).clamp(-1.0, 1.0)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Per-position softmax entropy averaged over batch+positions.

    logits: [B, P, C] — lower entropy = more confident predictions.
    """
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    H = -(p * log_p).sum(dim=-1)  # [B, P]
    return H.mean()


def confidence_loss(logits: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of the argmax class — sharper objective than entropy.

    Encourages the recognizer's top-class confidence to grow without committing
    to a specific class as ground truth.
    """
    log_p = F.log_softmax(logits, dim=-1)
    top, _ = log_p.max(dim=-1)  # [B, P]
    return -top.mean()
