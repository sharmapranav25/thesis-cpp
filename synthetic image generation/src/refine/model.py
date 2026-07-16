"""Mask-aware U-Net refiner + PatchGAN discriminator.

Generator:
  Input  4xHxW (RGB composite concatenated with text mask)
  Output 3xHxW (refined RGB)
  4 stride-2 down-blocks, 4 up-blocks with skip connections.
  Tanh on output -> [-1, 1].

Discriminator:
  Input  3xHxW (RGB image, real or fake)
  Output ~ patch grid of real/fake logits.
  Standard 70x70-style PatchGAN (4 conv blocks with stride 2).

Designed for input shape (B, ., 64, 208).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, down: bool, use_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    if down:
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
    else:
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
    if use_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetRefiner(nn.Module):
    """4-down / 4-up U-Net with skip concatenations.

    Channel plan: 4 -> 64 -> 128 -> 256 -> 512 (bottleneck) -> 256 -> 128 -> 64 -> 3.

    `residual=True`: U-Net's output is treated as a small delta added to the
    RGB part of the input, then clamped/squashed via tanh. Identity is the
    natural starting point so optimization is much easier.
    """
    def __init__(self, in_ch: int = 4, out_ch: int = 3, base: int = 64,
                 residual: bool = False, residual_scale: float = 0.3):
        super().__init__()
        self.residual = residual
        self.residual_scale = residual_scale
        # encoder
        self.d1 = _conv_block(in_ch, base, down=True, use_norm=False)        # 64x32x104
        self.d2 = _conv_block(base, base * 2, down=True)                     # 128x16x52
        self.d3 = _conv_block(base * 2, base * 4, down=True)                 # 256x8x26
        self.d4 = _conv_block(base * 4, base * 8, down=True)                 # 512x4x13

        # decoder (input channels = encoder output + skip from same level)
        self.u4 = _conv_block(base * 8, base * 4, down=False)                # 256x8x26
        self.u3 = _conv_block(base * 4 + base * 4, base * 2, down=False)     # 128x16x52
        self.u2 = _conv_block(base * 2 + base * 2, base, down=False)         # 64x32x104
        # Final layer: in residual mode emit a delta (tanh-bounded), in
        # standard mode emit the full image (also tanh-bounded).
        self.u1 = nn.Sequential(                                              # out_ch x 64 x 208
            nn.ConvTranspose2d(base + base, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        d4 = self.u4(e4)
        d3 = self.u3(torch.cat([d4, e3], dim=1))
        d2 = self.u2(torch.cat([d3, e2], dim=1))
        d1 = self.u1(torch.cat([d2, e1], dim=1))
        if self.residual:
            # x's first 3 channels are the input RGB in [-1, 1].
            return torch.clamp(x[:, :3] + self.residual_scale * d1, -1.0, 1.0)
        return d1


class PatchDiscriminator(nn.Module):
    """3 -> 64 -> 128 -> 256 -> 1; receptive field ≈ 70×70 on (208×64)."""
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),  # patch logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    g = UNetRefiner()
    d = PatchDiscriminator()
    x = torch.zeros(2, 4, 64, 208)
    real = torch.zeros(2, 3, 64, 208)
    out = g(x)
    dout = d(real)
    print(f"G(x): in {tuple(x.shape)} -> out {tuple(out.shape)}")
    print(f"D(real): in {tuple(real.shape)} -> out {tuple(dout.shape)}")
    print(f"G params: {sum(p.numel() for p in g.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in d.parameters()):,}")
