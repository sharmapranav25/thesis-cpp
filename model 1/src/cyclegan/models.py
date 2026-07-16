"""ResNet generator and PatchGAN discriminator for CycleGAN.

Adapted from junyanz/pytorch-CycleGAN-and-pix2pix (BSD-licensed). Stripped to
the essentials for the synth<->real captcha domain (200x60 RGB).
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn


def _norm_layer(kind: str = "instance"):
    if kind == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if kind == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    raise ValueError(kind)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """ResNet-9 (default) or ResNet-6 generator. Input/output: 3-channel image."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64,
                 n_blocks: int = 9, norm: str = "instance"):
        super().__init__()
        norm_layer = _norm_layer(norm)
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        # downsample x2
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2 ** n_down
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult, norm_layer)]
        # upsample x2
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN 70x70 discriminator."""

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3, norm: str = "instance"):
        super().__init__()
        norm_layer = _norm_layer(norm)
        kw = 4
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ndf, kw, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(m: nn.Module, gain: float = 0.02) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_models(device: torch.device, ngf: int = 64, ndf: int = 64, n_blocks: int = 9):
    G_AB = ResnetGenerator(3, 3, ngf, n_blocks).to(device)
    G_BA = ResnetGenerator(3, 3, ngf, n_blocks).to(device)
    D_A = NLayerDiscriminator(3, ndf).to(device)
    D_B = NLayerDiscriminator(3, ndf).to(device)
    for net in (G_AB, G_BA, D_A, D_B):
        net.apply(init_weights)
    return G_AB, G_BA, D_A, D_B
