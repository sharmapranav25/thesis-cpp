"""Self-supervised MAE pretraining of PARSeq's ViT encoder on the UNLABELED real
pool (GeeSolver, IEEE S&P 2023). Captcha distortions act like masks, so learning
to reconstruct masked patches of REAL captchas teaches the encoder the scheme's
true font/glyph features — label-free, source-free, scheme-general (swap the
folder). The pretrained encoder then initializes the recognizer so it already
"sees" the real font before any synth training.

GUARDRAIL: pretrain on the UNLABELED pool only (600 CryptBB unsolved), NEVER the
651 eval.

Output: an encoder state_dict compatible with PARSeq's `model.encoder`.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

_MODEL1 = Path(r"E:/pranav/model 1")
for p in (str(_MODEL1), str(_MODEL1 / "third_party" / "parseq")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import yaml

from strhub.models.parseq.modules import Encoder
from timm.models.vision_transformer import Block


class RealPatches(Dataset):
    """Unlabeled reals with mild photometric aug (no geometric aug that would
    break glyph structure). Length inflated for more views per epoch."""

    def __init__(self, real_dir, img_size, length=20000):
        self.paths = sorted(Path(real_dir).glob("*.png"))
        self.H, self.W = img_size
        self.length = length
        self.jit = T.ColorJitter(0.3, 0.3, 0.3, 0.08)
        self.tf = T.Compose([T.Resize((self.H, self.W), antialias=True),
                             T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        p = self.paths[i % len(self.paths)]
        img = Image.open(p).convert("RGB")
        if torch.rand(1).item() < 0.8:
            img = self.jit(img)
        return self.tf(img)


class MAE(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim=384, depth=12, heads=6,
                 dec_dim=192, dec_depth=4, dec_heads=6, mask_ratio=0.75):
        super().__init__()
        self.ph, self.pw = patch_size
        self.H, self.W = img_size
        self.gh, self.gw = self.H // self.ph, self.W // self.pw
        self.npatch = self.gh * self.gw
        self.mask_ratio = mask_ratio
        self.psize = self.ph * self.pw * 3

        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                               depth=depth, num_heads=heads)
        # decoder
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.npatch, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, 4.0, qkv_bias=True)
                                         for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.psize)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    def patchify(self, imgs):
        B = imgs.shape[0]
        x = imgs.reshape(B, 3, self.gh, self.ph, self.gw, self.pw)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.npatch, self.psize)
        return x

    def forward(self, imgs):
        B = imgs.shape[0]
        target = self.patchify(imgs)
        # patch embed + pos (encoder is a timm ViT, no class token)
        x = self.encoder.patch_embed(imgs)                 # B, N, C
        x = x + self.encoder.pos_embed
        # random mask: keep (1-mask_ratio)
        nkeep = int(self.npatch * (1 - self.mask_ratio))
        noise = torch.rand(B, self.npatch, device=imgs.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :nkeep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        # encode visible
        for blk in self.encoder.blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder.norm(x_vis)
        # decode: place visible tokens + mask tokens, unshuffle
        d = self.dec_embed(x_vis)
        mask_tok = self.mask_token.expand(B, self.npatch - nkeep, -1)
        d_full = torch.cat([d, mask_tok], dim=1)
        d_full = torch.gather(d_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d.shape[-1]))
        d_full = d_full + self.dec_pos
        for blk in self.dec_blocks:
            d_full = blk(d_full)
        d_full = self.dec_norm(d_full)
        pred = self.dec_pred(d_full)                       # B, N, psize
        # loss on MASKED patches only
        mask = torch.ones(B, self.npatch, device=imgs.device)
        mask[:, :nkeep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        loss = ((pred - target) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--parseq-config", type=Path, default=Path("configs/parseq_pool.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("data/ckpts/mae_encoder.pt"))
    args = ap.parse_args()

    corpus = yaml.safe_load(args.corpus.read_text())
    s = corpus["schemes"][args.scheme]
    pcfg = yaml.safe_load(args.parseq_config.read_text())
    img_size = tuple(pcfg["img_size"]); patch = tuple(pcfg["patch_size"])
    device = "cuda"

    ds = RealPatches(s["real_dir"], img_size, length=args.steps_per_epoch * args.batch)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                        pin_memory=True, persistent_workers=True, drop_last=True)
    mae = MAE(img_size, patch, embed_dim=pcfg["embed_dim"], depth=pcfg["enc_depth"],
              heads=pcfg["enc_num_heads"]).to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = args.epochs * len(loader); warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: t / max(1, warm) if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))

    print(f"MAE pretrain on {len(ds.paths)} unlabeled reals, {img_size} patch {patch}, "
          f"{args.epochs} ep x {len(loader)} steps", flush=True)
    step = 0
    for ep in range(args.epochs):
        run = 0.0
        for imgs in loader:
            loss = mae(imgs.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += float(loss); step += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:2d}  recon-mse {run/len(loader):.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": mae.encoder.state_dict(), "cfg": pcfg}, args.out)
    print(f"saved MAE encoder -> {args.out}")


if __name__ == "__main__":
    main()
