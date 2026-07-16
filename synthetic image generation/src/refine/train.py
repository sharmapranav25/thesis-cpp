"""Train the mask-aware refinement net.

Loss design:
  G total = lambda_adv * adv_loss + lambda_l1 * masked_l1
    - adv_loss: BCE(D(G(x)), 1)         encourages G(x) to look real
    - masked_l1: ||(G(x) - x_rgb) * mask||_1
                 (preserves original text pixels — the captcha label is the
                  text we picked; the refiner must not redraw it.)
  D total = 0.5 * (BCE(D(real), 1) + BCE(D(G(x).detach()), 0))

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/refine/train.py [--steps N] [--bs B]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generator.parametric import ParametricGenerator
from src.refine.data import (
    PAD_H, PAD_LEFT, PAD_TOP, PAD_W,
    RealCaptchaDataset, SyntheticPairDataset, crop_image,
)
from src.refine.model import PatchDiscriminator, UNetRefiner

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
DEFAULT_CKPT_DIR = ROOT / "checkpoints" / "darkbay_refiner"
DEFAULT_SAMPLE_DIR = ROOT / "mvp_results" / "refiner_samples"
DEFAULT_LOG_PATH = ROOT / "mvp_results" / "refiner_train_log.csv"


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor in [-1,1] CHW → PIL RGB at native (W, H), unpadded."""
    arr = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    arr = crop_image(arr)
    return Image.fromarray(arr)


def save_sample(sample_dir: Path, step: int, x: torch.Tensor, mask: torch.Tensor,
                fake: torch.Tensor, real: torch.Tensor) -> None:
    """Build a 4-row visual: input composite / mask / G(x) / a real for context."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    n = min(4, x.size(0))
    rows = []
    for i in range(n):
        comp = to_pil(x[i, :3])
        mk = ((mask[i, 0].cpu().numpy() * 255).astype(np.uint8))
        mk = mk[PAD_TOP:PAD_TOP + 50, PAD_LEFT:PAD_LEFT + 200]
        mk_im = Image.fromarray(mk).convert("RGB")
        gen_im = to_pil(fake[i])
        real_im = to_pil(real[i % real.size(0)])
        row = Image.new("RGB", (200 * 4 + 12, 50), (32, 32, 32))
        for j, im in enumerate([comp, mk_im, gen_im, real_im]):
            row.paste(im, (j * (200 + 4), 0))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 4 * (n - 1)),
                      (32, 32, 32))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    sheet.save(sample_dir / f"step_{step:06d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="G learning rate")
    ap.add_argument("--d-lr", type=float, default=None,
                    help="D learning rate (defaults to --lr)")
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="D real label = 1 - smooth (0.0=off; try 0.1)")
    ap.add_argument("--mask-erode", type=int, default=0,
                    help="erode the L1 mask inward by N px per side")
    ap.add_argument("--use-alpha-mask", action="store_true",
                    help="use exact glyph silhouette (alpha channel) as L1 mask")
    ap.add_argument("--mask-dilate", type=int, default=0,
                    help="dilate the (alpha) mask outward by N px per side")
    ap.add_argument("--residual", action="store_true",
                    help="UNet outputs delta added to input (identity start)")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    ap.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    d_lr = args.d_lr if args.d_lr is not None else args.lr
    print(f"device = {device}  ckpt_dir = {args.ckpt_dir}")
    print(f"steps = {args.steps}  batch = {args.bs}  lr_G = {args.lr}  lr_D = {d_lr}")
    print(f"lambda_l1 = {args.lambda_l1}  lambda_adv = {args.lambda_adv}  "
          f"label_smooth = {args.label_smooth}  mask_erode = {args.mask_erode}  "
          f"residual = {args.residual}")

    print("loading data ...")
    gen = ParametricGenerator(seed=args.seed)
    syn_ds = SyntheticPairDataset(gen, length=args.steps * args.bs, t=1.0,
                                  mask_erode=args.mask_erode,
                                  use_alpha_mask=args.use_alpha_mask,
                                  mask_dilate=args.mask_dilate)
    real_ds = RealCaptchaDataset(REAL_DIR)
    syn_loader = DataLoader(syn_ds, batch_size=args.bs, num_workers=0,
                            collate_fn=lambda b: {
                                "composite": torch.stack([x["composite"] for x in b]),
                                "mask":      torch.stack([x["mask"] for x in b]),
                            })
    real_loader = DataLoader(real_ds, batch_size=args.bs, shuffle=True, num_workers=0)

    G = UNetRefiner(residual=args.residual).to(device)
    D = PatchDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    syn_iter = iter(syn_loader)
    real_iter = cycle(real_loader)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = ["step,loss_G,loss_G_adv,loss_G_l1,loss_D,loss_D_real,loss_D_fake,sec"]

    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(syn_iter)
        except StopIteration:
            syn_iter = iter(syn_loader)
            batch = next(syn_iter)
        x_rgb = batch["composite"].to(device)
        mask = batch["mask"].to(device)
        x = torch.cat([x_rgb, mask], dim=1)  # 4-channel
        real = next(real_iter).to(device)

        # --- D step
        opt_D.zero_grad()
        with torch.no_grad():
            fake_d = G(x)
        d_real = D(real)
        d_fake = D(fake_d)
        real_target = torch.full_like(d_real, 1.0 - args.label_smooth)
        fake_target = torch.zeros_like(d_fake)
        loss_D_real = F.binary_cross_entropy_with_logits(d_real, real_target)
        loss_D_fake = F.binary_cross_entropy_with_logits(d_fake, fake_target)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        opt_D.step()

        # --- G step
        opt_G.zero_grad()
        fake = G(x)
        d_fake_for_g = D(fake)
        loss_G_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
        loss_G_l1 = ((fake - x_rgb).abs() * mask).mean()
        loss_G = args.lambda_adv * loss_G_adv + args.lambda_l1 * loss_G_l1
        loss_G.backward()
        opt_G.step()

        if step % args.log_every == 0 or step == 1:
            secs = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  "
                  f"G={loss_G.item():.3f}  G_adv={loss_G_adv.item():.3f}  "
                  f"G_l1={loss_G_l1.item():.4f}  "
                  f"D={loss_D.item():.3f} (r={loss_D_real.item():.3f} f={loss_D_fake.item():.3f})  "
                  f"{secs:.0f}s")
            log_lines.append(f"{step},{loss_G.item():.4f},{loss_G_adv.item():.4f},"
                             f"{loss_G_l1.item():.4f},{loss_D.item():.4f},"
                             f"{loss_D_real.item():.4f},{loss_D_fake.item():.4f},{secs:.0f}")

        if step % args.sample_every == 0 or step == 1:
            G.eval()
            with torch.no_grad():
                sample_fake = G(x[:4])
            G.train()
            save_sample(args.sample_dir, step, x[:4], mask[:4], sample_fake, real[:4])

        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "step": step},
                       args.ckpt_dir / f"step_{step:06d}.pt")
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "step": step},
                       args.ckpt_dir / "latest.pt")

    args.log_path.write_text("\n".join(log_lines) + "\n")
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "step": args.steps},
               args.ckpt_dir / "latest.pt")
    print(f"\ndone. ckpt at {args.ckpt_dir / 'latest.pt'}, log at {args.log_path}")


if __name__ == "__main__":
    main()
