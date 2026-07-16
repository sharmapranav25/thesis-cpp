"""Recognition-anchored CUT restyler — the core GAN (RESEARCH §7).

  source render (label ours) ⊕ mask  ──► G ──► ŷ (restyled captcha)
      reals ──► D (PatchGAN)                  adversarial: ŷ looks real
      PatchNCE(x, ŷ)                           content preserved (CUT)
      mask-L1                                   letters stay put
      R(ŷ) == label  (optional, M2)            letters stay LEGIBLE

M1 runs with rec weight 0 (no recognition loss). M2 sets rec>0 with a reader.
Same script; parameterized by configs/restyler.yaml + flags.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_source import SourceRenderer
from src.data import RealDataset, SourceDataset, source_collate
from src.model import build_models
from src.losses import (
    PatchNCELoss, patchnce_total, mask_l1, gan_g_loss, gan_d_loss,
)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def ada_augment(x: torch.Tensor, p: float) -> torch.Tensor:
    """Light differentiable-ish D augmentation (ADA spirit): per-sample flips of
    brightness/contrast + small noise, applied with prob p. Cheap stand-in for
    StyleGAN2-ADA to stabilise D on ~600 reals."""
    if p <= 0:
        return x
    B = x.shape[0]
    dev = x.device
    m = (torch.rand(B, 1, 1, 1, device=dev) < p).float()
    bright = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3
    noise = torch.randn_like(x) * 0.03
    aug = ((x * contrast + bright) + noise).clamp(-1, 1)
    return m * aug + (1 - m) * x


def to_pil(t: torch.Tensor, native_wh=None) -> Image.Image:
    arr = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    im = Image.fromarray(arr)
    if native_wh is not None:
        im = im.resize(native_wh, Image.BILINEAR)
    return im


def save_qa_panel(path, fake, real, native_wh, cols=4, rows=4):
    """4×8 real-vs-synth: left cols = G(x) synth, right cols = reals."""
    W, H = native_wh
    pad = 3
    total_cols = cols * 2
    sheet = Image.new("RGB", (total_cols * (W + pad) + pad, rows * (H + pad) + pad), (30, 30, 30))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < fake.shape[0]:
                sheet.paste(to_pil(fake[idx], native_wh), (pad + c * (W + pad), pad + r * (H + pad)))
            if idx < real.shape[0]:
                sheet.paste(to_pil(real[idx], native_wh), (pad + (cols + c) * (W + pad), pad + r * (H + pad)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_content_panel(path, src_rgb, fake, labels, preds, native_wh, n=8):
    """Rows of [source render | G(x) | label vs readback] — the legibility check."""
    W, H = native_wh
    pad = 3
    textw = 150
    sheet = Image.new("RGB", (2 * (W + pad) + textw + pad, n * (H + pad) + pad), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i in range(min(n, fake.shape[0])):
        y = pad + i * (H + pad)
        sheet.paste(to_pil(src_rgb[i], native_wh), (pad, y))
        sheet.paste(to_pil(fake[i], native_wh), (pad + W + pad, y))
        ok = labels[i] == preds[i]
        col = (120, 255, 120) if ok else (255, 120, 120)
        d.text((2 * (W + pad) + 4, y + 4), f"gt {labels[i]}", fill=(220, 220, 220))
        d.text((2 * (W + pad) + 4, y + 20), f"rd {preds[i]}", fill=col)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/restyler.yaml"))
    ap.add_argument("--corpus", type=Path, default=Path("configs/corpus.yaml"))
    ap.add_argument("--scheme", default="cryptbb")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="m1", help="run tag for ckpt/panel names")
    ap.add_argument("--rec-weight", type=float, default=None, help="override loss_weights.rec")
    ap.add_argument("--reader-ckpt", default=None, help="reader for L_rec (M2)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    corpus = yaml.safe_load(args.corpus.read_text())
    s = corpus["schemes"][args.scheme]
    device = "cuda"

    Wc, Hc = s["canvas"]                       # native (W,H)
    workH, workW = cfg["work_hw"]              # GAN working (H,W)
    steps = args.steps or cfg["steps"]
    lw = dict(cfg["loss_weights"])
    if args.rec_weight is not None:
        lw["rec"] = args.rec_weight
    nce_layers = tuple(cfg.get("nce_layers", [0, 1, 2, 3]))
    num_patches = cfg.get("num_patches", 256)

    renderer = SourceRenderer(canvas=(Wc, Hc), charset=s["charset"],
                              len_range=tuple(s["len_range"]), seed=0)
    src_ds = SourceDataset(
        renderer, (workH, workW), length=steps * cfg["batch"],
        bg_mode=cfg.get("bg_mode", "mix"), lama_dir=cfg.get("lama_dir"),
        p_procedural=cfg.get("p_procedural", 0.6),
    )
    real_ds = RealDataset(s["real_dir"], (workH, workW))
    src_loader = DataLoader(src_ds, batch_size=cfg["batch"], shuffle=False,
                            num_workers=6, collate_fn=source_collate,
                            pin_memory=True, persistent_workers=True)
    real_loader = DataLoader(real_ds, batch_size=cfg["batch"], shuffle=True,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    G, D, Fnet = build_models(base=cfg.get("base", 64), nce_layers=nce_layers,
                              nce_dim=cfg.get("nce_dim", 256), device=device,
                              residual=cfg.get("residual", False),
                              residual_scale=cfg.get("residual_scale", 0.5))
    opt_G = torch.optim.Adam(list(G.parameters()) + list(Fnet.parameters()),
                             lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.get("d_lr", cfg["lr"]), betas=(0.5, 0.999))
    nce_crit = PatchNCELoss(tau=cfg.get("nce_tau", 0.07))

    reader = None
    if lw.get("rec", 0) > 0:
        from src.parseq_reader import ParseqReader
        rc = args.reader_ckpt or cfg.get("reader_ckpt")
        reader = ParseqReader(ckpt=rc, train=False)
        print(f"[rec] L_rec ON with reader {rc}  weight={lw['rec']}")

    real_iter = cycle(real_loader)
    ckpt_dir = Path(cfg["out_dir"]) / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = Path(cfg["out_dir"]) / "qa_panels"
    ada_p = cfg.get("ada_p", 0.3) if cfg.get("use_ada", True) else 0.0

    print(f"scheme={args.scheme} steps={steps} batch={cfg['batch']} work={workH}x{workW} "
          f"weights={lw} nce_layers={nce_layers}")
    t0 = time.time()
    step = 0
    for x_rgb, mask, labels in src_loader:
        step += 1
        x_rgb = x_rgb.to(device); mask = mask.to(device)
        x = torch.cat([x_rgb, mask], dim=1)
        real = next(real_iter).to(device)

        # --- D step
        opt_D.zero_grad()
        with torch.no_grad():
            fake_d = G(x)
        d_real = D(ada_augment(real, ada_p))
        d_fake = D(ada_augment(fake_d, ada_p))
        loss_D, lr_, lf_ = gan_d_loss(d_real, d_fake, cfg.get("label_smooth", 0.1))
        loss_D.backward(); opt_D.step()

        # --- G step
        opt_G.zero_grad()
        fake = G(x)
        d_fake_g = D(ada_augment(fake, ada_p))
        l_adv = gan_g_loss(d_fake_g)
        fake_cat = torch.cat([fake, mask], dim=1)
        l_nce = patchnce_total(G, Fnet, x, fake_cat, nce_layers, num_patches, nce_crit)
        l_mask = mask_l1(fake, x_rgb, mask)
        l_rec = torch.tensor(0.0, device=device)
        if reader is not None:
            l_rec = reader.rec_loss(fake, labels)
        loss_G = (lw["adv"] * l_adv + lw["patchnce"] * l_nce
                  + lw["mask_l1"] * l_mask + lw.get("rec", 0.0) * l_rec)
        loss_G.backward(); opt_G.step()

        if step % 50 == 0 or step == 1:
            print(f"  {step:5d}/{steps} G={loss_G.item():.3f} adv={l_adv.item():.3f} "
                  f"nce={l_nce.item():.3f} mask={l_mask.item():.4f} rec={float(l_rec):.3f} "
                  f"D={loss_D.item():.3f}(r{lr_.item():.2f}/f{lf_.item():.2f}) {time.time()-t0:.0f}s",
                  flush=True)

        if step % cfg.get("qa_every", 500) == 0 or step == steps:
            G.eval()
            with torch.no_grad():
                fq = G(x)
            preds = (reader.read(fq, lengths=[len(l) for l in labels])
                     if reader is not None else ["" for _ in labels])
            G.train()
            save_qa_panel(qa_dir / f"{args.tag}_step{step:06d}.png", fq, real, (Wc, Hc))
            save_content_panel(qa_dir / f"{args.tag}_content_step{step:06d}.png",
                               x_rgb, fq, labels, preds, (Wc, Hc))

        if step % cfg.get("ckpt_every", 2000) == 0 or step == steps:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "F": Fnet.state_dict(),
                        "step": step, "cfg": cfg}, ckpt_dir / f"{args.tag}_latest.pt")
        if step >= steps:
            break

    print(f"done. ckpt -> {ckpt_dir / (args.tag + '_latest.pt')}")


if __name__ == "__main__":
    main()
