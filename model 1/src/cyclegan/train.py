"""CycleGAN training for synth <-> real CryptBB captchas.

Domain A = synthetic (renderer output)   — large pool, labels free
Domain B = real CryptBB captchas (data/raw) — small pool, no labels

Goal of training: produce G_AB such that G_AB(synth) looks like a real CryptBB
captcha while preserving the rendered text content. The cycle and identity
losses keep that text content intact while the discriminators push style.

Adapted from the standard CycleGAN training recipe (Zhu et al. 2017).
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from .models import build_models
from .dataset import make_loader


class ImagePool:
    """Replay buffer that stabilises D updates (Shrivastava et al.)."""
    def __init__(self, size: int = 50):
        self.size = size
        self.buf: list[torch.Tensor] = []

    def query(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return imgs
        out = []
        for img in imgs:
            img = img.unsqueeze(0)
            if len(self.buf) < self.size:
                self.buf.append(img.detach())
                out.append(img)
            elif random.random() > 0.5:
                idx = random.randrange(self.size)
                old = self.buf[idx].clone()
                self.buf[idx] = img.detach()
                out.append(old)
            else:
                out.append(img)
        return torch.cat(out, 0)


def _set_requires_grad(nets, flag: bool) -> None:
    for net in nets if isinstance(nets, (list, tuple)) else [nets]:
        for p in net.parameters():
            p.requires_grad = flag


def _gan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def linear_lr(epoch: int, n_epochs: int, n_decay: int) -> float:
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / max(1, n_decay))


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    G_AB, G_BA, D_A, D_B = build_models(
        device, ngf=cfg["ngf"], ndf=cfg["ndf"], n_blocks=cfg["n_blocks"]
    )

    lr_g = cfg.get("lr_g", cfg.get("lr", 2e-4))
    lr_d = cfg.get("lr_d", lr_g)
    opt_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=lr_g, betas=(0.5, 0.999),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=lr_d, betas=(0.5, 0.999),
    )
    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda e: linear_lr(e, cfg["n_epochs_const"], cfg["n_epochs_decay"])
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda e: linear_lr(e, cfg["n_epochs_const"], cfg["n_epochs_decay"])
    )

    loader = make_loader(
        synth_dir=Path(cfg["synth_dir"]),
        real_dir=Path(cfg["real_dir"]),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        virtual_len=cfg["iters_per_epoch"],
    )

    pool_A = ImagePool(cfg["pool_size"])
    pool_B = ImagePool(cfg["pool_size"])

    cri_l1 = nn.L1Loss()

    samples_dir = Path(cfg["samples_dir"]); samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = cfg["n_epochs_const"] + cfg["n_epochs_decay"]
    log_path = Path(cfg["samples_dir"]) / "train.log"
    log_f = log_path.open("a", encoding="utf-8")

    es_thresh = cfg.get("early_stop_d_loss", 0.0)
    es_patience = cfg.get("early_stop_patience_epochs", 0)
    es_count = 0

    for epoch in range(1, n_epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}/{n_epochs}", ncols=120)
        for step, batch in enumerate(pbar):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ---- Generators ----
            _set_requires_grad([D_A, D_B], False)
            opt_G.zero_grad(set_to_none=True)
            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)

            loss_id_A = cri_l1(G_BA(real_A), real_A) * cfg["lambda_identity"]
            loss_id_B = cri_l1(G_AB(real_B), real_B) * cfg["lambda_identity"]
            loss_gan_AB = _gan_loss(D_B(fake_B), True)
            loss_gan_BA = _gan_loss(D_A(fake_A), True)
            loss_cyc_A = cri_l1(rec_A, real_A) * cfg["lambda_cycle"]
            loss_cyc_B = cri_l1(rec_B, real_B) * cfg["lambda_cycle"]

            loss_G = loss_gan_AB + loss_gan_BA + loss_cyc_A + loss_cyc_B + loss_id_A + loss_id_B
            loss_G.backward()
            opt_G.step()

            # ---- Discriminators ----
            _set_requires_grad([D_A, D_B], True)
            opt_D.zero_grad(set_to_none=True)

            fake_A_q = pool_A.query(fake_A.detach())
            loss_D_A = 0.5 * (_gan_loss(D_A(real_A), True) + _gan_loss(D_A(fake_A_q), False))
            loss_D_A.backward()

            fake_B_q = pool_B.query(fake_B.detach())
            loss_D_B = 0.5 * (_gan_loss(D_B(real_B), True) + _gan_loss(D_B(fake_B_q), False))
            loss_D_B.backward()

            opt_D.step()

            pbar.set_postfix({
                "G": f"{loss_G.item():.3f}",
                "D_A": f"{loss_D_A.item():.3f}",
                "D_B": f"{loss_D_B.item():.3f}",
                "cyc": f"{(loss_cyc_A + loss_cyc_B).item():.3f}",
            })

        sched_G.step(); sched_D.step()

        epoch_d_mean = (loss_D_A.item() + loss_D_B.item()) / 2
        log_f.write(f"epoch {epoch} G={loss_G.item():.4f} D_A={loss_D_A.item():.4f} "
                    f"D_B={loss_D_B.item():.4f} cyc={(loss_cyc_A+loss_cyc_B).item():.4f}\n")
        log_f.flush()

        if es_thresh > 0 and epoch_d_mean < es_thresh:
            es_count += 1
            print(f"  [early-stop] D-loss mean {epoch_d_mean:.3f} < {es_thresh:.2f} "
                  f"({es_count}/{es_patience})")
        else:
            es_count = 0

        # readability sample: dump grids of real_A | fake_B | rec_A | real_B | fake_A | rec_B
        if epoch % cfg["sample_every"] == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                grid = torch.cat([real_A[:4], fake_B[:4], rec_A[:4],
                                  real_B[:4], fake_A[:4], rec_B[:4]], dim=0)
                save_image((grid + 1) / 2, samples_dir / f"epoch_{epoch:03d}.png", nrow=4)

        save_epochs = cfg.get("save_epochs", [])
        ckpt_every = cfg.get("ckpt_every", 0) or 0
        should_save = (
            (save_epochs and epoch in save_epochs)
            or (not save_epochs and ckpt_every and (epoch % ckpt_every == 0 or epoch == n_epochs))
        )
        if should_save:
            torch.save({
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if es_thresh > 0 and es_count >= es_patience and es_patience > 0:
            print(f"  [early-stop] triggered at epoch {epoch}; saving final ckpt")
            torch.save({
                "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                "epoch": epoch, "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}_earlystop.pt")
            break

    log_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/cyclegan.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)
