"""Train CharBoxNet on data/synth_mybb_bb.

Loss: GIoU on sorted-by-cx pairs. Plus a small smooth-L1 on (cx, cy)
for early stability. ~20 min on the A4000 at 30 epochs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.detector.data import CaptchaBoxDataset, split_rows
from src.detector.model import (
    CharBoxNet, cxcywh_to_xyxy, giou_loss, sort_by_cx,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    return ap.parse_args()


def compute_loss(pred, target):
    """pred, target: [B, 5, 4] in cxcywh."""
    pred_s = sort_by_cx(pred)
    targ_s = sort_by_cx(target)
    p_xyxy = cxcywh_to_xyxy(pred_s)
    t_xyxy = cxcywh_to_xyxy(targ_s)
    giou = giou_loss(p_xyxy, t_xyxy).mean()
    center_loss = F.smooth_l1_loss(pred_s[..., :2], targ_s[..., :2])
    return giou + 0.5 * center_loss, giou.detach(), center_loss.detach()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    iou_sum = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        pred = sort_by_cx(pred)
        y = sort_by_cx(y)
        # IoU per box
        p = cxcywh_to_xyxy(pred)
        t = cxcywh_to_xyxy(y)
        ix0 = torch.maximum(p[..., 0], t[..., 0])
        iy0 = torch.maximum(p[..., 1], t[..., 1])
        ix1 = torch.minimum(p[..., 2], t[..., 2])
        iy1 = torch.minimum(p[..., 3], t[..., 3])
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        pa = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
        ta = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
        iou = inter / (pa + ta - inter + 1e-9)
        iou_sum += iou.sum().item()
        n += iou.numel()
    model.train()
    return iou_sum / max(1, n)


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    out = Path(cfg["ckpt_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a", buffering=1)

    train_rows, val_rows = split_rows(
        Path(cfg["data_dir"]), val_frac=cfg["val_frac"], seed=cfg["seed"]
    )
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    train_ds = CaptchaBoxDataset(cfg["data_dir"], rows=train_rows, augment=True)
    val_ds = CaptchaBoxDataset(cfg["data_dir"], rows=val_rows, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    device = "cuda"
    model = CharBoxNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = cfg["warmup_steps"]

    def lr_at(step):
        if step < warmup:
            return cfg["lr"] * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(max(prog, 0), 1)
        return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    step = 0
    best_iou = 0.0
    t0 = time.time()
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(cfg["epochs"]):
        model.train()
        run_loss = run_giou = run_c = 0.0
        n = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(x)
                loss, giou, c = compute_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            run_giou += giou.item() * x.size(0)
            run_c += c.item() * x.size(0)
            n += x.size(0)
            step += 1

        iou = evaluate(model, val_loader, device)
        msg = (f"ep {epoch+1}/{cfg['epochs']}  loss {run_loss/n:.4f}  "
               f"giou_loss {run_giou/n:.4f}  c {run_c/n:.4f}  "
               f"val_iou {iou:.4f}  lr {optim.param_groups[0]['lr']:.2e}  "
               f"t {time.time()-t0:.0f}s")
        print(msg, flush=True)
        log_f.write(msg + "\n")

        if iou > best_iou:
            best_iou = iou
            torch.save({
                "model": model.state_dict(),
                "iou": iou,
                "epoch": epoch,
                "cfg": cfg,
            }, out / "best.pt")
        torch.save({
            "model": model.state_dict(),
            "iou": iou,
            "epoch": epoch,
            "cfg": cfg,
        }, out / "last.pt")

    print(f"best val_iou: {best_iou:.4f}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
