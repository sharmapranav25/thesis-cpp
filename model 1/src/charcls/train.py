"""Train ResNet-18 char classifier on PHP-synth crops."""
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

from src.charcls.data import CharCropDataset, split_rows
from src.charcls.model import build_charcls


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/charcls.yaml"))
    return ap.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    c_letter = c_case = total = 0
    for x, y_l, y_c in loader:
        x = x.to(device, non_blocking=True)
        y_l = y_l.to(device, non_blocking=True)
        y_c = y_c.to(device, non_blocking=True)
        logit_l, logit_c = model(x)
        c_letter += (logit_l.argmax(-1) == y_l).sum().item()
        c_case += (logit_c.argmax(-1) == y_c).sum().item()
        total += y_l.numel()
    model.train()
    return c_letter / max(1, total), c_case / max(1, total)


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    out = Path(cfg["ckpt_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a", buffering=1)

    train_rows, val_rows = split_rows(
        Path(cfg["data_dir"]), val_frac=cfg["val_frac"], seed=cfg["seed"]
    )
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}", flush=True)

    train_ds = CharCropDataset(
        cfg["data_dir"], rows=train_rows, crop_size=cfg["crop_size"],
        pad_frac=cfg["pad_frac"], jitter_frac=cfg["jitter_frac"],
        augment=True,
    )
    val_ds = CharCropDataset(
        cfg["data_dir"], rows=val_rows, crop_size=cfg["crop_size"],
        pad_frac=cfg["pad_frac"], jitter_frac=0.0, augment=False,
    )
    print(f"train samples={len(train_ds)} val samples={len(val_ds)}", flush=True)

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
    model = build_charcls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = cfg["warmup_steps"]

    def lr_at(step):
        if step < warmup:
            return cfg["lr"] * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(max(prog, 0), 1)
        return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    scaler = torch.amp.GradScaler("cuda")
    step = 0
    best_acc = 0.0
    t0 = time.time()

    case_weight = float(cfg.get("case_loss_weight", 0.5))

    for epoch in range(cfg["epochs"]):
        model.train()
        run_loss = run_letter_correct = run_case_correct = 0.0
        n = 0
        for x, y_l, y_c in train_loader:
            x = x.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            y_c = y_c.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit_l, logit_c = model(x)
                loss_l = F.cross_entropy(logit_l, y_l, label_smoothing=0.05)
                loss_c = F.cross_entropy(logit_c, y_c, label_smoothing=0.05)
                loss = loss_l + case_weight * loss_c
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            run_letter_correct += (logit_l.argmax(-1) == y_l).sum().item()
            run_case_correct += (logit_c.argmax(-1) == y_c).sum().item()
            n += x.size(0)
            step += 1

        acc_l, acc_c = evaluate(model, val_loader, device)
        msg = (f"ep {epoch+1}/{cfg['epochs']}  loss {run_loss/n:.4f}  "
               f"tr_letter {run_letter_correct/n:.4f}  tr_case {run_case_correct/n:.4f}  "
               f"val_letter {acc_l:.4f}  val_case {acc_c:.4f}  "
               f"lr {optim.param_groups[0]['lr']:.2e}  t {time.time()-t0:.0f}s")
        print(msg, flush=True)
        log_f.write(msg + "\n")

        # checkpoint by case acc — that's the head we actually deploy
        score = acc_c
        if score > best_acc:
            best_acc = score
            torch.save({"model": model.state_dict(), "acc_letter": acc_l,
                        "acc_case": acc_c, "epoch": epoch, "cfg": cfg},
                       out / "best.pt")
        torch.save({"model": model.state_dict(), "acc_letter": acc_l,
                    "acc_case": acc_c, "epoch": epoch, "cfg": cfg},
                   out / "last.pt")

    print(f"best val_acc: {best_acc:.4f}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
