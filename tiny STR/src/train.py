"""Train TinySTR on synth_mybb. Per ../CONTEXT.md:

- AdamW + cosine LR with 5% warmup, lr 5e-4, weight_decay 1e-4
- 15 epochs, batch 128, fp16 AMP
- Loss: sum of 5 cross-entropies (one per position)
- Same val split as PARSeq (val_frac=0.05, seed=0) so synth-val numbers compare
- Best ckpt by val word accuracy
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CaptchaDir, split_rows
from model import NUM_POSITIONS, TinySTR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path(r"E:\pranav\model 1\data\synth_mybb"))
    p.add_argument("--ckpt-dir", type=Path,
                   default=HERE.parent / "checkpoints")
    p.add_argument("--log-file", type=Path,
                   default=HERE.parent / "train.log")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50, help="steps between train loss prints")
    p.add_argument("--resume", type=Path, default=None,
                   help="Checkpoint to resume model weights from (start_epoch and global_step inferred from ckpt)")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=6)
    p.add_argument("--query-self-attn-layers", type=int, default=0)
    return p.parse_args()


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits [B, 5, C], targets [B, 5] -> scalar loss = sum_pos CE, mean over batch."""
    B, P, C = logits.shape
    return F.cross_entropy(logits.reshape(B * P, C), targets.reshape(B * P), reduction="mean") * P


@torch.no_grad()
def evaluate(model: TinySTR, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    n_imgs = 0
    n_chars_correct = 0
    n_chars_total = 0
    n_words_correct = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = compute_loss(logits, targets)
        bsz = imgs.size(0)
        total_loss += loss.item() * bsz
        n_imgs += bsz
        preds = logits.argmax(dim=-1)  # [B, 5]
        char_match = (preds == targets)
        n_chars_correct += char_match.sum().item()
        n_chars_total += bsz * NUM_POSITIONS
        n_words_correct += char_match.all(dim=-1).sum().item()
    model.train()
    return {
        "loss": total_loss / max(1, n_imgs),
        "char_acc": n_chars_correct / max(1, n_chars_total),
        "word_acc": n_words_correct / max(1, n_imgs),
    }


class TeeLogger:
    """Print to stdout and append to a file. Flushes on every line for live tailing."""
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def main():
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(args.log_file)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    train_rows, val_rows = split_rows(args.data_dir, args.val_frac, args.seed)
    log(f"data: train={len(train_rows)} val={len(val_rows)}")

    train_ds = CaptchaDir(args.data_dir, rows=train_rows, augment=True)
    val_ds = CaptchaDir(args.data_dir, rows=val_rows, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TinySTR(
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        query_self_attn_layers=args.query_self_attn_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: TinySTR params={n_params/1e6:.2f}M "
        f"embed_dim={args.embed_dim} encoder_depth={args.encoder_depth} "
        f"query_self_attn={args.query_self_attn_layers}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    log(f"schedule: steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup_steps}")

    best_word_acc = -1.0
    global_step = 0
    start_epoch = 1
    if args.resume is not None:
        log(f"resume: loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = (start_epoch - 1) * steps_per_epoch
        if "val" in ckpt and isinstance(ckpt["val"], dict) and "word_acc" in ckpt["val"]:
            best_word_acc = float(ckpt["val"]["word_acc"])
        log(f"resume: start_epoch={start_epoch} global_step={global_step} best_word_acc={best_word_acc:.4f}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_imgs = 0
        ep_start = time.time()
        for it, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            lr_now = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr_now

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = compute_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            bsz = imgs.size(0)
            epoch_loss += loss.item() * bsz
            epoch_imgs += bsz
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                log(f"epoch={epoch} step={global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e} elapsed={elapsed:.0f}s")

        train_loss = epoch_loss / max(1, epoch_imgs)
        val = evaluate(model, val_loader, device)
        ep_dt = time.time() - ep_start
        log(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_char_acc={val['char_acc']:.4f} "
            f"val_word_acc={val['word_acc']:.4f} epoch_time={ep_dt:.1f}s")

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "val": val,
            "args": vars(args),
        }, args.ckpt_dir / "last.pt")

        if val["word_acc"] > best_word_acc:
            best_word_acc = val["word_acc"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val": val,
                "args": vars(args),
            }, args.ckpt_dir / "best.pt")
            log(f"  -> saved new best (val_word_acc={best_word_acc:.4f})")

    total_dt = time.time() - t0
    log(f"done. best val_word_acc={best_word_acc:.4f} total_time={total_dt/60:.1f}min")
    log.close()


if __name__ == "__main__":
    main()
