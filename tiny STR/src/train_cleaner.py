"""Train TinyCleaner on disjoint unlabeled real CryptBB captchas.

Pipeline:
  1. SHA256-dedup the unlabeled CryptBB folder against the eval set; keep only
     images NOT present in eval.
  2. Load a frozen TinySTR recognizer.
  3. Train TinyCleaner with confidence-maximization loss (negative log-prob of
     argmax class through the frozen recognizer) + small L1 penalty on the
     cleaner's residual (prevents collapse to trivial / hallucinatory cleaning).
  4. Save cleaner state_dict.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cleaner import TinyCleaner, confidence_loss
from data import _to_rgb
from model import TinySTR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recognizer-ckpt", type=Path,
                   default=HERE.parent / "checkpoints" / "best.pt")
    p.add_argument("--raw-dir", type=Path,
                   default=Path(r"E:\pranav\CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001\CryptBBCaptchas(unsolved)07272023"))
    p.add_argument("--eval-dir", type=Path,
                   default=Path(r"E:\pranav\model 1\data\eval"))
    p.add_argument("--out", type=Path,
                   default=HERE.parent / "checkpoints" / "cleaner.pt")
    p.add_argument("--log-file", type=Path,
                   default=HERE.parent / "train_cleaner.log")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--residual-l1", type=float, default=0.05,
                   help="Weight on |cleaner_output - input| L1 penalty")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--residual-scale", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dedup_against_eval(raw_dir: Path, eval_dir: Path) -> tuple[list[Path], int]:
    """Return list of raw images whose pixel-bytes are disjoint from eval set."""
    eval_hashes: set[str] = set()
    for p in eval_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            eval_hashes.add(file_sha(p))
    safe: list[Path] = []
    n_overlap = 0
    for p in raw_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        if file_sha(p) in eval_hashes:
            n_overlap += 1
        else:
            safe.append(p)
    return safe, n_overlap


class RealDir(Dataset):
    def __init__(self, paths: list[Path], img_size=(60, 200)):
        self.paths = paths
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]))


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def main():
    args = parse_args()
    log = TeeLogger(args.log_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    log(f"dedup: hashing {args.raw_dir} vs {args.eval_dir}")
    t0 = time.time()
    safe_paths, n_overlap = dedup_against_eval(args.raw_dir, args.eval_dir)
    log(f"  raw_total={len(safe_paths)+n_overlap} overlap={n_overlap} safe={len(safe_paths)} ({time.time()-t0:.1f}s)")
    if not safe_paths:
        log("ERROR: no safe training images after dedup; aborting")
        sys.exit(2)

    log(f"loading recognizer: {args.recognizer_ckpt}")
    rckpt = torch.load(args.recognizer_ckpt, map_location=device, weights_only=False)
    rargs = rckpt.get("args", {})
    recog = TinySTR(
        embed_dim=rargs.get("embed_dim", 256),
        encoder_depth=rargs.get("encoder_depth", 6),
        query_self_attn_layers=rargs.get("query_self_attn_layers", 0),
    ).to(device)
    recog.load_state_dict(rckpt["model"])
    recog.eval()
    for p in recog.parameters():
        p.requires_grad_(False)
    n_recog = sum(p.numel() for p in recog.parameters())
    log(f"  recognizer: TinySTR params={n_recog/1e6:.2f}M from epoch {rckpt.get('epoch')}")

    cleaner = TinyCleaner(base_ch=args.base_ch, residual_scale=args.residual_scale).to(device)
    n_clean = sum(p.numel() for p in cleaner.parameters())
    log(f"  cleaner: TinyCleaner params={n_clean/1e6:.3f}M base_ch={args.base_ch} residual_scale={args.residual_scale}")

    ds = RealDir(safe_paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    optim = torch.optim.AdamW(cleaner.parameters(), lr=args.lr, weight_decay=1e-4)

    log(f"training: epochs={args.epochs} batch={args.batch_size} lr={args.lr} residual_l1={args.residual_l1}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        cleaner.train()
        ep_conf = ep_l1 = ep_n = 0.0
        # Track baseline: confidence on uncleaned input for comparison
        ep_conf_in = 0.0
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            cleaned = cleaner(imgs).clamp(-1.0, 1.0)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = recog(cleaned)
                with torch.no_grad():
                    logits_in = recog(imgs)
            conf = confidence_loss(logits)
            l1 = (cleaned - imgs).abs().mean()
            loss = conf + args.residual_l1 * l1
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cleaner.parameters(), 5.0)
            optim.step()

            bsz = imgs.size(0)
            ep_conf += conf.item() * bsz
            ep_conf_in += confidence_loss(logits_in).item() * bsz
            ep_l1 += l1.item() * bsz
            ep_n += bsz

        log(f"[epoch {epoch:02d}] conf_loss={ep_conf/ep_n:.4f} (input={ep_conf_in/ep_n:.4f}) "
            f"l1={ep_l1/ep_n:.4f} elapsed={time.time()-t0:.0f}s")

    torch.save({
        "model": cleaner.state_dict(),
        "args": vars(args),
        "n_train": len(safe_paths),
        "recognizer_ckpt": str(args.recognizer_ckpt),
        "recognizer_epoch": rckpt.get("epoch"),
    }, args.out)
    log(f"saved cleaner -> {args.out}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
