"""Evaluate a TinySTR checkpoint on the 651 real CryptBB captchas.

Reports word/char accuracy using the exact same scoring rules as PARSeq's
src/eval/score.py (exact-match for word, position-by-position with <gap>
filler for char), so numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cleaner import TinyCleaner
from data import _to_rgb, build_train_augment
from model import CHARSET, NUM_POSITIONS, TinySTR

# PARSeq baseline numbers on the same 651 real captchas (greedy)
PARSEQ_BASELINE = {"word_acc": 0.3118, "char_acc": 0.7437}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=HERE.parent / "checkpoints" / "best.pt")
    p.add_argument("--eval-dir", type=Path,
                   default=Path(r"E:\pranav\model 1\data\eval"))
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="defaults to {eval_dir}/labels.csv")
    p.add_argument("--out-csv", type=Path,
                   default=HERE.parent / "eval_predictions.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top", type=int, default=15, help="top-N confusions to print")
    p.add_argument("--tta", type=int, default=0,
                   help="Test-time augmentation: average softmaxes over N augmented versions per image (0 = off)")
    p.add_argument("--cleaner", type=Path, default=None,
                   help="Optional TinyCleaner checkpoint to apply before recognition")
    return p.parse_args()


class EvalDir(Dataset):
    """Read all images named in labels.csv. Returns (filename, image_tensor, label)."""
    def __init__(self, root: Path, labels_csv: Path, img_size: tuple[int, int] = (60, 200),
                 augment: bool = False):
        self.root = Path(root)
        with labels_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header
            self.rows = [(fn, lbl) for fn, lbl in reader if fn]
        self.aug = build_train_augment() if augment else None
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[str, torch.Tensor, str]:
        fn, lbl = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.aug is not None:
            import numpy as np
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
            img = Image.fromarray(arr)
        return fn, self.tf(img), lbl


def collate(batch):
    fns, imgs, lbls = zip(*batch)
    return list(fns), torch.stack(imgs, 0), list(lbls)


@torch.no_grad()
def predict(model: TinySTR, loader: DataLoader, device: torch.device,
            accum_softmax: dict[str, torch.Tensor] | None = None,
            labels_out: dict[str, str] | None = None,
            cleaner: TinyCleaner | None = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Forward pass over loader. Returns per-filename softmax tensors [5, 62] and labels.
    If accum_softmax is provided, adds to it (for TTA averaging)."""
    model.eval()
    if cleaner is not None:
        cleaner.eval()
    sm: dict[str, torch.Tensor] = accum_softmax if accum_softmax is not None else {}
    labels: dict[str, str] = labels_out if labels_out is not None else {}
    for fns, imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        if cleaner is not None:
            imgs = cleaner(imgs).clamp(-1.0, 1.0)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)  # [B, 5, 62]
        probs = logits.float().softmax(dim=-1).cpu()  # [B, 5, 62]
        for fn, row, lbl in zip(fns, probs, lbls):
            if fn in sm:
                sm[fn] = sm[fn] + row
            else:
                sm[fn] = row.clone()
            labels[fn] = lbl
    return sm, labels


def softmax_to_preds(sm: dict[str, torch.Tensor]) -> dict[str, str]:
    preds: dict[str, str] = {}
    for fn, probs in sm.items():
        idx = probs.argmax(dim=-1)  # [5]
        preds[fn] = "".join(CHARSET[i] for i in idx.tolist())
    return preds


def score(preds: dict[str, str], labels: dict[str, str], top_n: int):
    """Mirrors PARSeq's score.py exactly: exact-match for word, position-by-
    position with <gap> filler for char."""
    common = sorted(set(preds.keys()) & set(labels.keys()))
    n = len(common)
    n_exact = sum(1 for k in common if preds[k] == labels[k])
    char_total = 0
    char_correct = 0
    confusions: Counter = Counter()
    for k in common:
        p, l = preds[k], labels[k]
        for i in range(max(len(p), len(l))):
            char_total += 1
            pc = p[i] if i < len(p) else "<gap>"
            lc = l[i] if i < len(l) else "<gap>"
            if pc == lc:
                char_correct += 1
            else:
                confusions[(lc, pc)] += 1
    return {
        "n": n,
        "word_acc": n_exact / n,
        "char_acc": char_correct / char_total,
        "n_exact": n_exact,
        "char_correct": char_correct,
        "char_total": char_total,
        "confusions": confusions,
    }


def main():
    args = parse_args()
    if args.labels_csv is None:
        args.labels_csv = args.eval_dir / "labels.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Read architecture from saved args (defaults match original TinySTR)
    saved_args = ckpt.get("args", {})
    model_kwargs = {
        "embed_dim": saved_args.get("embed_dim", 256),
        "encoder_depth": saved_args.get("encoder_depth", 6),
        "query_self_attn_layers": saved_args.get("query_self_attn_layers", 0),
    }
    model = TinySTR(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    val = ckpt.get("val", {})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: TinySTR params={n_params/1e6:.2f}M from epoch {epoch}  {model_kwargs}")
    if val:
        print(f"  synth val at save: char_acc={val.get('char_acc',float('nan')):.4f} "
              f"word_acc={val.get('word_acc',float('nan')):.4f}")

    cleaner = None
    if args.cleaner is not None:
        cckpt = torch.load(args.cleaner, map_location=device, weights_only=False)
        cargs = cckpt.get("args", {})
        cleaner = TinyCleaner(
            base_ch=cargs.get("base_ch", 32),
            residual_scale=cargs.get("residual_scale", 0.5),
        ).to(device)
        cleaner.load_state_dict(cckpt["model"])
        n_cl = sum(p.numel() for p in cleaner.parameters())
        print(f"cleaner: TinyCleaner params={n_cl/1e6:.3f}M  trained on {cckpt.get('n_train','?')} disjoint real images")

    print(f"eval set: {args.eval_dir}  tta={args.tta}  cleaner={'on' if cleaner else 'off'}")
    sm: dict[str, torch.Tensor] = {}
    labels: dict[str, str] = {}

    # Always do one clean (no-augment) pass
    ds_clean = EvalDir(args.eval_dir, args.labels_csv, augment=False)
    loader_clean = DataLoader(ds_clean, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate)
    print(f"  clean pass: {len(ds_clean)} captchas")
    t0 = time.time()
    sm, labels = predict(model, loader_clean, device, sm, labels, cleaner=cleaner)
    print(f"  clean pass took {time.time()-t0:.2f}s")

    # TTA passes
    for k in range(args.tta):
        ds_aug = EvalDir(args.eval_dir, args.labels_csv, augment=True)
        loader_aug = DataLoader(ds_aug, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)
        t0 = time.time()
        sm, labels = predict(model, loader_aug, device, sm, labels, cleaner=cleaner)
        print(f"  tta pass {k+1}/{args.tta} took {time.time()-t0:.2f}s")

    preds = softmax_to_preds(sm)
    dt_total = time.time() - t0
    print(f"inference complete: {len(preds)} predictions ({1 + args.tta} passes total)")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prediction"])
        for fn in sorted(preds):
            w.writerow([fn, preds[fn]])
    print(f"predictions -> {args.out_csv}")

    s = score(preds, labels, args.top)
    print()
    print(f"=== TinySTR on {s['n']} real CryptBB captchas (greedy) ===")
    print(f"  word acc:  {s['word_acc']*100:6.2f}%   ({s['n_exact']}/{s['n']})")
    print(f"  char acc:  {s['char_acc']*100:6.2f}%   ({s['char_correct']}/{s['char_total']})")
    print()
    print(f"=== vs PARSeq baseline (greedy, same 651 captchas) ===")
    print(f"                  TinySTR    PARSeq    diff")
    dw = (s['word_acc'] - PARSEQ_BASELINE['word_acc']) * 100
    dc = (s['char_acc'] - PARSEQ_BASELINE['char_acc']) * 100
    print(f"  word acc       {s['word_acc']*100:6.2f}%   {PARSEQ_BASELINE['word_acc']*100:6.2f}%   {dw:+.2f}")
    print(f"  char acc       {s['char_acc']*100:6.2f}%   {PARSEQ_BASELINE['char_acc']*100:6.2f}%   {dc:+.2f}")
    char_pow5 = s['char_acc'] ** NUM_POSITIONS * 100
    print(f"  char^5 ceiling {char_pow5:6.2f}%   (TinySTR's word ceiling given its char acc)")
    print()
    print(f"top {args.top} confusions  (gt -> pred  : count)")
    for (gt, pr), c in s['confusions'].most_common(args.top):
        print(f"  {gt!r:>6} -> {pr!r:<6} : {c}")


if __name__ == "__main__":
    main()
