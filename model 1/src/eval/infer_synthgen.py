"""Standalone PARSeq inference + scoring for the parseq_synthgen ckpt.

Bypasses src.parseq.train (which imports albumentations -> broken cv2 install).
Reimplements the minimum: build_parseq, ImageDir, predict loop.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

from strhub.models.parseq.system import PARSeq


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_rgb(im):
    return im.convert("RGB")


def build_parseq(cfg: dict) -> PARSeq:
    return PARSeq(
        charset_train=CHARSET,
        charset_test=CHARSET,
        max_label_length=cfg["max_label_length"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        warmup_pct=cfg["warmup_pct"],
        weight_decay=cfg["weight_decay"],
        img_size=tuple(cfg["img_size"]),
        patch_size=tuple(cfg["patch_size"]),
        embed_dim=cfg["embed_dim"],
        enc_num_heads=cfg["enc_num_heads"],
        enc_mlp_ratio=cfg["enc_mlp_ratio"],
        enc_depth=cfg["enc_depth"],
        dec_num_heads=cfg["dec_num_heads"],
        dec_mlp_ratio=cfg["dec_mlp_ratio"],
        dec_depth=cfg["dec_depth"],
        perm_num=cfg["perm_num"],
        perm_forward=cfg["perm_forward"],
        perm_mirrored=cfg["perm_mirrored"],
        decode_ar=cfg["decode_ar"],
        refine_iters=cfg["refine_iters"],
        dropout=cfg["dropout"],
    )


class ImageDir(Dataset):
    def __init__(self, root: Path, img_size):
        self.paths = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and not p.name.startswith("_")
        )
        self.tf = T.Compose([
            T.Lambda(_to_rgb),
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.paths[i].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.cuda().eval()

    ds = ImageDir(args.src, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0, pin_memory=True)
    print(f"running PARSeq on {len(ds)} images", flush=True)

    preds_dict: dict[str, str] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prediction"])
        with torch.inference_mode():
            for imgs, names in loader:
                imgs = imgs.cuda(non_blocking=True)
                logits = model(imgs)
                preds, _probs = model.tokenizer.decode(logits.softmax(-1))
                for name, p in zip(names, preds):
                    w.writerow([name, p])
                    preds_dict[name] = p
    print(f"wrote {args.out}", flush=True)

    # --- score --------------------------------------------------------------
    labels: dict[str, str] = {}
    with args.labels.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                labels[row[0]] = row[1]

    common = sorted(set(preds_dict.keys()) & set(labels.keys()))
    n = len(common)
    n_exact = sum(1 for k in common if preds_dict[k] == labels[k])
    char_total = char_correct = 0
    confusions: Counter = Counter()
    for k in common:
        p, l = preds_dict[k], labels[k]
        for i in range(max(len(p), len(l))):
            char_total += 1
            pc = p[i] if i < len(p) else "<gap>"
            lc = l[i] if i < len(l) else "<gap>"
            if pc == lc:
                char_correct += 1
            else:
                confusions[(lc, pc)] += 1

    print(f"\n=== eval over {n} samples ===")
    print(f"exact-match (word) accuracy: {n_exact/n*100:6.2f}%   ({n_exact}/{n})")
    print(f"char-level accuracy:         {char_correct/char_total*100:6.2f}%   ({char_correct}/{char_total})")
    print(f"\ntop {args.top} confusions (gt -> pred : count)")
    for (gt, pr), c in confusions.most_common(args.top):
        print(f"  {gt!r:>6} -> {pr!r:<6} : {c}")


if __name__ == "__main__":
    main()
