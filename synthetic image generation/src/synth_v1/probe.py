"""Auto-detect scheme parameters from a directory of raw scraped captchas.

Inputs:  raw_dir (a directory of unlabeled captcha PNGs)
Outputs: probe.json with W, H, charset, length, sample stats

Cold-start OCR uses a generic STR checkpoint (defaults to M1 v2 PARSeq, which
is good enough on forum-style captchas). Predictions filtered at conf >= 0.5
form the "observed" charset and length distribution. For schemes where OCR
confidence is low, the auto-pipeline falls back to a configurable default
charset (a-zA-Z0-9) so the renderer doesn't ship gaps.

Standalone — does NOT import src.parseq.train (which pulls albumentations
-> broken cv2 in the current venv).

Usage:
  python -m src.synth_v1.probe \\
      --raw-dir "E:/pranav/model 1/data/raw" \\
      --ckpt "E:/pranav/model 1/checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt" \\
      --config "E:/pranav/model 1/configs/parseq_v2.yaml" \\
      --out probes/cryptbb.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(r"E:\pranav\model 1\third_party\parseq")
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from strhub.models.parseq.system import PARSeq


CHARSET_FULL = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_rgb(im):
    return im.convert("RGB")


def build_parseq(cfg: dict) -> PARSeq:
    return PARSeq(
        charset_train=CHARSET_FULL,
        charset_test=CHARSET_FULL,
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


class _RawDir(Dataset):
    def __init__(self, paths, img_size):
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
        return self.tf(Image.open(self.paths[i])), self.paths[i].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="dir of unlabeled raw captcha PNGs")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="PARSeq ckpt for cold-start OCR")
    ap.add_argument("--config", type=Path, required=True,
                    help="PARSeq config YAML matching the ckpt")
    ap.add_argument("--out", type=Path, required=True, help="probe.json path")
    ap.add_argument("--scheme", type=str, default=None,
                    help="optional scheme name (default: raw-dir parent name)")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="min per-position softmax product to count a prediction")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on raw images probed (0 = all)")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    # --- gather raw images ----------------------------------------------------
    paths = sorted(
        p for p in args.raw_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and not p.name.startswith("_")
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images in {args.raw_dir}")
    print(f"probing {len(paths)} raw images from {args.raw_dir}", flush=True)

    # --- image dims (from raw, before model resize) ---------------------------
    widths, heights, modes = [], [], Counter()
    for p in paths:
        with Image.open(p) as im:
            widths.append(im.width)
            heights.append(im.height)
            modes[im.mode] += 1
    W = int(median(widths))
    H = int(median(heights))
    dominant_mode = modes.most_common(1)[0][0]
    print(f"image dims: median W={W} H={H} mode={dominant_mode}  "
          f"(W range {min(widths)}-{max(widths)}, H range {min(heights)}-{max(heights)})",
          flush=True)

    # --- cold-start OCR -------------------------------------------------------
    cfg = yaml.safe_load(args.config.read_text())
    model = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.cuda().eval()

    ds = _RawDir(paths, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0, pin_memory=True)

    high_conf_preds: list[str] = []
    char_count: Counter = Counter()
    length_dist: Counter = Counter()
    confs_all: list[float] = []
    with torch.inference_mode():
        for imgs, _names in loader:
            imgs = imgs.cuda(non_blocking=True)
            logits = model(imgs)
            preds, probs = model.tokenizer.decode(logits.softmax(-1))
            for pred, prob in zip(preds, probs):
                if len(pred) == 0:
                    continue
                conf = float(prob[: len(pred)].prod().item())
                confs_all.append(conf)
                if conf >= args.conf_threshold:
                    high_conf_preds.append(pred)
                    length_dist[len(pred)] += 1
                    for c in pred:
                        char_count[c] += 1

    print(f"OCR done: {len(high_conf_preds)}/{len(paths)} preds at conf >= {args.conf_threshold}",
          flush=True)

    # --- aggregate ------------------------------------------------------------
    n_chars_mode = length_dist.most_common(1)[0][0] if length_dist else 0
    charset_observed = "".join(sorted(char_count.keys()))

    # Recommend a charset for the renderer:
    # - if observed has at least 30 distinct chars, use it as-is (target distribution)
    # - else fall back to full a-zA-Z0-9 (target probably has more chars than OCR caught)
    if len(charset_observed) >= 30:
        charset_recommended = charset_observed
        recommended_reason = "observed >=30 distinct chars"
    else:
        charset_recommended = CHARSET_FULL
        recommended_reason = "observed <30 distinct chars; falling back to alphanumeric"

    confs_all.sort()
    n = len(confs_all)
    conf_stats = {
        "mean": sum(confs_all) / n if n else 0.0,
        "median": confs_all[n // 2] if n else 0.0,
        "p10": confs_all[n // 10] if n >= 10 else 0.0,
        "p90": confs_all[(9 * n) // 10] if n >= 10 else 0.0,
    }

    probe = {
        "scheme": args.scheme or args.raw_dir.name,
        "raw_dir": str(args.raw_dir),
        "n_samples": len(paths),
        "image_dims": {"width": W, "height": H, "mode": dominant_mode},
        "ocr_backend": str(args.ckpt),
        "ocr_config": str(args.config),
        "ocr_conf_threshold": args.conf_threshold,
        "n_high_conf": len(high_conf_preds),
        "ocr_confidence_stats": conf_stats,
        "charset_observed": charset_observed,
        "charset_observed_size": len(charset_observed),
        "charset_per_char_count": dict(char_count.most_common()),
        "charset_recommended": charset_recommended,
        "charset_recommended_reason": recommended_reason,
        "length_distribution": dict(length_dist.most_common()),
        "n_chars_mode": n_chars_mode,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe, indent=2))
    print(f"wrote {args.out}", flush=True)
    print()
    print(f"  scheme:           {probe['scheme']}")
    print(f"  width:            {W}")
    print(f"  height:           {H}")
    print(f"  n_chars_mode:     {n_chars_mode}")
    print(f"  charset (obs):    {len(charset_observed)} chars  -> {charset_observed[:40]}{'...' if len(charset_observed) > 40 else ''}")
    print(f"  charset (reco):   {len(charset_recommended)} chars ({recommended_reason})")
    print(f"  high-conf preds:  {len(high_conf_preds)}/{len(paths)}  "
          f"(mean conf {conf_stats['mean']:.3f})")


if __name__ == "__main__":
    main()
