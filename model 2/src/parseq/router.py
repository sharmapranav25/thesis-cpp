"""End-to-end router: image -> length classifier -> specialist -> prediction.

Reads a directory of PNG images, runs the length classifier first, then
dispatches each image to the appropriate specialist:
  is_fixed_prob > THRESH  ->  fixed-N specialist (currently only fixed-5)
  is_fixed_prob <= THRESH ->  variable-length specialist (optional)

Emits predictions.csv with columns:
  filename, prediction, conf_joint, conf_min, conf_mean,
  is_fixed_pred, is_fixed_prob, route

Usage:
    python -m src.parseq.router \
        --classifier-ckpt checkpoints/length_classifier/lenclass-...ckpt \
        --fixed5-ckpt checkpoints/parseq_fixed5_specialist/parseq-...ckpt \
        --src data/eval --out data/eval/predictions_router.csv

If --variable-ckpt is omitted, images classified as variable-length are
returned with empty prediction (and a route="variable_unhandled" tag).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import yaml

from src.parseq.train import build_model, _to_rgb
from src.parseq.length_classifier import LengthClassifier
from src.parseq.infer import ImageDir


def _load_parseq(ckpt_path: Path, cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    return model.cuda().eval(), cfg


def _load_classifier(ckpt_path: Path):
    model = LengthClassifier()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    return model.cuda().eval()


def _decode_pred(model, imgs):
    """Run a PARSeq specialist and return (preds, conf_joint, conf_min, conf_mean) lists."""
    logits = model(imgs)
    probs = logits.softmax(-1)
    preds, char_probs_per_sample = model.tokenizer.decode(probs)
    out = []
    for p, prob in zip(preds, char_probs_per_sample):
        if len(p) == 0:
            out.append((p, 0.0, 0.0, 0.0))
        else:
            cp = prob[: len(p)]
            out.append((p, float(cp.prod().item()), float(cp.min().item()), float(cp.mean().item())))
    return out


def main(args):
    classifier = _load_classifier(args.classifier_ckpt)
    fixed5_model, fixed5_cfg = _load_parseq(args.fixed5_ckpt, args.fixed5_config)
    if args.variable_ckpt is not None:
        variable_model, variable_cfg = _load_parseq(args.variable_ckpt, args.variable_config)
    else:
        variable_model = variable_cfg = None

    img_size = tuple(fixed5_cfg["img_size"])
    ds = ImageDir(args.src, img_size)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_fixed = n_var = n_unhandled = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "prediction", "conf_joint", "conf_min", "conf_mean",
            "is_fixed_pred", "is_fixed_prob", "route",
        ])
        with torch.inference_mode():
            for imgs, names in tqdm(loader, desc="route"):
                imgs = imgs.cuda(non_blocking=True)
                # 1. Length classifier
                logit = classifier(imgs)
                is_fixed_prob = torch.sigmoid(logit).cpu().tolist()

                # Split batch by route
                fixed_mask = [p > args.threshold for p in is_fixed_prob]
                fixed_idxs = [i for i, m in enumerate(fixed_mask) if m]
                var_idxs = [i for i, m in enumerate(fixed_mask) if not m]

                preds_per_idx: dict[int, tuple] = {}

                if fixed_idxs:
                    out = _decode_pred(fixed5_model, imgs[fixed_idxs])
                    for i, res in zip(fixed_idxs, out):
                        preds_per_idx[i] = (*res, "fixed5")
                    n_fixed += len(fixed_idxs)

                if var_idxs:
                    if variable_model is not None:
                        out = _decode_pred(variable_model, imgs[var_idxs])
                        for i, res in zip(var_idxs, out):
                            preds_per_idx[i] = (*res, "variable")
                        n_var += len(var_idxs)
                    else:
                        for i in var_idxs:
                            preds_per_idx[i] = ("", 0.0, 0.0, 0.0, "variable_unhandled")
                        n_unhandled += len(var_idxs)

                for i, name in enumerate(names):
                    pred, cj, cmin, cmean, route = preds_per_idx[i]
                    isfx_p = is_fixed_prob[i]
                    w.writerow([
                        name, pred, f"{cj:.6f}", f"{cmin:.6f}", f"{cmean:.6f}",
                        int(isfx_p > args.threshold), f"{isfx_p:.6f}", route,
                    ])

    print(f"\nrouted: fixed5={n_fixed}, variable={n_var}, variable_unhandled={n_unhandled}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-ckpt", type=Path, required=True)
    ap.add_argument("--fixed5-ckpt", type=Path, required=True)
    ap.add_argument("--fixed5-config", type=Path, default=Path("configs/parseq_fixed5_specialist.yaml"))
    ap.add_argument("--variable-ckpt", type=Path, default=None)
    ap.add_argument("--variable-config", type=Path, default=Path("configs/parseq.yaml"))
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    main(args)
