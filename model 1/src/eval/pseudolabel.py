"""Generate pseudo-labels for unlabeled real captchas using Model 1.5.

Pipeline: PARSeq + detector + char-CNN, combined at log-prob α=0.70 per
position. Outputs CSV: filename, prediction, conf_joint, conf_min,
conf_mean. The result feeds the pseudo-label retraining step.

Usage:
    python -m src.eval.pseudolabel \
        --ckpt checkpoints/parseq/parseq-epoch=10-val_accuracy=76.980.ckpt \
        --detector-ckpt checkpoints/detector/best.pt \
        --charcls-ckpt checkpoints/charcls/best.pt \
        --src data/raw \
        --out data/raw/pseudolabels.csv \
        --alpha 0.7 --fixed-n 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

PARSEQ_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "parseq"
sys.path.insert(0, str(PARSEQ_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml
from torch.utils.data import DataLoader

from src.parseq.train import build_parseq, CHARSET
from src.parseq.infer import ImageDir
from src.eval.case_decode import collect_softmax, _charset_index_maps
from src.detector.infer import load_detector, predict_dir
from src.charcls.infer import load_charcls, charcls_softmax_for_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--detector-ckpt", type=Path, required=True)
    ap.add_argument("--charcls-ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="dir of unlabeled images")
    ap.add_argument("--out", type=Path, required=True, help="output pseudo-label CSV")
    ap.add_argument("--config", type=Path, default=Path("configs/parseq.yaml"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.70,
                    help="weight on PARSeq logits in the log-prob mix")
    ap.add_argument("--fixed-n", type=int, default=5,
                    help="output length per captcha (fixed)")
    return ap.parse_args()


def main():
    args = parse_args()

    # PARSeq
    cfg = yaml.safe_load(args.config.read_text())
    print(f"loading PARSeq {args.ckpt}", flush=True)
    parseq = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    parseq.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    parseq = parseq.cuda().eval()

    ds = ImageDir(args.src, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True)
    print(f"running PARSeq on {len(ds)} images...", flush=True)
    names, probs = collect_softmax(parseq, loader, max_len=cfg["max_label_length"] + 1)
    tok = parseq.tokenizer
    parseq_idx, chars = _charset_index_maps(tok, CHARSET)

    # Detector
    print(f"loading detector {args.detector_ckpt}", flush=True)
    det_model = load_detector(args.detector_ckpt)
    box_path = args.src.parent / f"_pseudo_boxes_{args.src.name}.jsonl"
    with box_path.open("w") as f:
        n = 0
        for nm, boxes in predict_dir(det_model, args.src):
            f.write(json.dumps({"filename": nm, "bboxes": boxes}) + "\n")
            n += 1
    print(f"detector: wrote {n} box records to {box_path}", flush=True)
    del det_model
    torch.cuda.empty_cache()

    # Char-CNN
    print(f"loading char-CNN {args.charcls_ckpt}", flush=True)
    cls_model, crop_size, pad_frac = load_charcls(args.charcls_ckpt)
    letter_sm, _case_sm = charcls_softmax_for_dir(
        cls_model, args.src, box_path, crop_size, pad_frac
    )
    print(f"char-CNN: softmax for {len(letter_sm)} images", flush=True)

    # Combine and write
    eps = 1e-9
    L = args.fixed_n
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prediction", "conf_joint", "conf_min", "conf_mean"])
        for nm, parseq_probs in zip(names, probs):
            if nm not in letter_sm:
                continue
            cls_probs = letter_sm[nm]
            # align PARSeq's full-vocab softmax to the 62-char layout
            p_p = parseq_probs[:L][:, parseq_idx]  # [L, 62]
            p_c = cls_probs[:L]                    # [L, 62]
            log_combined = (
                args.alpha * torch.log(p_p + eps)
                + (1 - args.alpha) * torch.log(p_c + eps)
            )
            combined = log_combined.softmax(dim=-1)  # [L, 62]
            top_p, top_idx = combined.max(dim=-1)
            pred = "".join(chars[int(i.item())] for i in top_idx)
            cj = float(top_p.prod().item())
            cmin = float(top_p.min().item())
            cmean = float(top_p.mean().item())
            w.writerow([nm, pred, f"{cj:.6f}", f"{cmin:.6f}", f"{cmean:.6f}"])

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
