"""Score synthetic captcha difficulty using Model 1.5 v2 ensemble.

For each sample:
  - Run PARSeq v2 + detector v2 + 3 char-CNN scales (64/96/128).
  - Combine per-position softmaxes via log-prob mix at α=0.30 (Model 1.5 v2 best).
  - Compute:
      P_correct = ∏ combined[t, true_char_t]   — joint likelihood of true label
      NLL       = -log(P_correct)              — sum of per-position cross-entropy
      correct   = (argmax matches true label across all 5 positions)

Output: an augmented labels.csv with columns
  filename, label, nll, p_correct, correct

Usage:
    python -m src.eval.score_synth_difficulty \
        --src "E:/pranav/model 2/data/model2_training_data/images" \
        --labels-in "E:/pranav/model 2/data/model2_training_data/images/labels.csv" \
        --out "E:/pranav/model 2/data/model2_training_data/labels_scored.csv" \
        --ckpt checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt \
        --detector-ckpt checkpoints/detector_v2/best.pt \
        --charcls-ckpt "checkpoints/charcls_64_v2/best.pt,checkpoints/charcls_96_v2/best.pt,checkpoints/charcls_128_v2/best.pt" \
        --alpha 0.30 --fixed-n 5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
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
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--labels-in", type=Path, required=True,
                    help="CSV with filename,label rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--detector-ckpt", type=Path, required=True)
    ap.add_argument("--charcls-ckpt", type=str, required=True,
                    help="Comma-separated char-CNN checkpoint paths for multi-scale ensemble")
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_v2.yaml"))
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--fixed-n", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    # Read input labels
    rows = []
    with args.labels_in.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["filename"], row["label"]))
    print(f"loaded {len(rows)} samples from {args.labels_in}", flush=True)
    label_map = dict(rows)

    # ---- PARSeq ----
    print(f"loading PARSeq {args.ckpt}", flush=True)
    parseq = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    parseq.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    parseq = parseq.cuda().eval()

    ds = ImageDir(args.src, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True)
    print(f"running PARSeq on {len(ds)} images...", flush=True)
    names, parseq_probs_list = collect_softmax(parseq, loader, max_len=cfg["max_label_length"] + 1)
    tok = parseq.tokenizer
    parseq_idx, chars = _charset_index_maps(tok, CHARSET)
    char_to_idx = {c: i for i, c in enumerate(CHARSET)}
    del parseq
    torch.cuda.empty_cache()

    # ---- Detector ----
    print(f"loading detector {args.detector_ckpt}", flush=True)
    det_model = load_detector(args.detector_ckpt)
    box_path = args.src.parent / "_score_boxes.jsonl"
    with box_path.open("w") as f:
        n = 0
        for nm, boxes in predict_dir(det_model, args.src):
            f.write(json.dumps({"filename": nm, "bboxes": boxes}) + "\n")
            n += 1
    print(f"detector: wrote {n} box records", flush=True)
    del det_model
    torch.cuda.empty_cache()

    # ---- Char-CNNs (multi-scale) ----
    ckpt_paths = [Path(p.strip()) for p in args.charcls_ckpt.split(",") if p.strip()]
    print(f"loading {len(ckpt_paths)} char-CNN scales", flush=True)
    letter_sm_avg: dict[str, torch.Tensor] = {}
    for ckpt_path in ckpt_paths:
        cls_model, crop_size, pad_frac = load_charcls(ckpt_path)
        ls_one, _ = charcls_softmax_for_dir(
            cls_model, args.src, box_path, crop_size, pad_frac
        )
        for nm, p in ls_one.items():
            if nm in letter_sm_avg:
                letter_sm_avg[nm] = letter_sm_avg[nm] + p
            else:
                letter_sm_avg[nm] = p.clone()
        del cls_model
        torch.cuda.empty_cache()
    for nm in letter_sm_avg:
        letter_sm_avg[nm] = letter_sm_avg[nm] / float(len(ckpt_paths))

    # ---- Score each sample ----
    print(f"scoring {len(names)} samples...", flush=True)
    t0 = time.time()
    eps = 1e-9
    L = args.fixed_n
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "nll", "p_correct", "correct"])
        for i, (nm, p_p_full) in enumerate(zip(names, parseq_probs_list)):
            true_label = label_map.get(nm)
            if true_label is None or len(true_label) != L:
                continue
            if nm not in letter_sm_avg:
                continue
            p_p = p_p_full[:L][:, parseq_idx]   # [L, 62]
            p_c = letter_sm_avg[nm][:L]          # [L, 62]
            log_combined = (
                args.alpha * torch.log(p_p + eps)
                + (1 - args.alpha) * torch.log(p_c + eps)
            )
            combined = log_combined.softmax(dim=-1)  # [L, 62]
            true_idx = [char_to_idx[c] for c in true_label]
            true_probs = combined[range(L), true_idx]   # [L]
            p_correct = float(true_probs.prod().item())
            nll = float((-torch.log(true_probs + eps)).sum().item())
            pred_idx = combined.argmax(dim=-1)
            pred_str = "".join(chars[int(j.item())] for j in pred_idx)
            correct = int(pred_str == true_label)
            w.writerow([nm, true_label, f"{nll:.6f}", f"{p_correct:.6e}", correct])
            if (i + 1) % 5000 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (len(names) - i - 1) / rate
                print(f"  {i+1}/{len(names)}  rate={rate:.0f}/s  eta={eta:.0f}s", flush=True)

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
