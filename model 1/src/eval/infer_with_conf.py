"""Run a recogniser (PARSeq alone OR ensemble) on a dir of images,
emit a predictions.csv with per-sample confidence (joint/min/mean).

Modes:
  --no-ensemble       PARSeq alone, force-5 decoding, conf from PARSeq softmax
  --ensemble          PARSeq + detector + multi-scale char-CNNs, log-prob mix
                      at --alpha, force-5 decoding, conf from combined softmax

Outputs CSV columns: filename, label_truth, prediction, conf_joint, conf_min, conf_mean, correct
(label_truth = filename stem, the standard convention for the eval set)

Usage:
    python -m src.eval.infer_with_conf \
        --ckpt <PARSeq ckpt> --src data/eval --out data/eval/preds_<model>.csv \
        --train-labels data/synth_mybb/labels.csv \
        --no-ensemble

    python -m src.eval.infer_with_conf \
        --ckpt <PARSeq ckpt> --src data/eval --out data/eval/preds_<model>.csv \
        --train-labels data/synth_mybb/labels.csv \
        --ensemble --alpha 0.30 \
        --detector-ckpt checkpoints/detector_v2/best.pt \
        --charcls-ckpt "ckpt64,ckpt96,ckpt128"
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
from src.eval.case_decode import (
    collect_softmax, _charset_index_maps, detect_fixed_length,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-labels", type=Path, default=Path("data/synth_mybb/labels.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_v2.yaml"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--no-ensemble", dest="ensemble", action="store_false")
    ap.set_defaults(ensemble=False)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--detector-ckpt", type=Path, default=None)
    ap.add_argument("--charcls-ckpt", type=str, default=None,
                    help="Comma-separated multi-scale char-CNN ckpts")
    ap.add_argument("--fixed-n", type=int, default=None,
                    help="Force this length; else auto-detect from train labels")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="If !=1, scale per-position softmax: softmax(log(p)/T). "
                         "Argmax preserved (predictions unchanged), only conf changes.")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    fixed_n = args.fixed_n
    if fixed_n is None:
        fixed_n = detect_fixed_length(args.train_labels) or 5
    print(f"using fixed length {fixed_n}", flush=True)

    # PARSeq
    print(f"loading PARSeq {args.ckpt}", flush=True)
    parseq = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    parseq.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    parseq = parseq.cuda().eval()
    tok = parseq.tokenizer
    parseq_idx, chars = _charset_index_maps(tok, CHARSET)

    ds = ImageDir(args.src, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=4, pin_memory=True)
    print(f"running PARSeq on {len(ds)} images...", flush=True)
    names, parseq_probs_list = collect_softmax(parseq, loader, max_len=cfg["max_label_length"] + 1)
    del parseq
    torch.cuda.empty_cache()

    cnn_letter_avg: dict[str, torch.Tensor] | None = None
    if args.ensemble:
        if args.detector_ckpt is None or args.charcls_ckpt is None:
            raise SystemExit("--ensemble requires --detector-ckpt and --charcls-ckpt")
        from src.detector.infer import load_detector, predict_dir
        from src.charcls.infer import load_charcls
        from src.charcls.infer_fast import charcls_softmax_for_dir_fast

        print(f"loading detector {args.detector_ckpt}", flush=True)
        det_model = load_detector(args.detector_ckpt)
        box_path = args.src.parent / f"_infer_boxes_{args.src.name}.jsonl"
        with box_path.open("w") as f:
            n = 0
            for nm, boxes in predict_dir(det_model, args.src):
                f.write(json.dumps({"filename": nm, "bboxes": boxes}) + "\n")
                n += 1
        print(f"detector: {n} box records", flush=True)
        del det_model
        torch.cuda.empty_cache()

        ckpt_paths = [Path(p.strip()) for p in args.charcls_ckpt.split(",") if p.strip()]
        cnn_letter_avg = {}
        for ckpt_path in ckpt_paths:
            cls_model, crop_size, pad_frac = load_charcls(ckpt_path)
            ls_one, _ = charcls_softmax_for_dir_fast(
                cls_model, args.src, box_path, crop_size, pad_frac
            )
            for nm, p in ls_one.items():
                cnn_letter_avg[nm] = p.clone() if nm not in cnn_letter_avg else cnn_letter_avg[nm] + p
            del cls_model
            torch.cuda.empty_cache()
        for nm in cnn_letter_avg:
            cnn_letter_avg[nm] = cnn_letter_avg[nm] / float(len(ckpt_paths))
        print(f"char-CNN ensemble ({len(ckpt_paths)} scales) ready", flush=True)

    # Score each sample. Force-5 decoding: zero EOS, argmax over 62 chars per pos.
    eps = 1e-9
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    rows_written = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label_truth", "prediction", "conf_joint", "conf_min", "conf_mean", "correct"])
        for nm, p_p_full in zip(names, parseq_probs_list):
            label_truth = Path(nm).stem
            p_p = p_p_full[:fixed_n][:, parseq_idx]   # [L, 62] PARSeq aligned to charset order
            if cnn_letter_avg is not None and nm in cnn_letter_avg:
                p_c = cnn_letter_avg[nm][:fixed_n]
                log_combined = (
                    args.alpha * torch.log(p_p + eps)
                    + (1 - args.alpha) * torch.log(p_c + eps)
                )
                combined = log_combined.softmax(dim=-1)
            else:
                combined = p_p / (p_p.sum(dim=-1, keepdim=True) + eps)  # renormalise after charset slice

            if args.temperature != 1.0 and args.temperature > 0:
                # softmax(log(combined) / T) — preserves argmax, recalibrates conf
                log_p = torch.log(combined + eps)
                combined = (log_p / args.temperature).softmax(dim=-1)

            top_p, top_idx = combined.max(dim=-1)
            pred = "".join(chars[int(j.item())] for j in top_idx)
            cj = float(top_p.prod().item())
            cmin = float(top_p.min().item())
            cmean = float(top_p.mean().item())
            ok = int(pred == label_truth)
            n_correct += ok
            w.writerow([nm, label_truth, pred, f"{cj:.6e}", f"{cmin:.6f}", f"{cmean:.6f}", ok])
            rows_written += 1

    acc = n_correct / max(1, rows_written)
    print(f"\nwrote {rows_written} rows to {args.out}", flush=True)
    print(f"word accuracy: {acc:.4f} ({n_correct}/{rows_written})", flush=True)


if __name__ == "__main__":
    main()
