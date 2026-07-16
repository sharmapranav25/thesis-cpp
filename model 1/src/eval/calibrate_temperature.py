"""Fit a per-position-softmax temperature for a recogniser using a held-out
calibration set (typically the OTHER model's training data — same distribution,
unseen during this model's training).

For each calibration sample we know the true label. We compute per-position
softmaxes (PARSeq alone OR the PARSeq+CNN ensemble combined softmax), then
fit a single scalar T that minimises NLL of the true label:

    L(T) = -∑_{sample} ∑_{t=1..N} log( softmax(log(p_t) / T) [y_t] )

Temperature scaling preserves argmax — calibrated predictions are identical
to uncalibrated ones; only confidences change. T > 1 cools (model was
over-confident); T < 1 sharpens (model was under-confident).

Output: a JSON {"temperature": float, "n_calibration_samples": int, ...}.

Usage:
  python -m src.eval.calibrate_temperature \
      --ckpt <PARSeq ckpt> \
      --calibration-src <dir of cross-train images> \
      --calibration-labels <cross-train labels.csv> \
      --out <T.json> \
      [--ensemble --alpha 0.50 --detector-ckpt ... --charcls-ckpt ...] \
      [--limit 5000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--calibration-src", type=Path, required=True)
    ap.add_argument("--calibration-labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/parseq_v2.yaml"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--detector-ckpt", type=Path, default=None)
    ap.add_argument("--charcls-ckpt", type=str, default=None)
    ap.add_argument("--fixed-n", type=int, default=5)
    ap.add_argument("--limit", type=int, default=5000,
                    help="cap calibration samples for speed")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    fixed_n = args.fixed_n
    char_to_idx = {c: i for i, c in enumerate(CHARSET)}

    # --- load calibration labels -----------------------------------------
    label_map: dict[str, str] = {}
    with args.calibration_labels.open() as f:
        r = csv.reader(f)
        header = next(r)
        # support both 2-col and 4-col layouts
        for row in r:
            if len(row) >= 2:
                label_map[row[0]] = row[1]
    print(f"loaded {len(label_map)} cross-train labels", flush=True)

    # --- PARSeq inference ------------------------------------------------
    print(f"loading PARSeq {args.ckpt}", flush=True)
    parseq = build_parseq(cfg)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    parseq.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    parseq = parseq.cuda().eval()
    tok = parseq.tokenizer
    parseq_idx, chars = _charset_index_maps(tok, CHARSET)

    # take first `limit` calibration images that exist on disk
    src = args.calibration_src
    cands = sorted([
        p for p in src.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and not p.name.startswith("_")
    ])
    if args.limit and len(cands) > args.limit:
        cands = cands[: args.limit]
    print(f"using {len(cands)} calibration images", flush=True)

    # subset the dir via a temp ImageDir-like
    class _Subset(torch.utils.data.Dataset):
        def __init__(self, paths, img_size):
            from torchvision import transforms as T
            from PIL import Image
            from src.parseq.train import _to_rgb
            self.paths = paths
            self.tf = T.Compose([
                T.Lambda(_to_rgb),
                T.Resize(img_size, antialias=True),
                T.ToTensor(),
                T.Normalize((0.5,) * 3, (0.5,) * 3),
            ])
            self.Image = Image
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            im = self.Image.open(self.paths[i])
            return self.tf(im), self.paths[i].name

    ds = _Subset(cands, tuple(cfg["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0, pin_memory=True)
    names, parseq_probs_list = collect_softmax(parseq, loader, max_len=cfg["max_label_length"] + 1)
    del parseq
    torch.cuda.empty_cache()

    # --- optional ensemble -----------------------------------------------
    cnn_letter_avg: dict[str, torch.Tensor] | None = None
    if args.ensemble:
        from src.detector.infer import load_detector, predict_dir
        from src.charcls.infer import load_charcls
        from src.charcls.infer_fast import charcls_softmax_for_dir_fast

        det_model = load_detector(args.detector_ckpt)
        box_path = src.parent / f"_calib_boxes_{src.name}.jsonl"
        with box_path.open("w") as f:
            for nm, boxes in predict_dir(det_model, src, batch=args.batch):
                if any(p.name == nm for p in cands):
                    f.write(json.dumps({"filename": nm, "bboxes": boxes}) + "\n")
        del det_model
        torch.cuda.empty_cache()

        ckpt_paths = [Path(p.strip()) for p in args.charcls_ckpt.split(",") if p.strip()]
        cnn_letter_avg = {}
        for ckpt_path in ckpt_paths:
            cls_model, crop_size, pad_frac = load_charcls(ckpt_path)
            ls_one, _ = charcls_softmax_for_dir_fast(
                cls_model, src, box_path, crop_size, pad_frac
            )
            for nm, p in ls_one.items():
                cnn_letter_avg[nm] = p.clone() if nm not in cnn_letter_avg else cnn_letter_avg[nm] + p
            del cls_model
            torch.cuda.empty_cache()
        for nm in cnn_letter_avg:
            cnn_letter_avg[nm] = cnn_letter_avg[nm] / float(len(ckpt_paths))
        print(f"char-CNN ensemble ({len(ckpt_paths)} scales) ready", flush=True)

    # --- collect (per-position softmax, true_index_per_position) ---------
    # We work in log-prob space then divide by T and softmax. To avoid recomputing
    # the ensemble combination per T, we build the FINAL combined log-probs once.
    eps = 1e-9
    log_probs_list: list[torch.Tensor] = []   # each: [N, 62] log probs per position
    true_idx_list: list[torch.Tensor] = []    # each: [N] long indices

    for nm, p_p_full in zip(names, parseq_probs_list):
        true_label = label_map.get(nm)
        if true_label is None or len(true_label) != fixed_n:
            continue
        p_p = p_p_full[:fixed_n][:, parseq_idx]
        if cnn_letter_avg is not None:
            if nm not in cnn_letter_avg:
                continue
            p_c = cnn_letter_avg[nm][:fixed_n]
            log_combined = (
                args.alpha * torch.log(p_p + eps)
                + (1 - args.alpha) * torch.log(p_c + eps)
            )
            log_p = torch.log_softmax(log_combined, dim=-1)
        else:
            # PARSeq alone — renormalise after charset slice
            log_p = torch.log(p_p / (p_p.sum(dim=-1, keepdim=True) + eps) + eps)
        log_probs_list.append(log_p)
        true_idx_list.append(
            torch.tensor([char_to_idx[c] for c in true_label], dtype=torch.long)
        )

    n = len(log_probs_list)
    print(f"calibration set: {n} samples", flush=True)
    if n == 0:
        raise SystemExit("no calibration samples — check labels file path")

    log_probs = torch.stack(log_probs_list, dim=0)   # [N, fixed_n, 62]
    true_idx = torch.stack(true_idx_list, dim=0)     # [N, fixed_n]

    # --- fit T -----------------------------------------------------------
    # NLL(T) = -∑ log( softmax(log_p / T) [y] )
    def nll(T_val: float) -> float:
        if T_val <= 0:
            return float("inf")
        scaled = log_probs / T_val
        # log_softmax along last dim
        log_q = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
        # gather true_idx
        gathered = log_q.gather(-1, true_idx.unsqueeze(-1)).squeeze(-1)  # [N, fixed_n]
        return float(-gathered.sum().item())

    # golden-section search over T in [0.05, 20]
    lo, hi = 0.05, 20.0
    phi = (1 + 5 ** 0.5) / 2
    inv_phi = 1 / phi
    a, b = lo, hi
    c = b - (b - a) * inv_phi
    d = a + (b - a) * inv_phi
    fc = nll(c); fd = nll(d)
    for _ in range(60):  # tight convergence
        if fc < fd:
            b = d; d = c; fd = fc
            c = b - (b - a) * inv_phi
            fc = nll(c)
        else:
            a = c; c = d; fc = fd
            d = a + (b - a) * inv_phi
            fd = nll(d)
    T_opt = (a + b) / 2
    nll_opt = nll(T_opt)
    nll_baseline = nll(1.0)

    # also compute calibration-set accuracy (predictions are unchanged by T)
    pred_idx = log_probs.argmax(dim=-1)  # [N, fixed_n]
    correct = (pred_idx == true_idx).all(dim=-1).float().mean().item()

    print(f"\nfit complete:")
    print(f"  T_optimal     = {T_opt:.4f}")
    print(f"  NLL(T=1)      = {nll_baseline:.2f}")
    print(f"  NLL(T={T_opt:.4f}) = {nll_opt:.2f}")
    print(f"  improvement   = {nll_baseline - nll_opt:.2f}")
    print(f"  word acc on calibration set: {correct*100:.2f}% ({n} samples)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "temperature": T_opt,
            "nll_T1": nll_baseline,
            "nll_T_opt": nll_opt,
            "n_calibration_samples": n,
            "calibration_word_acc": correct,
            "ckpt": str(args.ckpt),
            "ensemble": args.ensemble,
            "alpha": args.alpha if args.ensemble else None,
            "calibration_src": str(args.calibration_src),
        }, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
