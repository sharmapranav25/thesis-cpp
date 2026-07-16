"""Bucket scored synth samples by NLL percentile (option A: equal-sized buckets).

Reads labels_scored.csv (filename,label,nll,p_correct,correct), emits:
  - labels.csv (master, with bucket column appended)
  - easy.csv, medium.csv, hard.csv (subsets by bucket)

Buckets are computed by NLL percentile:
  easy    = bottom 33%   (lowest NLL = model finds these easiest)
  medium  = middle 34%
  hard    = top 33%      (highest NLL = model struggles)

Usage:
    python -m src.eval.bucket_synth \
        --scored "E:/pranav/model 2/data/model2_training_data/labels_scored.csv" \
        --out-dir "E:/pranav/model 2/data/model2_training_data"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    with args.scored.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "filename": row["filename"],
                "label": row["label"],
                "nll": float(row["nll"]),
                "p_correct": float(row["p_correct"]),
                "correct": int(row["correct"]),
            })
    n = len(rows)
    print(f"loaded {n} scored samples")

    nll_arr = np.array([r["nll"] for r in rows])
    # tertile cutoffs
    t1, t2 = np.quantile(nll_arr, [1/3, 2/3])
    print(f"NLL tertile cutoffs: {t1:.4f}, {t2:.4f}")
    print(f"NLL min/median/max: {nll_arr.min():.4f} / {np.median(nll_arr):.4f} / {nll_arr.max():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bucket_writers: dict[str, csv.writer] = {}
    bucket_files = {}
    counts = {"easy": 0, "medium": 0, "hard": 0}

    master_path = args.out_dir / "labels.csv"
    master_f = master_path.open("w", newline="")
    master_w = csv.writer(master_f)
    master_w.writerow(["filename", "label", "nll", "p_correct", "correct", "bucket"])

    for name in ("easy", "medium", "hard"):
        p = args.out_dir / f"{name}.csv"
        bucket_files[name] = p.open("w", newline="")
        bucket_writers[name] = csv.writer(bucket_files[name])
        bucket_writers[name].writerow(["filename", "label", "nll", "p_correct", "correct"])

    for r in rows:
        if r["nll"] <= t1:
            b = "easy"
        elif r["nll"] <= t2:
            b = "medium"
        else:
            b = "hard"
        counts[b] += 1
        master_w.writerow([r["filename"], r["label"], f"{r['nll']:.6f}",
                           f"{r['p_correct']:.6e}", r["correct"], b])
        bucket_writers[b].writerow([r["filename"], r["label"], f"{r['nll']:.6f}",
                                    f"{r['p_correct']:.6e}", r["correct"]])

    master_f.close()
    for f in bucket_files.values():
        f.close()

    print(f"\nwrote {master_path}")
    for name, count in counts.items():
        avg_nll = np.mean([r["nll"] for r in rows if (
            (name == "easy" and r["nll"] <= t1) or
            (name == "medium" and t1 < r["nll"] <= t2) or
            (name == "hard" and r["nll"] > t2)
        )])
        avg_correct = np.mean([r["correct"] for r in rows if (
            (name == "easy" and r["nll"] <= t1) or
            (name == "medium" and t1 < r["nll"] <= t2) or
            (name == "hard" and r["nll"] > t2)
        )])
        print(f"  {name:<6}: {count:>6} samples  avg_nll={avg_nll:.3f}  acc={avg_correct:.3f}")


if __name__ == "__main__":
    main()
