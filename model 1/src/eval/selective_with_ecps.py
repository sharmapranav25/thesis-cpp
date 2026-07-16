"""Selective-prediction analysis across confidence thresholds.

For each threshold T, compute:
  - shoot_rate: fraction of captchas attempted (conf >= T)
  - acc_on_attempts: of attempts, fraction predicted correctly
  - ECPS: expected captchas per success = 1 / (shoot_rate * acc_on_attempts)
        i.e. how many captchas you have to see in expectation to get one right

Usage:
    python -m src.eval.selective_with_ecps \
        --pred data/eval/preds_model1.csv \
        --thresholds 0.99 0.95 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10 0.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="CSV with columns filename,label_truth,prediction,conf_joint,...,correct")
    ap.add_argument("--conf-field", default="conf_joint",
                    help="Which confidence column to threshold on")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0])
    ap.add_argument("--name", default="model")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = []
    with args.pred.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "conf": float(row[args.conf_field]),
                "correct": int(row["correct"]),
            })
    n = len(rows)
    print(f"loaded {n} predictions from {args.pred}")
    print(f"\nselective prediction — {args.name}\n")
    print(f"{'threshold':>10} {'shoot rate':>11} {'acc on attempts':>17} {'ECPS':>8}")
    print(f"{'-'*10:>10} {'-'*11:>11} {'-'*17:>17} {'-'*8:>8}")

    for t in sorted(args.thresholds, reverse=True):
        attempted = [r for r in rows if r["conf"] >= t]
        n_attempt = len(attempted)
        n_correct = sum(r["correct"] for r in attempted)
        shoot_rate = n_attempt / n if n else 0.0
        acc = n_correct / n_attempt if n_attempt else 0.0
        ecps = 1.0 / (shoot_rate * acc) if (shoot_rate * acc) > 0 else float("inf")
        ecps_str = f"{ecps:.2f}" if ecps != float("inf") else "inf"
        print(f"{t:>10.3f} {shoot_rate*100:>10.2f}% {acc*100:>16.2f}% {ecps_str:>8}")


if __name__ == "__main__":
    main()
