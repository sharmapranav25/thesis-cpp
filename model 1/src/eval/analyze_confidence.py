"""Selective-prediction analysis. Joins predictions+confidence with ground
truth, finds the optimal confidence threshold for a target precision, and
reports coverage + simulated K-refresh effective solve rate.

Outputs:
  - console: operating points + K-refresh table
  - data/eval/_confidence_analysis/calibration.png
  - data/eval/_confidence_analysis/precision_coverage.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["filename"]] = {
                "prediction": row["prediction"],
                "conf_joint": float(row["conf_joint"]),
                "conf_min": float(row["conf_min"]),
                "conf_mean": float(row["conf_mean"]),
            }
    return out


def load_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                out[row[0]] = row[1]
    return out


def precision_coverage(confs: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by confidence desc; for each prefix length k, compute (threshold, precision, coverage)."""
    order = np.argsort(-confs)
    sorted_conf = confs[order]
    sorted_correct = correct[order]
    cum_correct = np.cumsum(sorted_correct)
    ks = np.arange(1, len(sorted_conf) + 1)
    precision = cum_correct / ks
    coverage = ks / len(sorted_conf)
    return sorted_conf, precision, coverage


def find_threshold(sorted_conf, precision, coverage, target_p: float) -> tuple[float, float, float]:
    """Lowest threshold (highest coverage) where precision ≥ target_p.
    If we walk from highest-conf downward, precision can dip below target as we
    include more (less-confident) samples. We want the maximum coverage where
    precision is still ≥ target."""
    valid = precision >= target_p
    if not valid.any():
        return float("nan"), 0.0, 0.0
    # Pick the largest k where precision is still ≥ target. (sequential: if it drops
    # and recovers, we still want the latest stable point.)
    last_valid_idx = np.where(valid)[0].max()
    return float(sorted_conf[last_valid_idx]), float(precision[last_valid_idx]), float(coverage[last_valid_idx])


def k_refresh_solve_rate(coverage: float, precision: float, ks=(1, 2, 3, 5, 10)) -> dict[int, float]:
    """Probability of at least one confident-correct prediction in K independent refreshes."""
    p_per_attempt = coverage * precision
    return {k: 1 - (1 - p_per_attempt) ** k for k in ks}


def calibration_plot(confs: np.ndarray, correct: np.ndarray, n_bins: int, out: Path) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(confs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    bin_avg_conf = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_idx == i
        bin_count[i] = mask.sum()
        if bin_count[i] > 0:
            bin_acc[i] = correct[mask].mean()
            bin_avg_conf[i] = confs[mask].mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    valid = bin_count > 0
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    ax.scatter(bin_avg_conf[valid], bin_acc[valid], s=bin_count[valid] * 2, alpha=0.7,
               label="bin (size = count)")
    ax.set_xlabel("avg confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def precision_coverage_plot(precision, coverage, out: Path, target_ps=(0.95, 0.90, 0.80)) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, precision, label="P-C curve", linewidth=2)
    for tp in target_ps:
        ax.axhline(tp, linestyle=":", alpha=0.5, label=f"target precision={tp}")
    ax.set_xlabel("coverage (fraction of captchas committed)")
    ax.set_ylabel("precision (P(correct | committed))")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Precision vs coverage")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(pred_csv: Path, labels_csv: Path, out_dir: Path,
         conf_field: str, target_precisions: tuple[float, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_preds(pred_csv)
    labels = load_labels(labels_csv)
    common = sorted(set(preds.keys()) & set(labels.keys()))
    if not common:
        print("ERROR: no overlap between predictions and labels")
        return

    pred_strs = np.array([preds[k]["prediction"] for k in common])
    label_strs = np.array([labels[k] for k in common])
    correct = (pred_strs == label_strs).astype(np.int32)
    confs = np.array([preds[k][conf_field] for k in common], dtype=np.float64)
    n = len(common)

    print(f"\n=== confidence field: {conf_field}  (n={n}) ===")
    print(f"raw word accuracy: {correct.mean()*100:.2f}%   ({int(correct.sum())}/{n})")
    print(f"confidence stats: min={confs.min():.4f}  "
          f"median={np.median(confs):.4f}  mean={confs.mean():.4f}  max={confs.max():.4f}")

    sorted_conf, precision, coverage = precision_coverage(confs, correct)

    print("\n=== abstain / attempt / accuracy ===")
    print("  Each row picks a confidence threshold T so the model only attempts captchas")
    print("  it's confident about. Lower target = more attempts but lower attempt accuracy.")
    print()
    print(f"  {'target_acc':<12}{'thr':<10}{'skipped':<18}{'attempted':<22}{'attempt_acc':<12}")
    for tp in target_precisions:
        t, p, c = find_threshold(sorted_conf, precision, coverage, tp)
        n_attempted = int(round(c * n))
        n_skipped = n - n_attempted
        if np.isnan(t):
            print(f"  {tp:<12.2f}(no threshold reaches this accuracy)")
        else:
            n_correct = int(round(n_attempted * p))
            skip_str = f"{n_skipped} ({(n_skipped/n)*100:.1f}%)"
            att_str = f"{n_attempted} ({(n_attempted/n)*100:.1f}%, {n_correct} OK)"
            print(f"  {tp:<12.2f}{t:<10.4f}{skip_str:<18}{att_str:<22}{p*100:.2f}%")
    print()
    print(f"  raw (no abstain):    attempted={n} (100.0%), accuracy={correct.mean()*100:.2f}%")

    print("\n=== effective K-refresh solve rate ===")
    print("Two strategies compared:")
    print("  ALWAYS COMMIT: take every prediction; success when any of K is correct.")
    print("  SELECTIVE: only commit when confidence > threshold; success when any")
    print("    of K is BOTH committed AND correct (else refresh).")
    print()
    ks = (1, 2, 3, 5, 10)
    raw_acc = correct.mean()
    print(f"  {'strategy':<28}", end="")
    for k in ks:
        print(f"K={k:<6}", end="")
    print()
    # always-commit baseline
    row = f"  {'always-commit (raw)':<28}"
    for k in ks:
        row += f"{(1 - (1 - raw_acc) ** k) * 100:5.1f}%  "
    print(row)
    # selective at each target precision
    for tp in target_precisions:
        t, p, c = find_threshold(sorted_conf, precision, coverage, tp)
        if not np.isnan(t):
            rates = k_refresh_solve_rate(c, p, ks)
            row = f"  {f'selective P>={tp:.2f}':<28}"
            for k in ks:
                row += f"{rates[k]*100:5.1f}%  "
            print(row)
        else:
            print(f"  selective P>={tp:.2f}: no threshold meets target")

    # plots
    calibration_plot(confs, correct, n_bins=10, out=out_dir / "calibration.png")
    precision_coverage_plot(precision, coverage, out=out_dir / "precision_coverage.png")
    print(f"\nplots saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, default=Path("data/eval/predictions_php_e10_conf.csv"))
    ap.add_argument("--labels", type=Path, default=Path("data/eval/labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/_confidence_analysis"))
    ap.add_argument("--conf-field", default="conf_joint",
                    choices=["conf_joint", "conf_min", "conf_mean"])
    ap.add_argument("--target-precisions", type=float, nargs="+",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70])
    args = ap.parse_args()
    main(args.pred, args.labels, args.out, args.conf_field, tuple(args.target_precisions))
