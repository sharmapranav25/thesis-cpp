"""Compare predictions.csv vs ground-truth labels.csv.

Reports:
- exact-match accuracy (the meaningful number)
- char-level accuracy (per-position match for same-length predictions)
- top-N most-confused character pairs
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _load_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return {row[0]: row[1] for row in reader if row}


def main(pred_csv: Path, label_csv: Path, top_n: int) -> None:
    preds = _load_csv(pred_csv)
    labels = _load_csv(label_csv)

    common = sorted(set(preds.keys()) & set(labels.keys()))
    missing_pred = sorted(set(labels.keys()) - set(preds.keys()))
    missing_label = sorted(set(preds.keys()) - set(labels.keys()))

    if missing_pred:
        print(f"WARN: {len(missing_pred)} files in labels but not predictions")
    if missing_label:
        print(f"WARN: {len(missing_label)} files in predictions but not labels")
    if not common:
        print("ERROR: no overlap; nothing to score")
        return

    n = len(common)
    n_exact = sum(1 for k in common if preds[k] == labels[k])
    char_total = 0
    char_correct = 0
    confusions = Counter()
    for k in common:
        p, l = preds[k], labels[k]
        for i in range(max(len(p), len(l))):
            char_total += 1
            pc = p[i] if i < len(p) else "<gap>"
            lc = l[i] if i < len(l) else "<gap>"
            if pc == lc:
                char_correct += 1
            else:
                confusions[(lc, pc)] += 1

    exact = n_exact / n * 100
    char_acc = char_correct / char_total * 100
    print(f"\n=== eval over {n} samples ===")
    print(f"exact-match accuracy:  {exact:6.2f}%   ({n_exact}/{n})")
    print(f"char-level accuracy:   {char_acc:6.2f}%   ({char_correct}/{char_total})")
    print(f"\ntop {top_n} confusions  (gt -> pred  : count)")
    for (gt, pr), c in confusions.most_common(top_n):
        print(f"  {gt!r:>6} -> {pr!r:<6} : {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    main(args.pred, args.labels, args.top)
