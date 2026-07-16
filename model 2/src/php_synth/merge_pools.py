"""Merge multiple synth pools into a single training directory.

Filenames are prefixed by source dir name. Output labels.csv has columns:
filename,label,difficulty,is_fixed_length,bucket — `bucket` is the source
dir name with the `synth_` prefix stripped (e.g., `easy`, `medium`, `hard`).
Used downstream by BucketCurriculumSampler to schedule per-epoch weighting.

Usage:
    python -m src.php_synth.merge_pools \
        --src data/synth_easy data/synth_medium data/synth_hard \
        --out data/synth_buckets
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def merge(srcs: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "labels.csv"

    total = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, lineterminator="\n")
        writer.writerow(["filename", "label", "difficulty", "is_fixed_length", "bucket"])
        for src in srcs:
            src = Path(src)
            bucket = src.name.replace("synth_", "")  # synth_easy -> easy
            with (src / "labels.csv").open("r", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                header = next(reader)
                if header[:4] != ["filename", "label", "difficulty", "is_fixed_length"]:
                    raise ValueError(f"unexpected header in {src}/labels.csv: {header}")
                src_count = 0
                for fn, lbl, d, isfx in reader:
                    new_fn = f"{bucket}_{fn}"
                    shutil.copy2(src / fn, out / new_fn)
                    writer.writerow([new_fn, lbl, d, isfx, bucket])
                    src_count += 1
                print(f"  {src}: {src_count} images (bucket={bucket})")
                total += src_count
    print(f"merged {total} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    merge(args.src, args.out)
