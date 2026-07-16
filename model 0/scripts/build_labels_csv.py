"""Aggregate per-shard labels.tsv files into a single labels.csv at the
train_corpus root, with shard-relative paths so the model 0 PARSeq trainer
can iterate the entire corpus from one CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(root: Path) -> int:
    shards = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    out_path = root / "labels.csv"
    rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["filename", "label"])
        for shard in shards:
            tsv = shard / "labels.tsv"
            if not tsv.exists():
                continue
            with tsv.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    fn, label = parts[0], parts[1]
                    writer.writerow([f"{shard.name}/{fn}", label])
                    rows += 1
    print(f"wrote {rows} rows to {out_path}")
    return rows


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train_corpus")
    main(root)
