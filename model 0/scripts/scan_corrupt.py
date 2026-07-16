"""Scan train_corpus for files whose magic bytes don't match PNG/JPEG,
delete them, and rewrite each shard's labels.tsv to drop those rows.

Run from model 0 root:
    python scripts/scan_corrupt.py
    python scripts/scan_corrupt.py --dry-run     # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


def is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8)
    except OSError:
        return False
    if path.suffix.lower() == ".png":
        return head.startswith(PNG_MAGIC)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return head.startswith(JPEG_MAGIC)
    return False


def main(root: Path, dry_run: bool) -> None:
    shards = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    total_files = 0
    total_bad = 0
    by_lib = {}

    for shard in shards:
        tsv = shard / "labels.tsv"
        if not tsv.exists():
            continue
        kept_lines = []
        bad_files = []
        with tsv.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                fn = parts[0]
                fpath = shard / fn
                total_files += 1
                if fpath.exists() and is_valid(fpath):
                    kept_lines.append(line)
                else:
                    total_bad += 1
                    bad_files.append(fpath)
                    # try to extract lib from params_json (4th column)
                    if len(parts) >= 4:
                        params = parts[3]
                        i = params.find('"lib":"')
                        if i != -1:
                            j = params.find('"', i + 7)
                            lib = params[i + 7:j]
                            by_lib[lib] = by_lib.get(lib, 0) + 1

        if bad_files and not dry_run:
            for bf in bad_files:
                try:
                    bf.unlink()
                except OSError:
                    pass
            with tsv.open("w", encoding="utf-8") as f:
                f.writelines(kept_lines)

    print(f"scanned {total_files} files in {len(shards)} shards")
    print(f"corrupt: {total_bad} ({100.0 * total_bad / max(total_files, 1):.1f}%)")
    if by_lib:
        print("by library:")
        for lib, cnt in sorted(by_lib.items(), key=lambda x: -x[1]):
            print(f"  {lib}: {cnt}")
    if dry_run:
        print("(dry run — no changes made)")
    else:
        print("removed corrupt files and updated per-shard labels.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("train_corpus"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.root, args.dry_run)
