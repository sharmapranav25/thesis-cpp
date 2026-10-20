"""Prepare a leakage-checked 9,500/500 split from the first 10k v3 records."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "pixel_sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("real_test", type=Path)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = args.source / "manifest.jsonl"
    records = []
    with manifest.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == args.limit:
                break
            item = json.loads(line)
            filename = item.get("filename") or item.get("path")
            image = (args.source / filename).resolve()
            label = item["label"]
            pixel_hash = item["pixel_sha256"].lower()
            if len(label) != 6 or set(label) - set(CHARSET):
                raise ValueError(f"record {index}: invalid label {label!r}")
            if not image.is_file():
                raise FileNotFoundError(image)
            records.append({"path": str(image), "label": label, "pixel_sha256": pixel_hash,
                            "bucket": item.get("bucket", "unknown"), "source_index": index})
    if len(records) != args.limit:
        raise ValueError(f"requested {args.limit} records, found {len(records)}")
    hashes = [row["pixel_sha256"] for row in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate image pixels within first 10k")

    # Keep every repeated label in one partition while selecting exactly 500 records.
    by_label = defaultdict(list)
    for row in records:
        by_label[row["label"]].append(row)
    labels = list(by_label)
    random.Random(args.seed).shuffle(labels)
    validation_labels = set()
    selected = 0
    for label in labels:
        size = len(by_label[label])
        if selected + size <= args.validation:
            validation_labels.add(label)
            selected += size
        if selected == args.validation:
            break
    if selected != args.validation:
        raise RuntimeError(f"could select only {selected} validation records")

    train = [row for row in records if row["label"] not in validation_labels]
    validation = [row for row in records if row["label"] in validation_labels]
    fields = ("path", "label", "pixel_sha256")
    train_out = [{key: row[key] for key in fields} for row in train]
    validation_out = [{key: row[key] for key in fields} for row in validation]

    with args.real_test.open(encoding="utf-8-sig", newline="") as stream:
        test = list(csv.DictReader(stream))
    test_hashes = {row["pixel_sha256"].lower() for row in test}
    overlap = set(hashes) & test_hashes
    if overlap:
        raise ValueError(f"synthetic-real pixel overlap: {len(overlap)}")

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "train.csv", train_out)
    write_csv(args.output / "validation.csv", validation_out)
    write_csv(args.output / "test.csv", test)
    audit = {
        "source": str(args.source.resolve()),
        "source_manifest": str(manifest.resolve()),
        "selection": f"first {args.limit:,} finalized manifest records before split",
        "split_method": "seed-42 shuffled label groups; repeated labels kept within one partition",
        "samples": {"selected": len(records), "train": len(train),
                    "validation": len(validation), "real_test": len(test)},
        "source_index_range": [0, args.limit - 1],
        "label_length_counts": dict(Counter(len(row["label"]) for row in records)),
        "bucket_counts": dict(sorted(Counter(row["bucket"] for row in records).items())),
        "unique_labels": len(by_label),
        "repeated_label_groups": sum(len(group) > 1 for group in by_label.values()),
        "unique_synthetic_pixel_hashes": len(set(hashes)),
        "synthetic_real_pixel_overlap": len(overlap),
        "charset": CHARSET,
        "manifests": {},
    }
    for name in ("train.csv", "validation.csv", "test.csv"):
        audit["manifests"][name] = {"sha256": file_sha256(args.output / name)}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
