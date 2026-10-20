"""Prepare leakage-audited City manifests for Level 1 PARSeq training."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


CHARSET = "0123456789"
LABEL_RE = re.compile(r"^[0-9]{5}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "pixel_sha256"])
        writer.writeheader()
        writer.writerows(rows)


def select_validation_labels(groups: dict[str, list[dict]], target: int, seed: int) -> set[str]:
    labels = list(groups)
    random.Random(seed).shuffle(labels)
    selected: set[str] = set()
    count = 0
    for label in labels:
        size = len(groups[label])
        if count + size <= target:
            selected.add(label)
            count += size
        if count == target:
            return selected
    raise RuntimeError(f"could select only {count} validation records; requested {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("real_images", type=Path)
    parser.add_argument("--validation", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_manifest = args.source / "manifest.jsonl"
    records = []
    with source_manifest.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            label = str(item["label"])
            if not LABEL_RE.fullmatch(label):
                raise ValueError(f"synthetic record {index}: invalid label {label!r}")
            image = (args.source / item["filename"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            records.append({
                "path": str(image),
                "label": label,
                "pixel_sha256": item["pixel_sha256"].lower(),
                "source_index": index,
            })
    if len(records) != 100_000:
        raise ValueError(f"expected 100,000 synthetic records, found {len(records)}")
    synthetic_hashes = [row["pixel_sha256"] for row in records]
    if len(set(synthetic_hashes)) != len(synthetic_hashes):
        raise ValueError("duplicate synthetic image pixels")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["label"]].append(row)
    validation_labels = select_validation_labels(groups, args.validation, args.seed)
    train = [row for row in records if row["label"] not in validation_labels]
    validation = [row for row in records if row["label"] in validation_labels]

    # Only five-digit filenames carry verified ground-truth labels. This excludes
    # the screenshot-named unlabeled half and the exact duplicate 91018(1).png.
    test = []
    for image in sorted(args.real_images.glob("*.png")):
        label = image.stem
        if not LABEL_RE.fullmatch(label):
            continue
        test.append({
            "path": str(image.resolve()),
            "label": label,
            "pixel_sha256": file_sha256(image),
        })
    if len(test) != 999:
        raise ValueError(f"expected 999 uniquely named labeled real images, found {len(test)}")
    test_hashes = [row["pixel_sha256"] for row in test]
    if len(set(test_hashes)) != len(test_hashes):
        raise ValueError("duplicate pixels among selected real test images")
    pixel_overlap = set(synthetic_hashes) & set(test_hashes)
    if pixel_overlap:
        raise ValueError(f"synthetic-real pixel overlap: {len(pixel_overlap)}")

    fields = ("path", "label", "pixel_sha256")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "train.csv", [{key: row[key] for key in fields} for row in train])
    write_csv(args.output / "validation.csv", [{key: row[key] for key in fields} for row in validation])
    write_csv(args.output / "test.csv", test)

    train_labels = {row["label"] for row in train}
    validation_label_set = {row["label"] for row in validation}
    test_labels = {row["label"] for row in test}
    audit = {
        "source": str(args.source.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "real_images": str(args.real_images.resolve()),
        "split_method": "seed-42 shuffled label groups; repeated labels kept in one partition",
        "samples": {"selected": len(records), "train": len(train),
                    "validation": len(validation), "real_test": len(test)},
        "charset": CHARSET,
        "label_length_counts": dict(Counter(len(row["label"]) for row in records)),
        "unique_synthetic_labels": len(groups),
        "repeated_synthetic_label_groups": sum(len(group) > 1 for group in groups.values()),
        "train_validation_label_overlap": len(train_labels & validation_label_set),
        "synthetic_real_label_overlap": len((train_labels | validation_label_set) & test_labels),
        "unique_synthetic_pixel_hashes": len(set(synthetic_hashes)),
        "unique_real_pixel_hashes": len(set(test_hashes)),
        "synthetic_real_pixel_overlap": len(pixel_overlap),
        "excluded_real_files": {
            "unlabeled_screenshot_names": 1000,
            "duplicate_noncanonical_name": "91018(1).png",
        },
        "manifests": {},
    }
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = args.output / name
        audit["manifests"][name] = {"sha256": file_sha256(path)}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
