"""Prepare the audited CryptBB synthetic corpus and preserved real split for Phase 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


READER_CODE = Path(r"E:\pranav\Final sem\sourcefree_synth\style_pipeline\phase3_reader_v2")
sys.path.insert(0, str(READER_CODE))
from common import CHARSET, SIZE, group_key, load_image, pixels, read, sha, write  # noqa: E402


FONT_IDS = re.compile(rb'"font_ids":\s*\[([^\]]*)\]')
PIXEL_HASH = re.compile(rb'"pixel_sha256":\s*"([0-9a-f]{64})"')


def partition_for(text: str) -> str:
    value = int(hashlib.sha256(("capri-phase3-cryptbb-v1:" + text).encode()).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 88:
        return "selection"
    if value < 94:
        return "calibration"
    return "test"


def manifest_headers(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            header = line[:1200]
            font_match = FONT_IDS.search(header)
            hash_match = PIXEL_HASH.search(header)
            if font_match is None or hash_match is None:
                raise RuntimeError("Could not read required manifest header fields")
            font_ids = [int(value.strip()) for value in font_match.group(1).split(b",") if value.strip()]
            yield font_ids, hash_match.group(1).decode("ascii")


def main(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "status.json", {"state": "auditing", "pid": os.getpid(), "images": 0})
    source_status = read(args.corpus / "status.json")
    source_audit = read(args.corpus / "audit.json")
    if source_status.get("state") != "complete" or not source_audit.get("passed"):
        raise RuntimeError("Source corpus is not complete and audited")
    if (args.corpus / "failure.json").exists():
        raise RuntimeError("Source corpus has a failure.json")

    fonts = read(args.project / "releases/CAPRI_typography_v3/fonts.json")
    held = sorted(
        range(len(fonts)),
        key=lambda index: hashlib.sha256(fonts[index]["family"].encode()).hexdigest(),
    )[:3]
    with (args.corpus / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 100000:
        raise RuntimeError(f"Expected 100,000 labels, found {len(csv_rows)}")

    cache = np.lib.format.open_memmap(
        args.out / "synthetic.npy", mode="w+", dtype=np.uint8, shape=(len(csv_rows), 32, 192, 3)
    )
    records = []
    seen: set[str] = set()
    roles: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    headers = manifest_headers(args.corpus / "manifest.jsonl")
    for index, (row, (font_ids, expected_pixel_hash)) in enumerate(zip(csv_rows, headers)):
        text = row["label"]
        if not (set(text) <= set(CHARSET) and 2 <= len(text) <= 8):
            raise RuntimeError(f"Invalid label at index {index}")
        path = args.corpus / row["filename"]
        image = load_image(path)
        pixel_hash = pixels(image)
        if pixel_hash != expected_pixel_hash or pixel_hash in seen:
            raise RuntimeError(f"Pixel integrity failure at index {index}")
        seen.add(pixel_hash)
        cache[index] = np.asarray(image.resize(SIZE, Image.Resampling.BICUBIC))
        text_role = partition_for(text)
        withheld = bool(set(font_ids) & set(held))
        role = ("font_test" if text_role == "test" else "excluded_font") if withheld else text_role
        roles[role] += 1
        groups[group_key(text)] += 1
        records.append(
            {
                "index": index,
                "path": str(path),
                "label": text,
                "pixel_sha256": pixel_hash,
                "partition": role,
                "text_partition": text_role,
                "group": group_key(text),
                "font_ids": font_ids,
            }
        )
        if (index + 1) % 2500 == 0:
            write(args.out / "status.json", {"state": "auditing", "pid": os.getpid(), "images": index + 1})
            print(json.dumps({"state": "auditing", "images": index + 1}), flush=True)
    if len(records) != len(csv_rows) or next(headers, None) is not None:
        raise RuntimeError("Manifest and labels.csv row counts differ")
    cache.flush()
    del cache

    with (args.out / "partitions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    split = read(args.split)
    if len(split["fit"]) != 480 or len(split["validation"]) != 120 or split.get("labels_read") is not False:
        raise RuntimeError("Unexpected CryptBB real split boundary")
    real_records = {}
    real_hashes: dict[str, set[str]] = {}
    for part in ("fit", "validation"):
        part_rows = []
        real_hashes[part] = set()
        array = np.lib.format.open_memmap(
            args.out / f"{part}.npy", mode="w+", dtype=np.uint8, shape=(len(split[part]), 32, 192, 3)
        )
        for index, record in enumerate(split[part]):
            path = Path(record["path"])
            if sha(path) != record["file_sha256"]:
                raise RuntimeError(f"Real file hash mismatch: {part}/{index}")
            image = load_image(path)
            pixel_hash = pixels(image)
            if pixel_hash != record["pixel_sha256"] or image.size != (record["width"], record["height"]):
                raise RuntimeError(f"Real pixel integrity mismatch: {part}/{index}")
            if pixel_hash in real_hashes[part]:
                raise RuntimeError(f"Duplicate real image in {part}")
            real_hashes[part].add(pixel_hash)
            array[index] = np.asarray(image.resize(SIZE, Image.Resampling.BICUBIC))
            part_rows.append(
                {
                    "id": f"{part}_{index:04d}",
                    "path": str(path),
                    "pixel_sha256": pixel_hash,
                    "file_sha256": record["file_sha256"],
                }
            )
        array.flush()
        del array
        real_records[part] = part_rows
    if real_hashes["fit"] & real_hashes["validation"]:
        raise RuntimeError("Real fit/validation overlap")
    if seen & (real_hashes["fit"] | real_hashes["validation"]):
        raise RuntimeError("Synthetic/real pixel overlap")
    write(args.out / "real_locators.json", real_records)
    shortcut = {
        "claim": (
            "Source renderer retains count-dependent layout. This run does not claim shortcut-resistant counting; "
            "layout stress evaluation remains required."
        ),
        "source_protocol": str(args.corpus / "protocol.json"),
    }
    write(args.out / "shortcut_audit.json", shortcut)
    audit = {
        "passed": True,
        "images": len(records),
        "unique_pixels": len(seen),
        "case_group_counts": dict(groups),
        "partitions": dict(roles),
        "heldout_font_ids": held,
        "heldout_font_families": [fonts[index]["family"] for index in held],
        "real_counts": {part: len(rows) for part, rows in real_records.items()},
        "real_labels_available": False,
        "real_accuracy_permitted": False,
        "manifest_sha256": sha(args.corpus / "manifest.jsonl"),
        "labels_sha256": sha(args.corpus / "labels.csv"),
        "protocol_sha256": sha(args.corpus / "protocol.json"),
        "split_sha256": sha(args.split),
        "partitions_sha256": sha(args.out / "partitions.jsonl"),
        "text_partition_rule": "SHA256 capri-phase3-cryptbb-v1, 80/8/6/6; heldout fonts excluded from fitting/selection/calibration",
        "shortcut": shortcut,
    }
    write(args.out / "audit.json", audit)
    write(args.out / "status.json", {"state": "complete", "pid": os.getpid(), "images": len(records)})
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parsed = parser.parse_args()
    try:
        main(parsed)
    except BaseException as error:
        if parsed.out.exists():
            write(parsed.out / "failure.json", {"error": repr(error)})
            write(parsed.out / "status.json", {"state": "failed", "error": repr(error)})
        raise
