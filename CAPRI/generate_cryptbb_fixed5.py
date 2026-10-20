"""Generate an exact-quota, five-character CryptBB corpus from reader predictions."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


GROUP_ALPHABETS = {
    "0": string.digits,
    "1": string.ascii_lowercase,
    "2": string.ascii_uppercase,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def partition(text: str) -> str:
    value = int(hashlib.sha256(("capri-phase3-cryptbb-v1:" + text).encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "selection" if value < 88 else "calibration" if value < 94 else "test"


def draw_text(rng: np.random.Generator, length: int, group_key: str) -> str:
    active = [GROUP_ALPHABETS[str(index)] for index, bit in enumerate(group_key) if bit == "1"]
    if not active or len(active) > length:
        raise ValueError(f"Impossible length/group cell: {length}/{group_key}")
    for _ in range(10000):
        chars = [str(rng.choice(list(alphabet))) for alphabet in active]
        chars.extend(rng.choice(list("".join(active)), size=length - len(chars)).tolist())
        rng.shuffle(chars)
        text = "".join(chars)
        old = int(hashlib.sha256(("phase12-text-v1:" + text).encode()).hexdigest()[:8], 16) % 20
        if old < 18 and partition(text) == "train":
            return text
    raise RuntimeError(f"Could not sample a supported text for group {group_key}")


def exact_quotas(predictions: Path, total: int) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError("No reader predictions supplied")
    observed = Counter("".join(map(str, row["decoded_groups"])) for row in rows)
    cells = sorted(observed)
    raw = {cell: observed[cell] * total / len(rows) for cell in cells}
    quotas = {cell: int(raw[cell]) for cell in cells}
    remainder = total - sum(quotas.values())
    for cell in sorted(cells, key=lambda value: (raw[value] - quotas[value], value), reverse=True)[:remainder]:
        quotas[cell] += 1
    return [
        {
            "length": 5,
            "groups": cell,
            "source_count": observed[cell],
            "source_fraction": observed[cell] / len(rows),
            "target_count": quotas[cell],
            "target_fraction": quotas[cell] / total,
        }
        for cell in cells
    ], {"samples": len(rows), "observed_group_combinations": dict(observed)}


def main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    package = args.package.resolve()
    phase1 = package / "phase1"
    phase2 = package / "phase2"
    profile_path = phase1 / "profiles" / "conformer" / "cryptbb.json"
    profile = read_json(profile_path)
    fonts = read_json(phase2 / "fonts.json")
    renderer = module(phase2 / "renderer.py", "cryptbb_phase3_fixed5_renderer")
    common_dir = package / "phase3" / "reader"
    sys.path.insert(0, str(common_dir))
    from common import load_image, pixels  # type: ignore

    cells, source_summary = exact_quotas(args.predictions, args.count)
    source_checkpoint = read_json(args.predictions.parent / "summary.json").get("checkpoint_sha256") if (args.predictions.parent / "summary.json").exists() else None

    old_hashes = set()
    for line in (args.data / "partitions.jsonl").open(encoding="utf-8"):
        old_hashes.add(json.loads(line)["pixel_sha256"])
    real_hashes = {
        row["pixel_sha256"]
        for values in read_json(args.data / "real_locators.json").values()
        for row in values
    }

    args.out.mkdir(parents=True)
    (args.out / "images").mkdir()
    rng = np.random.default_rng(args.seed)
    font_rng = np.random.default_rng([args.seed, 3201])
    deck: list[int] = []
    seen = set()
    cell_counts = Counter()
    font_counts = Counter()
    modes = Counter()
    minimum_visibility = 1.0

    def next_font() -> int:
        if not deck:
            deck.extend(map(int, font_rng.permutation(len(fonts))))
        return deck.pop()

    cell_sequence = [cell["groups"] for cell in cells for _ in range(cell["target_count"])]
    rng.shuffle(cell_sequence)
    protocol = {
        "stage": "CryptBB targeted Phase 3 fixed-length generation",
        "count": args.count,
        "length": 5,
        "seed": args.seed,
        "source_predictions": str(args.predictions.resolve()),
        "source_checkpoint_sha256": source_checkpoint,
        "source_summary": source_summary,
        "exact_group_quotas": cells,
        "package": str(package),
        "profile_sha256": sha(profile_path),
        "fonts": len(fonts),
        "mixed_fraction": args.mixed_fraction,
        "refinement": "off",
        "real_labels_used": False,
        "source_pixel_overlap_rejected": True,
    }
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "source_predictions_summary.json", source_summary)

    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as csv_file, (
        args.out / "manifest.jsonl"
    ).open("w", encoding="utf-8") as manifest:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])
        for index, group_key in enumerate(cell_sequence):
            for attempt in range(100):
                text = draw_text(rng, 5, group_key)
                entry = int(rng.integers(len(profile["entries"])))
                style = copy.deepcopy(profile["entries"][entry]["style"])
                seed = int(rng.integers(2**31 - 1))
                mixed = bool(rng.random() < args.mixed_fraction)
                ids = [next_font() for _ in text] if mixed else [next_font()] * len(text)
                if mixed and len(set(ids)) < 2:
                    ids[-1] = (ids[-1] + 1) % len(fonts)
                paths = [str(phase2 / fonts[font_id]["path"]) for font_id in ids]
                controls = [
                    {
                        "height_gain": float(rng.uniform(0.94, 1.06) if rng.random() < 0.2 else rng.uniform(0.80, 1.18)),
                        "width_gain": float(rng.uniform(0.95, 1.05) if rng.random() < 0.2 else rng.uniform(0.75, 1.25)),
                        "angle_delta": float(rng.uniform(-3, 3) if rng.random() < 0.2 else rng.uniform(-15, 15)),
                        "vertical_delta": float(rng.uniform(-0.015, 0.015) if rng.random() < 0.2 else rng.uniform(-0.06, 0.06)),
                        "stroke": 0 if rng.random() < 0.2 else int(rng.random() < 0.2),
                        "effect": "filled" if rng.random() < 0.2 else str(rng.choice(["filled", "bold", "hollow"], p=[0.75, 0.15, 0.10])),
                    }
                    for _ in text
                ]
                style["_typography"] = controls
                style["spacing_gain"] = float(style.get("spacing_gain", 1) * rng.uniform(0.9, 1.1))
                try:
                    image, _, recipe = renderer.render(style, text, paths, seed, profile["variant"])
                    pixel_hash = pixels(image)
                    if pixel_hash in seen or pixel_hash in old_hashes or pixel_hash in real_hashes:
                        raise ValueError("duplicate or source-overlapping pixels")
                    if "".join(glyph["character"] for glyph in recipe["glyphs"]) != text:
                        raise ValueError("glyph/label mismatch")
                    if min(glyph["visible_fraction"] for glyph in recipe["glyphs"]) < 0.55:
                        raise ValueError("glyph visibility")
                    break
                except ValueError:
                    if attempt == 99:
                        raise
            seen.add(pixel_hash)
            filename = f"images/{index:07d}.png"
            image.save(args.out / filename)
            writer.writerow([filename, text])
            record = {
                "filename": filename,
                "label": text,
                "length": 5,
                "groups": group_key,
                "partition": "train",
                "font_ids": ids,
                "font_families": [fonts[font_id]["family"] for font_id in ids],
                "mixed": mixed,
                "seed": seed,
                "profile_entry": entry,
                "pixel_sha256": pixel_hash,
                "recipe": recipe,
            }
            manifest.write(json.dumps(record) + "\n")
            cell_counts[group_key] += 1
            font_counts.update(record["font_families"])
            modes["mixed" if mixed else "single"] += 1
            minimum_visibility = min(minimum_visibility, *(glyph["visible_fraction"] for glyph in recipe["glyphs"]))
            if (index + 1) % 500 == 0:
                write_json(args.out / "status.json", {"state": "generating", "generated": index + 1, "target": args.count})

    total = sum(cell_counts.values())
    audit = {
        "passed": total == args.count and len(seen) == args.count,
        "images": total,
        "unique_pixels": len(seen),
        "fixed_length": 5,
        "exact_group_counts": dict(cell_counts),
        "exact_group_prevalence": {key: count / total for key, count in sorted(cell_counts.items())},
        "font_glyph_counts": dict(font_counts),
        "modes": dict(modes),
        "minimum_glyph_visibility": minimum_visibility,
        "zero_overlap_with_phase3_or_real": True,
        "labels_sha256": sha(args.out / "labels.csv"),
        "manifest_sha256": sha(args.out / "manifest.jsonl"),
        "real_accuracy_verified": False,
    }
    write_json(args.out / "audit.json", audit)
    write_json(args.out / "status.json", {"state": "complete", "generated": args.count, "audit_passed": audit["passed"]})
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=732)
    parser.add_argument("--mixed-fraction", type=float, default=0.5)
    main(parser.parse_args())
