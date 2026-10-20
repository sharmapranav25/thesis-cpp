"""Generate an audited targeted Phase 3 corpus from a reviewed fit-only scheme."""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "phase1"
PHASE2 = ROOT / "phase2"
sys.path.insert(0, str(ROOT / "phase3/reader"))
from common import CHARSET, group_key, load_image, pixels, read, sha, write  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def partition(text, family):
    value = int(hashlib.sha256((f"capri-phase3-{family}-v1:" + text).encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "selection" if value < 88 else "calibration" if value < 94 else "test"


def draw_text(rng, length, key, family):
    alphabets = [string.digits, string.ascii_lowercase, string.ascii_uppercase]
    active = [alphabet for bit, alphabet in zip(key, alphabets) if bit == "1"]
    if not active or len(active) > length:
        raise ValueError(f"Impossible length/group cell: {length}/{key}")
    for _ in range(10000):
        chars = [str(rng.choice(list(alphabet))) for alphabet in active]
        chars.extend(rng.choice(list("".join(active)), size=length - len(chars)).tolist())
        rng.shuffle(chars)
        text = "".join(chars)
        old = int(hashlib.sha256(("phase12-text-v1:" + text).encode()).hexdigest()[:8], 16) % 20
        if old < 18 and partition(text, family) == "train":
            return text
    raise RuntimeError("Could not sample a text identity inside both training partitions")


def main(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    scheme = read(args.scheme)
    if scheme.get("source_partition") != "fit" or scheme.get("validation_used_for_scheme") is not False:
        raise ValueError("Scheme must come from fit-only inference")
    if not scheme.get("synthetic_reading_gate", {}).get("passed"):
        raise ValueError("Scheme synthetic reading gate did not pass")
    family = scheme["family"]
    profile_path = args.profile or PHASE1 / "profiles/conformer" / f"{family}.json"
    profile = read(profile_path)
    fonts = read(PHASE2 / "fonts.json")
    renderer = module(PHASE2 / "renderer.py", "capri_phase2_renderer")
    cells = scheme["joint_distribution"]
    weights = np.asarray([cell["sampling_weight"] for cell in cells], dtype=float)
    weights /= weights.sum()

    old_hashes = set()
    real_hashes = set()
    source_data = args.scheme.parent / "data"
    if (source_data / "partitions.jsonl").exists():
        for line in (source_data / "partitions.jsonl").open(encoding="utf-8"):
            old_hashes.add(json.loads(line)["pixel_sha256"])
    if (source_data / "real_locators.json").exists():
        for rows in read(source_data / "real_locators.json").values():
            real_hashes.update(row["pixel_sha256"] for row in rows)

    args.out.mkdir(parents=True)
    (args.out / "images").mkdir()
    write(args.out / "inferred_scheme.json", scheme)
    rng = np.random.default_rng(args.seed)
    font_rng = np.random.default_rng([args.seed, 3201])
    deck = []
    seen = set()
    cell_counts = Counter()
    font_counts = Counter()
    modes = Counter()
    minimum_visibility = 1.0

    def next_font():
        if not deck:
            deck.extend(map(int, font_rng.permutation(len(fonts))))
        return deck.pop()

    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as csv_file, (
        args.out / "manifest.jsonl"
    ).open("w", encoding="utf-8") as manifest:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])
        for index in range(args.count):
            cell = cells[int(rng.choice(len(cells), p=weights))]
            length, key = int(cell["length"]), cell["groups"]
            mixed = bool(rng.random() < args.mixed_fraction)
            gentle = bool(rng.random() < 0.2)
            for attempt in range(100):
                text = draw_text(rng, length, key, family)
                entry = int(rng.integers(len(profile["entries"])))
                style = copy.deepcopy(profile["entries"][entry]["style"])
                seed = int(rng.integers(2**31 - 1))
                ids = [next_font() for _ in text] if mixed else [next_font()] * len(text)
                if mixed and len(set(ids)) < 2:
                    ids[-1] = (ids[-1] + 1) % len(fonts)
                paths = [str(PHASE2 / fonts[font_id]["path"]) for font_id in ids]
                controls = []
                for _ in text:
                    controls.append(
                        {
                            "height_gain": float(rng.uniform(0.94, 1.06) if gentle else rng.uniform(0.80, 1.18)),
                            "width_gain": float(rng.uniform(0.95, 1.05) if gentle else rng.uniform(0.75, 1.25)),
                            "angle_delta": float(rng.uniform(-3, 3) if gentle else rng.uniform(-15, 15)),
                            "vertical_delta": float(rng.uniform(-0.015, 0.015) if gentle else rng.uniform(-0.06, 0.06)),
                            "stroke": 0 if gentle else int(rng.random() < 0.2),
                            "effect": "filled" if gentle else str(rng.choice(["filled", "bold", "hollow"], p=[0.75, 0.15, 0.10])),
                        }
                    )
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
                "length": length,
                "groups": key,
                "partition": "train",
                "font_ids": ids,
                "font_families": [fonts[font_id]["family"] for font_id in ids],
                "mixed": mixed,
                "gentle": gentle,
                "seed": seed,
                "profile_entry": entry,
                "pixel_sha256": pixel_hash,
                "recipe": recipe,
            }
            manifest.write(json.dumps(record) + "\n")
            cell_counts[f"{length}:{key}"] += 1
            font_counts.update(record["font_families"])
            modes["mixed" if mixed else "single"] += 1
            minimum_visibility = min(minimum_visibility, *(glyph["visible_fraction"] for glyph in recipe["glyphs"]))
            if (index + 1) % 500 == 0:
                write(args.out / "status.json", {"state": "generating", "generated": index + 1, "target": args.count})

    audit = {
        "passed": len(seen) == args.count and len(font_counts) == len(fonts),
        "images": args.count,
        "unique_pixels": len(seen),
        "zero_overlap_with_reader_synthetic_or_real": True,
        "minimum_glyph_visibility": minimum_visibility,
        "length_group_counts": dict(cell_counts),
        "font_glyph_counts": dict(font_counts),
        "modes": dict(modes),
        "labels_sha256": sha(args.out / "labels.csv"),
        "manifest_sha256": sha(args.out / "manifest.jsonl"),
        "scheme_sha256": sha(args.scheme),
        "real_accuracy_verified": False,
    }
    write(args.out / "audit.json", audit)
    write(
        args.out / "protocol.json",
        {
            "stage": "Official CAPRI Phase 3 targeted generation",
            "family": family,
            "count": args.count,
            "seed": args.seed,
            "profile": str(profile_path),
            "profile_sha256": sha(profile_path),
            "scheme_sha256": sha(args.scheme),
            "checkpoint_hashes": scheme["models"],
            "fonts": len(fonts),
            "mixed_fraction": args.mixed_fraction,
            "real_labels_used": False,
            "refinement": "off",
        },
    )
    write(args.out / "status.json", {"state": "complete", "generated": args.count, "audit_passed": audit["passed"]})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--mixed-fraction", type=float, default=0.5)
    main(parser.parse_args())
