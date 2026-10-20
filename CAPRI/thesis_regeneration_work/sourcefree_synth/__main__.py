from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from importlib import metadata
from pathlib import Path

from PIL import Image

from .data import prepare_split, read_json, sha256_file
from .pipeline import fit, generate


def parse_lengths(value):
    try:
        values = [int(n) for n in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers, e.g. 4,5,6,7") from exc
    if not values or len(set(values)) != len(values) or any(n < 1 or n > 16 for n in values):
        raise argparse.ArgumentTypeError("Lengths must be distinct integers 1..16")
    return values


def doctor(args):
    result = {"python": sys.version, "executable": sys.executable, "packages": {}}
    for name in ("numpy", "Pillow", "fonttools", "torch", "transformers"):
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = "not installed"
    if args.models:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    from .fonts import discover_fonts
    fonts, rejected = discover_fonts(string.ascii_letters + string.digits)
    result["generic_fonts"] = len(fonts)
    result["rejected_fonts"] = rejected
    print(json.dumps(result, indent=2))


def inspect_inputs(args):
    split = prepare_split(args.real_dir, args.exclude_dir, args.seed)
    print(json.dumps({k: v for k, v in split.items() if k not in {"fit", "validation"}}, indent=2))
    print(f"Fit: {len(split['fit'])}; validation: {len(split['validation'])}. No files written.")


def audit(args):
    root = Path(args.run).resolve()
    status = read_json(root / "status.json")
    if status.get("status") != "complete":
        raise ValueError("Generation run is not complete; do not train on a partial run")
    with (root / "labels.csv").open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(labels) != status["requested"] or len(samples) != len(labels):
        raise ValueError("Manifest count mismatch")
    seen_names, seen_hashes = set(), {}
    duplicates = 0
    for label_row, sample in zip(labels, samples):
        name = label_row["filename"]
        path = (root / name).resolve()
        if root not in path.parents or name in seen_names:
            raise ValueError("Unsafe or duplicate filename in manifest")
        seen_names.add(name)
        if sample["filename"] != name or sample["label"] != label_row["label"]:
            raise ValueError("Image/label metadata alignment failure")
        if sha256_file(path) != sample["image_sha256"]:
            raise ValueError(f"Image changed: {name}")
        with Image.open(path) as im:
            im.verify()
        mask = root / "masks" / Path(name).name
        with Image.open(mask) as im:
            im.verify()
        h = sample["image_sha256"]
        if h in seen_hashes:
            duplicates += 1
            if seen_hashes[h] != sample["label"]:
                raise ValueError("Identical image is assigned contradictory labels")
        seen_hashes[h] = sample["label"]
        if len(sample["quality"]) != len(sample["label"]):
            raise ValueError("Character quality records don't match label length")
    if len(list((root / "images").glob("*.png"))) != len(labels):
        raise ValueError("Unlisted image files found")
    print(json.dumps({"status": "pass", "images": len(labels), "exact_duplicate_images": duplicates,
                      "checks": "manifest alignment, file hashes, decoding, masks, per-character record count",
                      "readability_or_real_accuracy_verified": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description="Fit a source-free procedural generator from unlabeled image pixels.")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Check dependencies and generic fonts; no downloads")
    d.add_argument("--models", action="store_true", help="Also inspect CUDA")
    d.set_defaults(func=doctor)
    check = sub.add_parser("inspect", help="Read-only input inventory, exact overlap check and proposed split")
    check.add_argument("--real-dir", required=True)
    check.add_argument("--exclude-dir", action="append", default=[])
    check.add_argument("--seed", type=int, default=42)
    check.set_defaults(func=inspect_inputs)
    f = sub.add_parser("fit", help="Search parameters, freeze a generator, write diagnostic previews")
    f.add_argument("--real-dir", required=True)
    f.add_argument("--exclude-dir", action="append", default=[], help="Evaluation image directory; repeatable, labels never opened")
    f.add_argument("--out", required=True, help="New output directory; existing paths are refused")
    f.add_argument("--mode", choices=["no-vlm", "light-vlm"], default="no-vlm")
    f.add_argument("--features", choices=["pixels", "dinov2"], default="pixels")
    f.add_argument("--font-dir", help="Optional directory of generic TTF/OTF fonts; target fonts prohibited")
    f.add_argument("--charset", default=string.ascii_letters + string.digits,
                   help="Explicit sampling prior; default: ASCII letters + digits, NOT inferred")
    f.add_argument("--lengths", type=parse_lengths, default=[4, 5, 6, 7], help="Explicit sampling prior; default 4,5,6,7")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--validation-fraction", type=float, default=0.2)
    f.add_argument("--max-images", type=int, help="Optional reproducible subset AFTER full overlap/dedup checks")
    f.add_argument("--clusters", type=int, default=3)
    f.add_argument("--trials", type=int, default=24, help="Candidates per style, including baseline")
    f.add_argument("--samples", type=int, default=32, help="Common label/seed probes per candidate")
    f.add_argument("--trace-search", action="store_true", help="Save accepted probe panels for every search candidate")
    f.add_argument("--vlm-images", type=int, default=24, help="Representative fitting images, max 64")
    f.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct", help="Cached model ID or local snapshot directory")
    f.add_argument("--cache-dir", default="E:/hf_cache/hub" if sys.platform == "win32" else None)
    f.add_argument("--allow-download", action="store_true", help="Permit public model downloads; default local cache only")
    f.set_defaults(func=fit)
    g = sub.add_parser("generate", help="CPU-only rendering from a completed frozen config")
    g.add_argument("--config", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--count", type=int, default=64)
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--retries", type=int, default=8)
    g.add_argument("--char-masks", action="store_true", help="Save individual masks in addition to union masks")
    g.add_argument("--trace-count", type=int, default=0, help="Save the rendering stages for the first N accepted samples")
    g.set_defaults(func=generate)
    a = sub.add_parser("audit", help="Read-only integrity check of a generated dataset")
    a.add_argument("--run", required=True)
    a.set_defaults(func=audit)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
