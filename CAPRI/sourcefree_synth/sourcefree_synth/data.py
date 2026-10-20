"""Image-only ingestion. Never reads CSV labels or decodes text from filenames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original)
        # Make transparent inputs unambiguous; never use an embedded text label.
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(base, rgba).convert("RGB")


def pixel_hash(image):
    h = hashlib.sha256(str(image.size).encode("ascii"))
    h.update(image.convert("RGB").tobytes())
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    records, errors = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image = load_image(path)
            w, h = image.size
            if not (16 <= w <= 2048 and 12 <= h <= 1024):
                raise ValueError(f"Unsupported canvas {w}x{h}")
            records.append({"path": str(path), "pixel_sha256": pixel_hash(image),
                            "file_sha256": sha256_file(path), "width": w, "height": h})
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if errors:
        raise ValueError("Unreadable/unsupported inputs; none silently skipped: " + json.dumps(errors[:10]))
    if not records:
        raise ValueError(f"No supported images in {root}")
    return records


def prepare_split(root, exclude_dirs=(), seed=42, validation_fraction=0.2, max_images=None):
    if not 0.1 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.1 and 0.4")
    records = inventory(root)
    excluded = set()
    for directory in exclude_dirs:
        excluded.update(r["pixel_sha256"] for r in inventory(directory))
    overlaps = [r["pixel_sha256"] for r in records if r["pixel_sha256"] in excluded]
    if overlaps:
        raise ValueError(f"Evaluation overlap: {len(overlaps)} input images have identical decoded pixels. "
                         "Use a disjoint input pool. No run was created.")
    unique = {}
    for record in records:
        unique.setdefault(record["pixel_sha256"], record)
    # Selection and splitting are invariant to filenames and directory order.
    rows = sorted(unique.values(), key=lambda r: r["pixel_sha256"])
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if max_images is not None:
        if max_images < 20:
            raise ValueError("max_images must be at least 20")
        rows = rows[:max_images]
    if len(rows) < 20:
        raise ValueError("At least 20 unique real images are required for a diagnostic fit; 500+ recommended.")
    n_val = max(4, round(len(rows) * validation_fraction))
    return {"fit": rows[n_val:], "validation": rows[:n_val], "seed": seed,
            "input_count": len(records), "unique_count": len(unique),
            "duplicates_removed": len(records) - len(unique), "used_count": len(rows),
            "excluded_directories": [str(Path(p).resolve()) for p in exclude_dirs],
            "exclusion_check": "decoded-pixel equality; transformed/near duplicates are not ruled out",
            "labels_read": False}


def new_output(path, protected=()):
    path = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        if path == root or root in path.parents or path in root.parents:
            raise ValueError(f"Output must be separate from input/excluded directories: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, obj):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
