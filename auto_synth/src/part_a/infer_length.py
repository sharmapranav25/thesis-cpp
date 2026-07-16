"""Infer captcha length from per-image segmentation counts.

Histogram the number of detected characters per image. Modal length is the
captcha's expected length. If multiple modes are within a band of each other,
return all of them with weights — compose can sample.

Usage:
    python -m src.part_a.infer_length --images <dir>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.part_a.segment import crop_characters


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def infer_length_distribution(images_dir: Path, max_images: int = 0):
    paths = list_images(images_dir)
    if max_images:
        paths = paths[:max_images]
    counts = Counter()
    for p in paths:
        try:
            im = Image.open(p)
            crops = crop_characters(im)
            counts[len(crops)] += 1
        except Exception:
            pass
    return counts


def length_distribution_to_modes(counts: Counter, top_share_threshold: float = 0.10,
                                  min_plausible_length: int = 3):
    """Return list of (length, weight) for lengths above share threshold.

    Filters out implausibly low counts (default <3), since segmentation tends
    to *merge* adjacent chars and under-count rather than over-count. Captchas
    with 1-2 char outputs are nearly always segmentation failures, not real.
    """
    total = sum(counts.values())
    if total == 0:
        return []
    # Filter out implausibly low counts
    plausible = {L: n for L, n in counts.items() if L >= min_plausible_length}
    plausible_total = sum(plausible.values())
    if plausible_total == 0:
        # All counts were low — segmentation totally failed; just take top
        L, n = counts.most_common(1)[0]
        return [(L, 1.0)]
    items = [(L, n / plausible_total) for L, n in plausible.items()
             if n / plausible_total >= top_share_threshold]
    if not items:
        L, n = max(plausible.items(), key=lambda kv: kv[1])
        items = [(L, 1.0)]
    s = sum(w for _, w in items)
    items = [(L, w / s) for L, w in items]
    items.sort()
    return items


def main(images_dir: Path, out: Path | None, max_images: int):
    counts = infer_length_distribution(images_dir, max_images=max_images)
    total = sum(counts.values())
    print(f"[infer_length] images sampled: {total}")
    print(f"[infer_length] count histogram:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]} ({100*counts[k]/total:.1f}%)")
    modes = length_distribution_to_modes(counts)
    print(f"[infer_length] kept modes: {modes}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"counts": dict(counts), "modes": modes}, indent=2))
        print(f"[infer_length] saved {out}")
    return modes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=500)
    args = ap.parse_args()
    main(args.images, args.out, args.max_images)
