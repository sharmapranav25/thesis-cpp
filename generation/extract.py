"""Extraction (pipeline step 3, light version) — segment glyph crops + infer length
from an unlabeled captcha. Compact connected-component segmenter (mirrors
auto_synth/src/part_a/segment.crop_characters: mask-applied glyph-on-white crops),
enough to (a) query the font atlas and (b) estimate the length distribution.

The heavy SAM/LaMa background extraction (for the real-crop fallback) is a separate
step; this module only needs clean single-glyph crops + a char count.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def _components(gray: np.ndarray, invert: bool):
    """Return (labels, [(x,y,w,h,idx)]) for one Otsu polarity."""
    flag = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, th = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    H, W = gray.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.30 * H or w > 0.45 * W or w < 2 or h < 4 or area < 18:
            continue
        comps.append((int(x), int(y), int(w), int(h), i))
    comps.sort(key=lambda c: c[0])
    return labels, comps


def _score(count: int, expected: tuple[int, int]) -> int:
    lo, hi = expected
    if lo <= count <= hi:
        return 100 - abs(count - (lo + hi) / 2)  # prefer counts inside range, near mid
    return -abs(count - (lo + hi) / 2)


def char_crops(img: Image.Image, expected: tuple[int, int] = (3, 8)) -> list[Image.Image]:
    """Mask-applied glyph-on-white crops, left-to-right. Picks the Otsu polarity whose
    component count best fits `expected`."""
    gray = np.asarray(img.convert("L"))
    best = None
    for invert in (False, True):
        labels, comps = _components(gray, invert)
        if not comps:
            continue
        sc = _score(len(comps), expected)
        if best is None or sc > best[0]:
            best = (sc, labels, comps)
    if best is None:
        return []
    _, labels, comps = best
    crops = []
    for (x, y, w, h, idx) in comps:
        sub = gray[y:y + h, x:x + w].copy()
        mask = labels[y:y + h, x:x + w] == idx
        sub[~mask] = 255
        crops.append(Image.fromarray(sub).convert("RGB"))
    return crops


def count_chars(img: Image.Image, expected: tuple[int, int] = (3, 8)) -> int:
    return len(char_crops(img, expected))


def sample_crops(image_paths: Sequence[str], per_image: int = 8, max_images: int = 60,
                 expected: tuple[int, int] = (3, 8)) -> list[Image.Image]:
    """Collect a bag of glyph crops across many captchas (for font-atlas queries)."""
    out: list[Image.Image] = []
    for p in image_paths[:max_images]:
        try:
            out.extend(char_crops(Image.open(p), expected)[:per_image])
        except Exception:
            continue
    return out


def length_modes(image_paths: Sequence[str], max_images: int = 300,
                  expected: tuple[int, int] = (3, 8), min_plausible: int = 3,
                  top_share: float = 0.10) -> list[tuple[int, float]]:
    """Weighted length modes from per-image char counts (mirrors infer_length)."""
    counts: Counter = Counter()
    for p in image_paths[:max_images]:
        try:
            c = count_chars(Image.open(p), expected)
        except Exception:
            continue
        if c >= min_plausible:
            counts[c] += 1
    if not counts:
        return [(5, 1.0)]
    total = sum(counts.values())
    modes = [(ln, n / total) for ln, n in counts.items() if n / total >= top_share]
    modes.sort(key=lambda t: -t[1])
    z = sum(w for _, w in modes) or 1.0
    return [(ln, w / z) for ln, w in modes]


# --------------------------------------------------------------------------- #
# CLI:  python -m extract --dir <folder> [--montage out.png]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of unlabeled captchas")
    ap.add_argument("--expected", default="3,8")
    ap.add_argument("--montage", default=None)
    args = ap.parse_args()
    exp = tuple(int(x) for x in args.expected.split(","))
    paths = sorted(glob.glob(str(Path(args.dir) / "**" / "*.png"), recursive=True)
                   + glob.glob(str(Path(args.dir) / "**" / "*.jpg"), recursive=True))
    print(f"{len(paths)} images")
    print("length modes:", length_modes(paths, expected=exp))
    if args.montage:
        crops = sample_crops(paths, per_image=6, max_images=12, expected=exp)
        C = 64
        cols = min(16, len(crops)) or 1
        rows = (len(crops) + cols - 1) // cols
        panel = Image.new("RGB", (cols * C, rows * C), "white")
        for i, c in enumerate(crops):
            panel.paste(c.resize((C, C)), ((i % cols) * C, (i // cols) * C))
        panel.save(args.montage)
        print("montage ->", args.montage, f"({len(crops)} crops)")
