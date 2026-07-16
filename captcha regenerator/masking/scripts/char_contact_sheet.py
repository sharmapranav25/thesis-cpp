"""Per-type character contact sheets.

Each row = one captcha:
  [original | overlay | char_0 | char_1 | ... | char_(MAX-1)]

Empty character slots padded with white. Each cell is row_h tall. Character
cells are square (row_h x row_h) with the crop letterboxed inside.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MAX_CHARS = 8
CELL_BG = (255, 255, 255)


def _letterbox(img: np.ndarray, side: int) -> np.ndarray:
    """Resize while preserving aspect ratio, pad to (side x side) on white."""
    if img is None or img.size == 0:
        return np.full((side, side, 3), CELL_BG, dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((side, side, 3), CELL_BG, dtype=np.uint8)
    scale = (side - 4) / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), CELL_BG, dtype=np.uint8)
    y = (side - new_h) // 2
    x = (side - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def _resize_strip(img: np.ndarray, h: int) -> np.ndarray:
    if img is None:
        return np.full((h, h, 3), CELL_BG, dtype=np.uint8)
    src_h, src_w = img.shape[:2]
    new_w = int(round(src_w * h / src_h))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _build_row(captcha_dir: Path, row_h: int, max_chars: int) -> tuple[np.ndarray, int]:
    orig = cv2.imread(str(captcha_dir / "original.png"))
    overlay = cv2.imread(str(captcha_dir / "overlay.png"))
    if orig is None or overlay is None:
        return None, 0
    orig_r = _resize_strip(orig, row_h)
    overlay_r = _resize_strip(overlay, row_h)

    # Char crops (left-to-right by filename index).
    chars_dir = captcha_dir / "characters"
    char_imgs = []
    if chars_dir.exists():
        crop_paths = sorted([p for p in chars_dir.iterdir()
                             if p.suffix.lower() == ".png" and "_mask" not in p.stem],
                            key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        for p in crop_paths[:max_chars]:
            ci = cv2.imread(str(p))
            if ci is not None:
                char_imgs.append(_letterbox(ci, row_h))
    n_chars = len(char_imgs)
    while len(char_imgs) < max_chars:
        char_imgs.append(np.full((row_h, row_h, 3), CELL_BG, dtype=np.uint8))

    sep = np.full((row_h, 4, 3), 200, dtype=np.uint8)
    pieces = [orig_r, sep, overlay_r, sep] + sum(([c, sep] for c in char_imgs), [])[:-1]
    row = np.hstack(pieces)
    return row, n_chars


def build_sheet(type_dir: Path, n: int, row_h: int, max_chars: int) -> tuple[np.ndarray, dict]:
    captcha_dirs = sorted([d for d in type_dir.iterdir() if d.is_dir()])[:n]
    rows = []
    char_counts = []
    for d in captcha_dirs:
        row, n_chars = _build_row(d, row_h=row_h, max_chars=max_chars)
        if row is None:
            continue
        rows.append(row)
        char_counts.append(n_chars)

    if not rows:
        return None, {}

    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((row_h, max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
    gap = np.full((4, max_w, 3), 255, dtype=np.uint8)
    out = padded[0]
    for r in padded[1:]:
        out = np.vstack([out, gap, r])

    stats = {
        "n_rows": len(rows),
        "char_count_distribution": dict(sorted({c: char_counts.count(c) for c in set(char_counts)}.items())),
        "mean_chars": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--row-h", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args()

    sheets_dir = args.out / "_char_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    for t in ["city", "courier", "cryptbb", "darkbay"]:
        type_dir = args.out / t
        if not type_dir.exists():
            print(f"skip {t}: no output dir")
            continue
        sheet, stats = build_sheet(type_dir, n=args.n, row_h=args.row_h, max_chars=args.max_chars)
        if sheet is None:
            continue
        path = sheets_dir / f"{t}.png"
        cv2.imwrite(str(path), sheet)
        print(f"{t}: wrote {path} ({sheet.shape[1]}x{sheet.shape[0]}) -- {stats}")


if __name__ == "__main__":
    main()
