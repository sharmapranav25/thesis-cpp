"""Build per-type contact sheets: 50 rows of (original | overlay | mask) stacked horizontally."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_sheet(type_dir: Path, n: int = 50, row_h: int = 80) -> np.ndarray | None:
    rows = []
    captcha_dirs = sorted([d for d in type_dir.iterdir() if d.is_dir()])[:n]
    for d in captcha_dirs:
        orig = cv2.imread(str(d / "original.png"))
        overlay = cv2.imread(str(d / "overlay.png"))
        mask = cv2.imread(str(d / "mask.png"))
        if orig is None or overlay is None or mask is None:
            continue
        scale = row_h / orig.shape[0]
        new_w = int(orig.shape[1] * scale)
        orig_r = cv2.resize(orig, (new_w, row_h), interpolation=cv2.INTER_AREA)
        overlay_r = cv2.resize(overlay, (new_w, row_h), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (new_w, row_h), interpolation=cv2.INTER_NEAREST)
        sep = np.full((row_h, 4, 3), 200, dtype=np.uint8)
        row = np.hstack([orig_r, sep, overlay_r, sep, mask_r])
        rows.append(row)

    if not rows:
        return None

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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--row-h", type=int, default=64)
    args = ap.parse_args()

    sheets_dir = args.out / "_contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    for t in ["city", "courier", "cryptbb", "darkbay"]:
        type_dir = args.out / t
        if not type_dir.exists():
            print(f"skip {t}: no output dir")
            continue
        sheet = build_sheet(type_dir, n=args.n, row_h=args.row_h)
        if sheet is None:
            print(f"skip {t}: no rows")
            continue
        path = sheets_dir / f"{t}.png"
        cv2.imwrite(str(path), sheet)
        print(f"{t}: wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
