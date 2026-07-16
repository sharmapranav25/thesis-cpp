"""Phase 2/3: composite PHP-rendered text RGBA over sampled extracted bgs.

For each text-only RGBA PNG (from PHP --text-only):
  1. Sample one of the extracted bgs (with replacement).
  2. Apply light augmentation (color/brightness/contrast jitter, small translation).
  3. Alpha-composite text over augmented bg → final RGB PNG.
  4. Write filename,label row to labels.csv (passed through from PHP).

Inputs (defaults match this project's layout):
  --text-dir   PHP --text-only output dir (PNGs + labels.csv)
  --bg-dir     extracted bg dir (PNGs only)
  --out-dir    output dir (created; PNGs + labels.csv written here)

Usage:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/composite/compose.py \
      --text-dir data/php_text_only_test --bg-dir data/cryptbb_bg_lama \
      --out-dir data/synth_compbg_test --limit 50
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

SEED = 0


def jitter_bg(bg_pil: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        bg_pil = ImageEnhance.Color(bg_pil).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.7:
        bg_pil = ImageEnhance.Brightness(bg_pil).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.7:
        bg_pil = ImageEnhance.Contrast(bg_pil).enhance(rng.uniform(0.9, 1.1))
    dx = rng.randint(-10, 10)
    dy = rng.randint(-3, 3)
    if dx or dy:
        arr = np.array(bg_pil)
        arr = np.roll(arr, (dy, dx), axis=(0, 1))
        bg_pil = Image.fromarray(arr)
    return bg_pil


def main(text_dir: Path, bg_dir: Path, out_dir: Path, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (text_dir / "labels.csv").open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    print(f"composing {len(rows)} samples")

    bg_paths = sorted(bg_dir.glob("*.png"))
    bgs = [Image.open(p).convert("RGB") for p in bg_paths]
    print(f"loaded {len(bgs)} bgs into RAM")

    rng = random.Random(SEED)
    out_csv = out_dir / "labels.csv"
    t0 = time.time()
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for i, (fn, label) in enumerate(rows):
            text = Image.open(text_dir / fn).convert("RGBA")
            bg = rng.choice(bgs).copy()
            if bg.size != text.size:
                bg = bg.resize(text.size)
            bg = jitter_bg(bg, rng)
            bg_rgba = bg.convert("RGBA")
            out_im = Image.alpha_composite(bg_rgba, text).convert("RGB")
            arr = np.array(out_im)
            arr[0, :] = 0
            arr[-1, :] = 0
            arr[:, 0] = 0
            arr[:, -1] = 0
            out_im = Image.fromarray(arr)
            out_im.save(out_dir / fn)
            w.writerow([fn, label])
            if (i + 1) % 1000 == 0 or (i + 1) == len(rows):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(rows) - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{len(rows)}  rate={rate:.0f}/s  eta={eta:.0f}s")
    print(f"done; saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", type=Path, required=True)
    ap.add_argument("--bg-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.text_dir, args.bg_dir, args.out_dir, args.limit)
