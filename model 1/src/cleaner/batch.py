"""Batch-process a directory of images through the cleaner. Writes cleaned
PNGs to an output directory and copies labels.csv unchanged if present.
Modeled on src/cyclegan/translate.py."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .pipeline import Cleaner, CleanerConfig


def main(src: Path, out: Path, batch: int) -> None:
    cfg = CleanerConfig(batch_size=batch)
    cleaner = Cleaner(cfg)

    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in src.glob("*.png") if not p.name.startswith("_"))
    if not paths:
        # fall back to .PNG (some real-data filenames are uppercase)
        paths = sorted(p for p in src.glob("*.PNG") if not p.name.startswith("_"))
    print(f"cleaning {len(paths)} images from {src} -> {out}")

    if (src / "labels.csv").exists():
        shutil.copy2(src / "labels.csv", out / "labels.csv")
        print("copied labels.csv")

    # Real captchas come in 200x60 OR ~400x120; resize all to 200x60 first so
    # cleaned outputs are consistent with synth (the training distribution).
    target = (200, 60)  # PIL (W, H)

    pbar = tqdm(range(0, len(paths), batch), desc="clean")
    for start in pbar:
        chunk = paths[start:start + batch]
        arrs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            if im.size != target:
                im = im.resize(target, Image.BICUBIC)
            arrs.append(np.array(im))
        arrs = np.stack(arrs, axis=0)
        cleaned, _ = cleaner.clean_batch(arrs)
        for p, out_arr in zip(chunk, cleaned):
            Image.fromarray(out_arr).save(out / p.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    main(args.src, args.out, args.batch)
