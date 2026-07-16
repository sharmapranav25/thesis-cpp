"""Inspect raw CryptBB captchas: dimensions, color mode, sample montage,
and isolate the 3 candidate-labeled files so we can verify filename==text."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def main(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(raw_dir.glob("*.png"))
    print(f"found {len(pngs)} pngs in {raw_dir}")

    dims = collections.Counter()
    modes = collections.Counter()
    sizes_bytes = []
    candidate_labeled = []

    for p in pngs:
        with Image.open(p) as im:
            dims[im.size] += 1
            modes[im.mode] += 1
        sizes_bytes.append(p.stat().st_size)
        if not p.stem.startswith("captcha_screenshot_"):
            candidate_labeled.append(p)

    print("\n--- dimensions (top 5) ---")
    for size, n in dims.most_common(5):
        print(f"  {size}: {n}")
    print("\n--- color modes ---")
    for mode, n in modes.most_common():
        print(f"  {mode}: {n}")
    arr = np.array(sizes_bytes)
    print(f"\n--- file size bytes --- min={arr.min()} mean={arr.mean():.0f} max={arr.max()}")

    print(f"\n--- candidate-labeled files ({len(candidate_labeled)}) ---")
    for p in candidate_labeled:
        with Image.open(p) as im:
            print(f"  {p.name:<20s} size={im.size} mode={im.mode}")

    montage_imgs = [Image.open(p).convert("RGB") for p in pngs[:25]]
    if montage_imgs:
        w, h = montage_imgs[0].size
        grid = Image.new("RGB", (w * 5, h * 5), (255, 255, 255))
        for i, im in enumerate(montage_imgs):
            grid.paste(im.resize((w, h)), ((i % 5) * w, (i // 5) * h))
        montage_path = out_dir / "_montage_25.png"
        grid.save(montage_path)
        print(f"\nsaved montage -> {montage_path}")

    if candidate_labeled:
        cl_imgs = [Image.open(p).convert("RGB") for p in candidate_labeled]
        max_w = max(im.size[0] for im in cl_imgs)
        max_h = max(im.size[1] for im in cl_imgs)
        pad = 30
        big = Image.new("RGB", (max_w, (max_h + pad) * len(cl_imgs)), (255, 255, 255))
        for i, im in enumerate(cl_imgs):
            big.paste(im, (0, i * (max_h + pad)))
        cl_path = out_dir / "_candidate_labeled.png"
        big.save(cl_path)
        print(f"saved candidate-labeled stack -> {cl_path}")
        for p in candidate_labeled:
            (out_dir / f"_cl_{p.stem}.png").write_bytes(p.read_bytes())

    widths = [d[0] for d in dims.elements()]
    heights = [d[1] for d in dims.elements()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(widths, bins=20); axes[0].set_title("width (px)")
    axes[1].hist(heights, bins=20); axes[1].set_title("height (px)")
    fig.tight_layout()
    fig.savefig(out_dir / "_dims_histogram.png", dpi=110)
    plt.close(fig)
    print(f"saved dims histogram -> {out_dir / '_dims_histogram.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/raw", type=Path)
    args = ap.parse_args()
    main(args.raw, args.out)
