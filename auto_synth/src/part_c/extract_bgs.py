"""Part C — extract clean backgrounds via LaMa inpainting.

For each real image: re-segment with the same CC pipeline used by Part A, take
the union of character masks (dilated), and inpaint that region with LaMa.
The output is a "background only" version of the captcha that we can
composite freshly-rendered glyphs onto in compose.py.

Reuses the segmentation logic so the mask is per-image rather than fixed-
geometry. Schemes with non-uniform char positions are handled automatically.

Usage:
    python -m src.part_c.extract_bgs \\
        --images "<source dir>" \\
        --out "data/<scheme>/bg_lama" \\
        --max-images 0     # 0 = all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, square

from src.part_a.segment import _binary_candidates, _components, _filter_components


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _detect_underlines(gray: np.ndarray, min_width_frac: float = 0.30,
                       max_height_frac: float = 0.18) -> list[np.ndarray]:
    """Find long thin horizontal-ish components (underlines, strikethroughs).

    These are NOT characters but should be included in the LaMa mask so they
    get inpainted away from the bg. Detected by: width > 30% of canvas AND
    height < 18% of canvas.
    """
    H, W = gray.shape
    out_masks = []
    for binary in _binary_candidates(gray):
        comps = _components(binary)
        for c in comps:
            cw = c["x2"] - c["x1"] + 1
            ch = c["y2"] - c["y1"] + 1
            if cw / W >= min_width_frac and ch / H <= max_height_frac:
                out_masks.append(c["mask"])
    return out_masks


def build_char_mask(img: Image.Image, dilate_px: int = 4,
                    include_underlines: bool = True) -> np.ndarray:
    """Return uint8 [H,W] mask where 255 covers char/underline regions, 0 elsewhere.

    Re-uses Part A's segmentation for chars. Optionally includes "long thin
    horizontal" components (underlines, strikethroughs) so LaMa wipes them too.
    """
    gray = np.array(img.convert("L"))
    H, W = gray.shape
    best = None
    best_score = float("inf")
    for binary in _binary_candidates(gray):
        comps = _components(binary)
        kept = _filter_components(
            comps, gray.shape,
            min_area=60, max_area_frac=0.30,
            min_aspect=0.10, max_aspect=5.0,
            max_height_frac=0.95, max_width_frac=0.40,
            min_height_frac=0.30,
        )
        # Score: prefer reasonable component count
        n = len(kept)
        if 3 <= n <= 8:
            score = 0
        elif n < 3:
            score = (3 - n) ** 2
        else:
            score = (n - 8) ** 2 * 0.4
        if score < best_score:
            best_score = score
            best = kept

    mask = np.zeros((H, W), dtype=bool)
    if best:
        for c in best:
            mask |= c["mask"]
    if include_underlines:
        for um in _detect_underlines(gray):
            mask |= um
    if dilate_px > 0:
        mask = dilation(mask, square(2 * dilate_px + 1))
    return (mask.astype(np.uint8) * 255)


def main(images_dir: Path, out_dir: Path, max_images: int, dilate_px: int):
    from simple_lama_inpainting import SimpleLama
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(list_images(images_dir))
    if max_images:
        paths = paths[:max_images]
    print(f"[part_c] images={len(paths)}, out={out_dir}")

    lama = SimpleLama()
    done = 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            mask = build_char_mask(img, dilate_px=dilate_px)
            mask_pil = Image.fromarray(mask, mode="L")
            bg = lama(img, mask_pil)
            # bg is RGB
            bg_pil = bg if isinstance(bg, Image.Image) else Image.fromarray(bg)
            bg_pil.save(out_dir / p.name)
            done += 1
            if done % 100 == 0:
                print(f"[part_c] {done}/{len(paths)}")
        except Exception as e:
            print(f"[part_c] skip {p.name}: {e}")
    print(f"[part_c] done. {done}/{len(paths)} bgs in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--dilate-px", type=int, default=4)
    args = ap.parse_args()
    main(args.images, args.out, args.max_images, args.dilate_px)
