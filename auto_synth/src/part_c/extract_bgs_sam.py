"""Part C v2 — extract clean backgrounds via MobileSAM-refined masks + LaMa.

Pipeline per image:
  1. CC segmentation -> rough character bboxes (existing Part A logic)
  2. MobileSAM with each bbox as a box-prompt -> clean per-character mask
  3. Union all masks -> dilated full-mask
  4. LaMa inpaints over the union mask -> clean background

The big improvement over the v1 Part C: SAM understands character shape
semantically, so anti-aliased edges, color halos, and curved underlines get
captured cleanly. CC alone fails on those.

Usage:
    python -m src.part_c.extract_bgs_sam \\
        --images "<source dir>" \\
        --out "data/<scheme>/bg_lama_sam"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.morphology import dilation, square

from src.part_a.segment import _binary_candidates, _components, _filter_components


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SAM_CKPT = "E:/pranav/auto_synth/configs/mobile_sam.pt"


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def cc_bboxes(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Same heuristic as Part A but only return bboxes, not crops or masks."""
    gray = np.array(img.convert("L"))
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
    if not best:
        return []
    return [(c["x1"], c["y1"], c["x2"] + 1, c["y2"] + 1) for c in best]


def main(images_dir: Path, out_dir: Path, max_images: int, dilate_px: int):
    from mobile_sam import sam_model_registry, SamPredictor
    from simple_lama_inpainting import SimpleLama

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[part_c_sam] loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=SAM_CKPT).cuda().eval()
    predictor = SamPredictor(sam)
    print(f"[part_c_sam] loading LaMa...")
    lama = SimpleLama()

    paths = sorted(list_images(images_dir))
    if max_images:
        paths = paths[:max_images]
    print(f"[part_c_sam] processing {len(paths)} images")

    done = 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
            bboxes = cc_bboxes(img)

            mask = np.zeros((img.height, img.width), dtype=bool)

            if bboxes:
                predictor.set_image(arr)
                # Box prompts: SAM accepts a stack of boxes [N, 4] in (x1, y1, x2, y2)
                boxes = torch.tensor(bboxes, device="cuda", dtype=torch.float32)
                # Transform boxes to SAM's input space
                transformed = predictor.transform.apply_boxes_torch(boxes, arr.shape[:2])
                masks_pred, _, _ = predictor.predict_torch(
                    point_coords=None, point_labels=None,
                    boxes=transformed, multimask_output=False,
                )
                # masks_pred: [N, 1, H, W] bool
                masks_np = masks_pred.squeeze(1).cpu().numpy()
                for m in masks_np:
                    mask |= m.astype(bool)

            # Add long-thin underlines too (Courier-style)
            from src.part_c.extract_bgs import _detect_underlines
            for um in _detect_underlines(np.array(img.convert("L"))):
                mask |= um

            if dilate_px > 0:
                mask = dilation(mask, square(2 * dilate_px + 1))
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_pil = Image.fromarray(mask_u8, mode="L")

            bg = lama(img, mask_pil)
            bg_pil = bg if isinstance(bg, Image.Image) else Image.fromarray(bg)
            bg_pil.save(out_dir / p.name)
            done += 1
            if done % 50 == 0:
                print(f"[part_c_sam] {done}/{len(paths)}")
        except Exception as e:
            print(f"[part_c_sam] skip {p.name}: {e}")

    print(f"[part_c_sam] done. {done}/{len(paths)} bgs in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--dilate-px", type=int, default=4)
    args = ap.parse_args()
    main(args.images, args.out, args.max_images, args.dilate_px)
