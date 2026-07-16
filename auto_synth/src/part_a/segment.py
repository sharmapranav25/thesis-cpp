"""Character segmentation via connected components.

Strategy: convert to grayscale, generate several binary candidates (Otsu both
polarities, with optional morphological opening to break thin underlines /
strikethrough lines), pick the candidate whose component count is closest to
the expected character count, then crop. Crops are returned WITH the mask
applied so the non-character pixels are uniform white — this makes the
downstream DINOv2 embedding focus on character shape rather than background
color/texture.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, square


def _gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def _binary_candidates(gray: np.ndarray) -> list[np.ndarray]:
    """Return multiple foreground=True binary masks for the same image."""
    out = []
    try:
        t = threshold_otsu(gray)
    except Exception:
        return out
    for inv in [False, True]:
        base = (gray < t) if not inv else (gray > t)
        out.append(base.copy())
        # Morph opening to break thin connections (e.g. wavy underline)
        # square(2) breaks 1-2px connections, disk(2) breaks slightly thicker
        try:
            out.append(opening(base, square(2)))
            out.append(opening(base, disk(2)))
        except Exception:
            pass
    return out


def _components(binary: np.ndarray):
    labels, n = ndimage.label(binary)
    out = []
    for i in range(1, n + 1):
        mask = labels == i
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        out.append({
            "y1": int(ys.min()), "y2": int(ys.max()),
            "x1": int(xs.min()), "x2": int(xs.max()),
            "area": int(len(ys)),
            "mask": mask,
        })
    return out


def _filter_components(comps, img_shape, min_area: int, max_area_frac: float,
                       min_aspect: float, max_aspect: float, max_height_frac: float,
                       max_width_frac: float, min_height_frac: float):
    H, W = img_shape
    max_area = int(max_area_frac * H * W)
    min_height = int(min_height_frac * H)
    keep = []
    for c in comps:
        if c["area"] < min_area or c["area"] > max_area:
            continue
        h = c["y2"] - c["y1"] + 1
        w = c["x2"] - c["x1"] + 1
        if h < min_height:           # reject short fragments (e.g. underline)
            continue
        if h > max_height_frac * H or w > max_width_frac * W:
            continue
        ratio = w / max(1, h)
        if ratio < min_aspect or ratio > max_aspect:
            continue
        keep.append(c)
    return keep


def _score_candidate(n: int, target_min: int, target_max: int) -> float:
    """Lower is better. 0 if in range; penalty grows quadratically outside."""
    if target_min <= n <= target_max:
        return 0.0
    if n < target_min:
        return (target_min - n) ** 2
    return (n - target_max) ** 2 * 0.4


def _apply_mask_to_crop(img: Image.Image, mask: np.ndarray, bbox: tuple) -> Image.Image:
    """Return a crop where pixels outside the mask are forced to white."""
    x1, y1, x2, y2 = bbox
    rgb = np.array(img.convert("RGB"))
    sub_rgb = rgb[y1:y2, x1:x2].copy()
    sub_mask = mask[y1:y2, x1:x2]
    sub_rgb[~sub_mask] = 255
    return Image.fromarray(sub_rgb)


def crop_characters(
    img: Image.Image,
    min_area: int = 60,
    max_area_frac: float = 0.30,
    min_aspect: float = 0.10,
    max_aspect: float = 5.0,
    max_height_frac: float = 0.95,
    max_width_frac: float = 0.40,
    min_height_frac: float = 0.30,
    pad: int = 2,
    expected_chars: tuple[int, int] = (3, 8),
) -> list[dict]:
    gray = _gray(img)
    H, W = gray.shape

    best_kept = None
    best_score = float("inf")
    best_binary = None

    for binary in _binary_candidates(gray):
        comps = _components(binary)
        kept = _filter_components(
            comps, gray.shape, min_area, max_area_frac,
            min_aspect, max_aspect, max_height_frac, max_width_frac,
            min_height_frac,
        )
        # Penalize tiny + huge component variance — we want roughly uniform char-sized blobs
        score = _score_candidate(len(kept), *expected_chars)
        if score < best_score:
            best_score = score
            best_kept = kept
            best_binary = binary

    if not best_kept:
        return []

    best_kept.sort(key=lambda c: c["x1"])

    crops = []
    for c in best_kept:
        y1 = max(0, c["y1"] - pad)
        y2 = min(H, c["y2"] + pad + 1)
        x1 = max(0, c["x1"] - pad)
        x2 = min(W, c["x2"] + pad + 1)
        # Apply mask: whiten pixels outside the component
        masked_crop = _apply_mask_to_crop(img, c["mask"], (x1, y1, x2, y2))
        crops.append({
            "bbox": (x1, y1, x2, y2),
            "crop": masked_crop,
            "area": c["area"],
        })
    return crops
