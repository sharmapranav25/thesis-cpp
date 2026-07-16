"""
Stage 3 — CRAFT detection + intersection validator.

Two-signal agreement:
  Signal A: CRAFT bboxes (regions where a pretrained CNN says "characters live here").
  Signal B: Stage 2.5 mask (pixels we believe are character-coloured / character-shaped).

Validator:
  - Build a polygon-fill mask from CRAFT bboxes.
  - validated_mask = stage25_mask AND craft_bbox_mask.
  - Connected components on the result -> per-character candidates.
  - Drop CCs that are too small (noise) or too large (background bled in / multi-char blob).
  - Sort left-to-right.

Per-character extraction:
  - For each surviving CC, crop the original image to the CC's bbox.
  - Zero out non-CC pixels in the crop (so the character appears on black).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .types import Stage2_5Output

_CRAFT_CACHE: dict[str, Any] = {}


@dataclass
class CharCrop:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in original coords
    cc_index: int
    area: int
    crop: np.ndarray                  # HxWx3, BGR, non-character pixels zeroed
    crop_mask: np.ndarray             # HxW, uint8 {0,255}


@dataclass
class Stage3Output:
    validated_mask: np.ndarray        # uint8 {0,255}, HxW — Stage 2.5 mask AND CRAFT bbox-fill
    craft_polys: list[np.ndarray]     # list of (4,2) int arrays in original image coords
    chars: list[CharCrop]
    n_craft_polys: int
    n_chars_kept: int
    n_chars_dropped: int


def run(
    image_bgr: np.ndarray,
    s25: Stage2_5Output,
    cfg: dict[str, Any],
) -> Stage3Output:
    H, W = image_bgr.shape[:2]
    try:
        polys = _craft_polys(image_bgr, cfg)
    except Exception:
        # CRAFT can produce ragged poly arrays on heavy-noise images,
        # which numpy 1.x rejects internally. Treat as "no CRAFT signal".
        polys = []

    bbox_fill = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        if poly.size == 0:
            continue
        cv2.fillPoly(bbox_fill, [poly.reshape(-1, 1, 2).astype(np.int32)], 255)

    if not polys:
        return Stage3Output(
            validated_mask=np.zeros_like(s25.mask),
            craft_polys=[],
            chars=[],
            n_craft_polys=0,
            n_chars_kept=0,
            n_chars_dropped=0,
        )

    validated = cv2.bitwise_and(s25.mask, bbox_fill)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(validated, connectivity=8)
    min_area = cfg.get("char_min_area_px", 30)
    max_area = int(cfg.get("char_max_area_pct", 0.40) * H * W)

    chars: list[CharCrop] = []
    n_dropped = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            n_dropped += 1
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1, y1, x2, y2 = x, y, x + w, y + h
        cc = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        crop = image_bgr[y1:y2, x1:x2].copy()
        crop[cc == 0] = 0
        chars.append(CharCrop(
            bbox=(x1, y1, x2, y2),
            cc_index=i,
            area=area,
            crop=crop,
            crop_mask=cc,
        ))

    chars.sort(key=lambda c: c.bbox[0])

    return Stage3Output(
        validated_mask=validated,
        craft_polys=polys,
        chars=chars,
        n_craft_polys=len(polys),
        n_chars_kept=len(chars),
        n_chars_dropped=n_dropped,
    )


def _craft_polys(image_bgr: np.ndarray, cfg: dict[str, Any]) -> list[np.ndarray]:
    craft_net, refine_net = _load_craft(cfg)
    from craft_text_detector import get_prediction

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    long_target = cfg.get("upscale_long_side", 720)
    scale = max(1.0, float(long_target) / max(H, W))
    if scale > 1.0:
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        rgb_in = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        rgb_in = rgb

    pred = get_prediction(
        image=rgb_in,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=cfg.get("text_threshold", 0.7),
        link_threshold=cfg.get("link_threshold", 0.4),
        low_text=cfg.get("low_text", 0.4),
        cuda=cfg.get("cuda", True),
        long_size=max(long_target, max(rgb_in.shape[:2])),
    )
    boxes = pred.get("boxes", [])

    polys_orig: list[np.ndarray] = []
    for b in boxes:
        try:
            arr = np.asarray(b, dtype=np.float32).reshape(-1, 2)
        except (ValueError, TypeError):
            # CRAFT occasionally returns malformed/ragged polys on heavy noise.
            continue
        if arr.shape[0] < 3:
            continue
        if scale > 1.0:
            arr = arr / scale
        polys_orig.append(arr.astype(np.int32))
    return polys_orig


def _load_craft(cfg: dict[str, Any]):
    key = "default"
    if key in _CRAFT_CACHE:
        return _CRAFT_CACHE[key]
    from craft_text_detector import load_craftnet_model, load_refinenet_model
    cuda = cfg.get("cuda", True)
    craft_net = load_craftnet_model(cuda=cuda)
    refine_net = load_refinenet_model(cuda=cuda)
    _CRAFT_CACHE[key] = (craft_net, refine_net)
    return _CRAFT_CACHE[key]
