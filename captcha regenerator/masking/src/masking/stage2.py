"""
Stage 2 — Character isolation.

Two paths:
  Path A (colour known + colour in HSV table):  HSV inRange + morphology.
  Path B (otherwise):                           DINO ViT-S/8 attention + Canny.

Public entry takes a chosen colour (Stage 2 doesn't decide routing — api.py does).
Stage 2.5 reuses `run_path_b_with_overrides` to retry threshold/CC without
re-running DINO.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .types import Stage2Output


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

def run(image_bgr: np.ndarray, chosen_color: str, cfg: dict[str, Any]) -> Stage2Output:
    hsv_table = cfg["path_a"]["hsv_table"]
    if chosen_color in hsv_table:
        return _run_path_a(image_bgr, chosen_color, cfg["path_a"])
    return _run_path_b(image_bgr, cfg["path_b"])


# --------------------------------------------------------------------------- #
# Path A — HSV inRange
# --------------------------------------------------------------------------- #

_ACHROMATIC = {"black", "white", "gray"}


def _run_path_a(image_bgr: np.ndarray, chosen_color: str, cfg: dict[str, Any]) -> Stage2Output:
    use_otsu = chosen_color in _ACHROMATIC and cfg.get("achromatic_use_otsu", True)
    if use_otsu:
        mask = _otsu_mask(image_bgr, chosen_color)
        sub = "otsu_gray"
    else:
        mask = _hsv_mask(image_bgr, chosen_color, cfg)
        sub = "hsv"

    k_close = _ellipse(cfg["morph_close_ksize"])
    k_open = _ellipse(cfg["morph_open_ksize"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if cfg.get("fill_holes", True):
        mask = _fill_holes(mask)

    cleaned = _apply_mask(image_bgr, mask)
    return Stage2Output(
        mask=mask,
        cleaned=cleaned,
        attention=None,
        path="A",
        chosen_color=chosen_color,
        params_used={"chosen_color": chosen_color, "sub_path": sub,
                     "fill_holes": bool(cfg.get("fill_holes", True))},
    )


def _hsv_mask(image_bgr: np.ndarray, chosen_color: str, cfg: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ranges = cfg["hsv_table"][chosen_color]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array([r["h_lo"], r["s_lo"], r["v_lo"]], dtype=np.uint8)
        hi = np.array([r["h_hi"], r.get("s_hi", 255), r.get("v_hi", 255)], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def _otsu_mask(image_bgr: np.ndarray, chosen_color: str) -> np.ndarray:
    """Auto-thresholded mask for achromatic character colours.

    - black: dark pixels   (THRESH_BINARY_INV + OTSU)
    - white: bright pixels (THRESH_BINARY     + OTSU)
    - gray:  dark-ish pixels constrained to low saturation (chars are achromatic,
             noise is sometimes coloured)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if chosen_color == "white":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    # black or gray: take the dark side of Otsu
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if chosen_color == "gray":
        # Restrict to genuinely achromatic pixels — kills coloured noise that
        # happens to be dark-grey-ish.
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        achromatic = hsv[:, :, 1] < 80
        mask = ((mask > 0) & achromatic).astype(np.uint8) * 255
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary mask. Outlined glyphs (rings) become
    solid silhouettes; open characters (5, 7, U) are unchanged.
    """
    if mask.size == 0:
        return mask
    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    inv = 255 - padded
    cv2.floodFill(inv, None, (0, 0), 0)   # mark outer background as 0
    holes = inv[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


# --------------------------------------------------------------------------- #
# Path B — DINO + Canny
# --------------------------------------------------------------------------- #

def _run_path_b(image_bgr: np.ndarray, cfg: dict[str, Any]) -> Stage2Output:
    attn_full = _dino_attention(image_bgr, cfg)
    edges_d = _canny_dilated(image_bgr, cfg)
    score = cfg["score_attn_weight"] * attn_full + cfg["score_edge_weight"] * edges_d
    score = np.clip(score, 0.0, 1.0)

    mask = _threshold_and_filter(
        score,
        method=cfg["threshold_method"],
        noise_pctile=cfg["noise_pctile"],
        cc_min_area_px=cfg["cc_min_area_px"],
        cc_max_area_pct=cfg["cc_max_area_pct"],
    )
    cleaned = _apply_mask(image_bgr, mask)
    return Stage2Output(
        mask=mask,
        cleaned=cleaned,
        attention=attn_full,
        path="B",
        chosen_color="unknown",
        params_used={
            "noise_pctile": cfg["noise_pctile"],
            "cc_max_area_pct": cfg["cc_max_area_pct"],
        },
    )


def run_path_b_with_overrides(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
    attn_full: np.ndarray,
    *,
    noise_pctile: Optional[float] = None,
    cc_max_area_pct: Optional[float] = None,
) -> Stage2Output:
    edges_d = _canny_dilated(image_bgr, cfg)
    score = cfg["score_attn_weight"] * attn_full + cfg["score_edge_weight"] * edges_d
    score = np.clip(score, 0.0, 1.0)

    mask = _threshold_and_filter(
        score,
        method=cfg["threshold_method"],
        noise_pctile=noise_pctile if noise_pctile is not None else cfg["noise_pctile"],
        cc_min_area_px=cfg["cc_min_area_px"],
        cc_max_area_pct=cc_max_area_pct if cc_max_area_pct is not None else cfg["cc_max_area_pct"],
    )
    cleaned = _apply_mask(image_bgr, mask)
    return Stage2Output(
        mask=mask,
        cleaned=cleaned,
        attention=attn_full,
        path="B",
        chosen_color="unknown",
        params_used={
            "noise_pctile": noise_pctile if noise_pctile is not None else cfg["noise_pctile"],
            "cc_max_area_pct": cc_max_area_pct if cc_max_area_pct is not None else cfg["cc_max_area_pct"],
        },
    )


# --------------------------------------------------------------------------- #
# DINO attention
# --------------------------------------------------------------------------- #

_DINO_CACHE: dict[str, Any] = {}


def _dino_attention(image_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    model = _load_dino(cfg["dino_model"])
    patch = cfg["patch_size"]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    H_p = (H + patch - 1) // patch * patch
    W_p = (W + patch - 1) // patch * patch
    padded = cv2.copyMakeBorder(rgb, 0, H_p - H, 0, W_p - W, cv2.BORDER_REPLICATE)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (padded.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        attentions = model.get_last_selfattention(tensor)
    n_h, n_w = H_p // patch, W_p // patch

    cls_attn = attentions[0, :, 0, 1:]
    cls_attn = cls_attn.reshape(-1, n_h, n_w).cpu().numpy()

    if cfg.get("head_std_keep_top_half", True) and cls_attn.shape[0] >= 2:
        stds = cls_attn.reshape(cls_attn.shape[0], -1).std(axis=1)
        keep = stds.argsort()[-(cls_attn.shape[0] // 2):]
        cls_attn = cls_attn[keep]

    attn = cls_attn.mean(axis=0)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)

    attn_t = torch.from_numpy(attn)[None, None].float()
    attn_up = F.interpolate(attn_t, size=(H_p, W_p), mode="bilinear", align_corners=False)
    attn_full = attn_up.squeeze().numpy()[:H, :W]
    return attn_full.astype(np.float32)


def _load_dino(name: str):
    if name in _DINO_CACHE:
        return _DINO_CACHE[name]
    import torch
    model = torch.hub.load("facebookresearch/dino:main", name, trust_repo=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _DINO_CACHE[name] = model
    return model


# --------------------------------------------------------------------------- #
# Canny + dilation
# --------------------------------------------------------------------------- #

def _canny_dilated(image_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg["canny_lo"], cfg["canny_hi"])
    k = _ellipse(cfg["edge_dilate_ksize"])
    edges = cv2.dilate(edges, k, iterations=1)
    return (edges.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------- #
# Threshold + connected-component filter
# --------------------------------------------------------------------------- #

def _threshold_and_filter(
    score: np.ndarray,
    *,
    method: str,
    noise_pctile: float,
    cc_min_area_px: int,
    cc_max_area_pct: float,
) -> np.ndarray:
    if method == "otsu":
        score_8u = (np.clip(score, 0, 1) * 255).astype(np.uint8)
        _, mask = cv2.threshold(score_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = float(np.percentile(score, noise_pctile))
        mask = (score >= thresh).astype(np.uint8) * 255

    H, W = mask.shape
    max_area = int(cc_max_area_pct * H * W)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cc_min_area_px:
            continue
        if area > max_area:
            continue
        out[labels == i] = 255
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ellipse(k: int) -> np.ndarray:
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cleaned = image_bgr.copy()
    cleaned[mask == 0] = 0
    return cleaned
