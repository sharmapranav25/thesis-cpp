"""Cleaner pipeline: DINOv2 saliency + Canny edges → soft union → connected
component filter → apply mask to original. Output: chars on white background."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .dinov2 import Dinov2Saliency
from .saturation import saturation_saliency, darkness_saliency


@dataclass
class CleanerConfig:
    # which saliency signals to combine.
    use_saturation: bool = True
    use_darkness: bool = True
    use_dinov2: bool = False

    # ABSOLUTE thresholds (signals NOT per-image normalised). Eval captchas have
    # both saturated chars and saturated bg — per-image min-max stretching killed
    # the signal. Absolute cutoffs work because chars consistently saturate
    # higher than background noise.
    t_sat: float = 0.45               # raw HSV saturation (chars typically > 0.5)
    t_dark: float = 0.55              # 1 - V (dark pixels relative to white bg)
    t_dino: float = 0.60
    t_intersect: float = 0.18

    # canny params
    canny_low: int = 60
    canny_high: int = 160
    canny_dilate: int = 3

    # morphological opening on the mask BEFORE CC filter — kills isolated
    # speckle dots (single-pixel noise) while keeping char strokes (3px+ thick)
    open_kernel: int = 3              # 0 to skip

    # connected-component filter
    min_component_px: int = 30
    max_component_frac: float = 0.40

    # morphological closing AFTER CC filter — fills small holes inside chars
    close_kernel: int = 3             # 0 to skip

    # final mask dilation to fill stroke interiors
    final_dilate: int = 3

    device: str = "cuda"
    batch_size: int = 16


def _canny(arr_rgb: np.ndarray, low: int, high: int, dilate: int) -> np.ndarray:
    """arr_rgb: (H, W, 3) uint8. Returns (H, W) float32 in [0, 1] — dilated canny."""
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        edges = cv2.dilate(edges, k)
    return (edges.astype(np.float32) / 255.0)


def _connected_component_filter(mask: np.ndarray, min_px: int, max_frac: float) -> np.ndarray:
    """mask: (H, W) bool. Drop components smaller than min_px or larger than max_frac of image."""
    H, W = mask.shape
    area = H * W
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):  # 0 = background
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_px:
            continue
        if a > max_frac * area:
            continue
        keep |= (labels == i)
    return keep


class Cleaner:
    """Stateful cleaner. Holds DINOv2 in memory if enabled."""

    def __init__(self, cfg: CleanerConfig):
        self.cfg = cfg
        self.dino = Dinov2Saliency(device=cfg.device) if cfg.use_dinov2 else None

    def clean_batch(self, arrs_uint8: np.ndarray) -> tuple[np.ndarray, dict]:
        """arrs_uint8: (B, H, W, 3) uint8 RGB. Returns (cleaned, debug)."""
        cfg = self.cfg
        B, H, W, _ = arrs_uint8.shape

        sat = saturation_saliency(arrs_uint8, normalize=False) if cfg.use_saturation else None
        dark = darkness_saliency(arrs_uint8) if cfg.use_darkness else None
        dino = self.dino.saliency(arrs_uint8) if self.dino is not None else None

        cleaned = np.empty_like(arrs_uint8)
        debug_out = {}
        for i in range(B):
            arr = arrs_uint8[i]
            c = _canny(arr, cfg.canny_low, cfg.canny_high, cfg.canny_dilate)

            mask = np.zeros((H, W), dtype=bool)
            if sat is not None:
                s = sat[i]
                mask |= (s > cfg.t_sat)
                mask |= (s * c > cfg.t_intersect)
            if dark is not None:
                d = dark[i]
                mask |= (d > cfg.t_dark)
                mask |= (d * c > cfg.t_intersect)
            if dino is not None:
                dn = dino[i]
                mask |= (dn > cfg.t_dino)
                mask |= (dn * c > cfg.t_intersect)

            # Morphological opening to remove isolated speckle dots
            if cfg.open_kernel > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

            mask = _connected_component_filter(mask, cfg.min_component_px, cfg.max_component_frac)

            # Morphological closing to fill small holes inside characters
            if cfg.close_kernel > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

            if cfg.final_dilate > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.final_dilate, cfg.final_dilate))
                mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)

            out = np.where(mask[..., None], arr, np.full_like(arr, 255))
            cleaned[i] = out

            if i == 0:
                debug_out = {
                    "saturation": sat[i] if sat is not None else np.zeros((H, W), np.float32),
                    "darkness": dark[i] if dark is not None else np.zeros((H, W), np.float32),
                    "dinov2": dino[i] if dino is not None else np.zeros((H, W), np.float32),
                    "canny": c,
                    "mask": mask.astype(np.uint8) * 255,
                }
        return cleaned, debug_out

    def clean_pil(self, img: Image.Image) -> tuple[Image.Image, dict]:
        arr = np.array(img.convert("RGB"))[None]
        out, dbg = self.clean_batch(arr)
        return Image.fromarray(out[0]), dbg
