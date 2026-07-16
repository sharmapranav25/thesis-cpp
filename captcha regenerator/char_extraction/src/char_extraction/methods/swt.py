"""Universal text-localization via Stroke-Width Transform.

Epshtein, Ofek, Wexler — "Detecting Text in Natural Scenes with Stroke
Width Transform" (CVPR 2010). The classical SOTA before deep learning
took over scene text detection.

Key insight: characters have **consistent per-glyph stroke widths**. Each
pixel is labeled with the width of the stroke it lies on. Pixels with
plausible-text stroke widths form connected components, which we filter
by per-CC stroke-width consistency + geometry.

Why this is a strong fit for our CAPTCHAs:
  - DarkBay: chars have ~3-5 px strokes; mosaic noise has 0 or random width.
  - City: digits have ~3-5 px strokes; wavy lines have 1-2 px strokes — separable.
  - Courier: digits have thick strokes (~4-6 px); scribbles are thin (~1-2 px).
  - CryptBB: chars have outlined/3D ~3-7 px strokes; mosaic tiles have 1-2 px.

Two passes per image: dark-on-light and light-on-dark polarity, masks
unioned. Per-CC stroke-width variance filtering ensures we keep only
text-like components (low variance) and reject random texture (high
variance).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from char_extraction.methods.base import BaseMethod, register_method
from char_extraction.methods.classical import remove_horizontal_lines, to_gray
from char_extraction.postproc.components import (
    component_bboxes,
    filter_bboxes,
    label_components,
)
from char_extraction.types import CharBox


def _compute_swt(
    gray: np.ndarray,
    dark_on_light: bool = True,
    max_stroke_width: int = 20,
    canny_low: int = 50,
    canny_high: int = 150,
    direction_tolerance: float = 0.5,  # cos(60°) ≈ 0.5; pairs within 60° of opposite direction
) -> np.ndarray:
    """Single-polarity Stroke-Width Transform.

    Returns float64 (H, W) where each pixel's value is the estimated stroke
    width of the stroke containing that pixel. Pixels not on any valid
    stroke have value `np.inf`.

    For dark-on-light: rays follow the gradient (gradient points from dark
    to light, so going *with* the gradient leads from text-edge into bg;
    going *against* leads into the text interior, then back out the other
    side).

    For light-on-dark: rays follow the negative gradient direction.
    """
    edges = cv2.Canny(gray, canny_low, canny_high)
    if edges.sum() == 0:
        return np.full(gray.shape, np.inf, dtype=np.float64)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag_safe = np.where(grad_mag == 0, 1.0, grad_mag)
    gx_n = gx / grad_mag_safe
    gy_n = gy / grad_mag_safe

    # Sign of ray direction. For dark-on-light text, gradient at the BOUNDARY
    # points outward (from text/dark to bg/light). To traverse text interior
    # toward the opposite boundary, follow the *negative* gradient.
    sign = -1.0 if dark_on_light else 1.0

    H, W = gray.shape
    swt = np.full((H, W), np.inf, dtype=np.float64)

    edge_ys, edge_xs = np.where(edges > 0)
    rays: list = []

    for ey, ex in zip(edge_ys.tolist(), edge_xs.tolist()):
        dx = sign * gx_n[ey, ex]
        dy = sign * gy_n[ey, ex]
        if dx == 0.0 and dy == 0.0:
            continue
        ray_pixels = [(ey, ex)]
        prev_y, prev_x = ey, ex
        for step in range(1, max_stroke_width + 1):
            fy = ey + step * dy
            fx = ex + step * dx
            ny, nx = int(round(fy)), int(round(fx))
            if ny == prev_y and nx == prev_x:
                continue
            if not (0 <= ny < H and 0 <= nx < W):
                break
            ray_pixels.append((ny, nx))
            prev_y, prev_x = ny, nx
            if edges[ny, nx] > 0:
                # Opposite-direction check: gradient at the far edge should
                # point ~opposite to the start (i.e. across the stroke).
                dot = dx * (-sign * gx_n[ny, nx]) + dy * (-sign * gy_n[ny, nx])
                if dot >= direction_tolerance:
                    width = float(np.hypot(ey - ny, ex - nx))
                    for py, px in ray_pixels:
                        if width < swt[py, px]:
                            swt[py, px] = width
                    rays.append((tuple(ray_pixels), width))
                break

    # Second pass: replace each ray's per-pixel SWT with the median along the
    # ray. Handles corners where one ray's width is wrong but its neighbours
    # converge on the correct value.
    for ray_pixels, _ in rays:
        widths = [swt[py, px] for py, px in ray_pixels if not np.isinf(swt[py, px])]
        if not widths:
            continue
        median_w = float(np.median(widths))
        for py, px in ray_pixels:
            if swt[py, px] > median_w:
                swt[py, px] = median_w

    return swt


def _filter_components_by_stroke_consistency(
    swt: np.ndarray,
    valid_mask: np.ndarray,
    max_sw_cv: float = 0.6,
    min_pixels: int = 30,
) -> np.ndarray:
    """Keep CCs whose per-pixel stroke-width has low coefficient-of-variation.

    Real character strokes have nearly-constant width within a glyph;
    random texture has high variance. CV = std / mean. Threshold ~0.5–0.7
    is typical per the SWT paper.
    """
    labels, n = label_components(valid_mask.astype(np.uint8) * 255)
    if n == 0:
        return np.zeros_like(valid_mask, dtype=np.uint8)
    out = np.zeros_like(valid_mask, dtype=np.uint8)
    for label_id in range(1, n + 1):
        comp = labels == label_id
        if comp.sum() < min_pixels:
            continue
        sws = swt[comp]
        sws = sws[np.isfinite(sws)]
        if sws.size < min_pixels // 2:
            continue
        mean_sw = float(np.mean(sws))
        if mean_sw <= 0:
            continue
        std_sw = float(np.std(sws))
        cv = std_sw / mean_sw
        if cv > max_sw_cv:
            continue
        out[comp] = 255
    return out


@register_method
class UniversalSWT(BaseMethod):
    """Stroke-Width Transform based universal text localizer.

    Two-pass SWT (dark-on-light + light-on-dark) → keep pixels whose stroke
    width is in [min_stroke_width, max_stroke_width] → CC + per-CC variance
    filter → geometry filter.
    """

    name = "universal_swt"

    def segment(self, img: np.ndarray) -> np.ndarray:
        gray = to_gray(img)

        max_sw = int(self.cfg.get("max_stroke_width", 10))
        min_sw = int(self.cfg.get("min_stroke_width", 3))
        canny_low = int(self.cfg.get("canny_low", 50))
        canny_high = int(self.cfg.get("canny_high", 150))
        max_cv = float(self.cfg.get("swt_max_cv", 0.9))
        min_cc_pixels = int(self.cfg.get("swt_min_cc_pixels", 20))

        # Two polarity passes; keep whichever produces a stroke at each pixel.
        swt_dol = _compute_swt(gray, dark_on_light=True,
                               max_stroke_width=max_sw,
                               canny_low=canny_low, canny_high=canny_high)
        swt_lod = _compute_swt(gray, dark_on_light=False,
                               max_stroke_width=max_sw,
                               canny_low=canny_low, canny_high=canny_high)

        # Combine: take the smaller (tighter) stroke estimate at each pixel.
        swt_combined = np.minimum(swt_dol, swt_lod)

        in_range = (swt_combined >= min_sw) & (swt_combined <= max_sw)

        # Per-CC consistency filter on each polarity, then union.
        mask_dol_clean = _filter_components_by_stroke_consistency(
            swt_dol,
            (swt_dol >= min_sw) & (swt_dol <= max_sw),
            max_sw_cv=max_cv,
            min_pixels=min_cc_pixels,
        )
        mask_lod_clean = _filter_components_by_stroke_consistency(
            swt_lod,
            (swt_lod >= min_sw) & (swt_lod <= max_sw),
            max_sw_cv=max_cv,
            min_pixels=min_cc_pixels,
        )
        mask = cv2.bitwise_or(mask_dol_clean, mask_lod_clean)

        # Strip long horizontal connections (underlines, wavy lines, side bars
        # if those happen to be horizontal). Cuts CC bridges between adjacent
        # characters so each character lands in its own connected component.
        line_kw = int(self.cfg.get("post_horiz_line_min_length", 14))
        line_kh = int(self.cfg.get("post_horiz_line_max_thickness", 2))
        if line_kw > 1:
            mask = remove_horizontal_lines(mask, min_length=line_kw, max_thickness=line_kh)

        # Strip long *vertical* connections too — DarkBay has structural bars
        # at L/R edges that look like very tall thin "characters" otherwise.
        vbar_kh = int(self.cfg.get("post_vert_bar_min_length", 25))
        vbar_kw = int(self.cfg.get("post_vert_bar_max_thickness", 3))
        if vbar_kh > 1:
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (vbar_kw, vbar_kh))
            bars = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v)
            mask = cv2.subtract(mask, bars)

        # Small closing to fill 1-2 px intra-glyph gaps left over.
        close_k = self.cfg.get("post_close_ksize", [3, 3])
        if close_k and close_k[0] > 1 and close_k[1] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(close_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def split(self, mask: np.ndarray, img: np.ndarray) -> List[CharBox]:
        labels, n = label_components(mask)
        bboxes = component_bboxes(labels, n)
        bboxes = filter_bboxes(
            bboxes,
            min_area=int(self.cfg.get("min_area", 50)),
            max_area=int(self.cfg.get("max_area", 100_000)),
            min_aspect_ratio=float(self.cfg.get("min_aspect_ratio", 0.05)),
            max_aspect_ratio=float(self.cfg.get("max_aspect_ratio", 10.0)),
            min_height=int(self.cfg.get("min_height", 5)),
            max_height=int(self.cfg.get("max_height", 10_000)),
            min_width=int(self.cfg.get("min_width", 1)),
            max_width=int(self.cfg.get("max_width", 10_000)),
        )
        sorted_b = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)
        return [
            CharBox(index=i, bbox=tuple(int(v) for v in b), confidence=0.85)
            for i, b in enumerate(sorted_b)
        ]
