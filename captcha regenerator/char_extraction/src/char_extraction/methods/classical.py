"""Image-level classical-CV utilities used by multiple methods.

Pure-CV building blocks (no torch). Methods compose these alongside the
postproc/components helpers.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB or RGBA uint8 -> grayscale uint8."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def otsu_mask(gray: np.ndarray, dark_on_light: bool = True) -> np.ndarray:
    """Otsu threshold. Returns uint8 mask with 255 = foreground.

    `dark_on_light=True` flags pixels darker than the threshold (the typical
    text-on-light-bg case). Set False when chars are lighter than bg.
    """
    if gray.ndim != 2:
        gray = to_gray(gray)
    thr_type = cv2.THRESH_BINARY_INV if dark_on_light else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, thr_type | cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def open_morphology(mask: np.ndarray, ksize: tuple = (3, 3)) -> np.ndarray:
    """Morphological opening (erosion then dilation) — removes thin noise."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def close_morphology(mask: np.ndarray, ksize: tuple = (3, 3)) -> np.ndarray:
    """Morphological closing (dilation then erosion) — fills small holes."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def detect_polarity(gray: np.ndarray) -> bool:
    """Heuristic: returns True if image is dark-text-on-light-bg.

    Compares median intensity to mean of intensity extremes; works when
    chars cover < ~50% of pixels (typical CAPTCHA case).
    """
    median = float(np.median(gray))
    return median > 127.0


def remove_horizontal_lines(mask: np.ndarray, min_length: int = 15, max_thickness: int = 1) -> np.ndarray:
    """Strip long horizontal connections from a binary mask.

    Morphological opening with a (min_length x max_thickness) horizontal
    kernel keeps only pixels that lie inside a horizontal segment of at
    least `min_length` long and `max_thickness` tall. Subtracting that
    from the mask removes wavy/straight lines that span the image while
    leaving character strokes (which are taller than `max_thickness` or
    shorter than `min_length` horizontally) intact.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, max_thickness))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(mask, lines)


def top_hat_dark_on_light(gray: np.ndarray, ksize: tuple = (15, 15)) -> np.ndarray:
    """Black top-hat: highlights dark features smaller than the kernel.

    closing(gray, k) - gray. Useful for surfacing thin dark strokes when
    the image has uneven illumination or grey background bands. Returns
    uint8 magnitude.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    diff = cv2.subtract(closed, gray)
    return diff
