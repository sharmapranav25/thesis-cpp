"""Pixel-based saliency channels for captcha character isolation.

- saturation_saliency: HSV S channel. Bright on saturated colors (synth chars).
- darkness_saliency: 1 - HSV V channel, smoothed. Bright on dark pixels (real
  captcha characters often have lower brightness than pastel backgrounds).

Both per-image normalised to [0, 1] so soft-union thresholds are comparable.
"""
from __future__ import annotations

import cv2
import numpy as np


def _norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def saturation_saliency(arr_rgb_uint8: np.ndarray, smooth: int = 3,
                        normalize: bool = True) -> np.ndarray:
    """If normalize=False, use absolute saturation values (not per-image stretched).
    Useful when the image has both saturated chars AND saturated background —
    per-image normalisation collapses the distinction."""
    single = arr_rgb_uint8.ndim == 3
    if single:
        arr_rgb_uint8 = arr_rgb_uint8[None]
    out = []
    for arr in arr_rgb_uint8:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        s = hsv[..., 1].astype(np.float32) / 255.0
        if smooth > 0:
            s = cv2.GaussianBlur(s, (smooth, smooth), 0)
        if normalize:
            s = _norm01(s)
        out.append(s)
    out = np.stack(out, axis=0)
    return out[0] if single else out


def darkness_saliency(arr_rgb_uint8: np.ndarray, smooth: int = 3,
                      normalize: bool = False) -> np.ndarray:
    """1 - V (brightness) — high on dark pixels.
    Default normalize=False (absolute darkness): a pixel is "dark" if it's
    objectively dark relative to the white captcha canvas, not relative to
    other pixels in the same image."""
    single = arr_rgb_uint8.ndim == 3
    if single:
        arr_rgb_uint8 = arr_rgb_uint8[None]
    out = []
    for arr in arr_rgb_uint8:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2].astype(np.float32) / 255.0
        d = 1.0 - v
        if smooth > 0:
            d = cv2.GaussianBlur(d, (smooth, smooth), 0)
        if normalize:
            d = _norm01(d)
        out.append(d)
    out = np.stack(out, axis=0)
    return out[0] if single else out
