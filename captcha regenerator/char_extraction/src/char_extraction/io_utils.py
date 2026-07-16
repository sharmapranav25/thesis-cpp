"""Image I/O + writing the spec's per-image output directory layout.

Everything that touches the filesystem on the output side lives here so
the spec's directory format is owned by one module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from char_extraction.types import CharBox, ExtractionResult


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image as RGB uint8 (H, W, 3)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 could not decode {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Write an image. Accepts RGB or RGBA uint8."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 2:
        cv2.imwrite(str(p), img)
        return
    if img.shape[2] == 4:
        bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(p), bgra)
        return
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)


def crop_char(img: np.ndarray, bbox: tuple, pad: int = 0) -> np.ndarray:
    """Crop a CharBox from the image with optional padding (clamped to bounds)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2].copy()


def make_overlay(img: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """Draw a mask on the image as a colored overlay."""
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        base = img.copy()
    overlay = base.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    return blended


def derive_captcha_id(image_input: Union[str, Path, np.ndarray], explicit: Optional[str] = None) -> str:
    """Derive a stable ID from a path stem; require explicit ID for ndarray inputs."""
    if explicit:
        return explicit
    if isinstance(image_input, (str, Path)):
        return Path(image_input).stem
    raise ValueError("captcha_id must be provided when extracting from an ndarray (no path to derive from)")


def write_extraction(
    result: ExtractionResult,
    image: np.ndarray,
    out_dir: Union[str, Path],
    crop_pad: int = 2,
) -> Path:
    """Write the spec's per-image output layout under out_dir/<captcha_id>/.

    Returns the path of the per-image directory.
    """
    base = Path(out_dir) / result.captcha_id
    base.mkdir(parents=True, exist_ok=True)
    chars_dir = base / "characters"
    chars_dir.mkdir(exist_ok=True)

    save_image(image, base / "original.png")
    save_image(result.mask, base / "mask.png")
    save_image(make_overlay(image, result.mask), base / "overlay.png")

    new_chars = []
    for c in result.characters:
        crop = crop_char(image, c.bbox, pad=crop_pad)
        rel_path = f"characters/{c.index}.png"
        save_image(crop, base / rel_path)
        new_chars.append(
            CharBox(
                index=c.index,
                bbox=c.bbox,
                confidence=c.confidence,
                crop_path=rel_path,
            )
        )
    result.characters = new_chars

    (base / "metadata.json").write_text(json.dumps(result.to_metadata(), indent=2))
    return base
