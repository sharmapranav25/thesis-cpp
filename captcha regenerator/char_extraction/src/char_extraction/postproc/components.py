"""Connected-component utilities used by classical methods.

Owns labeling, bbox extraction, and area/aspect filtering. Methods compose
these rather than reimplementing them.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from char_extraction.types import CharBox


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components in a binary mask. Returns (labels, n)."""
    binary = (mask > 0).astype(np.uint8)
    labels, n = ndi.label(binary)
    return labels.astype(np.int32), int(n)


def component_bboxes(labels: np.ndarray, n: int) -> List[Tuple[int, int, int, int]]:
    """Get (x1, y1, x2, y2) bbox per labeled component (1..n)."""
    if n <= 0:
        return []
    slices = ndi.find_objects(labels, max_label=n)
    out = []
    for s in slices:
        if s is None:
            continue
        ys, xs = s
        out.append((int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)))
    return out


def filter_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    *,
    min_area: int = 20,
    max_area: int = 100_000,
    min_aspect_ratio: float = 0.05,
    max_aspect_ratio: float = 10.0,
    min_height: int = 5,
    max_height: int = 10_000,
    min_width: int = 1,
    max_width: int = 10_000,
) -> List[Tuple[int, int, int, int]]:
    """Drop bboxes failing geometric constraints. Aspect ratio = w/h."""
    keep = []
    for b in bboxes:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        if w < min_width or w > max_width:
            continue
        if h < min_height or h > max_height:
            continue
        area = w * h
        if area < min_area or area > max_area:
            continue
        ar = w / max(h, 1)
        if ar < min_aspect_ratio or ar > max_aspect_ratio:
            continue
        keep.append(b)
    return keep


def bboxes_to_charboxes(
    bboxes: List[Tuple[int, int, int, int]],
    confidence: float = 1.0,
) -> List[CharBox]:
    """Wrap bboxes into CharBox dataclasses sorted left-to-right.

    Indices are assigned in left-to-right order before returning. The default
    BaseMethod.run will reassign indices after its own sort, so this is a
    convenience for early-stage methods.
    """
    sorted_b = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)
    return [
        CharBox(index=i, bbox=tuple(int(v) for v in b), confidence=confidence)
        for i, b in enumerate(sorted_b)
    ]
