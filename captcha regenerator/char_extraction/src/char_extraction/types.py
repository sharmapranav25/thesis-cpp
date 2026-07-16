"""Core dataclasses returned by the extraction pipeline.

Single source of truth for the on-disk metadata.json schema and the in-memory
result shape returned by `char_extraction.api.extract`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CharBox:
    """One detected character.

    bbox is (x1, y1, x2, y2) with x2 > x1 and y2 > y1, inclusive of x1/y1
    and exclusive of x2/y2 (numpy-slice convention).
    """

    index: int
    bbox: tuple
    confidence: float = 1.0
    crop_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "crop_path": self.crop_path,
        }

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0


@dataclass
class ExtractionResult:
    """Output of one `extract()` call.

    `mask` is a uint8 array (H, W) with 0/255 values; 255 = character pixel.
    `characters` is left-to-right ordered.
    """

    captcha_id: str
    captcha_type: Optional[str]
    image_size: tuple  # (W, H)
    mask: np.ndarray
    characters: list = field(default_factory=list)
    method: str = ""
    notes: str = ""

    def to_metadata(self) -> dict:
        return {
            "captcha_id": self.captcha_id,
            "captcha_type": self.captcha_type,
            "image_size": list(self.image_size),
            "characters": [c.to_dict() for c in self.characters],
            "method": self.method,
            "notes": self.notes,
        }
