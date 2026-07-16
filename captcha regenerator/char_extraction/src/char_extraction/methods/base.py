"""BaseMethod contract and registry.

Every extraction method implements `segment(img) -> mask` and `split(mask, img)
-> list[CharBox]`. The default `run` template glues them into an
ExtractionResult and orders chars left-to-right. Methods register via
@register_method and are looked up by `name` from the router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

from char_extraction.types import CharBox, ExtractionResult

_REGISTRY: Dict[str, Type["BaseMethod"]] = {}


def register_method(cls):
    """Class decorator that registers a method by its `name` class attribute."""
    name = getattr(cls, "name", "") or ""
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name` class attribute")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Method name collision: {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_methods() -> List[str]:
    return sorted(_REGISTRY)


class BaseMethod(ABC):
    """Abstract method contract.

    Subclasses set a unique `name`, decorate with @register_method, and
    implement `segment` + `split`. The default `run` template handles
    ordering and metadata so most subclasses won't override it.
    """

    name: str = ""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg: Dict[str, Any] = cfg or {}

    @abstractmethod
    def segment(self, img: np.ndarray) -> np.ndarray:
        """Return a (H, W) uint8 mask, 255 where character pixels are."""

    @abstractmethod
    def split(self, mask: np.ndarray, img: np.ndarray) -> List[CharBox]:
        """Turn a mask into ordered CharBox list."""

    def run(
        self,
        img: np.ndarray,
        captcha_id: str,
        captcha_type: Optional[str],
    ) -> ExtractionResult:
        h, w = img.shape[:2]
        mask = self.segment(img)
        if mask.shape[:2] != (h, w):
            raise RuntimeError(
                f"Method {self.name} returned mask of shape {mask.shape} "
                f"for image of shape {(h, w)}"
            )
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        chars = self.split(mask, img)
        chars = sorted(chars, key=lambda c: c.x_center)
        chars = [
            CharBox(index=i, bbox=tuple(int(v) for v in c.bbox), confidence=c.confidence, crop_path=c.crop_path)
            for i, c in enumerate(chars)
        ]
        return ExtractionResult(
            captcha_id=captcha_id,
            captcha_type=captcha_type,
            image_size=(w, h),
            mask=mask,
            characters=chars,
            method=self.name,
            notes="",
        )
