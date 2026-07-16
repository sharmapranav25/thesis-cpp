"""Pure-classical baseline: Otsu threshold + connected components.

Used as the universal-fallback default in `_default.yaml` until torch-based
universal baselines (CRAFT, MobileSAM) are wired up. Also useful as a
zero-dependency stress test for the pipeline plumbing.
"""

from __future__ import annotations

from typing import List

import numpy as np

from char_extraction.methods.base import BaseMethod, register_method
from char_extraction.methods.classical import (
    detect_polarity,
    open_morphology,
    otsu_mask,
    to_gray,
)
from char_extraction.postproc.components import (
    bboxes_to_charboxes,
    component_bboxes,
    filter_bboxes,
    label_components,
)
from char_extraction.types import CharBox


@register_method
class OtsuCC(BaseMethod):
    """Grayscale -> Otsu -> morph open -> CC -> filter by area/AR -> sort."""

    name = "otsu_cc"

    def segment(self, img: np.ndarray) -> np.ndarray:
        gray = to_gray(img)
        polarity = self.cfg.get("polarity", "auto")
        if polarity == "auto":
            dark_on_light = detect_polarity(gray)
        elif polarity == "dark_on_light":
            dark_on_light = True
        elif polarity == "light_on_dark":
            dark_on_light = False
        else:
            raise ValueError(f"Unknown polarity {polarity!r}")
        mask = otsu_mask(gray, dark_on_light=dark_on_light)
        open_k = self.cfg.get("morph_open_ksize", [2, 2])
        if open_k and open_k[0] > 1 and open_k[1] > 1:
            mask = open_morphology(mask, ksize=tuple(open_k))
        return mask

    def split(self, mask: np.ndarray, img: np.ndarray) -> List[CharBox]:
        labels, n = label_components(mask)
        bboxes = component_bboxes(labels, n)
        bboxes = filter_bboxes(
            bboxes,
            min_area=int(self.cfg.get("min_area", 30)),
            max_area=int(self.cfg.get("max_area", 100_000)),
            min_aspect_ratio=float(self.cfg.get("min_aspect_ratio", 0.05)),
            max_aspect_ratio=float(self.cfg.get("max_aspect_ratio", 10.0)),
            min_height=int(self.cfg.get("min_height", 5)),
            max_height=int(self.cfg.get("max_height", 10_000)),
            min_width=int(self.cfg.get("min_width", 1)),
            max_width=int(self.cfg.get("max_width", 10_000)),
        )
        return bboxes_to_charboxes(bboxes)
