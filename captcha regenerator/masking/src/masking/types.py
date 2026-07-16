from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

PathTaken = Literal["A", "B"]


@dataclass
class Characterisation:
    char_colors: list[str]
    noise_color: str
    noise_types: list[str]
    char_style: str = "unknown"   # solid | outlined | hollow | varies | unknown
    raw: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    available: bool = True


@dataclass
class Stage2Output:
    mask: np.ndarray
    cleaned: np.ndarray
    attention: Optional[np.ndarray] = None
    path: PathTaken = "A"
    chosen_color: str = "unknown"
    params_used: dict = field(default_factory=dict)


@dataclass
class Stage2_5Output:
    mask: np.ndarray
    cleaned: np.ndarray
    iterations: int = 0
    final_coverage: Optional[float] = None
    history: list[dict] = field(default_factory=list)


@dataclass
class CharBox:
    bbox: tuple[int, int, int, int]    # (x1, y1, x2, y2) in original coords
    area: int
    crop_path: str = ""                # populated by io_utils.write_result


@dataclass
class MaskResult:
    captcha_id: str
    captcha_type: str
    image_size: tuple[int, int]
    characterisation: Characterisation
    chosen_color: str
    path: PathTaken
    mask: np.ndarray
    cleaned: np.ndarray
    overlay: np.ndarray
    attention: Optional[np.ndarray] = None
    stage_2_5: dict = field(default_factory=lambda: {"iterations": 0, "final_coverage": None})

    # Stage 3 outputs
    validated_mask: Optional[np.ndarray] = None
    craft_polys: list = field(default_factory=list)        # list of (4,2) int arrays
    chars: list[CharBox] = field(default_factory=list)
    char_crops: list[np.ndarray] = field(default_factory=list)   # parallel to chars
    char_crop_masks: list[np.ndarray] = field(default_factory=list)
    stage3_stats: dict = field(default_factory=lambda: {"n_polys": 0, "n_chars_kept": 0, "n_chars_dropped": 0})
