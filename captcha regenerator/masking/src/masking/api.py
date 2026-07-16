"""High-level orchestration: Stage 0 (Qwen) -> Stage 2 -> Stage 2.5 -> Stage 3 -> MaskResult."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import stage0, stage2, stage2_5, stage3
from .io_utils import make_overlay
from .types import CharBox, Characterisation, MaskResult


def extract(
    image_bgr: np.ndarray,
    captcha_id: str,
    captcha_type: str,
    cfg: dict[str, Any],
    *,
    forced_color: Optional[str] = None,
    run_stage3: bool = True,
) -> MaskResult:
    H, W = image_bgr.shape[:2]

    if forced_color is not None:
        det = Characterisation(
            char_colors=[forced_color],
            noise_color="unknown",
            noise_types=["none"],
            char_style="unknown",
            raw={"_forced": forced_color},
            description="",
            available=False,
        )
    else:
        det = stage0.characterise(image_bgr, cfg["stage0"])

    chosen_color = _pick_color(det, cfg["stage2"]["path_a"]["hsv_table"])

    s2 = stage2.run(image_bgr, chosen_color, cfg["stage2"])
    s25 = stage2_5.run(image_bgr, s2, cfg["stage2"], cfg["stage2_5"])
    overlay = make_overlay(image_bgr, s25.mask)

    result = MaskResult(
        captcha_id=captcha_id,
        captcha_type=captcha_type,
        image_size=(W, H),
        characterisation=det,
        chosen_color=chosen_color,
        path=s2.path,
        mask=s25.mask,
        cleaned=s25.cleaned,
        overlay=overlay,
        attention=s2.attention,
        stage_2_5={
            "iterations": s25.iterations,
            "final_coverage": s25.final_coverage,
        },
    )

    if run_stage3 and cfg.get("stage3", {}).get("enabled", True):
        s3 = stage3.run(image_bgr, s25, cfg["stage3"])
        result.validated_mask = s3.validated_mask
        result.craft_polys = s3.craft_polys
        result.chars = [CharBox(bbox=c.bbox, area=c.area) for c in s3.chars]
        result.char_crops = [c.crop for c in s3.chars]
        result.char_crop_masks = [c.crop_mask for c in s3.chars]
        result.stage3_stats = {
            "n_polys": s3.n_craft_polys,
            "n_chars_kept": s3.n_chars_kept,
            "n_chars_dropped": s3.n_chars_dropped,
        }

    return result


def _pick_color(det: Characterisation, hsv_table: dict[str, Any]) -> str:
    if not det.char_colors:
        return "unknown"
    if len(det.char_colors) != 1:
        return "unknown"
    c = det.char_colors[0]
    if c in {"multi", "unknown"}:
        return "unknown"
    if c not in hsv_table:
        return "unknown"
    return c
