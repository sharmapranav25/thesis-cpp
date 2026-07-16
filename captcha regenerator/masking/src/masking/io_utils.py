from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .types import MaskResult


def load_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def load_config(captcha_type: str, configs_dir: str | Path) -> dict[str, Any]:
    configs_dir = Path(configs_dir)
    default = yaml.safe_load((configs_dir / "_default.yaml").read_text())
    type_path = configs_dir / f"{captcha_type}.yaml"
    if type_path.exists():
        override = yaml.safe_load(type_path.read_text()) or {}
        default = _deep_merge(default, override)
    return default


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_overlay(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150)
    overlay = original_bgr.copy()
    overlay[edges > 0] = (0, 0, 255)
    return overlay


def write_result(result: MaskResult, out_dir: str | Path, original_bgr: np.ndarray) -> Path:
    out_dir = Path(out_dir) / result.captcha_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "original.png"), original_bgr)
    cv2.imwrite(str(out_dir / "mask.png"), result.mask)
    cv2.imwrite(str(out_dir / "cleaned.png"), result.cleaned)
    cv2.imwrite(str(out_dir / "overlay.png"), result.overlay)
    if result.attention is not None:
        attn_8u = (np.clip(result.attention, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "attention.png"), attn_8u)

    if result.validated_mask is not None:
        cv2.imwrite(str(out_dir / "validated_mask.png"), result.validated_mask)

    # CRAFT polys as overlay on original.
    if result.craft_polys:
        craft_viz = original_bgr.copy()
        for poly in result.craft_polys:
            cv2.polylines(craft_viz, [poly.reshape(-1, 1, 2).astype(np.int32)],
                          isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.imwrite(str(out_dir / "craft_polys.png"), craft_viz)

    # Per-character crops.
    chars_meta = []
    if result.chars:
        chars_dir = out_dir / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        for i, (cb, crop, cmask) in enumerate(
            zip(result.chars, result.char_crops, result.char_crop_masks)
        ):
            crop_path = f"characters/{i}.png"
            mask_path = f"characters/{i}_mask.png"
            cv2.imwrite(str(out_dir / crop_path), crop)
            cv2.imwrite(str(out_dir / mask_path), cmask)
            cb.crop_path = crop_path
            chars_meta.append({
                "index": i,
                "bbox": list(cb.bbox),
                "area": cb.area,
                "crop_path": crop_path,
                "mask_path": mask_path,
            })

    metadata = {
        "captcha_id": result.captcha_id,
        "captcha_type": result.captcha_type,
        "image_size": list(result.image_size),
        "characterisation": {
            "char_colors": result.characterisation.char_colors,
            "noise_color": result.characterisation.noise_color,
            "noise_types": result.characterisation.noise_types,
            "char_style": result.characterisation.char_style,
            "available": result.characterisation.available,
            "description": result.characterisation.description,
        },
        "chosen_color": result.chosen_color,
        "path": result.path,
        "stage_2_5": result.stage_2_5,
        "stage_3": {
            **result.stage3_stats,
            "characters": chars_meta,
        },
        "method": "masking_v0.3_qwen_craft",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir
