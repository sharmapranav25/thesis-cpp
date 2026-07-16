"""
Stage 0 — Qwen2.5-VL characterisation.

One structured-JSON call returns:
  - char_colors:  list of distinct character colours (e.g. ["blue"], ["multi"])
  - noise_color:  dominant colour of distractors, or "none"
  - noise_types:  list of distractor kinds (lines, dots, texture, ...)

Per-image SHA256 cache: re-runs are free.
Falls back gracefully if torch/transformers not importable, returning
char_colors=["unknown"] so Stage 2 takes Path B.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .types import Characterisation

# Set of colour tokens we accept. Anything outside this set falls back to "unknown".
_ALLOWED_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "black", "white", "gray", "brown", "multi", "unknown",
}
_ALLOWED_NOISE_COLORS = _ALLOWED_COLORS | {"none"}
_ALLOWED_NOISE_TYPES = {
    "horizontal_lines", "diagonal_lines", "vertical_lines", "curved_lines",
    "dots", "background_texture", "gradient_background",
    "warping", "occlusion", "shadow", "none",
}
_ALLOWED_CHAR_STYLES = {"solid", "outlined", "hollow", "varies", "unknown"}

_PROMPT = """\
You are analyzing a CAPTCHA image. Identify the characters' colour, the noise's colour, \
the kinds of noise present, and the style of the character glyphs.

Return ONLY a JSON object with this exact shape:

{
  "char_colors":  [<one or more colour tokens>],
  "noise_color":  <one colour token>,
  "noise_types":  [<one or more noise tokens>],
  "char_style":   <one style token>
}

Allowed colour tokens: red, orange, yellow, green, blue, purple, pink, black, white, gray, brown, multi, unknown.
Allowed noise_color tokens: same as colour tokens, plus "none" if there are no distractors.
Allowed noise tokens: horizontal_lines, diagonal_lines, vertical_lines, curved_lines, dots, background_texture, gradient_background, warping, occlusion, shadow, none.
Allowed style tokens: solid, outlined, hollow, varies, unknown.

Rules:
- char_colors: distinct colours the alphanumeric characters are drawn in. Use ["multi"] only if characters have visibly different colours from each other.
- noise_color: dominant colour of distractors. "none" if there are no distractors.
- noise_types: every distractor type present. ["none"] if no distractors.
- char_style: "solid" if the glyphs are filled in their character colour; "outlined" if only a stroke ring with a different-coloured (often white) interior; "hollow" if mostly transparent inside the strokes; "varies" if mixed; "unknown" if unclear.

Respond with ONLY the JSON object. No commentary, no code fences."""

_JSON_OBJ = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_LOADED: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

def characterise(image_bgr: np.ndarray, cfg: dict[str, Any]) -> Characterisation:
    if not cfg.get("enabled", True):
        return _fallback(reason="disabled")

    cache_dir = cfg.get("cache_dir")
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{_image_hash(image_bgr)}.json"
        if cache_path.exists():
            try:
                return _from_cache(cache_path)
            except Exception:
                pass  # corrupt cache, recompute

    try:
        model, processor = _load_qwen(cfg["model"], cfg["dtype"])
    except Exception as exc:
        return _fallback(reason=f"load: {exc!r}")

    try:
        text = _infer(model, processor, image_bgr, max_new=cfg.get("max_new_tokens", 256))
    except Exception as exc:
        return _fallback(reason=f"infer: {exc!r}")

    parsed = _parse(text)
    char_colors = _validate_list(parsed.get("char_colors"), _ALLOWED_COLORS, default=["unknown"])
    noise_color = _validate_str(parsed.get("noise_color"), _ALLOWED_NOISE_COLORS, default="unknown")
    noise_types = _validate_list(parsed.get("noise_types"), _ALLOWED_NOISE_TYPES, default=["none"])
    char_style = _validate_str(parsed.get("char_style"), _ALLOWED_CHAR_STYLES, default="unknown")

    det = Characterisation(
        char_colors=char_colors,
        noise_color=noise_color,
        noise_types=noise_types,
        char_style=char_style,
        raw=parsed if isinstance(parsed, dict) else {},
        description=text.strip()[:1000] if isinstance(text, str) else "",
        available=True,
    )

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(_to_cache(det), indent=2))
        except Exception:
            pass

    return det


# --------------------------------------------------------------------------- #
# Qwen loading + inference
# --------------------------------------------------------------------------- #

def _load_qwen(model_id: str, dtype_name: str):
    key = f"{model_id}|{dtype_name}"
    if key in _LOADED:
        return _LOADED[key]
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_name, torch.bfloat16)

    print(f"[stage0] loading {model_id} (dtype={dtype_name})...", file=sys.stderr)
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[stage0] loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    _LOADED[key] = (model, processor)
    return _LOADED[key]


def _infer(model, processor, image_bgr: np.ndarray, max_new: int = 256) -> str:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": _PROMPT},
        ],
    }]
    text_in = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[text_in],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0]


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #

def _parse(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except Exception:
        m = _JSON_OBJ.search(s)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}


def _validate_list(value: Any, allowed: set[str], *, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    cleaned = []
    for v in value:
        if isinstance(v, str) and v.strip().lower() in allowed:
            cleaned.append(v.strip().lower())
    return cleaned or list(default)


def _validate_str(value: Any, allowed: set[str], *, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

def _image_hash(image_bgr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(image_bgr.shape.__repr__().encode("ascii"))
    h.update(image_bgr.tobytes())
    return h.hexdigest()


def _to_cache(det: Characterisation) -> dict[str, Any]:
    return {
        "char_colors": det.char_colors,
        "noise_color": det.noise_color,
        "noise_types": det.noise_types,
        "char_style": det.char_style,
        "raw": det.raw,
        "description": det.description,
        "available": det.available,
    }


def _from_cache(path: Path) -> Characterisation:
    obj = json.loads(path.read_text())
    return Characterisation(
        char_colors=obj["char_colors"],
        noise_color=obj["noise_color"],
        noise_types=obj["noise_types"],
        char_style=obj.get("char_style", "unknown"),
        raw=obj.get("raw", {}),
        description=obj.get("description", ""),
        available=obj.get("available", True),
    )


def _fallback(*, reason: str) -> Characterisation:
    return Characterisation(
        char_colors=["unknown"],
        noise_color="unknown",
        noise_types=["none"],
        char_style="unknown",
        raw={"_fallback_reason": reason},
        description="",
        available=False,
    )
