"""CLI: char-extract-qwen-baseline — Qwen2.5-VL zero-shot eval pseudo-GT.

Open-source replacement for the Claude-VLM oracle. Loads
`Qwen/Qwen2.5-VL-7B-Instruct` once and processes a directory of images,
saving char count + bbox predictions as JSON (same shape as
`claude_baseline`'s output, so the eval CLI consumes both interchangeably).

Model weights (~16 GB at fp16) download to `$HF_HOME` on first call.
`char_extraction.__init__` pins `HF_HOME` to `<repo>/.cache/hf` so all
weights stay on the same drive as the venv.

Important: this is part of the EVAL harness, not the production pipeline.
The pipeline (otsu_cc, universal_sam) never calls Qwen.
"""

# Importing the package first sets HF_HOME / TORCH_HOME before transformers loads.
import char_extraction  # noqa: F401

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_PROMPT_TEMPLATE = """\
You are analyzing a CAPTCHA image used for bot detection. Identify each individual \
character glyph and locate it precisely with a tight bounding box.

The image is {width} x {height} pixels (origin top-left, x grows right, y grows down).

Return a JSON object exactly like this:

{{
  "n_characters": <int>,
  "characters": [
    {{"bbox": [x1, y1, x2, y2], "char": "?"}},
    ...
  ]
}}

Rules:
- Coordinates are integer pixels in the original {width} x {height} image, with x1<x2 and y1<y2.
- Each bbox must tightly enclose ONE character glyph (all strokes, dots, tails — exclude empty whitespace).
- Order the "characters" array left-to-right by x1.
- Ignore decorative lines, scribbles, dot patterns, mosaic backgrounds, side bars, and any non-character noise.
- If a character is unreadable, still include the bbox; set "char" to "?".
- Return ONLY the JSON object, no commentary, no code fences."""


def _list_images(root: Path) -> list:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS])


def _build_messages(image: Image.Image, prompt: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]


_JSON_OBJ = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_response(text: str, img_w: int, img_h: int) -> Dict[str, Any]:
    """Parse Qwen's text into the predictions schema.

    Qwen sometimes wraps in ```json fences or adds prose. Robust strategy:
    grab the longest top-level {...} block and parse.
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        m = _JSON_OBJ.search(s)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                pass
    if not isinstance(obj, dict):
        raise ValueError(f"Cannot parse Qwen response as JSON: {text[:200]!r}")
    chars = obj.get("characters") or []
    if not isinstance(chars, list):
        raise ValueError("Response missing 'characters' list")
    cleaned = []
    for c in chars:
        if not isinstance(c, dict) or "bbox" not in c:
            continue
        bb = c["bbox"]
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
        except Exception:
            continue
        # Some Qwen models output normalized 0-1000 coords. Detect by max value.
        if max(x1, x2) <= 1000 and max(x1, x2) > img_w * 1.2 and img_w < 800:
            x1 = int(round(x1 / 1000.0 * img_w))
            y1 = int(round(y1 / 1000.0 * img_h))
            x2 = int(round(x2 / 1000.0 * img_w))
            y2 = int(round(y2 / 1000.0 * img_h))
        # Clip to image bounds.
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        entry = {"bbox": [x1, y1, x2, y2]}
        if "char" in c and isinstance(c["char"], str):
            entry["char"] = c["char"]
        cleaned.append(entry)
    cleaned.sort(key=lambda e: e["bbox"][0])
    return {"n_characters": len(cleaned), "characters": cleaned}


_LOADED: Dict[str, Any] = {}


def _load_qwen(model_id: str, dtype: torch.dtype):
    if model_id in _LOADED:
        return _LOADED[model_id]
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"Loading {model_id} (dtype={dtype}, device=cuda)...", file=sys.stderr)
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    _LOADED[model_id] = (model, processor)
    return _LOADED[model_id]


def _infer_one(model, processor, image: Image.Image, prompt: str, max_new: int = 1024) -> str:
    from qwen_vl_utils import process_vision_info
    msgs = _build_messages(image, prompt)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[text],
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


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="char-extract-qwen-baseline",
                                description="Qwen2.5-VL zero-shot pseudo-GT (count + bbox).")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--type", dest="captcha_type", default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    images = _list_images(args.input_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("No images found.", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if args.resume and args.out.exists():
        try:
            existing = json.loads(args.out.read_text())
        except Exception:
            existing = {}
    predictions = existing.get("images", {})

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model, processor = _load_qwen(args.model, dtype_map[args.dtype])

    n_called = 0
    n_skipped = 0
    n_errors = 0
    for path in tqdm(images, desc="qwen"):
        cid = path.stem
        if args.resume and cid in predictions and "characters" in predictions[cid]:
            n_skipped += 1
            continue
        try:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
                w, h = image.size
                prompt = _PROMPT_TEMPLATE.format(width=w, height=h)
                t0 = time.perf_counter()
                text = _infer_one(model, processor, image, prompt, max_new=args.max_new_tokens)
                dt = time.perf_counter() - t0
            try:
                pred = _parse_response(text, w, h)
            except Exception as parse_err:
                pred = {"error": f"parse: {parse_err}", "raw": text[:500]}
                n_errors += 1
            else:
                n_called += 1
            pred["image_size"] = [w, h]
            pred["latency_s"] = round(dt, 2)
            predictions[cid] = pred
        except Exception as e:
            print(f"\n  [error] {cid}: {e}", file=sys.stderr)
            predictions[cid] = {"error": str(e)}
            n_errors += 1
        # Persist incrementally so a crash doesn't lose work.
        out_obj = {
            "captcha_type": args.captcha_type,
            "model": args.model,
            "input_dir": str(args.input_dir),
            "n_total": len(images),
            "n_with_predictions": sum(1 for v in predictions.values() if "characters" in v),
            "images": predictions,
        }
        args.out.write_text(json.dumps(out_obj, indent=2))

    summary = {
        "out": str(args.out),
        "n_total": len(images),
        "n_called": n_called,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }
    print(json.dumps(summary, indent=2))
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
