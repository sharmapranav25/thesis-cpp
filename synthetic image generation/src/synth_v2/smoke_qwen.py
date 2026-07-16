"""Phase 0 smoke: Qwen2.5-VL on 5 CryptBB images.

Goal: confirm Qwen-VL loads, accepts the captcha, and emits parseable JSON
of the form {chars: [{char, bbox_2d}, ...]} or string + per-char boxes.

If Qwen's per-character boxes are usable, we may not need CRAFT at all
(simpler v2). If Qwen reads the string accurately but bboxes are word-level
or wrong, we'll need CRAFT (or detector_v2) for boxes.

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v2.smoke_qwen
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

RAW = MODEL1 / "data" / "raw"
OUT = ROOT / "mvp_results" / "v2_smoke"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # 3B fits comfortably in bf16

SAMPLE_NAMES = [
    "captcha_screenshot_1690279209.png",
    "captcha_screenshot_1690279217.png",
    "captcha_screenshot_1690279267.png",
    "captcha_screenshot_1690279299.png",
    "captcha_screenshot_1690279422.png",
]

PROMPT = (
    "This is a CAPTCHA image with 5 colored, distorted alphanumeric characters. "
    "For each character, output its identity and pixel bounding box. "
    "Return JSON only, no commentary, in this exact schema:\n"
    "{\"chars\": [{\"char\": \"X\", \"bbox_2d\": [x1, y1, x2, y2]}, ...]}\n"
    "Order the characters left-to-right. Coordinates are pixel positions in "
    "the original image."
)


def parse_json(text: str) -> dict | None:
    """Strip ```json fences if present, return parsed dict or None."""
    s = text.strip()
    if s.startswith("```"):
        # remove first fence line + trailing ```
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # try extracting the first { ... } block
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} (downloads ~7 GB on first run) ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    log_path = OUT / "qwen_smoke_log.txt"
    log_lines: list[str] = []

    for name in SAMPLE_NAMES:
        path = RAW / name
        if not path.exists():
            print(f"  skip missing: {name}")
            continue
        im = Image.open(path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)

        t1 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512,
                                           do_sample=False)
        elapsed = time.time() - t1
        gen_only = generated_ids[:, inputs.input_ids.shape[1]:]
        out_text = processor.batch_decode(gen_only, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

        parsed = parse_json(out_text)
        log_lines.append("=" * 60)
        log_lines.append(f"{name}  ({elapsed:.2f}s)")
        log_lines.append(f"raw output:")
        log_lines.append(out_text)
        log_lines.append(f"parsed: {parsed}")

        # overlay
        if parsed and "chars" in parsed:
            vis = im.copy()
            draw = ImageDraw.Draw(vis)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 12)
            except Exception:
                font = ImageFont.load_default()
            for entry in parsed["chars"]:
                bb = entry.get("bbox_2d") or entry.get("bbox")
                ch = entry.get("char", "?")
                if bb and len(bb) == 4:
                    draw.rectangle(bb, outline=(0, 255, 0), width=1)
                    draw.text((bb[0], max(0, bb[1] - 14)), ch,
                              fill=(0, 255, 0), font=font)
            vis.save(OUT / f"qwen_{name}")
            n_ch = len(parsed.get("chars", []))
            print(f"  {name}: {n_ch} chars  ({elapsed:.2f}s)")
        else:
            print(f"  {name}: PARSE FAIL  ({elapsed:.2f}s)")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nlog -> {log_path}")


if __name__ == "__main__":
    main()
