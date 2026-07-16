"""Qwen2.5-VL-7B helpers for Part C — scheme prompt generation + cleanliness oracle.

Loads Qwen2.5-VL-7B-Instruct lazily on first use and caches the instance.
Both functions share the same loaded model. Call unload() to free VRAM
(~14 GB fp16) when no further VLM calls are needed.

The oracle is intended to be invoked sparsely (only when the heuristic says
"candidate clean"), since each call is a full Qwen forward pass.
"""
from __future__ import annotations

import gc

import torch
from PIL import Image


PROMPT_GEN_INSTRUCTION = """You are looking at unlabeled CAPTCHA images from a single CAPTCHA scheme.
Write ONE short sentence describing the *characters* in these images for a text-to-bbox detector (GroundingDINO).

Focus on:
- color and contrast (e.g. "thin black", "white outlined", "red bold")
- weight and style (thin/thick, serif/sans-serif, italic, hand-written)
- case (uppercase, lowercase, digits, mixed)
- distinctive overlay noise that touches the characters (e.g. "with red curved noise lines", "with grid pattern")

No preamble, no markdown, no quotes. Lowercase. Example output:
thin black serif uppercase letters with red curved noise lines"""


ORACLE_INSTRUCTION = """This image is supposed to show ONLY background — no readable characters.

Answer YES if you see ANY readable character (letter, digit, symbol, even partial). Answer NO if it looks like clean background only (textures, noise, gradients, solid colors are fine).

Reply with exactly one word: YES or NO."""


_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
_qwen = None  # tuple[model, processor] when loaded


def available() -> bool:
    """Loose check: transformers importable and CUDA present. Weights auto-download on first load()."""
    try:
        import transformers  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_loaded() -> bool:
    return _qwen is not None


def load(model_id: str = _MODEL_ID, dtype=torch.float16, device_map: str = "cuda"):
    global _qwen
    if _qwen is not None:
        return _qwen
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    ).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    _qwen = (model, proc)
    return _qwen


def unload():
    global _qwen
    if _qwen is None:
        return
    model, proc = _qwen
    del model, proc
    _qwen = None
    gc.collect()
    torch.cuda.empty_cache()


def _generate(instruction: str, images: list[Image.Image], max_new_tokens: int) -> str:
    model, proc = load()
    content = [{"type": "image", "image": im.convert("RGB")} for im in images]
    content.append({"type": "text", "text": instruction})
    msgs = [{"role": "user", "content": content}]
    text_prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text_prompt], images=[im.convert("RGB") for im in images],
               padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    out_ids = gen[:, inp.input_ids.shape[1]:]
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def generate_scheme_prompt(sample_images: list[Image.Image]) -> str:
    out = _generate(PROMPT_GEN_INSTRUCTION, sample_images, max_new_tokens=200)
    return out.strip().strip('"').strip("'").strip()


def is_clean(image: Image.Image) -> bool:
    out = _generate(ORACLE_INSTRUCTION, [image], max_new_tokens=10)
    return out.strip().upper().startswith("NO")
