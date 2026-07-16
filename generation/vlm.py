"""Shared VLM (Qwen2.5-VL-3B) for the source-free captcha-synth generator.

ONE lazily-loaded model instance drives every VLM-in-the-loop role: the difficulty
director, the background ghost gate, the coarse font tagger, and the pairwise
font/discriminator critic. Weights live on E: (HF_HOME=E:/hf_cache); nothing touches C:.

Design discipline (from DESIGN.md §2, grounded in the judge research):
  * Never ask the VLM to transcribe a captcha (unreliable + violates label-free rule).
  * Rank, don't score: comparisons are pairwise A/B with order-swap + agreement.
  * Gates use k-sample consensus, not a single fallible yes/no.
  * Colours are read from pixels by callers, never from the VLM.

Loader pattern lifted from auto_synth/src/part_c/vlm_qwen.py; JSON parse/validate/cache
harness lifted from captcha regenerator/masking/src/masking/stage0.py.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# --- pin all HF cache to E: BEFORE transformers is imported (C: is nearly full) ---
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "E:/hf_cache/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# captcha canvases are tiny; keep the vision token budget small + bounded
_MAX_PIXELS = 768 * 28 * 28
_MIN_PIXELS = 128 * 28 * 28

_JSON_OBJ = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_qwen: Optional[tuple] = None  # (model, processor)
_CACHE_DIR: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Model lifecycle
# --------------------------------------------------------------------------- #
def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import torch as _t
        return _t.cuda.is_available()
    except Exception:
        return False


def is_loaded() -> bool:
    return _qwen is not None


def load(model_id: str = MODEL_ID):
    """Load Qwen2.5-VL-3B once (bf16, eager attn) and cache the instance."""
    global _qwen
    if _qwen is not None:
        return _qwen
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    proc = AutoProcessor.from_pretrained(
        model_id, max_pixels=_MAX_PIXELS, min_pixels=_MIN_PIXELS
    )
    _qwen = (model, proc)
    return _qwen


def unload():
    global _qwen
    if _qwen is None:
        return
    import torch
    model, proc = _qwen
    del model, proc
    _qwen = None
    gc.collect()
    torch.cuda.empty_cache()


def set_cache_dir(path: str | os.PathLike | None):
    """Enable an on-disk response cache (keyed by prompt+images). None disables it."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path) if path is not None else None
    if _CACHE_DIR is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Low-level generation
# --------------------------------------------------------------------------- #
def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def _generate(instruction: str, images: Sequence, max_new_tokens: int = 128,
              do_sample: bool = False, temperature: float = 0.7) -> str:
    import torch
    from qwen_vl_utils import process_vision_info
    model, proc = load()
    pil = [_to_pil(im) for im in images]
    content = [{"type": "image", "image": im} for im in pil]
    content.append({"type": "text", "text": instruction})
    msgs = [{"role": "user", "content": content}]
    text_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = proc(text=[text_in], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        kw["temperature"] = temperature
    with torch.inference_mode():
        gen = model.generate(**inputs, **kw)
    out = proc.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0].strip()


# --------------------------------------------------------------------------- #
# JSON harness (lifted from masking/stage0.py)
# --------------------------------------------------------------------------- #
def _parse(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
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


def validate_str(value: Any, allowed: Iterable[str], default: str) -> str:
    allow = {a.lower() for a in allowed}
    if isinstance(value, str) and value.strip().lower() in allow:
        return value.strip().lower()
    return default


def validate_list(value: Any, allowed: Iterable[str], default: list) -> list:
    allow = {a.lower() for a in allowed}
    if not isinstance(value, list):
        return list(default)
    out = [v.strip().lower() for v in value
           if isinstance(v, str) and v.strip().lower() in allow]
    return out or list(default)


def _images_hash(images: Sequence, tag: str) -> str:
    h = hashlib.sha256(tag.encode("utf-8"))
    for im in images:
        a = np.asarray(_to_pil(im))
        h.update(repr(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def ask_json(prompt: str, images: Sequence, *, max_new_tokens: int = 256,
             cache_tag: Optional[str] = None) -> dict:
    """Run the VLM and parse a JSON object out of the reply (never raises)."""
    cache_path = None
    if _CACHE_DIR is not None and cache_tag is not None:
        cache_path = _CACHE_DIR / f"json_{_images_hash(images, cache_tag + prompt)}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass
    try:
        text = _generate(prompt, images, max_new_tokens=max_new_tokens, do_sample=False)
    except Exception as exc:
        return {"_error": repr(exc)}
    obj = _parse(text)
    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(obj, indent=2))
        except Exception:
            pass
    return obj


# --------------------------------------------------------------------------- #
# Role primitives
# --------------------------------------------------------------------------- #
def ask_choice(prompt: str, images: Sequence, choices: Sequence[str],
               k: int = 3) -> tuple[str, dict[str, int]]:
    """Forced-choice tag / ordinal bucket via k-sample majority vote.

    Returns (winning_choice, vote_counts). Falls back to choices[0] on total miss.
    """
    low = {c.lower(): c for c in choices}
    votes: Counter = Counter()
    for i in range(k):
        raw = _generate(prompt, images, max_new_tokens=8,
                        do_sample=(k > 1 and i > 0), temperature=0.7).lower()
        hit = next((low[c] for c in low if c in raw), None)
        if hit is not None:
            votes[hit] += 1
    if not votes:
        return choices[0], {}
    return votes.most_common(1)[0][0], dict(votes)


def is_clean(image, k: int = 5, temperature: float = 0.7) -> bool:
    """Ghost-glyph gate: True == background has NO readable character residue.

    k-sample consensus; accept only if EVERY sample says clean (conservative, because
    evaluator VLMs miss subtle residue — a classical tripwire backs this up upstream).
    """
    instr = ("This image should be background ONLY. Answer YES if you can see ANY "
             "character, letter, digit, or partial glyph stroke; answer NO if it is "
             "clean background (texture, noise, gradient, solid colour are fine). "
             "Reply with exactly one word: YES or NO.")
    clean_votes = 0
    for i in range(k):
        out = _generate(instr, [image], max_new_tokens=4,
                        do_sample=(i > 0), temperature=temperature).upper()
        if out.startswith("NO"):
            clean_votes += 1
        elif out.startswith("YES"):
            pass
        else:
            return False  # unparseable → treat as unclean (conservative)
    return clean_votes == k


def _pick_ab(ref, a, b, aspect: str) -> str:
    instr = (f"Image 1 is the REFERENCE. Image 2 and Image 3 are candidates. "
             f"Which candidate matches the reference's {aspect} more closely? "
             f"Reply with exactly one character: 2 or 3 (or 0 if identical/tie).")
    out = _generate(instr, [ref, a, b], max_new_tokens=4, do_sample=False)
    if "2" in out and "3" not in out:
        return "A"
    if "3" in out and "2" not in out:
        return "B"
    return "tie"


def which_closer(ref, a, b, aspect: str = "overall visual style") -> str:
    """Pairwise A/B judged BOTH orderings; returns 'A', 'B', or 'tie'.

    Order-swap + agreement kills position bias — disagreement ⇒ 'tie' (abstain).
    """
    first = _pick_ab(ref, a, b, aspect)
    second = _pick_ab(ref, b, a, aspect)  # swapped: 'A' here means original b
    second = {"A": "B", "B": "A", "tie": "tie"}[second]
    if first == second and first != "tie":
        return first
    return "tie"


def is_match(ref, cand, aspect: str = "font/glyph style") -> bool:
    """Coarse yes/no: does `cand` plausibly match `ref`'s style? (Use sparingly;
    prefer embedding cosine for the quantitative decision, this only breaks ties.)"""
    instr = (f"Image 1 is a REFERENCE glyph/character. Image 2 is a CANDIDATE. "
             f"Do they share the same {aspect} (weight, serif/sans, slant)? "
             f"Reply with exactly one word: YES or NO.")
    out = _generate(instr, [ref, cand], max_new_tokens=4, do_sample=False).upper()
    return out.startswith("YES")


# --------------------------------------------------------------------------- #
# Smoke test:  python -m vlm  (from generation/)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="captcha image paths to probe")
    args = ap.parse_args()
    print(f"available={available()}  model={MODEL_ID}")
    imgs = [Image.open(p) for p in (args.images or [])]
    if imgs:
        load()
        print("describe:", _generate(
            "In one short lowercase sentence, describe the characters' colour, weight, "
            "and any noise in this captcha. No transcription of the text.", imgs[:1]))
        d = ask_json(
            'Return ONLY JSON {"bg":"pale|dark","char_style":"solid|outlined|hollow",'
            '"serif":"serif|sans|mono"} for this captcha.', imgs[:1])
        print("json:", d)
        print("choice(difficulty):", ask_choice(
            "How hard is this captcha to recreate? Reply one word: "
            "very-easy, easy, medium, hard, or very-hard.",
            imgs[:1], ["very-easy", "easy", "medium", "hard", "very-hard"], k=3))
        print("is_clean(self, k=1):", is_clean(imgs[0], k=1))
        unload()
        print("ok")
