"""Qwen2-VL-2B-Instruct (Apache 2.0). Open VLM. Prompt-driven captcha read."""
from __future__ import annotations

import os, sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from src.eval_common import run_predictor


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT = (
    "This is a 5-character captcha. The characters are case-sensitive and from "
    "the set [a-zA-Z0-9]. Output ONLY the 5 characters, nothing else, no "
    "explanation, no spaces."
)


def main():
    device = "cuda"
    dtype = torch.float16
    print(f"loading {MODEL_ID} on {device} ({dtype})", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    @torch.inference_mode()
    def predict(img):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        out = processor.batch_decode(gen, skip_special_tokens=True)[0]
        # strip whitespace and quotes; if model says "answer: XXXXX" pull the last token
        out = out.strip().strip('"').strip("'")
        if " " in out or ":" in out:
            out = out.replace(":", " ").split()[-1]
        return out[:5] if len(out) >= 5 else out

    run_predictor("qwen2vl_2b", predict)


if __name__ == "__main__":
    main()
