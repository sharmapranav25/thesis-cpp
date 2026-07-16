"""GOT-OCR2.0 (stepfun-ai/GOT-OCR2_0, Apache 2.0). Strong general OCR."""
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
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.eval_common import run_predictor


MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


def main():
    device = "cuda"
    dtype = torch.float16
    print(f"loading {MODEL_ID} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    @torch.inference_mode()
    def predict(img):
        inputs = processor(img, return_tensors="pt").to(device, dtype=dtype)
        ids = model.generate(**inputs, do_sample=False, tokenizer=processor.tokenizer,
                             stop_strings="<|im_end|>", max_new_tokens=20)
        text = processor.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    run_predictor("got_ocr2", predict)


if __name__ == "__main__":
    main()
