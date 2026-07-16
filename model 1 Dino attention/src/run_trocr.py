"""TrOCR-large-printed (Microsoft, MIT). Pretrained printed-text OCR."""
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.eval_common import run_predictor_batch


MODEL_ID = "microsoft/trocr-large-printed"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # newly-init pooler weights are fp32; mixing breaks generate()
    print(f"loading {MODEL_ID} on {device} ({dtype})")
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def batch_predict(imgs):
        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device, dtype=dtype)
        ids = model.generate(pixel_values, max_new_tokens=10)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    run_predictor_batch("trocr_large_printed", batch_predict, batch_size=32)


if __name__ == "__main__":
    main()
