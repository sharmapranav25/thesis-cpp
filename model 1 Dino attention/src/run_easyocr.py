"""EasyOCR (JaidedAI, Apache 2.0). Lightweight production OCR baseline."""
from __future__ import annotations

import os, sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import easyocr

from src.eval_common import run_predictor


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def main():
    reader = easyocr.Reader(["en"], gpu=True)

    def predict(img):
        arr = np.array(img)
        results = reader.readtext(
            arr,
            allowlist=CHARSET,
            paragraph=False,
            detail=1,
        )
        if not results:
            return ""
        # join detected boxes left-to-right
        results.sort(key=lambda r: r[0][0][0])
        return "".join(r[1] for r in results).strip()

    run_predictor("easyocr", predict)


if __name__ == "__main__":
    main()
