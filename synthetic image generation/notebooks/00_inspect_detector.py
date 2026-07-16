"""Phase 0a: visualize detector_v2 bboxes on 50 raw CryptBB images.

Output: notebooks/00_detector_overlay.png — a 5x10 contact sheet of raw
images with red bbox overlays.

Run:
    cd "E:/pranav/synthetic image generation"
    "E:/pranav/model 1/.venv/Scripts/python.exe" notebooks/00_inspect_detector.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(MODEL1))  # so `from src.detector...` resolves

from src.detector.infer import load_detector, predict_dir  # noqa: E402

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RAW = MODEL1 / "data" / "raw"
CKPT = MODEL1 / "checkpoints" / "detector_v2" / "best.pt"
OUT = Path(__file__).resolve().parent / "00_detector_overlay.png"
N = 50
SEED = 42


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"detector ckpt: {CKPT}")
    model = load_detector(CKPT, device=device)

    print(f"running detector on {RAW} ...")
    records = list(predict_dir(model, RAW, batch=128))
    print(f"got {len(records)} predictions")

    rnd = random.Random(SEED)
    chosen = rnd.sample(records, N)

    # 5 cols x 10 rows, each cell 200x60 with 2 px gutter
    cols, rows = 5, 10
    cw, ch = 200, 60
    g = 2
    sheet = Image.new(
        "RGB",
        (cols * cw + (cols + 1) * g, rows * ch + (rows + 1) * g),
        (32, 32, 32),
    )

    try:
        font = ImageFont.truetype("arial.ttf", 8)
    except OSError:
        font = ImageFont.load_default()

    for i, (name, boxes) in enumerate(chosen):
        img_path = RAW / name
        img = Image.open(img_path).convert("RGB")
        if img.size != (cw, ch):
            img = img.resize((cw, ch))
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 30, 30), width=1)
        # tiny filename watermark in top-left for traceability
        draw.text((1, 0), name[-12:-4], fill=(0, 0, 0), font=font)
        r, c = divmod(i, cols)
        sheet.paste(img, (g + c * (cw + g), g + r * (ch + g)))

    sheet.save(OUT)
    print(f"saved {OUT} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
