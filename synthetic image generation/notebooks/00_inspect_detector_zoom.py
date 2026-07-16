"""Phase 0a (zoom view): same 50 random images, but rendered as 5 separate
4x zoom contact sheets of 10 each. This lets me actually see the bbox
overlays at a useful resolution.

Run after 00_inspect_detector.py — uses the same SEED=42 selection.

Output: notebooks/00_detector_zoom_{1..5}.png

Each output is 2 cols x 5 rows of 800x240 cells = 1612x1212.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(MODEL1))

from src.detector.infer import load_detector, predict_dir  # noqa: E402

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RAW = MODEL1 / "data" / "raw"
CKPT = MODEL1 / "checkpoints" / "detector_v2" / "best.pt"
OUT_DIR = Path(__file__).resolve().parent
N = 50
SEED = 42
ZOOM = 4  # 200x60 -> 800x240


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(CKPT, device=device)
    records = list(predict_dir(model, RAW, batch=128))
    rnd = random.Random(SEED)
    chosen = rnd.sample(records, N)

    cw, ch = 200 * ZOOM, 60 * ZOOM  # 800, 240
    cols, rows = 2, 5  # 10 per sheet
    g = 4
    sheet_w = cols * cw + (cols + 1) * g
    sheet_h = rows * ch + (rows + 1) * g

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for sheet_idx in range(5):
        sheet = Image.new("RGB", (sheet_w, sheet_h), (32, 32, 32))
        for j in range(10):
            i = sheet_idx * 10 + j
            name, boxes = chosen[i]
            img = Image.open(RAW / name).convert("RGB")
            if img.size != (200, 60):
                img = img.resize((200, 60))
            img = img.resize((cw, ch), Image.NEAREST)  # block-pixel zoom
            draw = ImageDraw.Draw(img)
            for x0, y0, x1, y1 in boxes:
                draw.rectangle(
                    [x0 * ZOOM, y0 * ZOOM, x1 * ZOOM, y1 * ZOOM],
                    outline=(255, 30, 30),
                    width=2,
                )
            draw.text((4, 2), name[-13:-4], fill=(0, 0, 0), font=font)
            r, c = divmod(j, cols)
            sheet.paste(img, (g + c * (cw + g), g + r * (ch + g)))
        out = OUT_DIR / f"00_detector_zoom_{sheet_idx + 1}.png"
        sheet.save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
