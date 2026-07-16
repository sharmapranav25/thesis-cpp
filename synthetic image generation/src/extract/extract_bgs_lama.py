"""Phase 1 (LaMa version): extract clean CryptBB backgrounds via LaMa inpaint.

Same flow as extract_bgs.py but swaps cv2.inpaint for simple-lama-inpainting.
LaMa handles textured regions much better than TELEA.

Output:
  data/cryptbb_bg_lama/<name>.png    — bg-only, all images
  data/cryptbb_bg_lama_qa.png        — 20-sample (raw | bg | mask) QA sheet

Usage:
    "E:/pranav/model 1/.venv/Scripts/python.exe" src/extract/extract_bgs_lama.py [--limit N]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(MODEL1))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.detector.infer import load_detector, predict_dir  # noqa: E402
from simple_lama_inpainting import SimpleLama  # noqa: E402

RAW = MODEL1 / "data" / "raw"
CKPT = MODEL1 / "checkpoints" / "detector_v2" / "best.pt"
OUT_BG = ROOT / "data" / "cryptbb_bg_lama"
OUT_QA = ROOT / "data" / "cryptbb_bg_lama_qa.png"

DILATE_PX = 5
SEED = 42
N_QA = 20


def make_mask(boxes, w: int = 200, h: int = 60, dilate: int = DILATE_PX) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        x0i = max(0, int(round(x0)) - dilate)
        y0i = max(0, int(round(y0)) - dilate)
        x1i = min(w, int(round(x1)) + dilate)
        y1i = min(h, int(round(y1)) + dilate)
        mask[y0i:y1i, x0i:x1i] = 255
    return mask


def main(limit: int | None) -> None:
    OUT_BG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = load_detector(CKPT, device=device)
    print("running detector ...")
    records = list(predict_dir(model, RAW, batch=128))
    if limit is not None:
        records = records[:limit]
    print(f"will process {len(records)} images")

    print("loading LaMa (downloads weights on first run; ~200 MB) ...")
    t0 = time.time()
    lama = SimpleLama()
    print(f"  LaMa ready in {time.time() - t0:.1f}s")

    qa_pool: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    for i, (name, boxes) in enumerate(records):
        img_pil = Image.open(RAW / name).convert("RGB")
        if img_pil.size != (200, 60):
            img_pil = img_pil.resize((200, 60))
        img_rgb = np.array(img_pil)
        mask = make_mask(boxes)
        mask_pil = Image.fromarray(mask)
        bg_pil = lama(img_pil, mask_pil)
        if bg_pil.size != (200, 60):
            bg_pil = bg_pil.resize((200, 60))
        bg_pil.save(OUT_BG / name)
        if len(qa_pool) < 200 and (i % max(1, len(records) // 100) == 0):
            qa_pool[name] = (img_rgb, np.array(bg_pil), mask)
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  {i + 1}/{len(records)}  rate={rate:.1f} img/s  eta={eta:.0f}s")
    print(f"saved {len(records)} bgs to {OUT_BG}")

    if not qa_pool:
        return
    rnd = random.Random(SEED)
    names = rnd.sample(sorted(qa_pool.keys()), min(N_QA, len(qa_pool)))
    zoom = 2
    cw, ch = 200 * zoom, 60 * zoom
    g = 4
    cols = 4
    rows = (len(names) + cols - 1) // cols
    strip_h = ch * 3 + g * 2
    sheet = Image.new(
        "RGB",
        (cols * cw + (cols + 1) * g, rows * strip_h + (rows + 1) * g),
        (32, 32, 32),
    )
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        raw_rgb, bg_rgb, mask = qa_pool[name]
        raw_im = Image.fromarray(raw_rgb).resize((cw, ch), Image.NEAREST)
        bg_im = Image.fromarray(bg_rgb).resize((cw, ch), Image.NEAREST)
        mask_im = Image.fromarray(mask).convert("RGB").resize((cw, ch), Image.NEAREST)
        x = g + c * (cw + g)
        y = g + r * (strip_h + g)
        sheet.paste(raw_im, (x, y))
        sheet.paste(bg_im, (x, y + ch + g))
        sheet.paste(mask_im, (x, y + 2 * (ch + g)))
    sheet.save(OUT_QA)
    print(f"saved QA sheet to {OUT_QA}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.limit)
