"""Phase 1: extract clean backgrounds from CryptBB raw images.

For each raw image:
  1. Run trained detector → 5 char bboxes.
  2. Build text mask = union of bboxes dilated by 5 px.
  3. cv2.inpaint with TELEA flag, radius=3.
  4. Save bg-only result to data/cryptbb_bg/<original_filename>.png.

Also writes data/cryptbb_bg_qa.png — 20-sample (raw | bg | mask) contact sheet
for visual QA.

Run:
    cd "E:/pranav/synthetic image generation"
    "E:/pranav/model 1/.venv/Scripts/python.exe" src/extract/extract_bgs.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(MODEL1))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.detector.infer import load_detector, predict_dir  # noqa: E402

RAW = MODEL1 / "data" / "raw"
CKPT = MODEL1 / "checkpoints" / "detector_v2" / "best.pt"
OUT_BG = ROOT / "data" / "cryptbb_bg"
OUT_QA = ROOT / "data" / "cryptbb_bg_qa.png"

DILATE_PX = 5
INPAINT_RADIUS = 3
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


def extract_one(img_path: Path, boxes):
    img_pil = Image.open(img_path).convert("RGB")
    if img_pil.size != (200, 60):
        img_pil = img_pil.resize((200, 60))
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = make_mask(boxes)
    bg_bgr = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, bg_rgb, mask


def render_qa(extracted: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Image.Image:
    rnd = random.Random(SEED)
    names = rnd.sample(sorted(extracted.keys()), N_QA)
    zoom = 2
    cw, ch = 200 * zoom, 60 * zoom
    g = 4
    cols = 4
    rows = (N_QA + cols - 1) // cols
    strip_h = ch * 3 + g * 2
    sheet = Image.new(
        "RGB",
        (cols * cw + (cols + 1) * g, rows * strip_h + (rows + 1) * g),
        (32, 32, 32),
    )
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        raw_rgb, bg_rgb, mask = extracted[name]
        raw_im = Image.fromarray(raw_rgb).resize((cw, ch), Image.NEAREST)
        bg_im = Image.fromarray(bg_rgb).resize((cw, ch), Image.NEAREST)
        mask_im = Image.fromarray(mask).convert("RGB").resize((cw, ch), Image.NEAREST)
        x = g + c * (cw + g)
        y = g + r * (strip_h + g)
        sheet.paste(raw_im, (x, y))
        sheet.paste(bg_im, (x, y + ch + g))
        sheet.paste(mask_im, (x, y + 2 * (ch + g)))
    return sheet


def main() -> None:
    OUT_BG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = load_detector(CKPT, device=device)
    print(f"running detector on {RAW} ...")
    records = list(predict_dir(model, RAW, batch=128))
    print(f"got {len(records)} predictions; inpainting ...")

    qa_pool: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, (name, boxes) in enumerate(records):
        raw_rgb, bg_rgb, mask = extract_one(RAW / name, boxes)
        Image.fromarray(bg_rgb).save(OUT_BG / name)
        if len(qa_pool) < 200 and (i % 3 == 0):  # keep a sampling pool for QA
            qa_pool[name] = (raw_rgb, bg_rgb, mask)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}")

    print(f"saved {len(records)} bgs to {OUT_BG}")
    sheet = render_qa(qa_pool)
    sheet.save(OUT_QA)
    print(f"saved QA sheet to {OUT_QA}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
