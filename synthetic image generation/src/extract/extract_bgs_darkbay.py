"""Extract clean DarkBay backgrounds.

DarkBay bg is uniform light gray (~190 V) so masking is trivial:
  - text pixels are either DARK (gray < low_thr) or WHITE (gray > high_thr).
  - bg pixels are in the middle band.
Mask = foreground pixels (text + dark boxes) dilated; cv2.inpaint TELEA fills
with surrounding gray.

Output:
  data/darkbay_bg/<name>.png    — 200×50 RGB
  data/darkbay_bg_qa.png        — 20-sample (raw | bg | mask) QA sheet

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/extract/extract_bgs_darkbay.py [--limit N]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
OUT_BG = ROOT / "data" / "darkbay_bg"
OUT_QA = ROOT / "data" / "darkbay_bg_qa.png"

LOW_THR = 150          # below = dark text or dark box
HIGH_THR = 230         # above = white text inside dark box
DILATE_ITERS = 3
INPAINT_RADIUS = 3
SEED = 42
N_QA = 20


def make_mask(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = ((gray < LOW_THR) | (gray > HIGH_THR)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=DILATE_ITERS)


def extract_one(img_path: Path):
    img_pil = Image.open(img_path).convert("RGB")
    if img_pil.size != (200, 50):
        img_pil = img_pil.resize((200, 50))
    img_rgb = np.array(img_pil)
    mask = make_mask(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bg_bgr = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, bg_rgb, mask


def render_qa(extracted: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Image.Image:
    rnd = random.Random(SEED)
    names = rnd.sample(sorted(extracted.keys()), min(N_QA, len(extracted)))
    zoom = 2
    cw, ch = 200 * zoom, 50 * zoom
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


def main(limit: int | None) -> None:
    OUT_BG.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW.glob("*.png"))
    if limit is not None:
        paths = paths[:limit]
    print(f"extracting bgs from {len(paths)} DarkBay images")
    qa: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, p in enumerate(paths):
        raw_rgb, bg_rgb, mask = extract_one(p)
        Image.fromarray(bg_rgb).save(OUT_BG / p.name)
        if len(qa) < 200 and (i % max(1, len(paths) // 50) == 0):
            qa[p.name] = (raw_rgb, bg_rgb, mask)
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  {i + 1}/{len(paths)}")
    print(f"saved {len(paths)} bgs to {OUT_BG}")
    sheet = render_qa(qa)
    sheet.save(OUT_QA)
    print(f"saved QA sheet to {OUT_QA}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.limit)
