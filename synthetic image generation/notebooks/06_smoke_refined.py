"""Smoke artifact: real / rough parametric / refined comparison at t=1.

Three bands, each a 4x4 grid (16 samples):
  Band 1: real DarkBay (random sample of the labeled set)
  Band 2: parametric t=1 (rough composite — what the refiner saw as input)
  Band 3: refined t=1 (parametric output passed through G)

Same RNG seed used for parametric and refined bands so identical text and bg
sampling — the only difference is the refinement pass.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.parametric import ParametricGenerator    # noqa: E402
from src.generator.refined import RefinedGenerator          # noqa: E402

REAL = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
OUT = ROOT / "mvp_results" / "smoke_real_rough_refined.png"

N = 16
COLS = 4
ZOOM = 3
W, H = 200, 50
GAP = 4
LABEL_H = 32
BANDS = ("real DarkBay", "parametric t=1 (rough)", "refined t=1")


def grid_of_pil(images, draw_label: str, sheet, y_offset, font, sheet_w, label_h):
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, y_offset, sheet_w, y_offset + label_h], fill=(48, 48, 48))
    d.text((10, y_offset + 4), draw_label, fill=(255, 255, 255), font=font)
    cw, ch = W * ZOOM, H * ZOOM
    for i, im in enumerate(images):
        r, c = divmod(i, COLS)
        big = im.resize((cw, ch), Image.NEAREST)
        x = GAP + c * (cw + GAP)
        y = y_offset + label_h + GAP + r * (ch + GAP)
        sheet.paste(big, (x, y))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("loading parametric ...")
    para = ParametricGenerator(seed=0)
    print("loading refined ...")
    ref = RefinedGenerator(seed=0)

    print("sampling real ...")
    rng = random.Random(7)
    real_paths = sorted(REAL.glob("*.png"))
    real_chosen = rng.sample(real_paths, N)
    real_imgs = [Image.open(p).convert("RGB") for p in real_chosen]
    if real_imgs[0].size != (W, H):
        real_imgs = [im.resize((W, H)) for im in real_imgs]

    print("rendering parametric t=1 ...")
    para.rng.seed(0)
    para_samples = [im for im, _l, _b in para.render(1.0, N)]

    print("rendering refined t=1 ...")
    ref.rng.seed(0)
    ref_samples = [im for im, _l, _b in ref.render(1.0, N)]

    cw, ch = W * ZOOM, H * ZOOM
    rows = N // COLS
    band_grid_h = rows * ch + (rows + 1) * GAP
    band_h = LABEL_H + band_grid_h
    sheet_w = COLS * cw + (COLS + 1) * GAP
    sheet_h = len(BANDS) * band_h + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (32, 32, 32))

    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    bands_imgs = (real_imgs, para_samples, ref_samples)
    for band_idx, (label, imgs) in enumerate(zip(BANDS, bands_imgs)):
        y = band_idx * band_h + GAP
        grid_of_pil(imgs, label, sheet, y, font, sheet_w, LABEL_H)

    sheet.save(OUT)
    print(f"saved {OUT}  ({sheet_w}x{sheet_h})")


if __name__ == "__main__":
    main()
