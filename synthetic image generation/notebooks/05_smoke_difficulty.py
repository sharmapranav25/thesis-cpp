"""Smoke artifact: 16 samples each at t=0.0 / 0.5 / 1.0 in one grid.

Visual sanity that the difficulty knob does what it should:
- t=0.0 → clean, single font, no rotation, no boxes, plain bg sampling
- t=0.5 → some rotation, ~30% box prob, light jitter, multiple fonts
- t=1.0 → full HARD config (the v2 DINOv2 winner)

Each row is one t-level (16 samples wide).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.parametric import ParametricGenerator  # noqa: E402

OUT = ROOT / "mvp_results" / "smoke_t0_05_10.png"
N_PER_T = 16
COLS = 4
TS = (0.0, 0.5, 1.0)
ZOOM = 3
W, H = 200, 50
GAP = 4
LABEL_H = 32


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("loading generator ...")
    gen = ParametricGenerator(seed=0)

    cw, ch = W * ZOOM, H * ZOOM
    rows = N_PER_T // COLS
    grid_w = COLS * cw + (COLS + 1) * GAP
    grid_h = rows * ch + (rows + 1) * GAP
    band_h = LABEL_H + grid_h
    sheet_w = grid_w
    sheet_h = len(TS) * band_h + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (32, 32, 32))

    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(sheet)
    for band, t in enumerate(TS):
        print(f"  rendering {N_PER_T} samples at t={t}")
        # reset RNG per band so each band has identical text inputs (still
        # different visuals because the config differs)
        gen.rng.seed(0)
        samples = gen.render(t, N_PER_T)

        y_label = band * band_h + GAP
        d.rectangle([0, y_label, sheet_w, y_label + LABEL_H], fill=(48, 48, 48))
        d.text((10, y_label + 4), f"t = {t}", fill=(255, 255, 255), font=font)

        y_grid = y_label + LABEL_H
        for i, (im, _label, _bb) in enumerate(samples):
            r, c = divmod(i, COLS)
            big = im.resize((cw, ch), Image.NEAREST)
            x = GAP + c * (cw + GAP)
            y = y_grid + GAP + r * (ch + GAP)
            sheet.paste(big, (x, y))

    sheet.save(OUT)
    print(f"saved {OUT}  ({sheet_w}x{sheet_h})")


if __name__ == "__main__":
    main()
