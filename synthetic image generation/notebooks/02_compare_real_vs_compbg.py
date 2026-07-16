"""Visualise composite synth vs real CryptBB captchas side by side.

Output: notebooks/02_compbg_vs_real.png — 4 columns x 8 rows. Left half
is real captchas (4 cols x 4 rows = 16); right half is composite synth
(4 cols x 4 rows = 16), at 2x zoom. A divider strip separates them.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REAL = Path(r"E:\pranav\model 1\data\raw")
SYNTH = ROOT / "data" / "synth_compbg_test"
OUT = Path(__file__).resolve().parent / "02_compbg_vs_real.png"

N_PER_SIDE = 16
SEED = 7
ZOOM = 2


def main() -> None:
    real_paths = sorted(REAL.glob("*.png"))
    synth_paths = sorted(SYNTH.glob("*.png"))

    rnd = random.Random(SEED)
    real_chosen = rnd.sample(real_paths, N_PER_SIDE)
    synth_chosen = synth_paths[:N_PER_SIDE]

    cw, ch = 200 * ZOOM, 60 * ZOOM
    cols, rows = 4, 4
    g = 4
    side_w = cols * cw + (cols + 1) * g
    side_h = rows * ch + (rows + 1) * g
    divider_h = 32
    sheet = Image.new("RGB", (side_w, side_h * 2 + divider_h), (32, 32, 32))

    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    def paste_grid(paths, y_off):
        for i, p in enumerate(paths):
            r, c = divmod(i, cols)
            im = Image.open(p).convert("RGB")
            if im.size != (200, 60):
                im = im.resize((200, 60))
            im = im.resize((cw, ch), Image.NEAREST)
            sheet.paste(im, (g + c * (cw + g), y_off + g + r * (ch + g)))

    paste_grid(real_chosen, 0)
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, side_h, side_w, side_h + divider_h], fill=(64, 64, 64))
    draw.text((10, side_h + 6), "TOP: real CryptBB    BOTTOM: composite synth", fill=(255, 255, 255), font=font)
    paste_grid(synth_chosen, side_h + divider_h)

    sheet.save(OUT)
    print(f"saved {OUT}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
