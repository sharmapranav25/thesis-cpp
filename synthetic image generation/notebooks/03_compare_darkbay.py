"""Visual diff: real DarkBay vs my synth."""
from __future__ import annotations
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REAL = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
SYNTH = ROOT / "data" / "darkbay_synth_diag"
OUT = Path(__file__).resolve().parent / "03_darkbay_diff.png"

N = 16
ZOOM = 3
W, H = 200, 50


def main() -> None:
    real_paths = sorted(REAL.glob("*.png"))
    rng = random.Random(7)
    real_chosen = rng.sample(real_paths, N)
    synth_chosen = sorted(SYNTH.glob("*.png"))[:N]

    cw, ch = W * ZOOM, H * ZOOM
    cols, rows = 2, N // 2
    g = 4
    side_w = cols * cw + (cols + 1) * g
    side_h = rows * ch + (rows + 1) * g
    div_h = 32
    sheet = Image.new("RGB", (side_w, side_h * 2 + div_h), (32, 32, 32))

    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    def grid(paths, y_off):
        for i, p in enumerate(paths):
            r, c = divmod(i, cols)
            im = Image.open(p).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H))
            im = im.resize((cw, ch), Image.NEAREST)
            sheet.paste(im, (g + c * (cw + g), y_off + g + r * (ch + g)))

    grid(real_chosen, 0)
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, side_h, side_w, side_h + div_h], fill=(64, 64, 64))
    d.text((10, side_h + 6), "TOP: real DarkBay    BOTTOM: my synth", fill=(255, 255, 255), font=font)
    grid(synth_chosen, side_h + div_h)

    sheet.save(OUT)
    print(f"saved {OUT}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
