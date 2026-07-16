"""Side-by-side comparison: real CryptBB captchas vs synthetic preview.
Saves data/_compare_real_vs_synth.png so a human can eyeball the gap the GAN must close.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    real_dir = Path("data/raw")
    synth_dir = Path("data/synthetic/_preview")
    assert real_dir.exists() and synth_dir.exists()

    real_paths = [p for p in sorted(real_dir.glob("captcha_screenshot_*.png"))]
    synth_paths = sorted(synth_dir.glob("*.png"))
    n = min(20, len(real_paths), len(synth_paths))

    rng = random.Random(0)
    real_sample = rng.sample(real_paths, n)
    synth_sample = rng.sample(synth_paths, n)

    W, H = 200, 60
    pad = 8
    label_h = 22
    cols = 2
    grid_w = W * cols + pad * (cols + 1)
    grid_h = label_h + (H + pad) * n + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    d.text((pad, 4), "REAL", fill=(0, 0, 0), font=font)
    d.text((pad * 2 + W, 4), "SYNTH", fill=(0, 0, 0), font=font)

    for i, (rp, sp) in enumerate(zip(real_sample, synth_sample)):
        y = label_h + i * (H + pad) + pad
        canvas.paste(Image.open(rp).convert("RGB"), (pad, y))
        canvas.paste(Image.open(sp).convert("RGB"), (pad * 2 + W, y))

    out = Path("data/_compare_real_vs_synth.png")
    canvas.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
