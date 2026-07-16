"""Render N synthetic captchas at large display size for human inspection,
alongside randomly sampled real captchas at the same scale.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from render_synthetic import _load_fonts, _render_one, CHARSET, CAPTCHA_LEN


def main(n: int, scale: int, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fonts = _load_fonts()
    rng = random.Random(seed)

    # synth
    for i in range(n):
        text = "".join(rng.choices(CHARSET, k=CAPTCHA_LEN))
        img = _render_one(text, fonts, rng).convert("RGB")
        big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        big.save(out_dir / f"synth_{i:02d}_{text}.png")

    # real (random sample)
    real_paths = sorted(Path("data/raw").glob("captcha_screenshot_*.png"))
    sample = rng.sample(real_paths, n)
    for i, p in enumerate(sample):
        img = Image.open(p).convert("RGB")
        big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        big.save(out_dir / f"real_{i:02d}.png")

    print(f"wrote {2*n} images to {out_dir} at {200*scale}x{60*scale}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/_preview_large"))
    args = ap.parse_args()
    main(args.n, args.scale, args.seed, args.out)
