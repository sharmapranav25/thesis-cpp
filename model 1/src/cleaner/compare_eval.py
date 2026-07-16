"""Stack raw eval vs cleaned eval side by side to inspect cleaner quality."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main(raw: Path, cleaned: Path, n: int, scale: int, out: Path) -> None:
    paths = sorted(p for p in raw.glob("*.png") if not p.name.startswith("_"))
    rng = random.Random(0)
    sample = rng.sample(paths, min(n, len(paths)))

    W, H = 200, 60
    pad = 6
    label_h = 18
    grid_w = W * 2 + pad * 3
    grid_h = label_h + (H + pad) * len(sample) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (235, 235, 235))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    d.text((pad, 2), "RAW", fill=(0, 0, 0), font=font)
    d.text((pad * 2 + W, 2), "CLEANED", fill=(0, 0, 0), font=font)

    for i, p in enumerate(sample):
        y = label_h + i * (H + pad) + pad
        raw_img = Image.open(p).convert("RGB").resize((W, H), Image.BICUBIC)
        clean_img = Image.open(cleaned / p.name).convert("RGB").resize((W, H), Image.BICUBIC)
        canvas.paste(raw_img, (pad, y))
        canvas.paste(clean_img, (pad * 2 + W, y))

    if scale > 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"saved {out} ({canvas.size})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/eval"))
    ap.add_argument("--cleaned", type=Path, default=Path("data/cleaned_eval"))
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("data/_cleaner_inspect/eval_compare.png"))
    args = ap.parse_args()
    main(args.raw, args.cleaned, args.n, args.scale, args.out)
