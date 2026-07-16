"""Side-by-side comparison: MyBB-fonts synth vs real CryptBB captchas.
Quick visual sanity check before committing to retrain on the new synth."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=Path("data/synth_mybb_v2"))
    ap.add_argument("--v3", type=Path, default=Path("data/synth_mybb_v3"))
    ap.add_argument("--real", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/_v2_v3_vs_real.png"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    def pick(d, n, want_200x60=True):
        cands = [
            p for p in d.iterdir()
            if p.suffix.lower() == ".png" and not p.name.startswith("_")
        ]
        if want_200x60:
            picked = []
            random.shuffle(cands)
            for p in cands:
                im = Image.open(p)
                if im.size == (200, 60):
                    picked.append(p)
                if len(picked) >= n:
                    break
            return picked
        return random.sample(cands, n)

    v2_files = pick(args.v2, args.n, want_200x60=True)
    v3_files = pick(args.v3, args.n, want_200x60=True)  # only 200x60 for fair compare
    real_files = pick(args.real, args.n, want_200x60=True)

    iw, ih = 200, 60
    pad = 8
    head = 30
    canvas_w = args.n * (iw + pad) + pad
    canvas_h = 3 * (ih + pad) + pad + head
    canvas = Image.new("RGB", (canvas_w, canvas_h), "black")

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), "v2 synth (MyBB fonts) | v3 synth (MyBB+supersample) | real CryptBB",
              fill="white", font=font)

    for row_i, files in enumerate([v2_files, v3_files, real_files]):
        for col, p in enumerate(files):
            img = Image.open(p).convert("RGB")
            if img.size != (iw, ih):
                img = img.resize((iw, ih))
            canvas.paste(img, (pad + col * (iw + pad), head + row_i * (ih + pad)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
