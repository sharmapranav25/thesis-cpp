"""Sanity script: draw per-char bboxes on a few rendered captchas to verify
the renderer's bbox extraction is correct. Outputs side-by-side composite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/_synth_bb_test"))
    ap.add_argument("--out", type=Path, default=Path("data/_synth_bb_test/_viz_bboxes.png"))
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    records = []
    with (args.src / "bboxes.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= args.n:
                break

    cols = 4
    rows = (len(records) + cols - 1) // cols
    iw, ih = 200, 60
    pad = 6
    canvas = Image.new("RGB", (cols * (iw + pad) + pad, rows * (ih + pad) + pad), "black")
    colors = [(255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 200, 0), (200, 0, 200)]

    for k, rec in enumerate(records):
        img = Image.open(args.src / rec["filename"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, (x0, y0, x1, y1) in enumerate(rec["bboxes"]):
            draw.rectangle([x0, y0, x1, y1], outline=colors[i], width=1)
        rx, ry = k % cols, k // cols
        canvas.paste(img, (pad + rx * (iw + pad), pad + ry * (ih + pad)))

    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
