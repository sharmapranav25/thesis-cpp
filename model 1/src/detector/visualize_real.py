"""Run trained detector on a sample of real captchas and visualise the
predicted boxes. Sanity check: did the detector generalise from the
PHP-rendered training distribution to real CryptBB captchas?"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ANACONDA_BIN = r"C:\ProgramData\Anaconda3\Library\bin"
if os.path.isdir(_ANACONDA_BIN) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ANACONDA_BIN)
    os.environ["PATH"] = _ANACONDA_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw

from src.detector.infer import load_detector, predict_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/detector/best.pt"))
    ap.add_argument("--src", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/_detector_real_viz.png"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_detector(args.ckpt)

    # collect predictions and pick a random subset
    random.seed(args.seed)
    all_preds = list(predict_dir(model, args.src))
    sample = random.sample(all_preds, min(args.n, len(all_preds)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    iw, ih = 200, 60
    pad = 6
    canvas = Image.new(
        "RGB",
        (cols * (iw + pad) + pad, rows * (ih + pad) + pad),
        "black",
    )
    colors = [(255, 50, 50), (40, 220, 40), (50, 130, 255), (255, 200, 0), (220, 50, 220)]

    for k, (name, boxes) in enumerate(sample):
        img = Image.open(args.src / name).convert("RGB").resize((iw, ih))
        draw = ImageDraw.Draw(img)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            draw.rectangle([x0, y0, x1, y1], outline=colors[i], width=1)
        rx, ry = k % cols, k // cols
        canvas.paste(img, (pad + rx * (iw + pad), pad + ry * (ih + pad)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
