"""Generate text-only RGBA images mimicking DarkBay's rendering.

Spec (inferred from observation):
  - 200x50 RGBA, transparent bg
  - 6 chars, charset = digits + uppercase A-Z minus 'I'
  - Per char: ~50% chance of "boxed" style (white text on dark gray rect)
              ~50% chance of "plain" style (dark text, no box)
  - Slight rotation (~±12°) and small position jitter
  - Bold serif-ish font (Times Bold, Georgia Bold, etc.)

Inputs:
  --out DIR     output dir (creates if missing)
  --n N         how many samples
  --seed S      RNG seed

Outputs:
  <out>/<i:07d>.png   RGBA text on transparent
  <out>/labels.csv    filename,label
  <out>/bboxes.jsonl  filename,label,width,height,bboxes (per-char rotated-aware
                      axis-aligned bbox)

Run:
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/composite/render_darkbay_text.py \
      --out data/darkbay_text_only --n 100000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CHARSET = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no 'I'
N_CHARS = 6
W, H = 200, 50

WIN_FONTS = "C:/Windows/Fonts"
FONT_FILES = [
    "timesbd.ttf",      # Times New Roman Bold
    "georgiab.ttf",     # Georgia Bold
    "cambriab.ttf",     # Cambria Bold
    "consolab.ttf",     # Consolas Bold (mono)
]


def load_fonts(size_range=(22, 30)) -> list:
    fonts = {}
    for fname in FONT_FILES:
        path = Path(WIN_FONTS) / fname
        if not path.exists():
            continue
        for sz in range(size_range[0], size_range[1] + 1):
            fonts.setdefault(sz, []).append(ImageFont.truetype(str(path), sz))
    return fonts


def render_char(canvas: Image.Image, ch: str, cx: float, cy: float,
                font: ImageFont.FreeTypeFont, rng: random.Random,
                boxed: bool) -> tuple[int, int, int, int]:
    """Draw one rotated char (boxed or plain) onto canvas. Returns its
    axis-aligned bbox in canvas coords."""
    angle = rng.uniform(-12, 12)
    # Render char on its own transparent mini-canvas, then rotate, then paste
    bbox = font.getbbox(ch)
    cw = bbox[2] - bbox[0]
    chh = bbox[3] - bbox[1]
    # padding for rotation room + box
    pad = 8
    mini = Image.new("RGBA", (cw + 2 * pad, chh + 2 * pad), (0, 0, 0, 0))
    mdraw = ImageDraw.Draw(mini)
    # Box decoration
    if boxed:
        box_color = (rng.randint(60, 95), rng.randint(60, 95), rng.randint(60, 95), 255)
        # box around the char with small inner padding
        bx0 = pad - rng.randint(1, 3)
        by0 = pad - rng.randint(1, 3)
        bx1 = pad + cw + rng.randint(1, 3)
        by1 = pad + chh + rng.randint(1, 3)
        mdraw.rectangle([bx0, by0, bx1, by1], fill=box_color)
        text_color = (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255), 255)
    else:
        text_color = (rng.randint(40, 90), rng.randint(40, 90), rng.randint(40, 90), 255)
    mdraw.text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=text_color)
    rot = mini.rotate(angle, resample=Image.BICUBIC, expand=True)
    rw, rh = rot.size
    px = int(round(cx - rw / 2))
    py = int(round(cy - rh / 2))
    canvas.alpha_composite(rot, (px, py))
    return (max(0, px), max(0, py), min(W - 1, px + rw), min(H - 1, py + rh))


def render_one(text: str, fonts_by_size: dict, rng: random.Random):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spacing = W / N_CHARS
    bboxes = []
    for i, ch in enumerate(text):
        size = rng.randint(22, 28)
        font = rng.choice(fonts_by_size[size])
        cx = spacing * (i + 0.5) + rng.uniform(-2, 2)
        cy = H / 2 + rng.uniform(-2, 2)
        boxed = rng.random() < 0.5
        bbox = render_char(canvas, ch, cx, cy, font, rng, boxed)
        bboxes.append(list(bbox))
    return canvas, bboxes


def main(out_dir: Path, n: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fonts_by_size = load_fonts()
    if not fonts_by_size:
        raise RuntimeError(f"no fonts loaded from {WIN_FONTS}")
    print(f"loaded fonts at sizes: {sorted(fonts_by_size.keys())}")

    rng = random.Random(seed)
    labels_csv = out_dir / "labels.csv"
    bboxes_jsonl = out_dir / "bboxes.jsonl"

    t0 = time.time()
    with labels_csv.open("w", newline="") as flbl, bboxes_jsonl.open("w") as fbb:
        w = csv.writer(flbl)
        w.writerow(["filename", "label"])
        for i in range(n):
            text = "".join(rng.choice(CHARSET) for _ in range(N_CHARS))
            im, bboxes = render_one(text, fonts_by_size, rng)
            fn = f"{i:07d}.png"
            im.save(out_dir / fn)
            w.writerow([fn, text])
            fbb.write(json.dumps({
                "filename": fn, "label": text, "width": W, "height": H, "bboxes": bboxes,
            }) + "\n")
            if (i + 1) % 1000 == 0 or (i + 1) == n:
                rate = (i + 1) / (time.time() - t0)
                eta = (n - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{n}  rate={rate:.0f}/s  eta={eta:.0f}s")
    print(f"done; saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.out, args.n, args.seed)
