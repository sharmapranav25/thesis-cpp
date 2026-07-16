"""Render a grid of training crops with labels and case tags.
Sanity check: is the data the char-CNN sees usable for the case decision?"""
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

from PIL import Image, ImageDraw, ImageFont

from src.charcls.data import _crop_with_pad
from src.charcls.model import char_to_case


CASE_TAG = {0: "U", 1: "l", 2: "#"}  # UPPER / lower / digit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/synth_mybb_bb"))
    ap.add_argument("--out", type=Path, default=Path("data/synth_mybb_bb/_crop_grid.png"))
    ap.add_argument("--n", type=int, default=80, help="crops to show")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pad-frac", type=float, default=0.20)
    ap.add_argument("--jitter-frac", type=float, default=0.0,
                    help="set >0 to also show what training-time jitter looks like")
    ap.add_argument("--cell", type=int, default=64)
    args = ap.parse_args()

    random.seed(args.seed)
    rows: list[dict] = []
    with (args.src / "bboxes.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))

    # Pick random (row, char_idx) pairs
    sample = []
    while len(sample) < args.n:
        r = random.choice(rows)
        c_i = random.randrange(5)
        sample.append((r, c_i))

    cell = args.cell
    cols = 10
    rows_n = (len(sample) + cols - 1) // cols
    label_h = 14
    pad = 4
    canvas = Image.new(
        "RGB",
        (cols * (cell + pad) + pad, rows_n * (cell + label_h + pad) + pad),
        (32, 32, 32),
    )

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for k, (rec, c_i) in enumerate(sample):
        img = Image.open(args.src / rec["filename"]).convert("RGB")
        crop = _crop_with_pad(img, rec["bboxes"][c_i], args.pad_frac, args.jitter_frac)
        crop = crop.resize((cell, cell))
        ch = rec["label"][c_i]
        case_tag = CASE_TAG[char_to_case(ch)]
        rx = k % cols
        ry = k // cols
        cx = pad + rx * (cell + pad)
        cy = pad + ry * (cell + label_h + pad)
        canvas.paste(crop, (cx, cy))
        # label strip below the crop: char + case tag
        label_box = (cx, cy + cell, cx + cell, cy + cell + label_h)
        ImageDraw.Draw(canvas).rectangle(label_box, fill=(20, 20, 20))
        ImageDraw.Draw(canvas).text(
            (cx + 4, cy + cell + 1),
            f"{ch}  ({case_tag})",
            fill=(220, 220, 220),
            font=font,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
