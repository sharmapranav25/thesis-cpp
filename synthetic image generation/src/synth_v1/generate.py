"""synth_v1 CLI: render N labeled synth captchas for a scheme.

Reads the per-scheme char bank built by build_bank.py + the LaMa bg pool,
writes RGB PNGs + labels.csv + bboxes.jsonl + a QA sheet.

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v1.generate \\
      --scheme cryptbb --n 1000 --out data/synth_v1/cryptbb_1k

QA sheet (mvp_results/synth_v1_<scheme>_qa.png) compares 4×8 of the rendered
synth against 4×8 of real samples drawn from the scheme's raw_dir, so the
match (or mismatch) is visible at a glance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synth_v1.compose import Composer
from src.synth_v1.font_bank import CharBank
from src.synth_v1.scheme import SchemeSpec, get

ROOT = Path(__file__).resolve().parents[2]


def write_dataset(comp: Composer, n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"
    bb_path = out_dir / "bboxes.jsonl"
    t0 = time.time()
    with csv_path.open("w", newline="") as flbl, bb_path.open("w") as fbb:
        wcsv = csv.writer(flbl)
        wcsv.writerow(["filename", "label"])
        for i in range(n):
            im, label, bb = comp.render_one()
            fn = f"{i:07d}.png"
            im.save(out_dir / fn)
            wcsv.writerow([fn, label])
            fbb.write(json.dumps({
                "filename": fn, "label": label,
                "width": comp.scheme.W, "height": comp.scheme.H,
                "bboxes": bb,
            }) + "\n")
            if (i + 1) % 250 == 0 or (i + 1) == n:
                rate = (i + 1) / max(1e-9, time.time() - t0)
                eta = (n - i - 1) / rate
                print(f"  {i + 1}/{n}  rate={rate:.0f}/s  eta={eta:.0f}s")
    print(f"wrote {n} images + labels.csv + bboxes.jsonl to {out_dir}")


def make_qa_sheet(scheme: SchemeSpec, comp: Composer, qa_path: Path,
                  rows: int = 4, cols: int = 8) -> None:
    """Synth grid stacked above real grid, both rows×cols, scheme-sized cells."""
    W, H = scheme.W, scheme.H
    pad = 6
    label_h = 22
    sheet_w = cols * W + (cols + 1) * pad
    sheet_h = label_h + rows * H + (rows + 1) * pad + label_h + rows * H + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # synth top
    draw.text((pad, 2), f"synth_v1 ({scheme.name})  rows×cols={rows}×{cols}",
              font=font, fill=(220, 220, 220))
    y = label_h
    rng = random.Random(0)
    for r in range(rows):
        for c in range(cols):
            im, _, _ = comp.render_one()
            x = pad + c * (W + pad)
            sheet.paste(im, (x, y + r * (H + pad) + pad))

    # real bottom
    real_paths = [p for p in sorted(scheme.raw_dir.glob("*.png"))
                  if not p.name.startswith("_")]
    rng.shuffle(real_paths)
    real_paths = real_paths[: rows * cols]
    real_y = label_h + rows * H + (rows + 1) * pad
    draw.text((pad, real_y), f"real ({scheme.raw_dir.name})",
              font=font, fill=(220, 220, 220))
    real_y += label_h
    for i, p in enumerate(real_paths):
        r = i // cols
        c = i % cols
        im = Image.open(p).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H))
        x = pad + c * (W + pad)
        sheet.paste(im, (x, real_y + r * (H + pad) + pad))

    qa_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(qa_path)
    print(f"qa sheet -> {qa_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=5,
                    help="min crops per char to include in bank sampler")
    ap.add_argument("--qa-only", action="store_true",
                    help="only build the qa sheet, do not write the dataset")
    args = ap.parse_args()

    scheme = get(args.scheme)
    print(f"scheme   {scheme.name}")
    print(f"bank     {scheme.bank_dir}")
    print(f"bg       {scheme.bg_dir}")
    print(f"out      {args.out}")
    print(f"n        {args.n}  seed={args.seed}")

    bank = CharBank(scheme.bank_dir, min_count=args.min_count)
    print(f"bank charset ({len(bank.available_charset())}): {bank.available_charset()}")
    comp = Composer(scheme, bank, seed=args.seed)
    print(f"composer charset ({len(comp.charset)}): {comp.charset}")

    qa_path = ROOT / "mvp_results" / f"synth_v1_{scheme.name}_qa.png"
    make_qa_sheet(scheme, comp, qa_path)

    if args.qa_only:
        return

    write_dataset(comp, args.n, args.out)


if __name__ == "__main__":
    main()
