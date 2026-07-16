"""CLI: render N labeled DarkBay synth captchas at a given difficulty.

Usage:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/generator/render.py \
      --t 0.7 --n 1000 --out data/mvp_samples_t0.7

Outputs in <out>:
  0000000.png … <n>.png   — RGB 200×50 captchas
  labels.csv              — filename,label
  bboxes.jsonl            — per-sample {filename,label,width,height,bboxes}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auto_tune.render_darkbay_v2 import H, W
from src.generator.parametric import ParametricGenerator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, required=True, help="difficulty in [0, 1]")
    ap.add_argument("--n", type=int, required=True, help="number of samples")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-dir", type=Path, default=None)
    args = ap.parse_args()

    if not (0.0 <= args.t <= 1.0):
        raise SystemExit(f"--t must be in [0, 1]; got {args.t}")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading generator (bg_dir={args.bg_dir or 'default'}) ...")
    gen = ParametricGenerator(bg_dir=args.bg_dir, seed=args.seed)
    print(f"  loaded {len(gen.bgs)} bgs; fonts cached lazily")

    t0 = time.time()
    with (args.out / "labels.csv").open("w", newline="") as flbl, \
         (args.out / "bboxes.jsonl").open("w") as fbb:
        w = csv.writer(flbl); w.writerow(["filename", "label"])
        for i in range(args.n):
            im, text, bboxes = gen.render_sample(args.t)
            fn = f"{i:07d}.png"
            im.save(args.out / fn)
            w.writerow([fn, text])
            fbb.write(json.dumps({"filename": fn, "label": text,
                                  "width": W, "height": H,
                                  "bboxes": bboxes, "t": args.t}) + "\n")
            if (i + 1) % 1000 == 0 or (i + 1) == args.n:
                rate = (i + 1) / (time.time() - t0)
                eta = (args.n - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{args.n}  rate={rate:.0f}/s  eta={eta:.0f}s")
    print(f"done; saved to {args.out}")


if __name__ == "__main__":
    main()
