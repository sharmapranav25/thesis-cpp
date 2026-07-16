"""Phase 0 smoke: CRAFT on 5 CryptBB images → per-character bbox overlay.

Goal: confirm CRAFT loads, runs, and emits boxes that are roughly
per-character (not per-word/line) on captcha-style text. If CRAFT collapses
all 5 chars into one region, we'll know up front and switch detector.

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" -m src.synth_v2.smoke_craft
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MODEL1 = Path(r"E:\pranav\model 1")
sys.path.insert(0, str(ROOT))

# craft-text-detector pypi pkg uses torchvision's pre-0.13 `model_urls` API.
# Restore it before the CRAFT import so the VGG16-BN backbone can find weights.
import torchvision.models.vgg as _tvv  # noqa: E402
if not hasattr(_tvv, "model_urls"):
    _tvv.model_urls = {
        "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    }
from craft_text_detector import Craft  # type: ignore  # noqa: E402

RAW = MODEL1 / "data" / "raw"
OUT = ROOT / "mvp_results" / "v2_smoke"
OUT.mkdir(parents=True, exist_ok=True)

# 5 sample images we used through v1 development; visible CryptBB style.
SAMPLE_NAMES = [
    "captcha_screenshot_1690279209.png",
    "captcha_screenshot_1690279217.png",
    "captcha_screenshot_1690279267.png",
    "captcha_screenshot_1690279299.png",
    "captcha_screenshot_1690279422.png",
]


def main() -> None:
    print("loading CRAFT (downloads weights on first run, ~80MB) ...")
    # Aggressive per-character settings:
    #   - high link_threshold so chars don't merge into a word region
    #   - low low_text so faint stroke pixels still register
    #   - refiner=False to skip the affinity-merging refinement step
    #   - long_size=600 to upscale 200×60 captchas (gives CRAFT finer regions)
    craft = Craft(output_dir=str(OUT / "_craft_dump"), crop_type="poly",
                  cuda=True, refiner=False, long_size=600,
                  text_threshold=0.4, link_threshold=0.95, low_text=0.2)
    print("CRAFT loaded")

    for name in SAMPLE_NAMES:
        path = RAW / name
        if not path.exists():
            print(f"  skip missing: {name}")
            continue
        prediction_result = craft.detect_text(str(path))
        if name == SAMPLE_NAMES[0]:
            print(f"  prediction keys: {sorted(prediction_result.keys())}")
        # craft_text_detector returns dict with keys: boxes, boxes_as_ratios,
        # polys_as_ratios, heatmaps, text_crop_paths, times.
        boxes = prediction_result.get("boxes", [])
        # `boxes` is list of 4-point polygons (4 x 2 array each).
        # Convert to axis-aligned for visualization.
        im = Image.open(path).convert("RGB").copy()
        draw = ImageDraw.Draw(im)
        for poly in boxes:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            draw.rectangle([min(xs), min(ys), max(xs), max(ys)],
                           outline=(255, 0, 0), width=1)
        out_path = OUT / f"craft_{name}"
        im.save(out_path)
        print(f"  {name}: {len(boxes)} boxes -> {out_path.name}")
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()


if __name__ == "__main__":
    main()
