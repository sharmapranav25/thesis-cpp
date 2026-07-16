"""DINOv2-guided random search over RendererConfig.

Procedure:
  1. Encode all real DarkBay images once → mean feature vector R (anchor).
  2. For each of N_TRIALS random configs:
       a. Render BATCH_SIZE text-only RGBA via render_darkbay_v2
       b. Composite over BATCH_SIZE sampled real bgs (with light jitter + border)
       c. Encode synth batch → mean feature S
       d. Distance = 1 - cosine_similarity(R, S)
  3. Track best config; save winner to configs/darkbay_renderer.json.

Run:
  cd "E:/pranav/synthetic image generation"
  "E:/pranav/model 1/.venv/Scripts/python.exe" src/auto_tune/dinov2_search.py [--trials N] [--batch B]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor, Dinov2Model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.auto_tune.render_darkbay_v2 import (  # noqa: E402
    H, W, load_fonts_for_config, random_text, render_one,
)
from src.auto_tune.renderer_config import RendererConfig, sample_random  # noqa: E402

REAL_DIR = Path(r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled")
BG_DIR = ROOT / "data" / "darkbay_bg_lama"
OUT_CFG = ROOT / "configs" / "darkbay_renderer.json"
LOG_CSV = ROOT / "data" / "darkbay_search_log.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dinov2() -> tuple[Dinov2Model, AutoImageProcessor]:
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


@torch.no_grad()
def encode_batch(imgs: list[Image.Image], model: Dinov2Model,
                 proc: AutoImageProcessor, sub_batch: int = 64) -> torch.Tensor:
    feats = []
    for start in range(0, len(imgs), sub_batch):
        batch = imgs[start : start + sub_batch]
        inputs = proc(images=batch, return_tensors="pt")
        out = model(pixel_values=inputs["pixel_values"].to(DEVICE))
        feats.append(out.last_hidden_state[:, 0, :])  # CLS token
    return torch.cat(feats, 0)


def jitter_bg(bg: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        bg = ImageEnhance.Color(bg).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.7:
        bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.7:
        bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.9, 1.1))
    dx = rng.randint(-10, 10)
    dy = rng.randint(-3, 3)
    if dx or dy:
        arr = np.array(bg)
        arr = np.roll(arr, (dy, dx), axis=(0, 1))
        bg = Image.fromarray(arr)
    return bg


def render_and_compose(config: RendererConfig, bgs: list[Image.Image],
                       n: int, rng: random.Random) -> list[Image.Image]:
    """Render n text images with this config and composite over sampled bgs.
    Returns final RGB images (with 1-px black border, matching real DarkBay)."""
    fonts = load_fonts_for_config(config)
    out: list[Image.Image] = []
    for _ in range(n):
        text = random_text(rng)
        text_im, _ = render_one(text, config, fonts, rng)
        bg = jitter_bg(rng.choice(bgs).copy(), rng)
        if bg.size != (W, H):
            bg = bg.resize((W, H))
        composed = Image.alpha_composite(bg.convert("RGBA"), text_im).convert("RGB")
        arr = np.array(composed)
        arr[0, :] = 0; arr[-1, :] = 0; arr[:, 0] = 0; arr[:, -1] = 0
        out.append(Image.fromarray(arr))
    return out


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def main(trials: int, batch: int, seed: int) -> None:
    rng = random.Random(seed)

    print("loading DINOv2 ...")
    model, proc = load_dinov2()

    print(f"encoding {len(list(REAL_DIR.glob('*.png')))} real DarkBay ...")
    real_paths = sorted(REAL_DIR.glob("*.png"))
    real_imgs = [Image.open(p).convert("RGB") for p in real_paths]
    real_feats = encode_batch(real_imgs, model, proc)
    real_anchor = real_feats.mean(dim=0)
    print(f"  real anchor shape: {real_anchor.shape}")

    print(f"loading {len(list(BG_DIR.glob('*.png')))} extracted bgs ...")
    bgs = [Image.open(p).convert("RGB") for p in sorted(BG_DIR.glob("*.png"))]

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    best_dist = float("inf")
    best_config: RendererConfig | None = None
    log_lines = ["trial,dist,is_best,config_summary"]

    print(f"running random search: {trials} trials × batch {batch} ...")
    t0 = time.time()
    for trial in range(trials):
        config = sample_random(rng)
        try:
            synth = render_and_compose(config, bgs, batch, rng)
            synth_feats = encode_batch(synth, model, proc)
            synth_anchor = synth_feats.mean(dim=0)
            dist = cosine_distance(real_anchor, synth_anchor)
        except Exception as e:
            print(f"  trial {trial}: skipped ({type(e).__name__}: {e})")
            continue
        is_best = dist < best_dist
        if is_best:
            best_dist = dist
            best_config = config
            print(f"  trial {trial}: dist={dist:.5f}  [NEW BEST]  fonts={config.font_indices} "
                  f"size=[{config.size_min},{config.size_max}] rot=±{config.rot_max:.1f} "
                  f"box_p={config.box_prob:.2f}")
        elif (trial + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (trial + 1) / elapsed
            eta = (trials - trial - 1) / rate if rate else 0
            print(f"  trial {trial}: dist={dist:.5f}  best={best_dist:.5f}  eta={eta:.0f}s")
        summary = (f"fonts={config.font_indices} size=[{config.size_min},{config.size_max}] "
                   f"rot=±{config.rot_max:.1f} box_p={config.box_prob:.2f}")
        log_lines.append(f"{trial},{dist:.5f},{int(is_best)},\"{summary}\"")

    if best_config is None:
        print("no successful trials")
        return

    LOG_CSV.write_text("\n".join(log_lines) + "\n")
    best_config.to_json(OUT_CFG)
    print(f"\n=== search complete ===")
    print(f"best dist = {best_dist:.5f}")
    print(f"best config saved to {OUT_CFG}")
    print(f"log saved to {LOG_CSV}")
    print(f"\nbest config:")
    for k, v in vars(best_config).items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.trials, args.batch, args.seed)
