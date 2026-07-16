"""Generate stylized synthetic CryptBB-like captchas via GlyphControl.

Pipeline:
  label string ("rghe3")
  → per-char glyph instructions (5 bboxes evenly spaced, random yaw)
  → GlyphControl inference (SD 2.0 + glyph control, 512x512 output)
  → crop a horizontal strip + resize to 200x60
  → save with label as filename

The control branch hard-wires the 5 characters into the conditioning so the
recognizer's labels are guaranteed correct (CycleGAN's failure mode was that
it didn't preserve content).

Usage (after GlyphControl venv + checkpoint are in place):
  python -m src.diffusion_render --n 50 --seed 0 --out data/diffusion_synthetic
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
GLYPH_ROOT = REPO / "third_party" / "GlyphControl"
sys.path.insert(0, str(GLYPH_ROOT))
sys.path.insert(0, str(REPO))  # so `from src.lora_train import ...` works

CHARSET = string.ascii_letters + string.digits  # 62 chars
CAPTCHA_LEN = 5
TARGET_W, TARGET_H = 200, 60

# CryptBB style cues. These steer the diffusion's _texture_ — colored speckle,
# chromatic shadows, vivid per-char hues — without altering glyph shape (that's
# what the control branch is for).
STYLE_PROMPT = (
    "captcha, multicolored speckle background, vivid letters with chromatic "
    "drop shadow, mixed fonts, slight rotation, web security image"
)
NEG_PROMPT = (
    "blurry, photograph, scene, person, sign, billboard, smooth background, "
    "single color text, watermark, signature"
)


def build_glyph_instructions(
    label: str,
    rng: random.Random,
    margin_x: float = 0.05,
    char_width_rel: float = 0.13,  # fraction of canvas width per char glyph
    y_rel: float = 0.42,           # vertical placement (top-left y, relative)
) -> dict:
    """Convert a 5-char label into the per-char bbox/yaw lists GlyphControl
    expects. We render at 512x512 internally; the box layout fakes the 200x60
    captcha aspect by clustering all chars in a horizontal strip near the
    image's vertical centre. We crop to 200x60 in post-processing."""
    n = len(label)
    usable = 1.0 - 2 * margin_x
    stride = usable / n
    rendered_txt_values = list(label)
    width_values = [char_width_rel] * n
    ratio_values = [0] * n  # 0 -> use optimal aspect inside box
    top_left_x_values = [
        round(margin_x + i * stride + rng.uniform(-0.005, 0.005), 4) for i in range(n)
    ]
    top_left_y_values = [
        round(y_rel + rng.uniform(-0.02, 0.02), 4) for _ in range(n)
    ]
    yaw_values = [int(rng.uniform(-15, 15)) for _ in range(n)]
    num_rows_values = [1] * n
    return {
        "rendered_txt_values": rendered_txt_values,
        "width_values": width_values,
        "ratio_values": ratio_values,
        "top_left_x_values": top_left_x_values,
        "top_left_y_values": top_left_y_values,
        "yaw_values": yaw_values,
        "num_rows_values": num_rows_values,
    }


def crop_to_captcha(square_arr: np.ndarray) -> Image.Image:
    """square_arr: (H, W, 3) uint8 from a square diffusion output (e.g. 512x512).
    Crop a horizontal band centred vertically, then resize to 200x60."""
    H, W = square_arr.shape[:2]
    band_h = int(W * TARGET_H / TARGET_W)  # band that, when resized, hits target ratio
    y0 = max(0, (H - band_h) // 2)
    band = square_arr[y0 : y0 + band_h, :, :3]
    img = Image.fromarray(band).resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return img


def load_render_tool(ckpt_path: Path, cfg_path: Path, save_memory: bool = True,
                     lora_path: Path | None = None):
    from cldm.hack import disable_verbosity, enable_sliced_attention
    from scripts.rendertext_tool import Render_Text, load_model_from_config
    from omegaconf import OmegaConf
    import torch
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    cfg = OmegaConf.load(cfg_path)
    model = load_model_from_config(cfg, str(ckpt_path), verbose=False)
    if lora_path is not None:
        from src.lora_train import attach_lora_to_unet, apply_lora_state_dict
        ckpt = torch.load(lora_path, map_location="cpu")
        attach_lora_to_unet(model.model.diffusion_model,
                            rank=ckpt.get("rank", 8),
                            alpha=ckpt.get("alpha", 16.0))
        n = apply_lora_state_dict(model.model.diffusion_model, ckpt["state_dict"])
        print(f"loaded LoRA from {lora_path} ({n} layers, step {ckpt.get('step')}, train loss {ckpt.get('loss', '?'):.4f})")
        model = model.cuda()  # ensure LoRA params are on GPU after wrapping
    return Render_Text(model, save_memory=save_memory)


def render_one(render_tool, label: str, rng: random.Random,
               ddim_steps: int, scale: float, strength: float, seed: int,
               image_resolution: int = 512) -> Image.Image:
    g = build_glyph_instructions(label, rng)
    # process_multi returns: [glyph_image, sample_0, sample_1, ...]
    results = render_tool.process_multi(
        rendered_txt_values=g["rendered_txt_values"],
        shared_prompt=STYLE_PROMPT,
        width_values=g["width_values"],
        ratio_values=g["ratio_values"],
        top_left_x_values=g["top_left_x_values"],
        top_left_y_values=g["top_left_y_values"],
        yaw_values=g["yaw_values"],
        num_rows_values=g["num_rows_values"],
        shared_num_samples=1,
        shared_image_resolution=image_resolution,
        shared_ddim_steps=ddim_steps,
        shared_guess_mode=False,
        shared_strength=strength,
        shared_scale=scale,
        shared_seed=seed,
        shared_eta=0,
        shared_a_prompt="",
        shared_n_prompt=NEG_PROMPT,
    )
    sample = results[1]  # results[0] is the rendered glyph image
    if isinstance(sample, Image.Image):
        sample = np.array(sample)
    return crop_to_captcha(sample)


def random_label(rng: random.Random) -> str:
    return "".join(rng.choices(CHARSET, k=CAPTCHA_LEN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "diffusion_synthetic")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=9.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ckpt", type=Path,
                    default=GLYPH_ROOT / "checkpoints" / "textcaps5K_epoch_40_model_ema_only.ckpt")
    ap.add_argument("--cfg", type=Path, default=GLYPH_ROOT / "configs" / "config.yaml")
    ap.add_argument("--lora", type=Path, default=None,
                    help="optional LoRA checkpoint to attach to the U-Net")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"loading GlyphControl from {args.ckpt}" + (f" + LoRA {args.lora}" if args.lora else ""))
    render_tool = load_render_tool(args.ckpt, args.cfg, save_memory=True, lora_path=args.lora)

    labels_path = args.out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for i in tqdm(range(args.n), desc="diffusion render"):
            label = random_label(rng)
            img = render_one(
                render_tool, label, rng,
                ddim_steps=args.ddim_steps,
                scale=args.scale,
                strength=args.strength,
                seed=args.seed + i,
            )
            fn = f"{i:06d}_{label}.png"
            img.save(args.out / fn)
            w.writerow([fn, label])
    print(f"wrote {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
