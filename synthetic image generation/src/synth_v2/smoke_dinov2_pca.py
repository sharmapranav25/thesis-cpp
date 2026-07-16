"""DINOv2 patch features + PCA — second feasibility test for char segmentation.

Hypothesis: instead of using DINOv2's CLS-to-patch attention (which is too
diffuse), use the *patch features* directly (last hidden state minus CLS),
take the first principal component across patches, threshold. This is the
core idea of TokenCut/MaskCut — patches representing the same semantic
object cluster on the first PC; thresholding separates fg from bg.

If that doesn't isolate characters, DINOv2 isn't the right detector tool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoImageProcessor, Dinov2Model  # noqa: E402

MODEL1 = Path(r"E:\pranav\model 1")
OUT = ROOT / "mvp_results" / "v2_smoke"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_NAMES = [
    "captcha_screenshot_1690279209.png",
    "captcha_screenshot_1690279217.png",
    "captcha_screenshot_1690279267.png",
    "captcha_screenshot_1690279299.png",
    "captcha_screenshot_1690279422.png",
]

TARGET_W, TARGET_H = 504, 168
PATCH = 14
GRID_W, GRID_H = TARGET_W // PATCH, TARGET_H // PATCH


@torch.inference_mode()
def patch_features(model, proc, pil_img: Image.Image, device: str) -> np.ndarray:
    """Returns [GRID_H, GRID_W, dim] of post-LN patch features."""
    img = pil_img.resize((TARGET_W, TARGET_H), Image.BICUBIC)
    inputs = proc(images=img, return_tensors="pt", do_resize=False,
                  do_center_crop=False).to(device)
    out = model(**inputs)
    last_hidden = out.last_hidden_state[0]   # [n_tokens, dim]
    patches = last_hidden[1:]                 # drop CLS
    n_patches, dim = patches.shape
    if n_patches != GRID_H * GRID_W:
        raise RuntimeError(
            f"unexpected n_patches={n_patches}, expected {GRID_H * GRID_W}"
        )
    return patches.view(GRID_H, GRID_W, dim).cpu().float().numpy()


def first_pc_map(features: np.ndarray) -> np.ndarray:
    """First principal component as a HxW saliency map at patch grid resolution."""
    h, w, d = features.shape
    flat = features.reshape(h * w, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # SVD-based first PC (avoids sklearn dependency)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    pc1 = u[:, 0] * s[0]                      # [n_patches]
    # sign convention: more variance on the bigger side; flip if needed
    if (pc1 > 0).sum() > (pc1 < 0).sum():
        pc1 = -pc1
    pc1_grid = pc1.reshape(h, w)
    pc1_full = cv2.resize(pc1_grid.astype(np.float32),
                          (TARGET_W, TARGET_H),
                          interpolation=cv2.INTER_LINEAR)
    pc1_full = (pc1_full - pc1_full.min()) / (pc1_full.max() - pc1_full.min() + 1e-9)
    return pc1_full


def boxes_from_heatmap(heat: np.ndarray, threshold: float = 0.55,
                       min_area: int = 100) -> list[list[int]]:
    binary = (heat > threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for cid in range(1, n):
        if int(stats[cid, cv2.CC_STAT_AREA]) < min_area:
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + w, y + h])
    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)
    return boxes


def render_overlay(im: Image.Image, heat: np.ndarray,
                   boxes: list[list[int]]) -> Image.Image:
    upscaled = im.resize((TARGET_W, TARGET_H), Image.BICUBIC).convert("RGB")
    heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended = Image.blend(upscaled, Image.fromarray(heat_color), 0.55)
    overlay = upscaled.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, bb in enumerate(boxes):
        draw.rectangle(bb, outline=(0, 255, 0), width=2)
        draw.text((bb[0], max(0, bb[1] - 16)), str(i + 1),
                  fill=(0, 255, 0), font=font)
    pad = 6
    sheet = Image.new("RGB", (TARGET_W, TARGET_H * 3 + pad * 2), (32, 32, 32))
    sheet.paste(upscaled, (0, 0))
    sheet.paste(blended, (0, TARGET_H + pad))
    sheet.paste(overlay, (0, 2 * (TARGET_H + pad)))
    return sheet


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dino-pca] loading facebook/dinov2-base on {device} ...")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    model = Dinov2Model.from_pretrained(
        "facebook/dinov2-base",
        torch_dtype=torch.float32,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[dino-pca] ready")

    log_lines: list[str] = []
    for name in SAMPLE_NAMES:
        path = MODEL1 / "data" / "raw" / name
        if not path.exists():
            print(f"  skip missing: {name}")
            continue
        im = Image.open(path).convert("RGB")
        feats = patch_features(model, proc, im, device)
        heat = first_pc_map(feats)
        boxes = boxes_from_heatmap(heat)
        sheet = render_overlay(im, heat, boxes)
        sheet.save(OUT / f"dinopca_{name}")
        line = f"{name}  pc1_boxes={len(boxes)}"
        log_lines.append(line)
        print(f"  {line}")

    (OUT / "dinov2_pca_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
