"""DINOv2 self-attention as per-character detector — feasibility test.

Hypothesis: DINOv2's last-layer self-attention from the CLS token to patch
tokens highlights semantically distinct regions in an image (this is the
LOST/TokenCut/MaskCut emergent behavior). For a captcha, the "objects" are
the individual characters, so the attention map should produce N hot
regions for N characters — without any captcha-specific training.

This script:
  1. Loads facebook/dinov2-base (more attention heads + capacity than -small).
  2. Upscales CryptBB images to a clean multiple of the patch size (14×14 →
     504×168 = 36×12 patches).
  3. Extracts attention from the final block, CLS→patch direction.
  4. Renders three views per image:
       a) the input upscaled image
       b) the per-head averaged attention (one map)
       c) connected-components overlay on the threshold-binarised map
  5. Reports the box count vs the ground-truth 5 chars.

If we see ~5 distinct hot regions on most samples, DINOv2 attention is a
real per-character detector slot for v2. If the map is diffuse / coarse /
attends to bg, this approach is dead.

Run:
    cd "E:/pranav/synthetic image generation"
    HF_HOME=E:/hf_cache HF_HUB_CACHE=E:/hf_cache/hub \
        "E:/pranav/model 1/.venv/Scripts/python.exe" \
            -m src.synth_v2.smoke_dinov2_attn
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

# DINOv2 patch size = 14. Choose target size as a multiple — 504×168 gives
# 36 columns × 12 rows of patches, plenty of granularity for 5 chars wide.
TARGET_W, TARGET_H = 504, 168
PATCH = 14
GRID_W, GRID_H = TARGET_W // PATCH, TARGET_H // PATCH  # 36, 12


def load_model(device: str):
    print(f"[dino] loading facebook/dinov2-base on {device} ...")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    model = Dinov2Model.from_pretrained(
        "facebook/dinov2-base",
        torch_dtype=torch.float32,
        attn_implementation="eager",   # required to get attention probs out
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[dino] ready; n_heads={model.config.num_attention_heads}")
    return model, proc


@torch.inference_mode()
def attention_heatmap(model, proc, pil_img: Image.Image, device: str) -> np.ndarray:
    """Return a HxW float32 attention map at TARGET_W × TARGET_H resolution.
    Map = average over heads of CLS-to-patch attention from the last layer."""
    img = pil_img.resize((TARGET_W, TARGET_H), Image.BICUBIC)
    inputs = proc(images=img, return_tensors="pt", do_resize=False,
                  do_center_crop=False).to(device)
    out = model(**inputs, output_attentions=True)
    last_attn = out.attentions[-1][0]            # [n_heads, n_tokens, n_tokens]
    cls_to_patches = last_attn[:, 0, 1:]         # [n_heads, n_patches]
    n_heads = cls_to_patches.shape[0]
    n_patches = cls_to_patches.shape[1]
    if n_patches != GRID_W * GRID_H:
        raise RuntimeError(
            f"unexpected n_patches={n_patches}, expected {GRID_W * GRID_H}"
        )
    head_maps = cls_to_patches.view(n_heads, GRID_H, GRID_W)
    head_maps = head_maps.cpu().float().numpy()
    avg = head_maps.mean(axis=0)
    # rescale to [0,1] for visualization
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-9)
    full = cv2.resize(avg, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return full, head_maps  # head_maps so we can also pick best head


def boxes_from_heatmap(heat: np.ndarray, threshold: float = 0.45,
                       min_area: int = 80) -> list[list[int]]:
    binary = (heat > threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[list[int]] = []
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
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
    """Original (upscaled) | heatmap | boxes overlay, vertically stacked."""
    upscaled = im.resize((TARGET_W, TARGET_H), Image.BICUBIC).convert("RGB")

    heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heat_pil = Image.fromarray(heat_color)
    blended = Image.blend(upscaled, heat_pil, 0.55)

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
    model, proc = load_model(device)

    log_lines: list[str] = []
    for name in SAMPLE_NAMES:
        path = MODEL1 / "data" / "raw" / name
        if not path.exists():
            print(f"  skip missing: {name}")
            continue
        im = Image.open(path).convert("RGB")
        heat, head_maps = attention_heatmap(model, proc, im, device)
        boxes = boxes_from_heatmap(heat)

        sheet = render_overlay(im, heat, boxes)
        out_path = OUT / f"dinov2_{name}"
        sheet.save(out_path)

        # Also save best-of-heads — sometimes one head segments better than mean.
        best_head_score = -1.0
        best_head_idx = 0
        for h in range(head_maps.shape[0]):
            m = head_maps[h]
            m = (m - m.min()) / (m.max() - m.min() + 1e-9)
            full = cv2.resize(m, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            bb = boxes_from_heatmap(full)
            # crude: score by how close box count is to 5
            score = -abs(len(bb) - 5)
            if score > best_head_score:
                best_head_score = score
                best_head_idx = h
        m = head_maps[best_head_idx]
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        best_heat = cv2.resize(m, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        best_boxes = boxes_from_heatmap(best_heat)
        best_sheet = render_overlay(im, best_heat, best_boxes)
        best_sheet.save(OUT / f"dinov2_besthead_{name}")

        line = (f"{name}  avg_boxes={len(boxes)}  "
                f"best_head={best_head_idx} best_boxes={len(best_boxes)}")
        log_lines.append(line)
        print(f"  {line}")

    (OUT / "dinov2_attn_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nlog -> {OUT / 'dinov2_attn_log.txt'}")


if __name__ == "__main__":
    main()
