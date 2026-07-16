"""v2 perception: scheme-agnostic per-character detection + recognition.

Architecture (cf. v2_plan.md):
  - **Detection** (per-character boxes): run CRAFT's CraftNet forward pass,
    grab the raw region_score heatmap, threshold, run connected components.
    This is the per-character signal CRAFT was designed to predict; we
    bypass the affinity-grouping that produces CRAFT's normal `boxes` output
    because that grouping merges captcha chars into word regions.
  - **Recognition** (char identity per box): run Qwen2.5-VL-7B-Instruct on
    the full image with a structured prompt asking for the captcha string
    only. We pair the i-th left-to-right CC box with the i-th char of Qwen's
    output. Qwen's own bboxes are ignored — small-image bbox quality is poor.

Public API:
    p = Perceiver()
    result = p.perceive(image_path)
    # result is None on failure or:
    #   {"chars": [(char, bbox)], "string": "...", "n_boxes": int,
    #    "qwen_raw": "...", "fail_reason": None}
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------- CRAFT setup ----------
import torchvision.models.vgg as _tvv  # noqa: E402
if not hasattr(_tvv, "model_urls"):
    _tvv.model_urls = {
        "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    }
from craft_text_detector import image_utils  # noqa: E402
from craft_text_detector.craft_utils import load_craftnet_model  # noqa: E402

# ---------- Qwen-VL setup ----------
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

QWEN_PROMPT = (
    "Read this CAPTCHA. It contains 5 colored, distorted, hand-drawn-style "
    "alphanumeric characters in a single row left-to-right. Reply with "
    "JSON only, no commentary:\n"
    '{"text": "XXXXX"}\n'
    "where XXXXX is the 5 characters in order. Be exact — case matters."
)


@dataclass
class PerceiveResult:
    chars: list[tuple[str, list[int]]]      # [(char, [x0,y0,x1,y1]), ...]
    string: str                             # concatenation of chars
    n_boxes: int                            # how many CC boxes CRAFT found
    qwen_raw: str                           # full Qwen output (for debug)
    fail_reason: Optional[str]              # None if usable


# ---------- CRAFT heatmap → connected components ----------

def _craft_region_heatmap(image_bgr: np.ndarray, craft_net,
                           long_size: int = 600,
                           cuda: bool = True) -> tuple[np.ndarray, float, int, int]:
    """Run CraftNet forward and return the raw region_score map upsampled to
    original image size. Also returns (effective_ratio, orig_h, orig_w)."""
    img_resized, target_ratio, _ = image_utils.resize_aspect_ratio(
        image_bgr, long_size, interpolation=cv2.INTER_LINEAR,
    )
    x = image_utils.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    if cuda:
        x = x.cuda()
    with torch.no_grad():
        y, _ = craft_net(x)
    score_text = y[0, :, :, 0].cpu().numpy()  # H_resized/2 × W_resized/2
    h_orig, w_orig = image_bgr.shape[:2]
    score_full = cv2.resize(score_text, (w_orig, h_orig),
                            interpolation=cv2.INTER_LINEAR)
    return score_full, float(target_ratio), h_orig, w_orig


def _heatmap_to_boxes(score: np.ndarray, threshold: float = 0.30,
                      min_area: int = 12) -> list[list[int]]:
    """Threshold the region heatmap, run connected components, return
    axis-aligned per-component bboxes sorted left-to-right."""
    binary = (score > threshold).astype(np.uint8) * 255
    n, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
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


# ---------- Qwen recognition ----------

def _parse_qwen_text(raw: str) -> Optional[str]:
    """Pull the captcha text out of Qwen's JSON output."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[:-3]
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        # try fallback bracket extraction
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b < 0:
            return None
        try:
            d = json.loads(s[a : b + 1])
        except json.JSONDecodeError:
            return None
    return d.get("text") if isinstance(d, dict) else None


# ---------- Perceiver ----------

class Perceiver:
    def __init__(self, n_chars_expected: int = 5,
                 craft_threshold: float = 0.30,
                 craft_min_area: int = 12,
                 craft_long_size: int = 600,
                 device: Optional[str] = None,
                 verbose: bool = True):
        self.n_chars_expected = n_chars_expected
        self.craft_threshold = craft_threshold
        self.craft_min_area = craft_min_area
        self.craft_long_size = craft_long_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if verbose:
            print(f"[perceive] loading CRAFT ...")
        t0 = time.time()
        self.craft_net = load_craftnet_model(cuda=(self.device == "cuda"))
        if verbose:
            print(f"  CRAFT ready ({time.time() - t0:.1f}s)")

        if verbose:
            print(f"[perceive] loading {QWEN_ID} (downloads ~14 GB on first run) ...")
        t0 = time.time()
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_ID,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_ID)
        if verbose:
            print(f"  Qwen ready ({time.time() - t0:.1f}s); "
                  f"VRAM={torch.cuda.memory_allocated() / 1e9:.1f} GB")

    @torch.inference_mode()
    def _qwen_read(self, image_path: Path) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": QWEN_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.device)
        gen = self.qwen.generate(**inputs, max_new_tokens=64, do_sample=False)
        out = gen[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            out, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _craft_boxes(self, image_path: Path) -> tuple[list[list[int]], np.ndarray]:
        bgr = image_utils.read_image(str(image_path))
        score, _, _, _ = _craft_region_heatmap(
            bgr, self.craft_net,
            long_size=self.craft_long_size,
            cuda=(self.device == "cuda"),
        )
        boxes = _heatmap_to_boxes(
            score,
            threshold=self.craft_threshold,
            min_area=self.craft_min_area,
        )
        return boxes, score

    def perceive(self, image_path: Path) -> PerceiveResult:
        boxes, score = self._craft_boxes(image_path)
        raw = self._qwen_read(image_path)
        text = _parse_qwen_text(raw)

        if text is None:
            return PerceiveResult([], "", len(boxes), raw,
                                  fail_reason="qwen_parse_fail")
        if len(boxes) != len(text):
            return PerceiveResult([], text, len(boxes), raw,
                                  fail_reason=f"box_count={len(boxes)}_vs_str_len={len(text)}")

        chars = [(text[i], boxes[i]) for i in range(len(text))]
        return PerceiveResult(chars=chars, string=text, n_boxes=len(boxes),
                              qwen_raw=raw, fail_reason=None)


# ---------- smoke ----------

def _smoke_main() -> None:
    from PIL import ImageDraw, ImageFont
    MODEL1 = Path(r"E:\pranav\model 1")
    SAMPLE_NAMES = [
        "captcha_screenshot_1690279209.png",
        "captcha_screenshot_1690279217.png",
        "captcha_screenshot_1690279267.png",
        "captcha_screenshot_1690279299.png",
        "captcha_screenshot_1690279422.png",
    ]
    OUT = ROOT / "mvp_results" / "v2_smoke"
    OUT.mkdir(parents=True, exist_ok=True)

    p = Perceiver(n_chars_expected=5)

    log_lines: list[str] = []
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for name in SAMPLE_NAMES:
        path = MODEL1 / "data" / "raw" / name
        if not path.exists():
            print(f"  skip missing: {name}")
            continue
        t0 = time.time()
        r = p.perceive(path)
        elapsed = time.time() - t0

        # overlay
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        boxes_only, _ = p._craft_boxes(path)  # cheap re-run for visualization
        for bb in boxes_only:
            draw.rectangle(bb, outline=(255, 0, 0), width=1)
        if r.chars:
            for ch, bb in r.chars:
                draw.text((bb[0], max(0, bb[1] - 14)), ch,
                          fill=(0, 255, 0), font=font)
        out_path = OUT / f"perceive_{name}"
        im.save(out_path)

        status = "OK" if r.fail_reason is None else f"FAIL({r.fail_reason})"
        line = (f"{name}  {status}  string={r.string!r}  "
                f"n_boxes={r.n_boxes}  qwen_raw={r.qwen_raw[:80]!r}  "
                f"({elapsed:.1f}s)")
        log_lines.append(line)
        print(f"  {line}")

    (OUT / "perceive_smoke_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nlog -> {OUT / 'perceive_smoke_log.txt'}")


if __name__ == "__main__":
    _smoke_main()
