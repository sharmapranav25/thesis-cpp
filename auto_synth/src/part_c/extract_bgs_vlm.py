"""Part C v3 — VLM-driven (Qwen2.5-VL) bg extraction.

Pipeline per image:
  1. Qwen2.5-VL-3B prompted to "find all characters" → JSON of {bbox, label}
  2. Parse JSON, get axis-aligned bboxes
  3. MobileSAM with each bbox as box-prompt → clean masks
  4. Union all masks + dilate
  5. LaMa inpaints over the union mask → clean bg

Bonus: the VLM also returns the char *identity*, which we save as a side-output
for use in alphabet discovery (potentially replacing Part A entirely).

Model: Qwen2.5-VL-3B-Instruct (~3.8B params, ~7.5 GB fp16)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.morphology import dilation, square


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SAM_CKPT = "E:/pranav/auto_synth/configs/mobile_sam.pt"


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def parse_vlm_output(text: str) -> list[dict]:
    """Extract list of {bbox_2d: [x1,y1,x2,y2], label: str} from VLM output.

    Qwen tends to output a JSON array, sometimes wrapped in ```json fences.
    Handle both forms.
    """
    # Strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        body = m.group(1)
    else:
        # Try raw JSON array
        m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        body = m.group(0) if m else None
    if not body:
        return []
    try:
        data = json.loads(body)
    except Exception:
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d") or item.get("bbox")
        label = item.get("label") or item.get("text") or item.get("character") or ""
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        out.append({"bbox": (x1, y1, x2, y2), "label": str(label)})
    return out


def main(images_dir: Path, out_dir: Path, max_images: int, dilate_px: int):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from mobile_sam import sam_model_registry, SamPredictor
    from simple_lama_inpainting import SimpleLama

    out_dir.mkdir(parents=True, exist_ok=True)
    vlm_log = out_dir / "vlm_predictions.jsonl"

    print(f"[part_c_vlm] loading Qwen2.5-VL-3B...")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    print(f"[part_c_vlm] loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=SAM_CKPT).cuda().eval()
    predictor = SamPredictor(sam)

    print(f"[part_c_vlm] loading LaMa...")
    lama = SimpleLama()

    paths = sorted(list_images(images_dir))
    if max_images:
        paths = paths[:max_images]
    print(f"[part_c_vlm] processing {len(paths)} images")

    detection_text = (
        "Detect all characters (letters, digits) in this captcha. "
        "Output a JSON list where each item is "
        '{"bbox_2d": [x1, y1, x2, y2], "label": "<the character>"}. '
        "Use exact pixel coordinates of each character's tight bounding box. "
        "Return ONLY the JSON array."
    )

    done = 0
    with vlm_log.open("w", encoding="utf-8") as flog:
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                W, H = img.size
                arr = np.array(img)

                # Qwen2.5-VL inference
                msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": detection_text},
                    ],
                }]
                text_prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = proc(text=[text_prompt], images=[img], padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    gen = vlm.generate(**inp, max_new_tokens=384, do_sample=False)
                vlm_text = proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
                preds = parse_vlm_output(vlm_text)

                flog.write(json.dumps({
                    "src": p.name,
                    "raw": vlm_text,
                    "preds": preds,
                }) + "\n")

                # SAM box-prompt with VLM bboxes
                mask = np.zeros((H, W), dtype=bool)
                if preds:
                    bboxes = []
                    for pr in preds:
                        x1, y1, x2, y2 = pr["bbox"]
                        x1 = max(0, min(W - 1, x1))
                        x2 = max(x1 + 1, min(W, x2))
                        y1 = max(0, min(H - 1, y1))
                        y2 = max(y1 + 1, min(H, y2))
                        bboxes.append((x1, y1, x2, y2))
                    if bboxes:
                        predictor.set_image(arr)
                        boxes_t = torch.tensor(bboxes, device="cuda", dtype=torch.float32)
                        transformed = predictor.transform.apply_boxes_torch(boxes_t, arr.shape[:2])
                        masks_pred, _, _ = predictor.predict_torch(
                            point_coords=None, point_labels=None,
                            boxes=transformed, multimask_output=False,
                        )
                        masks_np = masks_pred.squeeze(1).cpu().numpy()
                        for m_ in masks_np:
                            mask |= m_.astype(bool)

                # Underline catch (Courier-style)
                from src.part_c.extract_bgs import _detect_underlines
                for um in _detect_underlines(np.array(img.convert("L"))):
                    mask |= um

                if dilate_px > 0:
                    mask = dilation(mask, square(2 * dilate_px + 1))
                mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                bg = lama(img, mask_pil)
                bg_pil = bg if isinstance(bg, Image.Image) else Image.fromarray(bg)
                bg_pil.save(out_dir / p.name)

                done += 1
                if done % 20 == 0:
                    print(f"[part_c_vlm] {done}/{len(paths)}")
            except Exception as e:
                print(f"[part_c_vlm] skip {p.name}: {e}")

    print(f"[part_c_vlm] done. {done}/{len(paths)} bgs in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--dilate-px", type=int, default=4)
    args = ap.parse_args()
    main(args.images, args.out, args.max_images, args.dilate_px)
