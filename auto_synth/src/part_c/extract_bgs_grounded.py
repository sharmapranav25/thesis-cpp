"""Part C v4 — Grounded SAM (text-prompted GroundingDINO + MobileSAM) + LaMa, in a loop.

Pipeline per image, repeated up to --max-iters:
    1. GroundingDINO with text prompt -> bboxes for char-like things.
    2. If 0 boxes above threshold:
         heuristic clean. Optional Qwen oracle confirms; break on agreement.
    3. MobileSAM with bboxes -> per-char masks; union them.
    4. Add long-thin underline components (Courier-style noise).
    5. Dilate union mask, LaMa inpaint -> new bg.

Text prompt sources (in order of precedence):
    --prompt "..."    explicit string
    --vlm-prompt      Qwen2.5-VL-7B describes chars from a sample of inputs
    (default)         "characters. letters. digits."

Usage:
    python -m src.part_c.extract_bgs_grounded \\
        --images "E:/pranav/Courier Market (live)" \\
        --out   "E:/pranav/auto_synth/data/courier_market/bg_grounded" \\
        [--prompt "thin black serif letters with red noise lines"] \\
        [--vlm-prompt] [--vlm-oracle] \\
        [--max-iters 4] [--max-images 50]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.morphology import dilation, square


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SAM_CKPT = "E:/pranav/auto_synth/configs/mobile_sam.pt"
DEFAULT_PROMPT = "characters. letters. digits."


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _filter_oversized(boxes, scores, labels, img_size, max_frac: float = 0.7):
    W, H = img_size
    img_area = W * H
    out_b, out_s, out_l = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = b
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw * bh > max_frac * img_area:
            continue
        out_b.append(b); out_s.append(s); out_l.append(l)
    return out_b, out_s, out_l


def _segment_sam(predictor, arr: np.ndarray, bboxes: list[list[float]]) -> np.ndarray:
    H, W = arr.shape[:2]
    if not bboxes:
        return np.zeros((H, W), dtype=bool)
    predictor.set_image(arr)
    boxes_t = torch.tensor(bboxes, device=predictor.device, dtype=torch.float32)
    transformed = predictor.transform.apply_boxes_torch(boxes_t, arr.shape[:2])
    masks_pred, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=transformed, multimask_output=False,
    )
    masks_np = masks_pred.squeeze(1).cpu().numpy()
    union = np.zeros((H, W), dtype=bool)
    for m in masks_np:
        union |= m.astype(bool)
    return union


def main(
    images_dir: Path,
    out_dir: Path,
    prompt: str,
    use_vlm_prompt: bool,
    use_vlm_oracle: bool,
    max_iters: int,
    box_threshold: float,
    text_threshold: float,
    dilate_px: int,
    sample_k: int,
    max_images: int,
    max_box_frac: float,
):
    from mobile_sam import sam_model_registry, SamPredictor
    from simple_lama_inpainting import SimpleLama
    from src.part_c.grounded_dino import GroundingDinoDetector
    from src.part_c.extract_bgs import _detect_underlines

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(list_images(images_dir))
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"No images under {images_dir}")
    print(f"[grounded] {len(paths)} images, out={out_dir}")

    if use_vlm_prompt:
        from src.part_c import vlm_qwen
        if not vlm_qwen.available():
            raise SystemExit("--vlm-prompt requires CUDA + transformers")
        sample = [Image.open(p).convert("RGB") for p in paths[:sample_k]]
        print(f"[grounded] loading Qwen2.5-VL-7B for prompt gen...")
        prompt = vlm_qwen.generate_scheme_prompt(sample)
        print(f"[grounded] Qwen prompt: {prompt!r}")
        if not use_vlm_oracle:
            vlm_qwen.unload()
            print(f"[grounded] freed Qwen (oracle off)")
    else:
        print(f"[grounded] using prompt: {prompt!r}")

    oracle = None
    if use_vlm_oracle:
        from src.part_c import vlm_qwen
        if not vlm_qwen.available():
            raise SystemExit("--vlm-oracle requires CUDA + transformers")
        if not vlm_qwen.is_loaded():
            print(f"[grounded] loading Qwen2.5-VL-7B for oracle...")
            vlm_qwen.load()
        oracle = vlm_qwen.is_clean

    print(f"[grounded] loading GroundingDINO...")
    gd = GroundingDinoDetector()
    print(f"[grounded] loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=SAM_CKPT).cuda().eval()
    predictor = SamPredictor(sam)
    print(f"[grounded] loading LaMa...")
    lama = SimpleLama()

    log_path = out_dir / "_run.jsonl"
    done = 0
    with log_path.open("w", encoding="utf-8") as flog:
        for i, p in enumerate(paths, 1):
            try:
                bg = Image.open(p).convert("RGB")
                history = []
                final_n = 0
                iters_used = 0

                for it in range(1, max_iters + 1):
                    iters_used = it
                    arr = np.array(bg)
                    det = gd.detect(bg, prompt,
                                    box_threshold=box_threshold,
                                    text_threshold=text_threshold)
                    boxes, scores, labels = _filter_oversized(
                        det["boxes"], det["scores"], det["labels"], bg.size, max_box_frac
                    )
                    final_n = len(boxes)
                    history.append({
                        "iter": it, "n_boxes": final_n,
                        "scores": [round(s, 3) for s in scores],
                    })

                    if final_n == 0:
                        if oracle is None or oracle(bg):
                            iters_used = it - 1
                            break
                        history[-1]["oracle"] = "dirty"
                        break

                    sam_mask = _segment_sam(predictor, arr, boxes)
                    for um in _detect_underlines(np.array(bg.convert("L"))):
                        sam_mask |= um
                    if dilate_px > 0:
                        sam_mask = dilation(sam_mask, square(2 * dilate_px + 1))
                    mask_pil = Image.fromarray((sam_mask.astype(np.uint8) * 255), mode="L")
                    bg_out = lama(bg, mask_pil)
                    bg = bg_out if isinstance(bg_out, Image.Image) else Image.fromarray(bg_out)
                    bg = bg.convert("RGB")

                bg.save(out_dir / p.name)
                flog.write(json.dumps({
                    "src": p.name,
                    "iters": iters_used,
                    "final_box_count": final_n,
                    "history": history,
                }) + "\n")
                done += 1
                if done % 25 == 0 or done == len(paths):
                    print(f"[grounded] {done}/{len(paths)}")
            except Exception as e:
                print(f"[grounded] skip {p.name}: {type(e).__name__}: {e}")

    summary = {
        "prompt": prompt, "n_images": len(paths), "n_done": done,
        "max_iters": max_iters,
        "box_threshold": box_threshold, "text_threshold": text_threshold,
        "dilate_px": dilate_px, "max_box_frac": max_box_frac,
        "vlm_prompt": use_vlm_prompt, "vlm_oracle": use_vlm_oracle,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[grounded] done. {done}/{len(paths)} -> {out_dir}")
    print(f"[grounded] log: {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="GroundingDINO text prompt (ignored if --vlm-prompt)")
    ap.add_argument("--vlm-prompt", action="store_true",
                    help="Generate prompt via Qwen2.5-VL-7B from a sample of inputs")
    ap.add_argument("--vlm-oracle", action="store_true",
                    help="Use Qwen2.5-VL-7B as early-exit oracle when GroundingDINO returns no boxes")
    ap.add_argument("--sample-k", type=int, default=6,
                    help="N images for VLM prompt generation")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--dilate-px", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--max-box-frac", type=float, default=0.7,
                    help="Drop GroundingDINO boxes covering more than this fraction of the image")
    args = ap.parse_args()
    main(
        images_dir=args.images, out_dir=args.out,
        prompt=args.prompt,
        use_vlm_prompt=args.vlm_prompt, use_vlm_oracle=args.vlm_oracle,
        max_iters=args.max_iters,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        dilate_px=args.dilate_px,
        sample_k=args.sample_k, max_images=args.max_images,
        max_box_frac=args.max_box_frac,
    )
