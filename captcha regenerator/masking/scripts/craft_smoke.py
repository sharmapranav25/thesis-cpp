"""Smoke test: run CRAFT on 1 image of each type, save heatmaps + bbox overlays."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import masking  # noqa: F401  (sets HF_HOME etc. before torch)

import cv2
import numpy as np
from craft_text_detector import (
    get_prediction,
    load_craftnet_model,
    load_refinenet_model,
    read_image,
)

SAMPLES = [
    ("city",    r"E:\pranav\City (live)-20260503T045450Z-3-001\City (live)\00243.png"),
    ("courier", r"E:\pranav\Courier Market (live)-20260503T045531Z-3-001\Courier Market (live)\captcha_screenshot_1771976704.png"),
    ("cryptbb", r"E:\pranav\CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001\CryptBBCaptchas(unsolved)07272023\captcha_screenshot_1690279209.png"),
    ("darkbay", r"E:\pranav\DarkBay_Labeled-20260503T045615Z-3-001\DarkBay_Labeled\00BS69.png"),
]


def main() -> None:
    out_root = Path(__file__).resolve().parents[1] / "tmp" / "craft_smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    print("[craft] loading models...")
    craft_net = load_craftnet_model(cuda=True)
    refine_net = load_refinenet_model(cuda=True)

    for type_name, path in SAMPLES:
        img = read_image(path)
        h, w = img.shape[:2]
        # CRAFT was trained on ~720+ px. Upscale small captchas.
        scale = max(1.0, 720.0 / max(h, w))
        if scale > 1.0:
            img_in = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            img_in = img

        pred = get_prediction(
            image=img_in,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=True,
            long_size=max(720, max(img_in.shape[:2])),
        )

        boxes = pred["boxes"]              # list of [4,2] points (in img_in coords)
        text_heatmap = pred["heatmaps"]["text_score_heatmap"]   # already a viz BGR image
        link_heatmap = pred["heatmaps"]["link_score_heatmap"]

        # Map boxes back to original coords.
        boxes_orig = [(np.array(b) / scale).astype(int) for b in boxes]

        # Draw bboxes on original.
        viz = img.copy()
        viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
        for poly in boxes_orig:
            pts = poly.reshape(-1, 1, 2)
            cv2.polylines(viz, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        out_dir = out_root / type_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "original.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "boxes_overlay.png"), viz)
        cv2.imwrite(str(out_dir / "text_heatmap.png"), text_heatmap)
        cv2.imwrite(str(out_dir / "link_heatmap.png"), link_heatmap)

        print(f"[{type_name}] {len(boxes)} boxes detected -> {out_dir}")


if __name__ == "__main__":
    main()
