"""Universal Phase 1 method backed by pretrained PSSTRNet.

PSSTRNet (Lyu et al., ICME 2022) is a scene-text-removal network with a
two-decoder architecture: one for text segmentation, one for removal.
We use only the segmentation output (`mask_final`, the 8th tensor in
`forward`) — the removal output is discarded since Phase 2A handles that.

Pipeline:
  1. Resize input to 1024×1024 (the model's training resolution).
  2. Forward pass on GPU.
  3. Extract `mask_final` (sigmoid output, [0,1]).
  4. Resize mask back to input dimensions.
  5. Threshold at 0.5 → binary uint8 mask.
  6. Run CC + filter → CharBox list.

Domain-shift caveat: PSSTRNet is trained on natural-scene text (SCUT-EnsText
+ SCUT-Syn). Our CAPTCHAs are adversarial. Quality on each type is an
empirical question — see eval reports.

Pretrained weights expected at `<repo>/checkpoints/psstrnet/scut_syn.pth`
(or override via config `psstrnet_checkpoint`). Run `scripts/fetch_psstrnet.py`
to download. Note: the upstream code hardcodes `.cuda()`, so this method
requires a CUDA build of torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from char_extraction.methods.base import BaseMethod, register_method
from char_extraction.postproc.components import (
    component_bboxes,
    filter_bboxes,
    label_components,
)
from char_extraction.types import CharBox, ExtractionResult

_MODEL_CACHE: Dict[str, object] = {}


def _resolve_checkpoint(cfg_path: Optional[str]) -> Path:
    if cfg_path:
        p = Path(cfg_path)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "checkpoints" / "psstrnet" / "scut_syn.pth"


def _get_model(ckpt_path: Path):
    key = str(ckpt_path.resolve())
    if key not in _MODEL_CACHE:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"PSSTRNet checkpoint not found at {ckpt_path}. "
                f"Run scripts/fetch_psstrnet.py to download."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PSSTRNet upstream code hardcodes .cuda(); CUDA torch is required. "
                "Install torch with CUDA support or use a different method."
            )
        from char_extraction.external.psstrnet import PSSTRNet
        net = PSSTRNet()
        ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
        # Some checkpoints wrap weights in a 'model_state_dict' key (per upstream inference.py).
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        net.load_state_dict(sd)
        net = net.cuda()
        net.eval()
        _MODEL_CACHE[key] = net
    return _MODEL_CACHE[key]


def _segment_mask(net, img_rgb: np.ndarray, infer_size: int = 1024) -> np.ndarray:
    """Run PSSTRNet on RGB image, return text mask aligned to original dims.

    Returns uint8 (H, W) mask with 0/255 values; 255 = predicted text pixel.
    """
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).cuda()
    with torch.inference_mode():
        outputs = net(tensor)
    mask_final = outputs[7]  # 8th return value: averaged mask across 3 stages
    mask_np = mask_final[0, 0].detach().cpu().numpy().astype(np.float32)
    # PSSTRNet's mask convention: 1 = background, 0 = text. Invert for our convention.
    text_prob = 1.0 - mask_np
    mask_resized = cv2.resize(text_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask_resized


@register_method
class UniversalPSSTRNet(BaseMethod):
    """Pretrained PSSTRNet text-segmentation head used as a universal Phase 1 method."""

    name = "universal_psstrnet"

    def segment(self, img: np.ndarray) -> np.ndarray:
        ckpt = _resolve_checkpoint(self.cfg.get("psstrnet_checkpoint"))
        net = _get_model(ckpt)

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img

        infer_size = int(self.cfg.get("psstrnet_infer_size", 1024))
        threshold = float(self.cfg.get("psstrnet_threshold", 0.5))

        text_prob = _segment_mask(net, img_rgb, infer_size=infer_size)
        binary = (text_prob >= threshold).astype(np.uint8) * 255

        # Optional cleanup: small open to drop pinprick FP, small close to fill 1-px gaps.
        open_k = self.cfg.get("post_open_ksize", [2, 2])
        close_k = self.cfg.get("post_close_ksize", [3, 3])
        if open_k and open_k[0] > 1 and open_k[1] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(open_k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if close_k and close_k[0] > 1 and close_k[1] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(close_k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def split(self, mask: np.ndarray, img: np.ndarray) -> List[CharBox]:
        labels, n = label_components(mask)
        bboxes = component_bboxes(labels, n)
        bboxes = filter_bboxes(
            bboxes,
            min_area=int(self.cfg.get("min_area", 50)),
            max_area=int(self.cfg.get("max_area", 100_000)),
            min_aspect_ratio=float(self.cfg.get("min_aspect_ratio", 0.05)),
            max_aspect_ratio=float(self.cfg.get("max_aspect_ratio", 10.0)),
            min_height=int(self.cfg.get("min_height", 5)),
            max_height=int(self.cfg.get("max_height", 10_000)),
            min_width=int(self.cfg.get("min_width", 1)),
            max_width=int(self.cfg.get("max_width", 10_000)),
        )
        sorted_b = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)
        return [
            CharBox(index=i, bbox=tuple(int(v) for v in b), confidence=0.85)
            for i, b in enumerate(sorted_b)
        ]
