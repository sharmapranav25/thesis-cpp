"""Universal baseline: spectral-residual saliency + MobileSAM point-prompted segmentation.

Strategy:
  1. Compute spectral-residual saliency on the input image.
  2. Threshold + connected-components to get coarse character regions.
  3. Send the centroid of each region as a point prompt to MobileSAM (vit_t).
  4. Per-prompt mask = one candidate character. Filter by area/aspect.
  5. Union of accepted masks = the segmentation mask.

Module-level caches the SamPredictor so batch runs only load weights once.
The cache key is the resolved checkpoint path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from char_extraction.methods.base import BaseMethod, register_method
from char_extraction.methods.classical import detect_polarity, otsu_mask, to_gray
from char_extraction.postproc.components import (
    component_bboxes,
    filter_bboxes,
    label_components,
)
from char_extraction.types import CharBox, ExtractionResult

# Module-level cache: ckpt path -> (sam_model, predictor). Loading takes
# ~0.5s on CPU; doing it per-image in batch would be a 10x slowdown.
_SAM_CACHE: Dict[str, object] = {}


def _resolve_checkpoint(cfg_path: Optional[str]) -> Path:
    """Resolve checkpoint path. Defaults to <repo>/checkpoints/mobile_sam.pt."""
    if cfg_path:
        p = Path(cfg_path)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "checkpoints" / "mobile_sam.pt"


def _get_predictor(ckpt_path: Path):
    """Load (or return cached) MobileSAM predictor for this checkpoint."""
    key = str(ckpt_path.resolve())
    if key not in _SAM_CACHE:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found at {ckpt_path}. "
                f"Run scripts/fetch_checkpoints.py to download it."
            )
        from mobile_sam import SamPredictor, sam_model_registry  # lazy import
        sam = sam_model_registry["vit_t"](checkpoint=str(ckpt_path))
        sam.eval()
        _SAM_CACHE[key] = SamPredictor(sam)
    return _SAM_CACHE[key]


def _bbox_pad_clip(boxes, pad, H, W):
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(W, x2 + pad),
            min(H, y2 + pad),
        ))
    return out


def _saliency_prompts(
    img_rgb: np.ndarray,
    pctile: float,
    min_area: int,
    max_width_frac: float = 0.5,
    close_ksize: tuple = (5, 7),
) -> list:
    """Spectral-residual saliency -> close+threshold -> CC bboxes."""
    gray = to_gray(img_rgb)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = saliency.computeSaliency(gray)
    if not ok or sal_map is None:
        return []
    sal_map = sal_map.astype(np.float32)
    thr = float(np.percentile(sal_map, pctile))
    bin_mask = ((sal_map >= thr) & (sal_map > 0)).astype(np.uint8) * 255
    if close_ksize and close_ksize[0] > 1 and close_ksize[1] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_ksize)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    labels, n = label_components(bin_mask)
    if n == 0:
        return []
    bboxes = component_bboxes(labels, n)
    H, W = img_rgb.shape[:2]
    max_w_abs = int(W * max_width_frac)
    return [b for b in bboxes if (b[2] - b[0]) * (b[3] - b[1]) >= min_area and (b[2] - b[0]) <= max_w_abs]


def _otsu_prompts(
    img_rgb: np.ndarray,
    min_area: int,
    max_width_frac: float = 0.5,
    close_ksize: tuple = (3, 3),
) -> list:
    """Otsu threshold -> close -> CC bboxes. Used as a fallback prompt source
    for low-saliency images (uniform light bg with thin chars)."""
    gray = to_gray(img_rgb)
    dol = detect_polarity(gray)
    bin_mask = otsu_mask(gray, dark_on_light=dol)
    if close_ksize and close_ksize[0] > 1 and close_ksize[1] > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_ksize)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    labels, n = label_components(bin_mask)
    if n == 0:
        return []
    bboxes = component_bboxes(labels, n)
    H, W = img_rgb.shape[:2]
    max_w_abs = int(W * max_width_frac)
    return [b for b in bboxes if (b[2] - b[0]) * (b[3] - b[1]) >= min_area and (b[2] - b[0]) <= max_w_abs]


def _candidate_prompts(
    img_rgb: np.ndarray,
    saliency_pctile: float,
    min_area: int,
    max_width_frac: float,
    pad: int,
    saliency_close_ksize: tuple,
    otsu_close_ksize: tuple,
) -> np.ndarray:
    """Combine saliency and Otsu prompts; pad each, dedupe by IoU."""
    sal = _saliency_prompts(img_rgb, saliency_pctile, min_area, max_width_frac, saliency_close_ksize)
    ots = _otsu_prompts(img_rgb, min_area, max_width_frac, otsu_close_ksize)
    H, W = img_rgb.shape[:2]
    combined = _bbox_pad_clip(sal + ots, pad, H, W)
    if not combined:
        return np.empty((0, 4), dtype=np.float32)
    # Dedupe by IoU between candidate prompts (they often overlap).
    keep = []
    for b in combined:
        ok = True
        for k in keep:
            if _bbox_iou(b, k) >= 0.7:
                ok = False
                break
        if ok:
            keep.append(b)
    return np.array(keep, dtype=np.float32)


@register_method
class UniversalSAM(BaseMethod):
    """SAM with saliency-derived point prompts. Generic; no per-type tuning."""

    name = "universal_sam"

    def segment(self, img: np.ndarray) -> np.ndarray:
        # Default `run` uses segment->split, but we override `run` below so
        # this method exists only to satisfy the abstract contract. Returns
        # the same union mask that `run` produces, computed from scratch.
        result = self._run_inference(img)
        return result["mask"]

    def split(self, mask: np.ndarray, img: np.ndarray) -> List[CharBox]:
        # Same: bypassed by overridden `run`. Falls back to plain CC.
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

    def _run_inference(self, img: np.ndarray) -> dict:
        """Saliency CCs -> bbox-prompted SAM -> union mask -> CC -> char bboxes.

        After SAM, the union of per-prompt masks is run through CC again so
        adjacent SAM segments belonging to the same character merge into one
        bbox, even if saliency split them across multiple prompts.
        """
        ckpt = _resolve_checkpoint(self.cfg.get("sam_checkpoint"))
        predictor = _get_predictor(ckpt)

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img

        h, w = img_rgb.shape[:2]
        prompts = _candidate_prompts(
            img_rgb,
            saliency_pctile=float(self.cfg.get("saliency_pctile", 75.0)),
            min_area=int(self.cfg.get("min_saliency_area", 20)),
            max_width_frac=float(self.cfg.get("max_prompt_width_frac", 0.4)),
            pad=int(self.cfg.get("prompt_pad_px", 4)),
            saliency_close_ksize=tuple(self.cfg.get("saliency_close_ksize", [5, 7])),
            otsu_close_ksize=tuple(self.cfg.get("otsu_close_ksize", [3, 3])),
        )

        if prompts.shape[0] == 0:
            return {"mask": np.zeros((h, w), dtype=np.uint8), "bboxes": []}

        predictor.set_image(img_rgb)
        union_mask = np.zeros((h, w), dtype=bool)
        per_prompt_scores: list = []  # one score per accepted prompt
        min_score = float(self.cfg.get("min_sam_score", 0.5))
        max_mask_frac = float(self.cfg.get("max_mask_area_frac", 0.4))

        for prompt_bbox in prompts:
            box = np.array(prompt_bbox, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )
            score = float(scores[0])
            if score < min_score:
                continue
            m = masks[0].astype(bool)
            mask_area = int(m.sum())
            if mask_area > h * w * max_mask_frac or mask_area < 1:
                continue
            union_mask |= m
            per_prompt_scores.append(score)

        # Run CC on the union to recover per-character bboxes regardless of
        # how prompts were distributed. This handles fragmented saliency.
        union_uint8 = (union_mask.astype(np.uint8) * 255)
        # Optional small closing on the union to bridge 1-2px gaps.
        union_close_k = tuple(self.cfg.get("union_close_ksize", [3, 3]))
        if union_close_k and union_close_k[0] > 1 and union_close_k[1] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, union_close_k)
            union_uint8 = cv2.morphologyEx(union_uint8, cv2.MORPH_CLOSE, kernel)

        labels, n = label_components(union_uint8)
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
        # Use mean per-prompt score as a coarse confidence — no per-bbox score from CC.
        mean_conf = float(np.mean(per_prompt_scores)) if per_prompt_scores else 0.0
        out_bboxes = [(x1, y1, x2, y2, mean_conf) for x1, y1, x2, y2 in bboxes]
        return {"mask": union_uint8, "bboxes": out_bboxes}

    def run(
        self,
        img: np.ndarray,
        captcha_id: str,
        captcha_type,
    ) -> ExtractionResult:
        h, w = img.shape[:2]
        out = self._run_inference(img)
        bboxes = out["bboxes"]
        sorted_b = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)
        chars = [
            CharBox(
                index=i,
                bbox=(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                confidence=float(b[4]),
            )
            for i, b in enumerate(sorted_b)
        ]
        return ExtractionResult(
            captcha_id=captcha_id,
            captcha_type=captcha_type,
            image_size=(w, h),
            mask=out["mask"],
            characters=chars,
            method=self.name,
            notes=f"prompts={len(out['bboxes'])}",
        )


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe_overlapping(bboxes: list, iou_thresh: float = 0.5) -> list:
    """NMS by IoU. Keeps higher-score bbox in each pair."""
    sorted_b = sorted(bboxes, key=lambda b: -b[4])  # descending score
    keep = []
    for cand in sorted_b:
        ok = True
        for k in keep:
            if _bbox_iou(cand, k) >= iou_thresh:
                ok = False
                break
        if ok:
            keep.append(cand)
    return keep


def _bbox_contains_frac(inner, outer) -> float:
    """Fraction of the smaller bbox's area that lies inside the larger bbox.

    Returns max(inner_in_outer, outer_in_inner) so containment is symmetric:
    if either box mostly fits inside the other, we treat them as duplicates.
    """
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_inner = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    a_outer = max(1, (outer[2] - outer[0]) * (outer[3] - outer[1]))
    return max(inter / a_inner, inter / a_outer)


def _dedupe_containment(bboxes: list, contain_thresh: float = 0.85) -> list:
    """Keep the smaller bbox when one is mostly contained in another.

    This handles SAM producing both a per-char mask and a multi-char mask
    near the same prompt — they have moderate IoU but the smaller one
    is mostly inside the larger one.
    """
    sorted_b = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # ascending area
    keep = []
    for cand in sorted_b:
        ok = True
        for k in keep:
            if _bbox_contains_frac(cand, k) >= contain_thresh:
                # cand is similar in extent to k. Since we iterate small-first,
                # cand is the smaller one — but if k is also small we may keep
                # both. Drop cand only if k is similar size or smaller.
                a_cand = (cand[2] - cand[0]) * (cand[3] - cand[1])
                a_k = (k[2] - k[0]) * (k[3] - k[1])
                if a_k <= a_cand * 1.2:
                    ok = False
                    break
        # If cand survives, also remove any kept k that contains cand significantly.
        if ok:
            keep = [k for k in keep if _bbox_contains_frac(cand, k) < contain_thresh
                    or (k[2] - k[0]) * (k[3] - k[1]) <= (cand[2] - cand[0]) * (cand[3] - cand[1]) * 1.2]
            keep.append(cand)
    return keep
