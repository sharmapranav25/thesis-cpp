"""
Stage 2.5 — Coverage validator.

Find hot blobs in the attention map (>=0.55, >=50 px). For each blob check
whether Stage 2's mask kept any pixels there. If <80% of blobs are covered,
re-run Stage 2's threshold+CC filter with progressively looser params.

  iter 1: noise_pctile=50, max_area_pct=0.30
  iter 2: noise_pctile=40, max_area_pct=0.40
  iter 3: noise_pctile=30, max_area_pct=0.50

Stops as soon as coverage hits 80%, otherwise keeps the best result tried.
Path A skips this stage (no attention map).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .stage2 import run_path_b_with_overrides
from .types import Stage2_5Output, Stage2Output


def run(
    image_bgr: np.ndarray,
    stage2_out: Stage2Output,
    stage2_cfg: dict[str, Any],
    cfg: dict[str, Any],
) -> Stage2_5Output:
    if stage2_out.path == "A" or stage2_out.attention is None:
        return Stage2_5Output(
            mask=stage2_out.mask,
            cleaned=stage2_out.cleaned,
            iterations=0,
            final_coverage=None,
        )

    if not cfg.get("enabled", True):
        return Stage2_5Output(
            mask=stage2_out.mask,
            cleaned=stage2_out.cleaned,
            iterations=0,
            final_coverage=None,
        )

    target = cfg["coverage_target"]
    blobs = _find_blobs(stage2_out.attention, cfg["blob_attn_threshold"], cfg["blob_min_area_px"])

    cov0 = _coverage(blobs, stage2_out.mask)
    history = [{"iteration": 0, "coverage": cov0, "params": stage2_out.params_used}]

    if cov0 >= target or not blobs:
        return Stage2_5Output(
            mask=stage2_out.mask,
            cleaned=stage2_out.cleaned,
            iterations=0,
            final_coverage=cov0,
            history=history,
        )

    best = stage2_out
    best_cov = cov0
    path_b_cfg = stage2_cfg["path_b"]
    retries = cfg["retries"]

    for i, retry in enumerate(retries, start=1):
        attempt = run_path_b_with_overrides(
            image_bgr,
            path_b_cfg,
            attn_full=stage2_out.attention,
            noise_pctile=retry["noise_pctile"],
            cc_max_area_pct=retry["max_area_pct"],
        )
        cov = _coverage(blobs, attempt.mask)
        history.append({"iteration": i, "coverage": cov, "params": retry})
        if cov > best_cov:
            best, best_cov = attempt, cov
        if cov >= target:
            return Stage2_5Output(
                mask=attempt.mask,
                cleaned=attempt.cleaned,
                iterations=i,
                final_coverage=cov,
                history=history,
            )

    return Stage2_5Output(
        mask=best.mask,
        cleaned=best.cleaned,
        iterations=len(retries),
        final_coverage=best_cov,
        history=history,
    )


def _find_blobs(attention: np.ndarray, threshold: float, min_area: int) -> list[np.ndarray]:
    hot = (attention >= threshold).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    blobs: list[np.ndarray] = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        blobs.append(labels == i)
    return blobs


def _coverage(blobs: list[np.ndarray], mask: np.ndarray) -> float:
    if not blobs:
        return 1.0
    mask_bool = mask > 0
    covered = sum(1 for b in blobs if (b & mask_bool).any())
    return covered / len(blobs)
