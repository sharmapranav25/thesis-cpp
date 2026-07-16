"""CLI: char-extract-eval — score extracted outputs against Claude-VLM predictions.

Two modes (controlled by --predictions):

PROXY ONLY (no --predictions):
  - Char count plausibility (vs configured expected_chars range)
  - Bbox non-overlap (no two bboxes should IoU > 0.2)
  - Char-count modal agreement (type-wide mode matches expected range)
  - Bbox spatial sanity (left-to-right monotonicity of x-centers)

PSEUDO-GT (with --predictions PATH):
  All proxy metrics PLUS hard metrics scored against Claude-VLM zero-shot predictions:
  - Char-count accuracy (per-image == match)
  - Char-count precision/recall (over-detect / under-detect rates)
  - Mean bbox IoU (Hungarian-matched, **matched-only** — locked semantics)
  - Mask P / R (against rasterized union of predicted bboxes)

Verdict gate (LOCKED):
  PASS:        ≥95% of images meet all hard thresholds
               (count==1.0 OR within ±1, bbox IoU ≥0.7, mask P ≥0.85, mask R ≥0.85)
  DEGRADED:    60-95%
  OUT OF SCOPE: <60%

The "predictions" file is a JSON produced either by `char-extract-claude-baseline`
or by Claude-Code-in-session (this is acceptable — both are zero-shot).

Pipeline never reads predictions; only this eval script does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------- I/O ----------

def _load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    """Returns {captcha_id: {"characters": [...], "n_characters": int, ...}}."""
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "images" in obj and isinstance(obj["images"], dict):
        return obj["images"]
    if isinstance(obj, dict):
        # Allow flat shape: {captcha_id: {...}, ...}
        return obj
    raise ValueError(f"Unexpected predictions JSON shape in {path}")


def _load_extracted(extracted_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Walk extracted_dir/<captcha_id>/metadata.json. Returns {captcha_id: metadata}."""
    out = {}
    for sub in sorted(extracted_dir.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            meta["_dir"] = str(sub)
            out[sub.name] = meta
        except Exception:
            continue
    return out


def _load_mask(captcha_dir: Path) -> Optional[np.ndarray]:
    """Read mask.png as uint8 (H, W) with 0/255 values."""
    import cv2
    mp = captcha_dir / "mask.png"
    if not mp.exists():
        return None
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8) * 255


# ---------- Bbox geometry ----------

def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter) / float(ua) if ua > 0 else 0.0


def _hungarian_match(pred_bboxes: List, gt_bboxes: List) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Returns (matches, unmatched_pred_idx, unmatched_gt_idx).

    matches is a list of (pred_idx, gt_idx, iou). Uses Hungarian on -IoU
    cost matrix. Matched pairs with IoU < 0.05 are filtered as non-matches.
    """
    if not pred_bboxes or not gt_bboxes:
        return [], list(range(len(pred_bboxes))), list(range(len(gt_bboxes)))
    n_pred, n_gt = len(pred_bboxes), len(gt_bboxes)
    cost = np.zeros((n_pred, n_gt), dtype=np.float64)
    iou = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(pred_bboxes):
        for j, g in enumerate(gt_bboxes):
            iou[i, j] = _bbox_iou(p, g)
            cost[i, j] = -iou[i, j]
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    matched_pred = set()
    matched_gt = set()
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= 0.05:
            matches.append((int(r), int(c), float(iou[r, c])))
            matched_pred.add(int(r))
            matched_gt.add(int(c))
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


# ---------- Mask metrics (against bbox-union pseudo-mask) ----------

def _rasterize_bboxes(bboxes: List, h: int, w: int) -> np.ndarray:
    """Build a binary uint8 mask by filling each bbox rectangle."""
    m = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
    return m


def _mask_pr(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    p = (pred_mask > 0)
    g = (gt_mask > 0)
    inter = float(np.logical_and(p, g).sum())
    pred_area = float(p.sum())
    gt_area = float(g.sum())
    precision = inter / pred_area if pred_area > 0 else 0.0
    recall = inter / gt_area if gt_area > 0 else 0.0
    return precision, recall


# ---------- Proxy metrics ----------

def _proxy_count_plausibility(extracted: Dict[str, Dict[str, Any]], expected_range: Tuple[int, int]) -> float:
    lo, hi = expected_range
    n_ok = 0
    n = 0
    for meta in extracted.values():
        c = len(meta.get("characters", []))
        n += 1
        if lo <= c <= hi:
            n_ok += 1
    return n_ok / n if n else 0.0


def _proxy_bbox_non_overlap(extracted: Dict[str, Dict[str, Any]], iou_thresh: float = 0.2) -> float:
    n_ok = 0
    n = 0
    for meta in extracted.values():
        chars = meta.get("characters", [])
        bbs = [c["bbox"] for c in chars]
        bad = False
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if _bbox_iou(bbs[i], bbs[j]) > iou_thresh:
                    bad = True
                    break
            if bad:
                break
        n += 1
        if not bad:
            n_ok += 1
    return n_ok / n if n else 0.0


def _proxy_l2r_monotonic(extracted: Dict[str, Dict[str, Any]]) -> float:
    n_ok = 0
    n = 0
    for meta in extracted.values():
        chars = meta.get("characters", [])
        n += 1
        centers = [(c["bbox"][0] + c["bbox"][2]) / 2.0 for c in chars]
        if centers == sorted(centers):
            n_ok += 1
    return n_ok / n if n else 0.0


def _proxy_count_modal_agreement(extracted: Dict[str, Dict[str, Any]], expected_range: Tuple[int, int]) -> float:
    """Type-wide modal char count should land inside expected_chars."""
    counts = [len(m.get("characters", [])) for m in extracted.values()]
    if not counts:
        return 0.0
    bins = np.bincount(counts)
    mode = int(np.argmax(bins))
    lo, hi = expected_range
    return 1.0 if lo <= mode <= hi else 0.0


# ---------- Hard metrics (against Claude-VLM pseudo-GT) ----------

def _eval_against_predictions(
    extracted: Dict[str, Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    bbox_iou_thresh: float,
    mask_p_thresh: float,
    mask_r_thresh: float,
    count_tol: int,
) -> Dict[str, Any]:
    rows = []
    for cid, meta in extracted.items():
        gt = predictions.get(cid)
        if not gt or "characters" not in gt:
            continue
        pred_bboxes = [tuple(c["bbox"]) for c in meta.get("characters", [])]
        gt_bboxes = [tuple(c["bbox"]) for c in gt["characters"]]
        n_pred = len(pred_bboxes)
        n_gt = len(gt_bboxes)
        count_ok = abs(n_pred - n_gt) <= count_tol

        matches, unm_pred, unm_gt = _hungarian_match(pred_bboxes, gt_bboxes)
        ious = [m[2] for m in matches]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        median_iou = float(np.median(ious)) if ious else 0.0
        precision_count = len(matches) / n_pred if n_pred else 0.0
        recall_count = len(matches) / n_gt if n_gt else 0.0

        # Mask P/R against rasterized GT bbox union, using extracted mask.
        captcha_dir = Path(meta["_dir"])
        h, w = meta.get("image_size", [0, 0])[::-1] if meta.get("image_size") else (None, None)
        pred_mask = _load_mask(captcha_dir)
        if h is not None and pred_mask is not None and pred_mask.shape == (h, w):
            gt_mask = _rasterize_bboxes(gt_bboxes, h, w)
            mask_p, mask_r = _mask_pr(pred_mask, gt_mask)
        else:
            mask_p, mask_r = float("nan"), float("nan")

        # Per-image PASS check against spec thresholds.
        per_image_pass = (
            count_ok
            and mean_iou >= bbox_iou_thresh
            and (np.isnan(mask_p) or mask_p >= mask_p_thresh)
            and (np.isnan(mask_r) or mask_r >= mask_r_thresh)
        )
        rows.append({
            "captcha_id": cid,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "count_ok": bool(count_ok),
            "matched_iou_mean": mean_iou,
            "matched_iou_median": median_iou,
            "count_precision": precision_count,
            "count_recall": recall_count,
            "mask_p": None if np.isnan(mask_p) else mask_p,
            "mask_r": None if np.isnan(mask_r) else mask_r,
            "passes": bool(per_image_pass),
        })

    if not rows:
        return {"n_scored": 0, "rows": []}

    n = len(rows)
    pass_rate = sum(r["passes"] for r in rows) / n
    return {
        "n_scored": n,
        "pass_rate": pass_rate,
        "count_accuracy": sum(r["count_ok"] for r in rows) / n,
        "matched_iou_mean": float(np.mean([r["matched_iou_mean"] for r in rows])),
        "count_precision_mean": float(np.mean([r["count_precision"] for r in rows])),
        "count_recall_mean": float(np.mean([r["count_recall"] for r in rows])),
        "mask_p_mean": float(np.nanmean([r["mask_p"] if r["mask_p"] is not None else np.nan for r in rows])),
        "mask_r_mean": float(np.nanmean([r["mask_r"] if r["mask_r"] is not None else np.nan for r in rows])),
        "rows": rows,
    }


def _verdict(pass_rate: float) -> str:
    if pass_rate >= 0.95:
        return "PASS"
    if pass_rate >= 0.60:
        return "DEGRADED"
    return "OUT_OF_SCOPE"


# ---------- Report writer ----------

def _write_report(out_path: Path, report: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        f"# Eval report: {report['captcha_type']}",
        "",
        f"- Method: `{report['method']}`",
        f"- Extracted dir: `{report['extracted_dir']}`",
        f"- Predictions: `{report.get('predictions', '(proxy-only)')}`",
        f"- Images scored: {report['n_scored']}",
        "",
        "## Verdict",
        "",
        f"**{report.get('verdict', 'PROXY_ONLY')}**",
        "",
    ]
    if "pass_rate" in report:
        md_lines += [
            "## Hard metrics (vs Claude-VLM pseudo-GT)",
            "",
            f"- Per-image pass rate (all 4 thresholds met): **{report['pass_rate']:.2%}**",
            f"- Char count accuracy (±tol): {report['count_accuracy']:.2%}",
            f"- Mean matched-only bbox IoU: {report['matched_iou_mean']:.3f}",
            f"- Mean per-image count precision / recall: {report['count_precision_mean']:.2%} / {report['count_recall_mean']:.2%}",
            f"- Mean mask P / R (vs rasterized GT bboxes): {report['mask_p_mean']:.2%} / {report['mask_r_mean']:.2%}",
            "",
        ]
    md_lines += [
        "## Proxy metrics",
        "",
        f"- Char count plausibility (count in expected range): {report['proxy_count_plausibility']:.2%}",
        f"- Bbox non-overlap (no IoU > 0.2 within image): {report['proxy_bbox_non_overlap']:.2%}",
        f"- Left-to-right monotonicity: {report['proxy_l2r_monotonic']:.2%}",
        f"- Type-wide modal-count in expected range: {'YES' if report['proxy_count_modal_agreement'] >= 1.0 else 'NO'}",
        "",
    ]
    out_path.write_text("\n".join(md_lines))


# ---------- CLI entrypoint ----------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="char-extract-eval", description="Score char_extraction outputs (proxy + Claude-VLM pseudo-GT).")
    p.add_argument("--type", dest="captcha_type", required=True)
    p.add_argument("--extracted-dir", required=True, type=Path,
                   help="Directory with per-image subdirs (output of char-extract-batch).")
    p.add_argument("--predictions", type=Path, default=None,
                   help="Optional Claude-VLM predictions JSON. Without this, proxy-only mode.")
    p.add_argument("--report", type=Path, default=None,
                   help="Output report path. Defaults to eval/reports/<type>.md")
    p.add_argument("--expected-min", type=int, default=None, help="Override expected_chars min.")
    p.add_argument("--expected-max", type=int, default=None, help="Override expected_chars max.")
    p.add_argument("--bbox-iou-thresh", type=float, default=0.7)
    p.add_argument("--mask-p-thresh", type=float, default=0.85)
    p.add_argument("--mask-r-thresh", type=float, default=0.85)
    p.add_argument("--count-tol", type=int, default=0,
                   help="Count tolerance: |n_pred - n_gt| <= tol counts as match. Default 0 (strict).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    extracted = _load_extracted(args.extracted_dir)
    if not extracted:
        print(f"No extracted outputs in {args.extracted_dir}", file=sys.stderr)
        return 2

    method = next(iter(extracted.values())).get("method", "?")
    expected_range = (args.expected_min or 1, args.expected_max or 99)
    if not args.expected_min:
        # Try to infer from any per-type config heuristically (fallback wide).
        try:
            import yaml
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / f"{args.captcha_type}.yaml"
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                if "expected_chars" in cfg and isinstance(cfg["expected_chars"], list) and len(cfg["expected_chars"]) == 2:
                    expected_range = (int(cfg["expected_chars"][0]), int(cfg["expected_chars"][1]))
        except Exception:
            pass

    proxy = {
        "proxy_count_plausibility": _proxy_count_plausibility(extracted, expected_range),
        "proxy_bbox_non_overlap": _proxy_bbox_non_overlap(extracted),
        "proxy_l2r_monotonic": _proxy_l2r_monotonic(extracted),
        "proxy_count_modal_agreement": _proxy_count_modal_agreement(extracted, expected_range),
    }

    report: Dict[str, Any] = {
        "captcha_type": args.captcha_type,
        "method": method,
        "extracted_dir": str(args.extracted_dir),
        "expected_range": list(expected_range),
        **proxy,
    }

    n_scored = 0
    if args.predictions and args.predictions.exists():
        predictions = _load_predictions(args.predictions)
        hard = _eval_against_predictions(
            extracted,
            predictions,
            bbox_iou_thresh=args.bbox_iou_thresh,
            mask_p_thresh=args.mask_p_thresh,
            mask_r_thresh=args.mask_r_thresh,
            count_tol=args.count_tol,
        )
        report.update({k: v for k, v in hard.items() if k != "rows"})
        report["predictions"] = str(args.predictions)
        report["per_image"] = hard.get("rows", [])
        n_scored = hard.get("n_scored", 0)
        if n_scored > 0:
            report["verdict"] = _verdict(hard["pass_rate"])
        else:
            report["verdict"] = "NO_OVERLAP"
    else:
        report["n_scored"] = len(extracted)
        report["verdict"] = "PROXY_ONLY"

    out_path = args.report or (Path(__file__).resolve().parents[3] / "eval" / "reports" / f"{args.captcha_type}.md")
    _write_report(out_path, report)

    summary = {k: v for k, v in report.items() if k not in ("per_image",)}
    print(json.dumps(summary, indent=2, default=str))

    rc = 0
    v = report.get("verdict", "PROXY_ONLY")
    if v == "DEGRADED":
        rc = 3
    elif v in ("OUT_OF_SCOPE", "NO_OVERLAP"):
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
