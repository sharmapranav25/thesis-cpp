# char_extraction

**Phase 1 of the CAPTCHA-solver project.** Given a real CAPTCHA image, produce a binary character-pixel mask and per-character crops with bounding boxes. Outputs feed two downstream tracks: background reconstruction (`bg_syth/`) and font reconstruction (`Font_syth/`).

## Status

Under active development. See `eval/reports/summary.md` for current per-CAPTCHA-type pass/fail verdicts.

## Constraints

- **Open-source models only** in the pipeline — no proprietary APIs.
- **Label-free at inference** — no per-image labeling on new CAPTCHA types.
- **Hand-labeled eval** (50-100 imgs/type) drives the PASS verdict; never enters training.

## Quickstart

```powershell
# Create venv + install (run once)
py -3.9 -m venv .venv
.venv\Scripts\pip install -e .[craft,sam,dev]
.venv\Scripts\python scripts\fetch_checkpoints.py

# Run on a single image
.venv\Scripts\char-extract --image path\to\captcha.png --type city --out tmp\

# Batch run
.venv\Scripts\char-extract-batch --input-dir <captcha-dir> --type city --out output\city\ --limit 100

# Hand-label eval set (one-time per type)
.venv\Scripts\char-extract-label --type city --image-dir <captcha-dir> --out eval\labeled\city\labels.json --n 50

# Score against eval
.venv\Scripts\char-extract-eval --type city --extracted-dir output\city\ --labels eval\labeled\city\labels.json
```

## Outputs

For each image, the pipeline writes:

```
output/<captcha_id>/
  original.png
  mask.png            # binary, same dims as input
  overlay.png         # mask drawn on original
  characters/         # left-to-right
    0.png
    1.png
    ...
  metadata.json
```

`metadata.json` shape:

```json
{
  "captcha_id": "...",
  "captcha_type": "...",
  "image_size": [W, H],
  "characters": [
    {"index": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.92, "crop_path": "characters/0.png"}
  ],
  "method": "universal_craft",
  "notes": ""
}
```

## CAPTCHA types and verdicts

Verdicts populated by `char-extract-eval` against hand-labeled bbox sets. See `eval/reports/<type>.md` for full per-type reports.

**M2 first pass (n=10/type, vs Claude-VLM zero-shot pseudo-GT):**

| Type | Best method | Verdict | Count acc | Mean IoU | Mask P / R |
|---|---|---|---|---|---|
| DarkBay | otsu_cc | OUT_OF_SCOPE | 20% | 0.28 | 42% / 46% |
| City | otsu_cc | OUT_OF_SCOPE | 80% | 0.15 | 36% / 14% |
| Courier | otsu_cc | OUT_OF_SCOPE | 20% | 0.35 | 79% / 40% |
| CryptBB | universal_sam | OUT_OF_SCOPE | 50% | 0.36 | 73% / 34% |

All four types fail spec thresholds (IoU ≥0.7, mask P/R ≥0.85). Per-type fallbacks (M3) needed. See `eval/reports/summary.md`. Numbers are a noisy lower bound — pseudo-GT bbox precision is ~5-10 px; ranking between methods is reliable, absolute IoU is not.

Verdict gates:
- **PASS**: ≥95% of eval images meet all four spec thresholds (count ≥0.95, IoU ≥0.7, mask P ≥0.85, mask R ≥0.85).
- **DEGRADED**: 60-95%. Phase ships for this type with documented failure modes.
- **OUT OF SCOPE**: <60%. Phase ships *without* this type.

## Universal-baseline pass rates (M1, ≥1 char detected — weak proxy only)

These numbers tell us how often each method produces *some* output. They do not measure correctness; that's M2.

| Type | `otsu_cc` (classical) | `universal_sam` (MobileSAM + saliency/Otsu prompts) |
|---|---|---|
| DarkBay | 90% (45/50) | 68% (17/25) |
| City | 88% (44/50) | TBD |
| Courier | 86% (43/50) | TBD |
| CryptBB | 100% (50/50) | TBD |

**Throughput on CPU (RTX A4000 idle):** otsu_cc ~50 img/sec; universal_sam ~1 img/sec (MobileSAM ViT-Tiny CPU). For batch use on >1000 images, otsu_cc is 50× faster.

## Available methods

- `otsu_cc` — Otsu threshold + connected components + geometry filters. Zero deps beyond OpenCV/scipy. Default in `_default.yaml`.
- `universal_sam` — MobileSAM (ViT-Tiny) prompted with saliency-CC bboxes and Otsu-CC bboxes (hybrid prompt source). Use via `--method universal_sam`. Requires `pip install -e .[sam]` and `scripts/fetch_checkpoints.py`.
- *(planned)* `universal_craft` — CRAFT text detector, when M2 measurement reveals it would help.
- *(planned)* per-type fallbacks (city/courier/darkbay/cryptbb specific methods) — built only for types that fail M2's verdict gate against `otsu_cc` and `universal_sam`.
