# Past context — gathered from `../char_extraction/` on 2026-05-06

Neutral record of what exists in the sibling `char_extraction/` folder at the moment the `masking/` folder was created. No recommendations, no opinions — just the observed state.

## Repo layout (parent: `captcha regenerator/`)

Sibling folders:
- `bg_syth/`
- `char_extraction/`
- `edgetest1/`
- `Font_syth/`
- `masking/` (this folder)

## `char_extraction/` purpose (per its README)

Phase 1 of a CAPTCHA-solver project. Input: a real CAPTCHA image. Output: a binary character-pixel mask, per-character crops, and bounding boxes. Outputs feed two downstream tracks:
- `bg_syth/` — background reconstruction
- `Font_syth/` — font reconstruction

## Stated constraints

- Open-source models only in the pipeline (no proprietary APIs).
- Label-free at inference (no per-image labeling on new CAPTCHA types).
- Hand-labeled eval set (50–100 imgs/type) drives the PASS verdict; never enters training.

## Folder structure of `char_extraction/`

```
char_extraction/
  README.md
  research.md
  pyproject.toml
  checkpoints/
  configs/         # _default.yaml, _schema.yaml, city.yaml, courier.yaml, cryptbb.yaml, darkbay.yaml
  eval/
    claude_predictions/
    labeled/
    proxy/
    qwen_predictions/
    reports/       # summary.md + per-type per-method reports
  notebooks/
  output/
  scripts/         # fetch_checkpoints.py, fetch_psstrnet.py
  src/
    char_extraction/
      __init__.py
      api.py
      cli/
      external/
      io_utils.py
      methods/
      postproc/
      router.py
      types.py
    char_extraction.egg-info/
  tests/
  tmp/
```

## CAPTCHA types in scope

DarkBay, City, Courier, CryptBB.

## Methods present

- `otsu_cc` — Otsu threshold + connected components + geometry filters. OpenCV/scipy only. Default in `_default.yaml`.
- `universal_sam` — MobileSAM (ViT-Tiny) prompted with saliency-CC and Otsu-CC bboxes (hybrid prompt source).
- *(planned)* `universal_craft` — CRAFT text detector, not yet built.
- *(planned)* per-type fallbacks — to be built only for types that fail M2 verdict gate.

## Output format per image (per README)

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

`metadata.json` shape: `captcha_id`, `captcha_type`, `image_size [W, H]`, `characters[]` with `index`, `bbox [x1,y1,x2,y2]`, `confidence`, `crop_path`, plus `method` and `notes`.

## Verdict gate (per README + `eval/reports/summary.md`)

- **PASS:** ≥95% of eval images meet all four spec thresholds — count ±1, IoU ≥0.7, mask P ≥0.85, mask R ≥0.85.
- **DEGRADED:** 60–95%.
- **OUT OF SCOPE:** <60%.

## M2 first-pass results (n=10/type, scored vs Claude-VLM zero-shot pseudo-GT)

| Type | Method | Count acc | Mean IoU | Mask P | Mask R | Verdict |
|---|---|---|---|---|---|---|
| DarkBay | otsu_cc | 20% | 0.276 | 42.2% | 45.6% | OUT_OF_SCOPE |
| DarkBay | universal_sam | 10% | 0.214 | 41.7% | 34.3% | OUT_OF_SCOPE |
| City | otsu_cc | 80% | 0.146 | 35.7% | 13.8% | OUT_OF_SCOPE |
| City | universal_sam | 50% | 0.167 | 21.3% | 23.8% | OUT_OF_SCOPE |
| Courier | otsu_cc | 20% | 0.352 | 79.1% | 39.8% | OUT_OF_SCOPE |
| Courier | universal_sam | 0% | 0.126 | 64.0% | 3.8% | OUT_OF_SCOPE |
| CryptBB | otsu_cc | 90% | 0.300 | 52.7% | 19.2% | OUT_OF_SCOPE |
| CryptBB | universal_sam | 50% | 0.361 | 72.6% | 34.2% | OUT_OF_SCOPE |

All four types fail spec thresholds at first pass. Caveats noted in the report: pseudo-GT bbox precision ~5–10 px, n=10 is small, IoU values are a noisy lower bound — ranking between methods is reliable, absolute IoU is not.

## Universal-baseline coverage (M1 — "≥1 char detected", weak proxy only)

| Type | otsu_cc | universal_sam |
|---|---|---|
| DarkBay | 90% (45/50) | 68% (17/25) |
| City | 88% (44/50) | TBD |
| Courier | 86% (43/50) | TBD |
| CryptBB | 100% (50/50) | TBD |

Throughput on the project's CPU: otsu_cc ~50 img/sec; universal_sam ~1 img/sec.

## Research file (`research.md`) — recorded outcomes

- Stage A (pretrained PSSTRNet, both `scut_syn.pth` and `scut_enstext.pth` checkpoints, tested on one image per type): both checkpoints produced `mask_final.mean() ≈ 0.999` — model output "all background." `text_prob ≥ 0.5` fraction was 0.0–0.6% across all 16 (image × checkpoint) pairs.
- Conclusion recorded in `research.md`: domain gap between natural-scene text training data (SCUT-EnsText / Flickr-ST) and adversarial CAPTCHAs is severe.
- Three forward paths listed in `research.md` (no decision recorded yet): try Tang 2021 stroke-based synthetic-data pretrained model; fine-tune on synthetic CAPTCHAs; build per-type custom methods.
- Other STR papers surveyed: PSSTRNet, PERT, FETNet, Stroke-Based (Tang 2021), DiffSTR, TMIM, OTR, EnsNet, EraseNet, "Surprisingly Straightforward STR".

## Eval reports present

`eval/reports/` contains: `summary.md`, plus per-type per-method reports — `darkbay.md`, and `{type}_{method}.md` for each combination of {darkbay, city, courier, cryptbb} × {otsu, sam, swt}.

## Python package details

- Package name: `char_extraction`, version 0.1.0, requires Python ≥3.9.
- CLI entry points: `char-extract`, `char-extract-batch`, `char-extract-eval`, `char-extract-qwen-baseline`.
- Optional install groups: `[craft]`, `[sam]`, `[eval]` (Qwen2.5-VL-7B-Instruct as eval pseudo-GT, not in production), `[dev]`.
- Core deps: numpy<2, opencv-contrib-python, scikit-image, scipy, scikit-learn, pillow, pyyaml, pydantic, matplotlib, tqdm.

## What is *not* in this context

- Contents of `src/char_extraction/methods/`, `postproc/`, `router.py` — not read.
- Contents of per-type configs (`city.yaml`, `courier.yaml`, etc.) — not read.
- Contents of `notebooks/`, `tests/`, `output/`, `tmp/` — not read.
- Per-type per-method eval reports beyond `summary.md` — not read.
- The relationship (if any) between this `masking/` folder and `char_extraction/` — not yet stated by the user.
