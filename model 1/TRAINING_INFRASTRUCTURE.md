# Model 1 — Training Infrastructure

How Model 1 (and Model 1.5 = Model 1 + ensemble) is built end-to-end, from
data generation to deployment. Reading order: top-down.

---

## Overview

Model 1.5 has 5 trained components:

```
   PHP renderer  ──►  100k synth captchas (data/synth_mybb_v2/)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    PARSeq v2          Detector v2        Char-CNNs (×3 scales)
   (recogniser)       (5-bbox regressor)    (per-char classifier)
        │                   │                   │
        └───────┬───────────┴───────────┬───────┘
                │                       │
                ▼                       ▼
          (single model)          (ensemble)
        Model 1: 39.94%       Model 1.5: 49.16%
       on 651 real eval        on 651 real eval
```

All 5 components are trained from scratch, label-free, on 100k PHP-rendered
captchas using MyBB's actual bundled fonts. The 651 real CryptBB captchas at
`data/eval/` are eval-only.

---

## Step 1 — Synthetic data generation

### Tool
`src/php_synth/generate_mybb_captcha.php`

### What it does
Line-by-line port of MyBB's `captcha.php` — produces 200×60 PNG captchas
visually indistinguishable from CryptBB real captchas. Each render is a
fresh seeded RNG sample of:
- 5 random alphanumeric chars (62-class: `[A-Za-z0-9]`)
- Random bg pattern: lines | circles | squares (one per image)
- 1px black border, ~2400 light-pastel dot overlay
- Per-char: random font from `data/_mybb_fonts/`, size 20-32, rotation ±30°,
  RGB 0-200 each channel, ±3px shadow in slightly-lighter tone

Outputs: PNG files + `labels.csv` + (optional) `bboxes.jsonl`.

### Run
```bash
C:/xampp/php/php.exe src/php_synth/generate_mybb_captcha.php \
    --out data/synth_mybb_v2 \
    --n 100000 \
    --seed 11 \
    --bboxes \
    --fontset mybb \
    --fonts data/_mybb_fonts
```

**Wall time:** ~11 min on the workstation (PHP single-threaded, ~150 img/s).
**Output size:** 100k images ≈ 2.0 GB + bboxes.jsonl ≈ 40 MB.

### Constraint inheritance
- Charset and length are configurable (`--charset`, `--captcha-len`,
  `--width`, `--height`) — used by Model 2's same-script pipeline.
- `data/_mybb_fonts/` contains the 2 fonts MyBB ships
  (`MINYN___.ttf`, `edmunds.ttf`). Downloaded once from MyBB's GitHub.

---

## Step 2 — PARSeq recogniser

### Architecture
Standard PARSeq from `third_party/parseq/strhub/models/parseq/system.py`
with parameters tuned for our short-fixed-length task:

| Hyperparameter | Value |
|---|---|
| Backbone | ViT-Small encoder, 12 layers, 6 heads, embed 384 |
| Decoder | 1 cross-attention layer, 12 heads |
| Input image | 32 × 128 RGB (resized from 200×60) |
| Patch size | 4 × 8 (→ 8 × 16 patch grid) |
| Charset | 62 classes |
| Max label length | 5 |
| `perm_num` | 1 (no permutation training — task is fixed-length random) |
| Total params | 23.8 M |

### Training settings
| | |
|---|---|
| Optim | AdamW, lr 5.0e-4, weight_decay 0, betas default |
| LR schedule | Linear warmup 5% → cosine decay |
| Batch size | 128 |
| Augmentation | Albumentations: ColorJitter, blur (3 kinds), GaussNoise, ImageCompression, Affine, CoarseDropout |
| Precision | fp32 |
| Epochs | 15 (with `save_top_k=3, save_last=true`) |
| Loss | Sum-of-positions char CE |
| Hardware | RTX A4000, single GPU |

### Config
`configs/parseq_v2.yaml`

### Run
```bash
.venv/Scripts/python.exe -m src.parseq.train --config configs/parseq_v2.yaml \
    > checkpoints/parseq_v2_train.log 2>&1
```

**Wall time:** ~3 hours (~12 min/epoch × 15).
**Disk:** top-3 ckpts + last + lightning_logs ≈ 400 MB.
**Best ckpt selection:** retrospective by `val_accuracy` — peak was
epoch 11 at 57.26% synth val word.

### Why 15 epochs / fp32 / heavy aug
PARSeq's permutation training is unstable — val_word swings 30+ pts between
adjacent epochs. Trying to peak at epoch 11 requires running at least 12 to
have it. fp16 doesn't help (caused divergence in earlier attempts).
Heavy aug is non-negotiable; without it the model overfits within 2 epochs.

---

## Step 3 — Detector (5-bbox regressor)

### Architecture
`src/detector/model.py::CharBoxNet` — small CNN (~1.18M params).
- 4 conv blocks (32→64→128→256, stride-2 downsamples)
- Adaptive avg pool → flatten → Linear(256 → 20)
- Sigmoid output → 5 boxes × (cx, cy, w, h) ∈ (0,1)
- Sorted left-to-right by `cx` in the loss (no permutation ambiguity)

### Training settings
| | |
|---|---|
| Optim | AdamW lr 1e-3 cosine, wd 1e-4 |
| Batch | 256 |
| Loss | GIoU on sorted boxes + 0.5 × smooth-L1 on (cx,cy) |
| Aug | Photometric only (color jitter); NO geometric (would break bbox labels) |
| Precision | fp16 mixed |
| Epochs | 30 |

### Config
`configs/detector_v2.yaml`

### Run
```bash
.venv/Scripts/python.exe -m src.detector.train --config configs/detector_v2.yaml \
    > checkpoints/detector_v2_train.log 2>&1
```

**Wall time:** ~45 min.
**Final val_IoU:** 0.7522 (loose — chars sometimes overlap detector boxes,
but char-CNN's pad+jitter training tolerates that).

---

## Step 4 — Char-CNN multi-scale ensemble (3 scales)

### Architecture
ResNet-18 (timm, no pretrained) with multi-task heads:
- **Letter head**: 62-class char identity
- **Case head**: 3-class {UPPER, LOWER, DIGIT}
- Joint loss: `letter CE + 0.5 × case CE`, label-smoothing 0.05

### Three independently-trained scales
Same architecture, different input size (CNN sees char crop after detector
bbox + 20% pad + resize):

| Scale | Input | Batch | Epochs | Wall time | val_letter | val_case |
|---|---|---|---|---|---|---|
| 64 | 64 × 64 | 256 | 8 | ~45 min | 99.12% | 99.14% |
| 96 | 96 × 96 | 192 | 8 | ~50 min | 99.18% | 99.21% |
| 128 | 128 × 128 | 128 | 8 | ~80 min | 99.05% | 99.20% |

Crops are sampled on-the-fly during training: per-image GT bboxes from the
PHP renderer's `bboxes.jsonl` → crop with 20% pad → ±15% bbox jitter
(robustness to detector imperfection) → resize.

### Training settings
| | |
|---|---|
| Optim | AdamW lr 5e-4 cosine, wd 1e-4 |
| Aug | ColorJitter, GaussianBlur, RandomErasing |
| Precision | fp16 mixed |

### Config
`configs/charcls_64_v2.yaml`, `charcls_96_v2.yaml`, `charcls_128_v2.yaml`

### Run (per scale)
```bash
.venv/Scripts/python.exe -m src.charcls.train --config configs/charcls_64_v2.yaml \
    > checkpoints/charcls_64_v2_train.log 2>&1
# repeat for 96, 128
```

**Wall time total:** ~3 hours (sum of three sequential runs).

---

## Step 5 — Inference / decoding

### Tool
`src/eval/case_decode.py`

### What it does
Runs Model 1.5's full pipeline against an eval directory:
1. PARSeq → per-position softmax over 62 chars (force-5 length)
2. Detector → 5 bboxes per image
3. Char-CNNs (3 scales) → cropped per-position softmaxes, averaged
4. Combine: `softmax(α·log P_parseq + (1-α)·log P_cnn)` per position
5. Apply chosen decoding strategy → predicted string per image
6. Score against ground truth labels

### Decoding strategies tested
A. greedy
B. force-detected length (auto-detected from training labels)
C. case-marginal
D. detected + marginal
E. ensemble at α ∈ {0.30, 0.50, 0.70, 0.90}
F. letter override (CNN replaces PARSeq if CNN confident)
G. case-flip override (CNN's case head replaces PARSeq's case)

### Run
```bash
.venv/Scripts/python.exe -m src.eval.case_decode \
    --ckpt "checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt" \
    --src data/eval --labels data/eval/labels.csv \
    --train-labels data/synth_mybb/labels.csv \
    --detector-ckpt checkpoints/detector_v2/best.pt \
    --charcls-ckpt "checkpoints/charcls_64_v2/best.pt,checkpoints/charcls_96_v2/best.pt,checkpoints/charcls_128_v2/best.pt"
```

**Wall time:** ~1 min 45 sec for all 10 strategies on 651 images
(uses fast char-CNN inference path with DataLoader prefetch).

---

## End-to-end training time

| Step | Component | Wall time |
|---|---|---|
| 1 | PHP synth (100k) | ~11 min |
| 2 | PARSeq | ~3 hours |
| 3 | Detector | ~45 min |
| 4a | Char-CNN @64 | ~45 min |
| 4b | Char-CNN @96 | ~50 min |
| 4c | Char-CNN @128 | ~80 min |
| 5 | Eval | ~2 min |
| **Total** | end-to-end | **~7 hours** |

Sequenced via `run_v2_overnight.sh` (chains all steps + final eval).

---

## Disk footprint

| Path | Size |
|---|---|
| `data/synth_mybb_v2/` | ~2 GB |
| `checkpoints/parseq_v2/` | ~400 MB (3 ckpts × ~95 MB + lightning_logs) |
| `checkpoints/detector_v2/` | ~5 MB |
| `checkpoints/charcls_64_v2/` | ~45 MB |
| `checkpoints/charcls_96_v2/` | ~45 MB |
| `checkpoints/charcls_128_v2/` | ~45 MB |
| `data/eval/` (real captchas) | ~3 MB |
| **Total** | **~2.5 GB** |

---

## Reproducibility seeds

| Component | Seed |
|---|---|
| Synth render | 11 |
| PARSeq train/val split | 0 |
| Detector train/val split | 0 |
| Char-CNN @64 | 0 |
| Char-CNN @96 | 1 |
| Char-CNN @96 | 2 |

PARSeq trajectory itself is **not deterministic** even with same seed —
permutation training has stochastic gradient choices that depend on
worker scheduling. Best ckpt at peak epoch is the meaningful artefact.

---

## Files index

### Sources
- `src/php_synth/generate_mybb_captcha.php`
- `src/parseq/{train,infer}.py`
- `src/detector/{model,data,train,infer}.py`
- `src/charcls/{model,data,train,infer,infer_fast}.py`
- `src/eval/{case_decode,score,infer_with_conf,calibrate_temperature,selective_with_ecps}.py`

### Configs
- `configs/parseq_v2.yaml`
- `configs/detector_v2.yaml`
- `configs/charcls_{64,96,128}_v2.yaml`

### Scripts
- `run_v2_overnight.sh` — full pipeline launcher (all 5 trainings + eval)

### Documentation
- `CONTEXT.md` — current state, latest results, what's next
- `model_one_v1_results.md` — original Model 1 (Windows fonts)
- `model_one_v2_results.md` — Model 1 v2 (MyBB fonts swap)
- `model_one_v1.5_results.md` — Model 1.5 v1 (ensemble on Windows-fonts synth)
- `model_one_v1.5_v2_results.md` — Model 1.5 v2 (current flagship-from-here)
- `analysis.md` — selective prediction analysis (with calibration)
- `FUTURE_NOTES.md` — backlog
- `TRAINING_INFRASTRUCTURE.md` — this file
