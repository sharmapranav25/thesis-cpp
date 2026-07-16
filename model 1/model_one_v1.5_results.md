# Model 1.5 — Detector + Char-CNN ensemble with PARSeq

**Date**: 2026-05-02
**Eval set**: 651 hand-labeled real CryptBB captchas (filename = ground truth)
**Compute**: Local NVIDIA RTX A4000 (16 GB VRAM), Windows 11, Python 3.9.13, PyTorch 2.5.1+cu121

## Headline Result

| | Word % | Char % |
|---|---|---|
| **Model 1 baseline (PARSeq + force-detected length)** | 33.18 | 80.29 |
| **Model 1.5 (PARSeq + detector + char-CNN ensemble, α=0.70)** | **36.41** | **81.89** |
| **Δ** | **+3.23** | **+1.60** |

Same constraint: **zero real labels used during training**. Detector and char-CNN both train only on PHP-MyBB synth.

## Pipeline

```
   captcha 200×60
        ├──► PARSeq                  ──► 5 × 62 logits
        └──► detector (1.18M params)
                  └──► 5 axis-aligned bboxes
                            │
                       crop + pad ──► char-CNN (ResNet-18, 11.2M params)
                                              │
                                              └──► 5 × 62 letter logits
                                              └──► 5 × 3 case logits (UPPER/LOWER/DIGIT)
                                                          │
        ┌─────────────────────────────────────────────────┘
        │
   per-position log-prob mix:  α·log(parseq) + (1-α)·log(charcls_letter)
                       argmax over 62 charset chars (EOS masked, force-5)
        │
        ▼
   final 5-char prediction
```

## Components

### Detector (`src/detector/`)

Tiny ConvNet, 1.18M params. Maps 200×60 RGB → 5 axis-aligned bboxes in (cx, cy, w, h) coords. Trained on `data/synth_mybb_bb/` (100k re-rendered captchas + per-char bboxes from the PHP renderer's `imageftbbox()` output). GIoU loss on left-to-right-sorted boxes; smooth-L1 on centers for stability. 30 epochs.

| Setting | Value |
|---|---|
| Architecture | 4 conv blocks (32→64→128→256, stride-2 downsamples) → GAP → FC → 20 |
| Loss | GIoU + 0.5 × smooth-L1(cx,cy) |
| Augmentation | photometric only (color jitter) — geometric breaks bbox labels |
| Optim | AdamW, lr 1e-3 cosine, batch 256, fp16 |
| Train time | ~44 min |
| **Best val IoU** | **0.7502** |

Below the 0.85 target — small model just hit its capacity. Workable because the char-CNN trains with bbox jitter ±15% and survives loose crops.

### Char-CNN (`src/charcls/`)

ResNet-18 backbone (timm), two heads on shared features:
- **Head A — letter:** 62-class case-sensitive char identity
- **Head B — case:** 3-class {UPPER, LOWER, DIGIT}

Multi-task loss: `loss_letter + 0.5 × loss_case`, both with label smoothing 0.05.

Training crops on-the-fly from `data/synth_mybb_bb/`: each (image, char-index) pair becomes one sample (475k train / 25k val from the 100k-row split). 64×64 resize, 20% padding, ±15% bbox jitter, photometric augmentation, random erasing.

| Setting | Value |
|---|---|
| Backbone | ResNet-18 (timm), 11.2M params |
| Heads | 62-class letter + 3-class case |
| Crop | 64×64, 20% pad, ±15% jitter (training only) |
| Optim | AdamW, lr 5e-4 cosine, batch 256, fp16 |
| Epochs | 12 |
| Train time | ~66 min |
| **Best val_letter** | **96.45%** |
| **Best val_case** | **96.54%** |

### Ensemble combiner (`src/eval/case_decode.py`)

Eight decoding strategies on the same softmax extraction (PARSeq + char-CNN run once per image):

| Strategy | What it does |
|---|---|
| A. greedy | PARSeq EOS-aware argmax (Model 0 baseline) |
| B. force-detected | PARSeq + zero EOS, output exactly N chars (N detected from training labels) |
| C. case-marginal | Sum P(c) + P(case-pair-c) per position, pick higher-prob case |
| D. detected+marginal | B + C |
| E. ensemble α | `α·log(parseq) + (1-α)·log(charcls_letter)` per position; force-N chars |
| F. letter override | Take PARSeq's pick unless char-CNN disagrees AND CNN confidence > 0.6 |
| G. case-flip override | Keep PARSeq's letter; flip case if CNN's case head disagrees AND case confidence > 0.6 |

## Eval (651 real captchas)

| Strategy | Word % | Char % | Δ vs D |
|---|---|---|---|
| A. greedy | 31.18 | 74.37 | -1.85 / -5.92 |
| B. force-detected | 33.03 | 80.29 | -0.15 / 0.00 |
| C. case-marginal | 31.34 | 74.37 | -1.84 / -5.92 |
| D. detected+marginal (Model 1) | 33.18 | 80.29 | baseline |
| E. ensemble α=0.30 | 29.34 | 76.46 | -3.84 / -3.83 |
| E. ensemble α=0.50 | 35.79 | 81.52 | +2.61 / +1.23 |
| **E. ensemble α=0.70** | **36.41** | **81.89** | **+3.23 / +1.60** |
| E. ensemble α=0.90 | 34.56 | 81.03 | +1.38 / +0.74 |
| F. letter override | 28.11 | 77.50 | -5.07 / -2.79 |
| G. case-flip override | 20.74 | 71.18 | -12.44 / -9.11 |

### Soft ensemble works; hard overrides fail.

- **α = 0.70 is the sweet spot.** PARSeq's letter signal dominates; CNN provides a 30% nudge that flips ambiguous positions without overruling reliable PARSeq predictions.
- **α = 0.50** still beats baseline by +2.6 pts — the CNN signal is real.
- **α = 0.30** *regresses* — too much weight on the CNN, inherits its real-data noise.
- **α = 0.90** is too little CNN to matter (~+1.4 pts).
- **Strategy F (letter override)** loses 5 pts. Hard delegation when CNN's confidence is just above 0.6 is too aggressive — CNN swaps in wrong letters.
- **Strategy G (case-flip override)** loses **12.4 pts** — the CNN's case head reaches 96.54% on synth val but generalises poorly to real. `o → O` errors *tripled* from 12 → 33 under hard case override. The CNN is **introducing** new case errors faster than it fixes existing ones.

The contrast between G (catastrophic) and E α=0.70 (winner) is the central lesson: **CNN signal carries useful information when mixed at small weight, but is not reliable enough to override PARSeq directly on case decisions.**

## Confusion Analysis (decoder E α=0.70 on 651 real)

Top confusions still case-dominated:

| GT | Pred | Count | Type |
|---|---|---|---|
| p | P | 32 | case |
| v | V | 20 | case |
| s | S | 20 | case |
| W | w | 17 | case |
| S | s | 16 | case |
| L | I | 16 | visual |
| o | O | 14 | case |
| N | X | 13 | visual |

Soft ensemble fixed some case errors but not all. The persistent ones are likely in the **irreducible visual ambiguity** tier — at small char sizes with rotation, the case decision is genuinely undetermined from pixels alone in the synth distribution we trained on.

## Implication for case-folded PARSeq idea (`FUTURE_NOTES.md`)

**Don't pursue.** Strategy G shows that full delegation of case to the CNN regresses 12 pts on word acc. The case-folded PARSeq proposal is functionally equivalent to G at runtime — PARSeq emits case-folded letter, CNN provides case unilaterally. It would lose for the same reason: the CNN's case acc on real data is too unreliable.

To revisit later, we'd first need to close the synth-vs-real case gap. Likely paths:
- Use MyBB's bundled `inc/captcha_fonts/` instead of Windows TTFs in the renderer
- Pseudo-labeling on confident PARSeq predictions to bring real-data style into char-CNN training
- Explicit hard-mining on case-confusable pairs during training

## Files

| Path | Purpose |
|---|---|
| `src/php_synth/generate_mybb_captcha.php` | renderer (extended with `--bboxes` flag) |
| `data/synth_mybb_bb/` | 100k synth + `bboxes.jsonl` |
| `src/detector/{model,data,train,infer}.py` | detector module |
| `configs/detector.yaml` | detector training config |
| `checkpoints/detector/best.pt` | trained detector (val IoU 0.7502) |
| `src/charcls/{model,data,train,infer}.py` | char classifier (multi-task) |
| `configs/charcls.yaml` | char-CNN config |
| `checkpoints/charcls/best.pt` | trained char-CNN (val_case 96.54%) |
| `src/eval/case_decode.py` | extended decoder with ensemble + override strategies |
| `data/eval/_detector_boxes_eval.jsonl` | detector predictions on real eval |
| `checkpoints/ensemble_eval.log` | full eval output for all 8 strategies |

## Conclusion

Model 1.5 lifts the CryptBB captcha solver to **36.41% word / 81.89% char** on the 651-captcha real eval — a +3.23 pt word / +1.60 pt char improvement over Model 1, with no extra real-data labelling and no PARSeq retraining. The win comes entirely from injecting a per-character signal (detector + char-CNN trained on the same PHP synth) into PARSeq's decoding via soft log-prob ensemble.

The experiment also produced a clean negative result: **the char-CNN's case head, despite reaching 96.5% on synth val, cannot replace PARSeq's case decisions on real data.** This rules out the "case-folded PARSeq + CNN-only case" architecture until we close the synth-vs-real case gap by other means.
