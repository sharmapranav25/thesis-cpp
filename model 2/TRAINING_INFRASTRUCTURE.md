# Model 2 — Training Infrastructure

How Model 2 (= Model 1's PARSeq replaced with curriculum-trained variant)
and Model 2.5 (= Model 2's PARSeq + Model 1's CNN ensemble) are built
end-to-end. Model 2 reuses **everything from Model 1 except the recogniser** —
detector and char-CNNs are imported as-is.

---

## Overview

```
   100k MyBB-fonts CryptBB synth (rendered fresh, seed=1234)
   data/model2_training_data/images/
                       │
   ┌───────────────────┴────────────────────┐
   │ STEP A: bucket the synth pool          │
   │  Run M1.5 v2 ensemble (PARSeq+detector │
   │  +3 char-CNNs) on every sample.        │
   │  Compute per-sample (NLL, P_correct,   │
   │  correct). Write into labels.csv with  │
   │  a `bucket` column = NLL tertile.      │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   labels.csv with 5+ cols:
   filename, label, difficulty=1.0, is_fixed_length=1, bucket
                       │
   ┌───────────────────┴────────────────────┐
   │ STEP B: train PARSeq with bucket       │
   │  curriculum sampler.                   │
   │  Same architecture as Model 1. Only    │
   │  the sampler differs.                  │
   │  Curriculum schedule: 1/1/13           │
   │   ep 0:    easy=1.0, med=0, hard=0     │
   │   ep 1:    easy=1.0, med=1.0, hard=0   │
   │   ep 2-14: easy=0.15, med=0.25,        │
   │            hard=0.60                   │
   └───────────────────┬────────────────────┘
                       │
                       ▼
   Recogniser ckpt: parseq-epoch=03-val_accuracy=78.260.ckpt
                       │
   ┌───────────────────┴────────────────────┐
   │ STEP C: ensemble eval                  │
   │  Same as Model 1 v2 — feed the new     │
   │  PARSeq into case_decode.py with       │
   │  Model 1's detector_v2 + char-CNN      │
   │  ensemble. Best α=0.50.                │
   └───────────────────┬────────────────────┘
                       │
                       ▼
        Model 2:  54.22% word / 87.88% char (PARSeq alone)
        Model 2.5: 57.60% word / 89.04% char (with ensemble)
                  on 651 real CryptBB
```

**Detector and char-CNNs are NOT retrained for Model 2** — same checkpoints
from Model 1's `model 1/checkpoints/detector_v2/` and
`charcls_{64,96,128}_v2/` are reused unchanged.

---

## Step 0 — Synthetic data generation (fresh pool)

Model 2 renders a **separate** 100k pool from Model 1 (seed=1234 vs 11) so
that the bucketing's "calibration" data is disjoint from anything Model 1's
PARSeq saw during gradient updates.

```bash
C:/xampp/php/php.exe \
  "../model 1/src/php_synth/generate_mybb_captcha.php" \
  --out data/model2_training_data/images \
  --n 100000 --seed 1234 \
  --fontset mybb --fonts ../model\ 1/data/_mybb_fonts
```

**Wall time:** ~9 min.
**Output:** 100k PNGs in `images/`, plus `images/labels.csv` (2-col:
filename, label).

---

## Step A — Score + bucket the synth pool

### Tools
- `../model 1/src/eval/score_synth_difficulty.py` — Model 1.5 v2 inference
  on every synth sample, emits per-sample (NLL, P_correct, correct)
- `../model 1/src/eval/bucket_synth.py` — sorts by NLL tertile, emits bucket CSVs

### Score (~30 min)
```bash
"../model 1/.venv/Scripts/python.exe" \
  -m src.eval.score_synth_difficulty \
  --src "../model 2/data/model2_training_data/images" \
  --labels-in "../model 2/data/model2_training_data/images/labels.csv" \
  --out "../model 2/data/model2_training_data/labels_scored.csv" \
  --ckpt "checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt" \
  --detector-ckpt checkpoints/detector_v2/best.pt \
  --charcls-ckpt "checkpoints/charcls_64_v2/best.pt,checkpoints/charcls_96_v2/best.pt,checkpoints/charcls_128_v2/best.pt" \
  --alpha 0.30 --fixed-n 5
```

Per-sample math: `NLL = -log(∏_t softmax(α·logP_parseq + (1-α)·logP_cnn)[true_t])`.

### Bucket (~30 sec)
```bash
"../model 1/.venv/Scripts/python.exe" \
  -m src.eval.bucket_synth \
  --scored data/model2_training_data/labels_scored.csv \
  --out-dir data/model2_training_data
```

Emits `easy.csv`, `medium.csv`, `hard.csv` (each ~33,333 rows).
Bucket sizes are equal-third by NLL.

### Bucket statistics (Model 1.5 v2 as the teacher)
| Bucket | NLL range | Samples | Teacher accuracy |
|---|---|---|---|
| Easy | [0.007, 0.035] | 33,334 | 100.0% |
| Medium | [0.035, 0.161] | 33,333 | 100.0% |
| Hard | [0.161, 6.583] | 33,333 | 79.1% |

The hard bucket is the only one where the deployed teacher itself fails —
~7k samples that even the flagship gets wrong on synth val.

---

## Step B — Curriculum PARSeq training

### Architecture
**Identical to Model 1's PARSeq v2.** No length head, no scheduled sampling
(`length_head_enabled: false`). Just the `BucketCurriculumSampler` swapped
in for `shuffle=True`.

### Training settings (vs Model 1 v2)
| | Model 1 v2 | Model 2 v2 (this) |
|---|---|---|
| Sampler | shuffle, full pool every epoch | `BucketCurriculumSampler` (per-epoch bucket weights) |
| Schedule | n/a | 1 / 1 / 13 stages |
| Epochs | 15 | 15 |
| Augmentation | (default ON) | **`augment: true`** explicit (off causes overfit collapse) |
| Stage-end checkpoints | none | epochs 0, 1, 14 saved separately |
| Everything else | same | same |

### Curriculum schedule
| Epoch | easy | medium | hard | Effective composition (95k samples) |
|---|---|---|---|---|
| 0 | 1.0 | 0.0 | 0.0 | 95k from easy bucket (~3× reps per unique) |
| 1 | 1.0 | 1.0 | 0.0 | 47.5k easy + 47.5k medium (~1.5× reps each) |
| 2-14 | 0.15 | 0.25 | 0.60 | 14.25k easy + 23.75k medium + 57k hard |

Sampling is **with replacement**, normalized so the per-epoch sample count
equals the train set size (95k after 5% val holdout).

Cumulative exposure (×15 epochs):
- Easy (31,666 unique): ~328k seen → ~10× per unique
- Medium (31,663): ~356k seen → ~11×
- Hard (31,671): ~741k seen → **~23×**

### Config
`configs/parseq_arc_m2_teacher_v2.yaml`

### Run
```bash
"../model 1/.venv/Scripts/python.exe" -m src.parseq.train \
    --config configs/parseq_arc_m2_teacher_v2.yaml \
    > checkpoints/parseq_arc_m2_teacher_v2.log 2>&1
```

**Wall time:** ~3 hours (~12 min/epoch × 15).

### Critical config decisions
| Choice | Why |
|---|---|
| `augment: true` | `false` causes catastrophic overfit (val_word stuck at 1.30%). Sampling-with-replacement on the small (~31k) easy bucket in epoch 0 means each sample shows up 3× per epoch with no aug variation → memorisation. With aug on, the stochastic per-batch perturbation prevents this. |
| 1 / 1 / 13 schedule | Earlier 5/5/5 and 3/3/9 schedules locked the model into an over-easy local optimum before mixing in the full distribution. Keep scaffolding short; spend most of training on the full mix. |
| Bucket sizes equal-third | Cleanest tertile split. Other splits (e.g. correct-vs-wrong) under-populate the "hard" bucket. |
| Teacher = M1.5 v2 ensemble | Stronger teacher → cleaner difficulty signal. Earlier attempts using PARSeq alone as teacher gave noisier buckets. |

### Best-checkpoint selection
| Epoch | val_NED | val_word | val_loss |
|---|---|---|---|
| 0 | 2.86 | 0.00 | 3.43 |
| 1 | 71.47 | 1.48 | 1.72 |
| 2 | 92.70 | 67.68 | 0.19 |
| **3** | **95.21** | **78.26** | **0.12** ← peak |
| 4 | 81.99 | 18.00 | 1.17 |
| ... | ... | ... | ... |
| 14 | 83.70 | 22.74 | 1.31 |

Peak val_word lands at **epoch 3** — ~4× faster than Model 1 v2's epoch 11
peak (57.26%). Subsequent epochs bounce between 2-25% without recovering.
**Retrospective best-ckpt selection is essential** — the peak is rare and
not detectable a priori.

---

## Step C — Ensemble eval (reuses Model 1 components)

Same `case_decode.py` as Model 1, just pointed at the new PARSeq ckpt.
Detector and char-CNNs are unchanged.

```bash
"../model 1/.venv/Scripts/python.exe" -m src.eval.case_decode \
    --ckpt "checkpoints/parseq_arc_m2_teacher_v2/parseq-epoch=03-val_accuracy=78.260.ckpt" \
    --src ../model\ 1/data/eval --labels ../model\ 1/data/eval/labels.csv \
    --train-labels data/model2_training_data/labels.csv \
    --detector-ckpt ../model\ 1/checkpoints/detector_v2/best.pt \
    --charcls-ckpt "../model 1/checkpoints/charcls_64_v2/best.pt,../model 1/checkpoints/charcls_96_v2/best.pt,../model 1/checkpoints/charcls_128_v2/best.pt"
```

**Wall time:** ~1.5 min for all 10 decoding strategies.

**Optimal α changed from 0.30 (Model 1.5) to 0.50 (Model 2.5)** — the
curriculum-trained PARSeq is now strong enough to merit equal weight with
the CNN ensemble.

---

## End-to-end Model 2 training time

| Step | Component | Wall time |
|---|---|---|
| 0 | Render fresh 100k synth | ~9 min |
| A1 | Score with M1.5 v2 ensemble | ~30 min |
| A2 | Bucket | ~30 sec |
| B | PARSeq curriculum train (15 ep) | ~3 hours |
| C | Eval | ~2 min |
| **Total** | end-to-end | **~3.7 hours** |

Detector and char-CNNs are reused from Model 1 — **add ~3.7 hours** if
training those from scratch (one-time cost; they're scheme-stable).

---

## Disk footprint

| Path | Size |
|---|---|
| `data/model2_training_data/images/` | ~2 GB (100k PNGs) |
| `data/model2_training_data/labels*.csv` | ~10 MB |
| `data/model2_training_data/_score_boxes.jsonl` | ~40 MB |
| `checkpoints/parseq_arc_m2_teacher_v2/` | ~400 MB (3 top + last + 3 stage-end) |
| **Total (Model 2 only)** | **~2.5 GB** |
| Model 1 components reused | ~150 MB (detector_v2 + charcls_*_v2) |

---

## Reproducibility seeds

| Component | Seed |
|---|---|
| Synth render | 1234 |
| Curriculum scoring (M1.5 inference) | n/a (deterministic) |
| Bucket assignment | deterministic from NLL ranks |
| PARSeq train/val split | 0 |
| Sampler RNG | numpy default — not seeded explicitly |

The peak at epoch 3 is **stochastic** — different sampler RNG seeds may not
reproduce the same epoch-3 breakthrough. The fact that peak val_word can
reach >70% is reproducible across seeds; which epoch it lands at is not.

---

## Why this curriculum worked when prior attempts failed

| Property | Failed M2 v1 | Working M2 v2 |
|---|---|---|
| Bucket source | Human d-knob (renderer's `--difficulty-mode`, samples from different distributions) | Teacher per-sample NLL on a single distribution (d=1.0 only) |
| Augmentation | Mostly off (carry-over from fine-tune-from-converged experiments) | **On** |
| Schedule | 5/5/5 (long scaffolding) or scheduled-sampling | 1/1/13 (short scaffolding) |
| Length variability | Trained on 3-7 char range with length head | Fixed-5 only, no length head |
| Architecture | `PARSeqWithLengthHead` | Vanilla PARSeq |

The dominant fix was **bucket source**: training on samples drawn from
*different* synth distributions (low-d easy vs high-d hard) means the model
never sees the deployment distribution until late training. Teacher buckets
keep the distribution constant and just *order* samples — same target
distribution, smarter learning order.

---

## Files index

### Sources (Model 2 specific)
- `src/parseq/train.py` — adds `BucketCurriculumSampler`,
  `BucketCurriculumCallback`, `StageEndCheckpointCallback`. `CaptchaDir` reads 5-col CSV.
- `src/php_synth/merge_pools.py` — utility for merging multiple pools (used
  by failed M2 v1 d-knob run; preserved for reproducibility).
- `src/parseq/infer.py` — emits per-position confidences, length-head outputs
  optional.

### Configs
- **`configs/parseq_arc_m2_teacher_v2.yaml`** — flagship
- `configs/parseq_arc_m2_teacher.yaml` — first attempt (`augment: false` —
  failed, kept as a negative-result reference)
- `configs/parseq_arc_m2.yaml` — d-knob bucket attempt (failed)
- Other `parseq_m2_*.yaml` — earlier negative results

### Sources (reused from Model 1)
Imported via parent venv (`../model 1/.venv/`):
- `src/eval/score_synth_difficulty.py`
- `src/eval/bucket_synth.py`
- `src/eval/case_decode.py`
- `src/detector/`, `src/charcls/`

### Documentation
- `CONTEXT.md` — current state, latest results, what's next
- `model_two_v1_results.md` — earlier negative-result writeup (length-head, d-knob)
- `model_two_v2_curriculum_results.md` — current flagship writeup (Model 2.5)
- `analysis.md` — selective prediction analysis (mirrors model 1's)
- `TRAINING_INFRASTRUCTURE.md` — this file
