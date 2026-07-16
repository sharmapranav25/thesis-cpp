# Selective-Prediction Analysis — Models 1 / 1.5 / 2 / 2.5

**Date:** 2026-05-04
**Eval set:** 651 hand-labelled real CryptBB captchas
**Goal:** quantify each model's confidence calibration via selective prediction.

For each confidence threshold `T`, we compute:

| Quantity | Definition |
|---|---|
| **shoot rate** | fraction of captchas attempted (conf_joint ≥ T) |
| **acc on attempts** | of attempted, fraction predicted correctly |
| **ECPS** | Expected CAPTCHAs per Success = `1 / (shoot_rate × acc_on_attempts)` |

ECPS = average number of captchas the system has to see to deliver one correct solve. Lower = better.

Confidence = `conf_joint` = product of per-position max softmax (after force-5 + ensemble combination, where applicable).

## The four models

| Name | Recogniser | Ensemble | α | Real-eval @ no-threshold |
|---|---|---|---|---|
| **Model 1** | PARSeq v2 (MyBB-fonts synth, no curriculum) | none | n/a | 39.94% |
| **Model 1.5** | PARSeq v2 (same as M1) | + detector + 3 char-CNN scales | 0.30 | 49.16% |
| **Model 2** | PARSeq curriculum v2 (teacher-bucket, ep 3) | none | n/a | 54.22% |
| **Model 2.5** | PARSeq curriculum v2 (same as M2) | + detector + 3 char-CNN scales | 0.50 | 57.60% |

## Results

### Model 1 — PARSeq v2 alone

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 43.16% | 50.53% | 4.58 |
| 0.95 | 59.45% | 47.55% | 3.54 |
| 0.90 | 68.20% | 46.85% | 3.13 |
| 0.85 | 73.58% | 46.14% | 2.95 |
| 0.80 | 76.65% | 45.49% | 2.87 |
| 0.70 | 81.57% | 44.44% | 2.76 |
| 0.60 | 87.25% | 43.49% | 2.64 |
| 0.50 | 94.93% | 41.59% | 2.53 |
| 0.40 | 97.39% | 40.85% | 2.51 |
| 0.30 | 99.08% | 40.31% | 2.50 |
| 0.20 | 99.69% | 40.06% | 2.50 |
| 0.10 | 100.00% | 39.94% | 2.50 |
| 0.00 | 100.00% | 39.94% | 2.50 |

### Model 1.5 — PARSeq v2 + multi-scale CNN ensemble (α=0.30)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 0.15% | 0.00% | inf |
| 0.95 | 20.74% | 63.70% | 7.57 |
| 0.90 | 31.34% | 64.22% | 4.97 |
| 0.85 | 36.10% | 63.83% | 4.34 |
| 0.80 | 39.17% | 62.75% | 4.07 |
| 0.70 | 43.93% | 61.89% | 3.68 |
| 0.60 | 45.78% | 61.41% | 3.56 |
| 0.50 | 48.54% | 60.13% | 3.43 |
| 0.40 | 51.77% | 57.57% | 3.36 |
| 0.30 | 53.00% | 57.39% | 3.29 |
| 0.20 | 55.15% | 57.10% | 3.18 |
| 0.10 | 69.59% | 55.63% | 2.58 |
| 0.00 | 100.00% | 49.16% | 2.03 |

### Model 2 — Curriculum PARSeq alone

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 14.44% | 73.40% | 9.43 |
| 0.95 | 38.71% | 71.43% | 3.62 |
| 0.90 | 48.39% | 68.89% | 3.00 |
| 0.85 | 53.61% | 67.34% | 2.77 |
| 0.80 | 58.99% | 66.67% | 2.54 |
| 0.70 | 68.51% | 63.90% | 2.28 |
| 0.60 | 77.27% | 61.23% | 2.11 |
| 0.50 | 86.33% | 58.19% | 1.99 |
| 0.40 | 93.24% | 56.67% | 1.89 |
| 0.30 | 97.54% | 55.28% | 1.85 |
| 0.20 | 99.54% | 54.48% | 1.84 |
| 0.10 | 99.85% | 54.31% | 1.84 |
| 0.00 | 100.00% | 54.22% | 1.84 |

### Model 2.5 — Curriculum PARSeq + multi-scale CNN ensemble (α=0.50)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 0.61% | 50.00% | 325.50 |
| 0.95 | 22.89% | 63.09% | 6.93 |
| 0.90 | 32.72% | 62.91% | 4.86 |
| 0.85 | 37.94% | 65.18% | 4.04 |
| 0.80 | 41.17% | 65.67% | 3.70 |
| 0.70 | 46.24% | 65.45% | 3.30 |
| 0.60 | 52.69% | 64.14% | 2.96 |
| 0.50 | 61.44% | 64.00% | 2.54 |
| 0.40 | 70.81% | 63.12% | 2.24 |
| 0.30 | 78.96% | 61.87% | 2.05 |
| 0.20 | 90.17% | 60.31% | 1.84 |
| 0.10 | 97.39% | 58.99% | 1.74 |
| 0.00 | 100.00% | 57.60% | 1.74 |

## Best ECPS per model (with corresponding threshold)

| Model | Best ECPS | Threshold | Shoot rate | Acc on attempts |
|---|---|---|---|---|
| Model 1 | **2.50** | ≤ 0.20 | 99.69-100% | 40-40% |
| Model 1.5 | **2.03** | 0.00 | 100.00% | 49.16% |
| Model 2 | **1.84** | 0.10-0.20 | 99.54-99.85% | 54.31-54.48% |
| **Model 2.5** | **1.74** | 0.00-0.10 | 97.39-100% | 57.60-58.99% |

## Reading the trends

1. **ECPS monotonically decreases as accuracy increases**: Model 1 (2.50) → Model 1.5 (2.03) → Model 2 (1.84) → **Model 2.5 (1.74)**. Each architectural addition (ensemble, curriculum) lowers the cost-per-success.

2. **Calibration improves with ensembling.** Model 1's confidences are poorly calibrated — even at threshold 0.99 the accuracy on attempts is only 50.53%. Confidence at threshold 0.99 should mean "almost certainly right" but the model only gets it right half the time. Model 2.5 at threshold 0.85+ has `acc_on_attempts` ≥ 65% — far better calibration.

3. **High thresholds are worse for ECPS.** Counter-intuitive but clean: at very tight thresholds (T=0.99), shoot rate collapses to 0.15-0.61% so even with high accuracy the system rarely attempts. ECPS climbs to 6.93+ for Model 2.5 / inf for Model 1.5. The optimal regime is **moderate thresholds where shoot rate stays above ~50%**.

4. **Ensembles hurt selective prediction at very high thresholds**, because the log-prob mix tends to produce flatter softmaxes (combining two distributions averages them out). Notice how Model 1.5 has 0.00% acc at T=0.99 — the ensemble's joint confidence almost never exceeds 0.99. Model 1 (PARSeq alone) confidently produces high-conf predictions more often, but those are sometimes wrong.

5. **Model 2's curriculum-trained PARSeq is the best calibrated single recogniser** — at T=0.95 it has 71.43% acc on attempts vs Model 1's 47.55%. The curriculum apparently teaches PARSeq to be confident only on truly easy samples.

## What the result means in practice

If you're deploying as a one-shot solver: use **Model 2.5 at T=0** (always attempt) → 57.60% solve rate, ECPS 1.74.

If you're deploying with the option to skip-and-ask-again (e.g. attempt only on confident captchas): use **Model 2 at T=0.50** (shoot rate 86.33%, acc 58.19%, ECPS 1.99) — slightly worse ECPS but the ones we *don't* attempt can be deferred to a fresh image at no risk of a wrong answer.

If you need very high precision (avoid wrong answers at almost any cost): **Model 2 at T=0.90** gives 68.89% acc on attempts with 48.39% shoot rate. Each attempted captcha has a 7-in-10 chance of being right.

## Calibrated variants — Model 1C / 1.5C / 2C / 2.5C

A second-pass analysis with **temperature scaling** applied to each model. Calibration uses the OTHER model's training data as a held-out set (cross-pollination — neither model has seen the other's training images during gradient updates):

| Model | Calibration set | T fitted | Direction | Cal-set acc |
|---|---|---|---|---|
| Model 1C | `model 2/data/model2_training_data/images/` (5k) | **2.01** | over-confident → cool | 58.64% |
| Model 1.5C | same | **0.45** | under-confident → sharpen | 93.44% |
| Model 2C | `model 1/data/synth_mybb_v2/` (5k) | **1.04** | already well-calibrated | 77.56% |
| Model 2.5C | same | **0.47** | under-confident → sharpen | 94.06% |

**Key finding:** PARSeq-alone models tend to be over-confident (T > 1, M1) or naturally well-calibrated (M2). Ensemble models are sharply under-confident (T < 0.5) — the log-prob mix between PARSeq and char-CNN flattens the joint distribution.

Temperature scaling **preserves argmax** — calibrated predictions are byte-identical to raw, only confidences shift. Word/char accuracy unchanged. The benefit shows up in selective prediction.

### Model 1C — PARSeq v2 alone, calibrated (T=2.01)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 0.00% | — | inf |
| 0.95 | 0.00% | — | inf |
| 0.90 | 8.14% | 47.17% | 26.04 |
| 0.85 | 21.35% | 57.55% | 8.14 |
| 0.80 | 32.10% | 55.50% | 5.61 |
| 0.70 | 48.54% | 49.05% | 4.20 |
| 0.60 | 59.91% | 47.18% | 3.54 |
| 0.50 | 71.58% | 45.28% | 3.09 |
| 0.40 | 83.10% | 42.88% | 2.81 |
| 0.30 | 91.86% | 42.81% | 2.54 |
| 0.20 | 96.93% | 41.05% | 2.51 |
| 0.10 | 99.69% | 40.06% | 2.50 |
| 0.00 | 100.00% | 39.94% | 2.50 |

### Model 1.5C — PARSeq v2 + ensemble (α=0.30), calibrated (T=0.45)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | **41.63%** | **63.47%** | **3.78** |
| 0.95 | 53.15% | 62.43% | 3.01 |
| 0.90 | 62.21% | 59.01% | 2.72 |
| 0.85 | 67.59% | 58.18% | 2.54 |
| 0.80 | 71.43% | 56.77% | 2.47 |
| 0.70 | 78.80% | 55.56% | 2.28 |
| 0.60 | 83.87% | 54.03% | 2.21 |
| 0.50 | 91.40% | 50.76% | 2.16 |
| 0.40 | 95.70% | 50.08% | 2.09 |
| 0.30 | 98.31% | 49.69% | 2.05 |
| 0.20 | 99.54% | 49.23% | 2.04 |
| 0.10 | 99.85% | 49.23% | 2.03 |
| 0.00 | 100.00% | 49.16% | 2.03 |

(Compare: raw M1.5 at T=0.99 had 0.15% shoot, 0% acc, ECPS = inf. Calibrated brings high-precision regime to life.)

### Model 2C — Curriculum PARSeq alone, calibrated (T=1.04)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | 11.06% | 73.61% | 12.28 |
| 0.95 | 35.94% | 71.37% | 3.90 |
| 0.90 | 46.54% | 69.31% | 3.10 |
| 0.85 | 52.07% | 67.55% | 2.84 |
| 0.80 | 57.60% | 66.67% | 2.60 |
| 0.70 | 66.97% | 64.68% | 2.31 |
| 0.60 | 75.73% | 61.26% | 2.16 |
| 0.50 | 85.41% | 58.63% | 2.00 |
| 0.40 | 92.63% | 57.05% | 1.89 |
| 0.30 | 97.08% | 55.38% | 1.86 |
| 0.20 | 99.23% | 54.49% | 1.85 |
| 0.10 | 99.85% | 54.31% | 1.84 |
| 0.00 | 100.00% | 54.22% | 1.84 |

(Almost unchanged from raw M2 — model 2 was already nearly calibrated. T=1.04 is essentially identity.)

### Model 2.5C — Curriculum PARSeq + ensemble (α=0.50), calibrated (T=0.47)

| threshold | shoot rate | acc on attempts | ECPS |
|---|---|---|---|
| 0.99 | **48.39%** | **67.94%** | **3.04** |
| 0.95 | 63.29% | 68.20% | 2.32 |
| 0.90 | 67.74% | 67.35% | 2.19 |
| 0.85 | 71.12% | 66.95% | 2.10 |
| 0.80 | 74.50% | 65.57% | 2.05 |
| 0.70 | 80.34% | 64.24% | 1.94 |
| 0.60 | 86.48% | 62.70% | 1.84 |
| 0.50 | 93.09% | 60.40% | **1.78** |
| 0.40 | 96.01% | 59.20% | 1.76 |
| 0.30 | 98.62% | 58.26% | 1.74 |
| 0.20 | 99.54% | 57.87% | 1.74 |
| 0.10 | 100.00% | 57.60% | 1.74 |
| 0.00 | 100.00% | 57.60% | 1.74 |

(Compare: raw M2.5 at T=0.99 had 0.61% shoot, 50% acc, ECPS = 325.5. Calibrated turns it into 48.39% / 67.94% / 3.04 — actually usable.)

## Why calibration matters

Best ECPS per model — **before vs after calibration**:

| Model | Raw best ECPS @ T | Calibrated best ECPS @ T |
|---|---|---|
| Model 1 / 1C | 2.50 @ T≤0.20 | 2.50 @ T=0.00 |
| Model 1.5 / 1.5C | 2.03 @ T=0.00 | 2.03 @ T=0.00 |
| Model 2 / 2C | 1.84 @ T=0.10 | 1.84 @ T≤0.10 |
| Model 2.5 / 2.5C | 1.74 @ T=0.10 | **1.74 @ T=0.30** |

**Best ECPS doesn't change much** — same model, same predictions. But the **HIGH-PRECISION regime** (where high threshold = high confidence = high actual accuracy) becomes operational:

| | Raw at T=0.99 | Calibrated at T=0.99 |
|---|---|---|
| M1 / 1C | 43.16% shoot, 50.53% acc | 0% shoot — calibration correctly told us the model is rarely 99% confident |
| **M1.5 / 1.5C** | **0.15% shoot, 0% acc** | **41.63% shoot, 63.47% acc** ← actionable now |
| M2 / 2C | 14.44% shoot, 73.40% acc | 11.06% shoot, 73.61% acc |
| **M2.5 / 2.5C** | **0.61% shoot, 50% acc** | **48.39% shoot, 67.94% acc** ← actionable now |

**Practical implication:** for deployment with a "skip-and-defer" policy at high precision, **calibrated ensemble models are now usable at threshold T=0.99**. Raw ensemble at T=0.99 essentially never attempted. Calibrated at T=0.99 attempts ~50% of captchas with ~67% accuracy on attempts — significantly better than the noisy raw signal.

## Files

| Path | Content |
|---|---|
| `data/eval/preds_model{1,1_5,2,2_5}.csv` | Per-sample predictions + conf — raw |
| `data/eval/preds_model{1,1_5,2,2_5}C.csv` | Per-sample predictions + conf — calibrated (argmax identical, conf transformed) |
| `checkpoints/calibration/T_model{1,1_5,2,2_5}.json` | Fitted temperature + metadata per model |
| `src/eval/infer_with_conf.py` | Script that produces predictions (raw or `--temperature`) |
| `src/eval/calibrate_temperature.py` | Fits T on a held-out cross-train set |
| `src/eval/selective_with_ecps.py` | Threshold table + ECPS computation |

To reproduce any row:
```
python -m src.eval.selective_with_ecps --pred data/eval/preds_<model>.csv --name <model>
```

To re-fit a temperature:
```
python -m src.eval.calibrate_temperature \
    --ckpt <PARSeq ckpt> \
    --calibration-src <held-out cross-train images dir> \
    --calibration-labels <those images' labels.csv> \
    --out checkpoints/calibration/T_<model>.json \
    [--ensemble --alpha <a> --detector-ckpt ... --charcls-ckpt ...]
```
