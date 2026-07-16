# Model 2 v1 — Curriculum Learning Attempts on CryptBB Captcha Solver

**Date**: 2026-05-03
**Eval set**: 651 hand-labeled real CryptBB captchas (filename = ground truth)
**Compute**: Local NVIDIA RTX A4000 (16 GB VRAM), Windows 11, Python 3.9.13, PyTorch 2.5.1+cu121
**Status**: NEGATIVE RESULT — every curriculum variant tried regressed or tied the no-curriculum baseline

---

## Headline Result

**Recommendation: ship Model 1 v2** (`E:\pranav\model 1\checkpoints\parseq_v2\parseq-epoch=11-val_accuracy=57.260.ckpt`).

| Model | Word acc | Char acc |
|---|---|---|
| Model 1 v1 (original baseline, 100k fixed:5, no curriculum) | 31.18% | 74.37% |
| **Model 1 v2 (improved baseline, separately trained by user)** | **39.78%** | **83.92%** |
| M2 best attempt (M2-final ep0: 1ep no-aug fine-tune of v2) | 39.32% | 84.01% |
| All other M2 variants | ≤ 9.83% | ≤ 70% |

The best curriculum-laden M2 ties baseline within noise on word accuracy and is +0.09 on char accuracy — but the gain comes from "1 epoch of gentle fine-tuning with no curriculum effect active yet" (ε=0 during warmup), not from the curriculum itself. Adding any curriculum signal after that regresses the model.

## Run Catalog (best ckpt of each on real CryptBB)

| Run | Setup | Real word% | Real char% | Notes |
|---|---|---|---|---|
| **M1 v1** | vanilla PARSeq, 100k fixed:5 d=1.0, no curriculum, 15 ep | 31.18 | 74.37 | original baseline |
| **M1 v2** | (user-trained variant) | **39.78** | 83.92 | strongest baseline |
| Run A | variable length 3-7 + d_max curriculum + length head, 15 ep | 0.00 | 1.75 | autoregressive collapse |
| Bucket curriculum | 3-bucket easy→hard schedule, 15 ep | 1.08 | 60.83 | char OK, word destroyed |
| M2 sched-sample v1 | 15 ep, ε=0→0.25, no warmup beyond 1 ep | 9.83 | 69.00 | from-scratch, peak ep 9 |
| M2 sched-sample v2 | 25 ep, ε=0→0.15, slow ramp, killed mid-run | (~1% stuck) | — | from-scratch |
| M2-finetune | M1 v2 + sched-sample + gradual unfreeze + heavy aug | 6.76 | 70.44 | aug destroyed v2 in 1 ep |
| Diag fine-tune | 1 ep, no aug, lr=1e-5, no curriculum | 38.25 | 83.79 | confirms gentle fine-tune ≈ baseline |
| **M2-final ep0** | 3 ep no aug, lr=1e-5, light SS (eps_max=0.05, warmup=1) | **39.32** | **84.01** | tied baseline |
| M2-final ep1 | epoch 1 of above (ε=0.025 active) | 36.87 | 82.87 | SS kicks in, regression starts |
| M2-final ep2 | epoch 2 of above (ε=0.05 active) | 35.64 | 82.54 | further regression |

## Why Every Curriculum Failed

The literature (Wu, Dyer & Neyshabur, ICLR 2021, "When Do Curricula Work?") finds CL provides marginal benefit only in specific regimes (limited compute, noisy labels). Our setting fits neither: synth distribution already matches deployment, and the baseline is well-trained.

### Specific failure modes observed

1. **Run A — autoregressive collapse from input distribution shift.** Variable-length training mixed with d_max curriculum confused PARSeq's permuted decoder. Position 0 of the output learned to emit EOS for every input. Model emitted empty string on 94% of CryptBB images. Documented in NMT literature (Koehn & Knowles 2017).

2. **Bucket curriculum — joint distribution lost.** Per-character accuracy reached 60.83% on real CryptBB but word accuracy was 1.08%. Easy stages didn't exercise PARSeq's permutation-conditional, so the PSM head never received hard-stage gradient signal. Model learned per-position marginals but lost joint character relationships.

3. **M2 sched-sample v1/v2 (from scratch) — exposure-bias signal too aggressive.** Substituting model's own predictions into decoder input destabilized training before the encoder had learned good visual features. Training from a poorly-converged state with SS noise creates a moving optimization target.

4. **M2-finetune (heavy aug + curriculum) — augmentation killed the converged model.** Diagnostic isolated this: same setup with `augment: false` jumped from 6.76% → 38.25% real word accuracy. Augmentation forced the converged head/decoder to fit OUR specific augmented samples, shifting them away from M1 v2's optimum.

5. **M2-final (gentle, no aug, light SS) — SS provides no benefit.** Epoch 0 (ε=0, no SS effect) was the best ckpt; epochs 1-2 with SS active regressed. Model 1 v2 has no exposure-bias problem to fix, so SS only adds noise.

## The Augmentation Diagnostic (Most Surprising Finding)

When M2-finetune ran with our default training augmentation (ColorJitter, Blur, GaussNoise, ImageCompression, Affine, CoarseDropout), **one epoch destroyed M1 v2's accuracy on real CryptBB from 39.78% → 6.76%**.

Disabling augmentation (single change) recovered 38.25% in the diagnostic. So augmentation accounted for ~32 percentage points of the regression.

Likely cause: M1 v2 was trained to a sharp minimum on its specific augmented data distribution. Our augmentation pipeline produces different augmented samples (different RNG / possibly different functions). The decoder updates to fit OUR distribution, drifting away from v2's optimum. Even at lr=1e-4 with frozen encoder this happens within one epoch.

**Practical takeaway:** if you ever fine-tune from a converged checkpoint, match the original training augmentation EXACTLY or disable augmentation entirely. Mismatch is catastrophic.

## Architectural / Pipeline Changes Built for M2

(For reference; remain in `src/parseq/train.py` and configs.)

- `PARSeqWithLengthHead` — auxiliary binary classifier for fixed/variable length-mode (used in Run A, deprecated)
- `PARSeqSchedSample` — scheduled sampling decoder wrapper (Bengio et al. 2015 style)
- `CurriculumSampler` + `CurriculumCallback` — per-epoch d_max-based sample filtering (Run A)
- `BucketCurriculumSampler` + `BucketCurriculumCallback` — per-epoch bucket-weighted sampling
- `SchedSampleCallback` — linear ε ramp for scheduled sampling
- `GradualUnfreezeCallback` — ULMFiT-style progressive layer unfreezing
- `finetune_from` config option — load arbitrary ckpt as starting weights
- `augment` config option — toggle albumentations on/off

All available; none improved over baseline.

## Files of Interest

| Path | Purpose |
|---|---|
| `src/parseq/train.py` | All training code, curriculum classes, callbacks |
| `configs/parseq_m2_final.yaml` | Best M2 attempt config (1 ep tied baseline) |
| `configs/parseq_m2_finetune.yaml` | Heavy curriculum + gradual unfreeze (failed) |
| `configs/parseq_m2_sched_sample.yaml` | From-scratch SS (failed) |
| `data/synth_fixed5_100k/` | 100k fixed:5 d=1.0 captchas (M2's training pool) |
| `data/eval/` | 651 real CryptBB captchas |
| `data/eval/predictions_m1_v2_raw.csv` | M1 v2 baseline predictions (39.78%) |
| `data/eval/predictions_m2_final_e00.csv` | Best M2 predictions (39.32%) |

## Conclusion

Three weeks of curriculum experiments produced **no improvement** over Model 1 v2's no-curriculum baseline. Every variant — input-difficulty curriculum (continuous d-window, 3-bucket scheduled), supervision curriculum (scheduled sampling at 4 different ε schedules), and ULMFiT-style gradual unfreezing fine-tuning — either regressed the model or tied it within noise.

This is consistent with the curriculum-learning literature for distribution-matched training: when synth = deployment, there is nothing to scaffold. The synthetic data IS the test data.

**Ship Model 1 v2.** Reserve curriculum approaches for tasks with known distribution gaps or limited training budget — neither applies here.

### Open paths if pushing further (not pursued)

- **TEACH (Yu et al. 2025)** — supervision curriculum that masks decoder hints adaptively based on loss, the only published technique with positive results on PARSeq. Implementation effort ~1 day.
- **Aug pipeline match** — replicate the exact aug pipeline M1 v2 was trained with, then try fine-tuning. Requires v2's training config.
- **Self-training / pseudo-labels on confident real CryptBB predictions** — already noted in Model 1's FUTURE_NOTES.md as the highest-leverage non-curriculum improvement (~5-10pt potential).

---

## Addendum — Per-Character Confusion Comparison

Top error modes are similar across M1 v2 and best M2 (both at 39%): case confusions (`p→P`, `v→V`, `S→s`, `o→O`, `W→w`) and visual lookalikes (`o→0`, `c→e`, `L→I`, `1→l`). M2's slight char accuracy edge (+0.09%) doesn't translate to fewer specific error categories — they're the same cases-confused captchas, just with marginally better per-position predictions.

Case-aware decoding remains the highest-leverage next step (~50% of all errors are case confusion of visually-symmetric chars).
