# Model 1 v1 — CryptBB Captcha Solver Results

**Date**: 2026-05-01
**Eval set**: 651 hand-labeled real CryptBB captchas (filename = ground truth)
**Compute**: Local NVIDIA RTX A4000 (16 GB VRAM), Windows 11, Python 3.9.13, PyTorch 2.5.1+cu121

---

## Headline Result

| Metric | Final v1 |
|---|---|
| **Exact-match (word) accuracy** | **31.18% (203 / 651)** |
| **Char-level accuracy** | **74.37% (2423 / 3258)** |
| **Random baseline** | 0% word, 1.6% char (5-char × 62-class) |
| **Earlier baseline (PIL-rendered synth + aug)** | 0.92% word, 40.6% char |

**Improvement over earlier baseline: ~34× on word accuracy, ~1.8× on char accuracy.**

The final pipeline is a one-step inference: raw real captcha → PARSeq → text. No cleaner, no GAN, no fine-tuning on real labels.

---

## Goal

Build a CAPTCHA solver for CryptBB-style text captchas with **zero human labeling effort during training**. The training pipeline consumes only synthetic data; real-data labels are reserved for evaluation only.

## Captcha Format

- 200×60 RGB PNG
- 5 characters, case-sensitive alphanumeric (`[a-zA-Z0-9]`, 62 classes)
- Server-rendered by **MyBB's default GD captcha** (`mybb/captcha.php`):
  - White background
  - One of three randomly-chosen background patterns: lines, circles, or filled squares (light-pastel colors RGB ≥ 150)
  - Light-pastel dot overlay (~2400 random pixels)
  - 5 TTF chars: random font, size 20–32, rotation ±30°, color RGB 0–200, with a ±3px shadow in a slightly lighter tone
  - 1px black border

## What Worked: Final Pipeline

```
   PHP-MyBB renderer (port of mybb/captcha.php)
              │
   100k (image, label) pairs (synth_mybb/)
              │
              ▼
       PARSeq training
       (15 epochs, fp32, perm_num=1, batch=128, albumentations aug)
              │
       best ckpt: epoch 10 (synth val 77.0% word)
              │
              ▼
       Inference on raw 651 real captchas
              │
              ▼
       31.18% word, 74.37% char
```

Key components:
- **`src/php_synth/generate_mybb_captcha.php`** — standalone port of MyBB's captcha algorithm, parameterised over font pool and image count. Uses Windows TTF fonts (~30 from `C:\Windows\Fonts`).
- **`src/parseq/train.py`** — wraps PARSeq's Lightning module from `third_party/parseq/strhub/models/parseq/system.py`. Custom dataset reads `(filename,label)` from `labels.csv`, applies albumentations.
- **`src/parseq/infer.py`** — batch inference over a directory.
- **`src/eval/score.py`** — exact-match + char accuracy + top-N confusion table.

## Failed Approaches (Documented for Future)

### CycleGAN unsupervised domain translation (v1, v2)

Attempted: render synthetic with our PIL renderer, train CycleGAN to translate synth → real-style, train PARSeq on translated data.

- **v1** (λ_cycle=12, default LR): later epochs produced black-bar artifacts and content-faded characters as the discriminators dominated (D-loss → 0.03). Best checkpoint (epoch 20) had OK content but mild style transfer. PARSeq trained on it never learned (val_loss stuck at random for 5 epochs). **Dropped.**
- **v2** (λ_cycle=20, λ_id=10, lr_d=1e-4, early-stop): ran 8 epochs before D-loss collapsed below threshold and early-stopped. Translation preserved characters but barely shifted style. PARSeq trained on these translations was worse than training on raw synth. **Dropped.**

Conclusion: CycleGAN's adversarial pressure consistently fights against content preservation for this task. Translation introduces noise that hurts more than the style transfer helps.

### bg-free synth + heuristic cleaner

Attempted: render synth without backgrounds (chars on white), build a saturation+darkness+Canny+morphology cleaner, train PARSeq on bg-free synth, run cleaner on real at inference.

- Cleaner worked OK on real captchas (chars survived; some bg bleed-through).
- PARSeq on bg-free synth + augmentation was unstable: train→val gap widened severely, word accuracy bounced 0.7-1.5% across epochs.
- **Dropped** when PHP-MyBB synth produced distribution-identical training data instead.

### PARSeq decoder-only pretraining (DPTR)

Implemented but not run. For uniform-random alphanumeric strings the decoder has no language structure to learn beyond uniform priors. Listed in FUTURE_NOTES.md as low-expected-value experiment.

### Wider PIL renderer

Initial PIL renderer matched 3 observed real-data backgrounds (speckle, grid, blocks) and char features (chromatic shadow, mixed fonts). PARSeq trained on it reached 87% synth val word accuracy but only 0.92% real-data word accuracy. The synth-vs-real distribution gap was the dominant blocker. Solved by switching to PHP-MyBB renderer (distribution-identical to actual server algorithm).

## Why PHP-MyBB Synth Worked

**CryptBB's captchas are produced by the exact MyBB default captcha algorithm.** Once we identified that and ported the PHP source code line-by-line (`mybb/captcha.php` → `src/php_synth/generate_mybb_captcha.php`), the synth distribution and real distribution became **visually identical** — same random color sampling, same shadow offset distribution, same background pattern logic, same pixel-level GD rendering quirks.

Side-by-side comparison (`data/_php_vs_real.png`) showed PHP synth and real captchas to be effectively interchangeable to the eye.

## Training Trajectory

PARSeq val word accuracy by epoch (synth val 5% holdout):

| Epoch | char % | word % | val_loss |
|---|---|---|---|
| 0 | 1.06 | 0.0 | 3.86 |
| 1 | ~0 | 0.0 | 2.98 |
| 2 | 10.9 | 5.6 | 1.36 |
| 3 | 48.1 | 33.6 | 0.50 |
| 4 | 82.4 | 61.6 | 0.20 |
| 5 | 20.6 | 15.8 | 0.72 |
| 6 | 41.5 | 32.8 | 0.48 |
| 7 | 81.8 | 65.7 | 0.18 |
| 8 | 77.5 | 62.4 | 0.20 |
| 9 | 66.0 | 54.2 | 0.29 |
| **10** | **92.5** | **77.0** | **0.10** ← best |
| 11 | 75.1 | 62.3 | 0.19 |
| 12 | 67.1 | 56.4 | 0.24 |
| 13 | 70.6 | 59.3 | 0.22 |
| 14 | 72.2 | 60.6 | 0.21 |

Trajectory was unstable (peaks and dips of 30+ percentage points between adjacent epochs) — likely due to permutation training interacting with the relatively small training set or to fp32 + lr schedule. Epoch 10 was a clear peak; subsequent epochs never recovered to that level.

Total training time: ~25 min on RTX A4000.

## Confusion Analysis (top 15 on real eval)

| GT char | Predicted | Count | Type |
|---|---|---|---|
| p | P | 30 | case |
| v | V | 18 | case |
| L | I | 18 | visual |
| s | S | 18 | case |
| S | s | 17 | case |
| m | `<gap>` | 15 | length error |
| N | X | 15 | visual |
| o | O | 12 | case |
| 1 | l | 12 | visual |
| y | V | 12 | visual |
| W | w | 12 | case |
| 2 | `<gap>` | 11 | length error |
| 4 | `<gap>` | 11 | length error |
| y | `<gap>` | 11 | length error |
| 1 | `<gap>` | 10 | length error |

**~50% of top errors are case confusions.** Lower-case ↔ upper-case for visually-symmetric characters (`p`/`P`, `v`/`V`, `s`/`S`, `o`/`O`, `W`/`w`). A case-aware loss term (or post-processing rule that picks the higher-prior case when ambiguous) would reduce these.

**Length-error (`<gap>`)** errors mean the model emitted fewer than 5 chars — usually losing edge characters. Could be addressed by enforcing 5-char output decoding (fixed-length decode) since we know the format.

## Architecture & Hyperparameters

### Synth renderer (`src/php_synth/generate_mybb_captcha.php`)
- 100k (image, label) pairs, seeded
- Image size 200×60
- Charset: `[0-9A-Za-z]` (62)
- Font pool: ~30 Windows TTF fonts (Arial, Calibri, Cambria, Candara, Comic, Consolas, Courier, Georgia, Impact, Lucida, Tahoma, Times, Trebuchet, Verdana — bold/italic variants)

### PARSeq training (`configs/parseq.yaml`)
- Backbone: vanilla PARSeq (no pretrained STR weights)
- ViT encoder: 12 layers, 6 heads, embed_dim 384
- Decoder: 1 layer, 12 heads, embed_dim 384
- Patch size [4, 8]; img_size [32, 128] (resized from 200×60)
- max_label_length: 5
- charset_train / charset_test: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- perm_num=1 (no permutation training; faster, more stable for short fixed-length sequences)
- decode_ar=true, refine_iters=1
- Optimizer: AdamW, lr 5e-4, warmup 5%, weight_decay 0
- Batch size 128, 15 epochs, fp32 (no AMP)
- Gradient clip val 5
- Augmentation (albumentations): ColorJitter, Gaussian/motion/median blur, GaussNoise, ImageCompression (50-95% quality), Affine (rot ±3°, translate ±2-4%, scale 95-105%, shear ±3°), CoarseDropout (1-3 holes, 2-12px)

### Hardware footprint
- VRAM during train: ~7-8 GB
- 100k synth gen wall-time: ~11 min (PHP-CLI single-threaded, ~150 img/s on this machine)
- PARSeq train wall-time: ~25 min for 15 epochs
- Inference on 651 captchas: ~45 sec at batch 64

## Limitations

1. **Train-time instability** — val word accuracy bounced 15-77% across consecutive epochs. The "best" checkpoint was found by retrospective selection, not by stable convergence. A different seed could miss the peak.
2. **Case confusion** — accounts for ~50% of remaining errors. Visually-symmetric characters confuse the model.
3. **Edge-character drops** — some predictions are 4-char ones with a leading or trailing `<gap>`.
4. **Eval set heterogeneity** — 350 of 651 are 200×60 (training distribution), 301 are 400×120 (~2× scale, possibly different forum source). Resizing to 32×128 internally; some distribution shift remains.
5. **Font set may not be exact** — we used Windows TTFs; CryptBB uses MyBB's bundled `captcha_fonts/` directory. Visually similar bold serif/sans fonts but not identical.

## Files of Interest

| Path | Purpose |
|---|---|
| `src/php_synth/generate_mybb_captcha.php` | Synth renderer (port of MyBB captcha.php) |
| `src/parseq/train.py` | PARSeq training entry point |
| `src/parseq/infer.py` | Batch inference over a directory |
| `src/eval/score.py` | Score predictions vs labels |
| `configs/parseq.yaml` | All PARSeq hyperparameters |
| `data/synth_mybb/` | 100k PHP synth + labels.csv |
| `data/eval/` | 651 real captchas + labels.csv (eval set) |
| `checkpoints/parseq/parseq-epoch=10-val_accuracy=76.980.ckpt` | Best model checkpoint |
| `data/eval/predictions_php_e10.csv` | Final predictions on 651 |
| `FUTURE_NOTES.md` | Backlog of follow-up experiments |
| `CONTEXT.md` | High-level project overview |

## Next-Step Candidates (in order of expected leverage)

1. **Case-aware decoding / loss** — biggest single error category. Either add label smoothing across same-letter case pairs at training time, or post-process predictions using char-level confidence scores.
2. **Fixed-length decoding** — force exactly 5-char outputs to eliminate `<gap>` errors.
3. **Self-training / pseudo-labeling** — run epoch-10 model on a fresh batch of unlabeled real captchas, take top-K confidence predictions as pseudo-labels, fine-tune. Bursztein-style attacks reach 90%+ this way.
4. **Train-time stability** — try perm_num=6, longer warmup, lower peak LR. Find a config that reliably hits 75%+ synth val without bouncing.
5. **TTA (test-time augmentation)** — at inference, run model on 3-5 augmented versions of each captcha, majority vote. Free 2-5pt boost.
6. **Use MyBB's bundled fonts** — clone MyBB's `inc/captcha_fonts/` into our renderer and stop using Windows TTFs. Closer font-set match might lift accuracy 5-10pt.
7. **Qwen2.5-VL fine-tune** — tested off-the-shelf VLMs hit 21-40% on captcha benchmarks; fine-tuned on our distribution-matched synth they should beat PARSeq.

## Conclusion

A label-free CryptBB captcha solver achieving **31.2% exact-match accuracy** on a held-out 651-captcha real-world eval set, trained only on PHP-rendered synthetic data that matches the server's actual MyBB captcha algorithm. The key insight: when the synth distribution is identical to the target distribution, the synth-vs-real gap collapses and standard supervised learning suffices.

The earlier dead-ends (CycleGAN, custom heuristic cleaner, PIL renderer + augmentation) were all attempts to bridge an synth-real gap that didn't need to exist if we just used the right renderer in the first place.

---

## Addendum — Selective Prediction (skip-and-refresh)

Modified `src/parseq/infer.py` to emit a per-prediction confidence (`conf_joint` = product of per-position softmax probabilities). Empty predictions (model emits only EOS) are assigned `conf=0` since no characters were predicted. Re-ran on the same 651 real captchas → `data/eval/predictions_php_e10_conf.csv`.

`src/eval/analyze_confidence.py` joins predictions with ground-truth labels and reports, for any chosen confidence threshold T, how many captchas the model would skip vs attempt and the accuracy on the attempted ones.

Under a "skip = refresh" framing (the model passes on captchas it isn't confident about and gets a fresh one):

| confidence threshold | attempted accuracy | avg skip rate |
|---|---|---|
| 0.70 | 57.37% | 70.8% |
| 0.75 | 61.78% | 75.9% |
| 0.80 | 61.83% | 79.9% |
| 0.85 | 64.08% | 84.2% |
| 0.90 | 68.06% | 88.9% |
| 0.95 | 70.00% | 93.9% |

Tradeoff: higher threshold → fewer attempts but higher accuracy on attempts. PARSeq's confidence on real data plateaus around 70% attempt accuracy at the top end — confidence is somewhat informative but not strongly calibrated, so 90%+ attempt accuracy is not reachable without model-level changes (temperature scaling, calibration training, etc.).

---

## Addendum — Inference-time decoding sweep (no retraining)

Same epoch-10 checkpoint, same 651 real captchas. Tried four decoding strategies on the per-position softmax to see which post-processing tricks the existing model can absorb. Implementation: `src/eval/case_decode.py`.

| Strategy | Word % | Char % | Δ word | Δ char |
|---|---|---|---|---|
| A. greedy (EOS-aware, baseline) | 31.18 | 74.37 | — | — |
| **B. force-detected length**     | **33.03** | **80.29** | **+1.85** | **+5.92** |
| C. case-marginal                | 31.34 | 74.37 | +0.16 | 0.00 |
| **D. force-detected + marginal** | **33.18** | **80.29** | **+2.00** | **+5.92** |

**B / D — fixed-length forcing.** At each position we zero the EOS logit before argmax and stop after exactly *N* characters, where *N* is auto-detected from the training labels (single length covering >= 90% of rows). For PHP-MyBB synth this resolves to N=5; the decoder generalises to other schemes by reading the threshold from `--train-labels` so it isn't a CryptBB-only shortcut. The +5.92 pts char-accuracy lift comes from the `<gap>` confusions (`m → <gap>`, `1 → <gap>`, etc.) — the model was emitting EOS too early on ~12% of captchas, and forcing 5 chars recovers most of those positions with the model's actual best non-EOS guess.

**C — case marginalisation.** For visually-symmetric letter pairs (Cc/Kk/Mm/Oo/Pp/Ss/Uu/Vv/Ww/Xx/Yy/Zz), pick the case-folded letter with the highest summed probability, then output the higher-probability case. Effect: negligible (+0.16 pts word). Implication: the model's case errors are **not** ambiguous splits between case-pairs in the softmax — they are decisive wrong-case predictions. The visual decision boundary itself is biased toward upper-case for {p, v, o} and lower-case for {W}; you can't fix that by reading the softmax differently.

**Top remaining confusions (decoder D):** `p → P` (32), `v → V` (20), `L → I` (19), `S → s` (18), `s → S` (18), `N → X` (15), `1 → l` (14), `W → w` (14). Case errors still dominate. Closing them requires per-character logit calibration learned from a held-out synth val set, not just smarter decoding.

**Headline:** **31.18% → 33.18% word, 74.37% → 80.29% char**, zero retraining. Same checkpoint, smarter decoder. Length detection is data-driven (`detect_fixed_length()` in `case_decode.py`) so the trick stays valid for any captcha format whose label distribution is concentrated; it falls back to EOS-aware greedy for variable-length schemes.

---

## Addendum — Qwen2.5-VL-3B LoRA fine-tune (for reference)

Tested whether a fine-tuned VLM beats the specialist PARSeq recogniser on this task. **Conclusion: it doesn't.**

Setup: Qwen2.5-VL-3B-Instruct with LoRA (rank 16, alpha 32) on `q_proj/k_proj/v_proj/o_proj`. Trained 2000 optimizer steps × effective batch 8 on a 30k subset of `data/synth_mybb/`. ~2.8h on the A4000.

| Checkpoint | Word % | Char % | vs PARSeq word |
|---|---|---|---|
| Qwen2-VL-2B zero-shot (no fine-tune) | 2.47 | 27.49 | −28.71 pts |
| Qwen2.5-VL-3B LoRA step 500 | 21.97 | 69.36 | −9.21 pts |
| **Qwen2.5-VL-3B LoRA step 1000 (peak)** | **24.12** | 72.74 | **−7.06 pts** |
| Qwen2.5-VL-3B LoRA step 1500 | 22.43 | 72.89 | −8.75 pts |
| Qwen2.5-VL-3B LoRA step 2000 | 23.35 | 73.23 | −7.83 pts |
| **PARSeq Model 1 (greedy baseline)** | **31.18** | **74.37** | **— winner —** |

LoRA gave a ~10× lift over zero-shot (2.47% → 24.12%) so fine-tuning *worked*. But even the peak checkpoint sits ~7 pts below PARSeq on word accuracy. Char accuracy is within 1.6 pts of PARSeq (73 vs 74) — meaning Qwen's vision tower reads chars about as well as PARSeq, but the autoregressive BPE decoder costs per-position accuracy on case-sensitive 5-char random strings.

Word accuracy peaks around 1 effective epoch (step 1000) and degrades after — classic overfitting to the synth distribution while synth loss continues to drop.

**Verdict:** LoRA on a 3B VLM is not the path to beat PARSeq. The bottleneck is decoder/tokenizer mismatch, not visual capacity. A larger backbone (7B QLoRA) would mainly test "does a stronger visual encoder buy us another ~1 pt char acc?" — unlikely to close the word-accuracy gap. Higher-leverage moves stay on PARSeq itself.

Code: `src/qwenvl/{data,train,infer}.py`. Adapters: `checkpoints/qwenvl_lora/step_{500,1000,1500,2000}/`. Predictions: `data/eval/predictions_qwen25vl3b_lora_step{500,1000,1500,2000}.csv`.
