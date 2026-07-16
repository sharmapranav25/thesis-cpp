# Model 1 v2 — PARSeq alone with MyBB-bundled fonts

**Date**: 2026-05-03
**Eval set**: 651 hand-labeled real CryptBB captchas
**Compute**: Local NVIDIA RTX A4000 (16 GB VRAM), Windows 11, Python 3.9.13, PyTorch 2.5.1+cu121

---

## Headline Result

| Metric | v1 (Windows fonts) | **v2 (MyBB fonts)** | Δ |
|---|---|---|---|
| Synth val word acc | 77.0% | **57.26%** | −19.74 pts (lower ceiling, **by design**) |
| Synth val char acc (NED) | 92.5% | 89.31% | −3.19 pts |
| **Real eval word acc (greedy)** | 31.18% | **39.78%** | **+8.60 pts** |
| **Real eval word acc (force-detected)** | 33.18% | **39.94%** | **+6.76 pts** |
| **Real eval char acc** | 80.29% | **84.28%** | **+3.99 pts** |
| **Synth-real word gap** | 43.8 pts | **17.4 pts** | **−26.4 pts** |

Improvement over v1 baseline: **+6.76 pts word, +3.99 pts char**, no architecture change, no real labels. The synth-real distribution gap closed by ~26 pts.

---

## Goal

Close the synth-real distribution gap by replacing the renderer's font pool with **MyBB's actual bundled captcha fonts**. Model 1 (v1) used 32 Windows TTFs as approximations for MyBB's font set; we found MyBB's source ships with only **2 fonts** (`MINYN___.ttf` and `edmunds.ttf`), and `captcha.php` simply scans `inc/captcha_fonts/` for any TTFs at render time.

The 32-Windows-font synth distribution was **wider** than reality — many of those fonts have clearly distinguishable upper/lower case shapes that the actual 2 MyBB fonts don't. This let v1's PARSeq learn a *case-friendly* distribution that didn't exist in the real captchas, hitting 77% on synth val but only 33% on real (43.8-pt gap).

The hypothesis: by training on synth that uses **only the MyBB fonts**, the synth distribution becomes a tight match for real, even if it makes the task visually harder (lower synth val word acc).

## Pipeline

```
   MyBB fonts (MINYN___.ttf, edmunds.ttf)
              │
              ▼
   src/php_synth/generate_mybb_captcha.php  --fontset mybb --fonts data/_mybb_fonts
              │
   100k (image, label) pairs (data/synth_mybb_v2/)
              │
              ▼
       PARSeq training (configs/parseq_v2.yaml, 15 ep, fp32)
              │
       best ckpt: parseq_v2-epoch=11-val_accuracy=57.260
              │
              ▼
       infer.py on data/eval/  (force-detected length = 5)
              │
              ▼
       39.94% word, 84.28% char on 651 real
```

Same 32→128 PARSeq input size, same 4×8 patch grid, same architecture as Model 1. **The only change is the font set used by the PHP renderer.**

## Why this worked

**The synth-real gap is fundamentally a font-distribution gap.** When the synth uses 32 visually-diverse fonts and reality uses 2, the model trained on synth learns case-discriminating features (size differences, x-heights) calibrated to the wide font set. On real images those features fail because the visual cues are different (and weaker).

**With v2**, the synth font set IS the real font set. The case ambiguity that exists in real is now explicitly present in training. The model learns features that generalise without distribution shift. Lower synth val ceiling, much higher real eval transfer.

This validates the underlying intuition: **distribution-matched synth dominates wider/more-diverse synth for downstream transfer.**

## Synth Renderer Changes

`src/php_synth/generate_mybb_captcha.php` — added `--fontset {windows|mybb}` flag.
- `windows` (default, used in v1): pool of 32 curated Windows TTFs.
- `mybb` (v2): only the 2 fonts MyBB actually ships.

Backwards-compatible — Model 1 retraining on Windows fonts still works the same way. `--bboxes` flag still emits per-char bounding boxes for the v1.5/v2 ensemble pipeline.

## Training Trajectory

PARSeq val word/char accuracy by epoch on 5% synth-val holdout from `data/synth_mybb_v2/`:

| Epoch | val_NED | val_word |
|---|---|---|
| 0 | 2.10 | 0.00 |
| 1 | 29.97 | 0.00 |
| 2 | 77.05 | 1.54 |
| 3 | 78.20 | 1.54 |
| 4 | 85.01 | 31.40 |
| 5 | 85.42 | 32.58 |
| 6 | 81.27 | 15.80 |
| 7 | 85.83 | 35.40 |
| 8 | 83.52 | 32.90 |
| 9 | 83.01 | 27.46 |
| 10 | 86.56 | 47.18 |
| **11** | **89.31** | **57.26** ← best |
| 12 | 85.62 | 45.08 |
| 13 | 86.76 | 43.94 |
| 14 | 87.23 | 46.40 |

Same instability pattern as Model 1 (peaks and dips of 30+ pts). Best checkpoint at epoch 11.

Total training time: ~2.7 hours for 15 epochs (slower than v1's 25 min — likely a different system load state during the overnight run; same fp32 + 100k synth + batch 128).

## Real Eval Results (no ensemble)

PARSeq v2 alone, all 4 single-recogniser decoding strategies:

| Strategy | Word % | Char % |
|---|---|---|
| A. greedy | 39.78 | 83.92 |
| **B. force-detected length** | **39.94** | **84.28** |
| C. case-marginal | 39.78 | 83.92 |
| D. detected+marginal | 39.94 | 84.28 |

Force-detected (length-5 forced) gives a tiny lift over greedy on this checkpoint (+0.16 word, +0.36 char). Smaller than v1's force5 lift (+1.85 word, +5.92 char) — because the v2 model is naturally outputting 5 chars more often (it learned the structure cleaner). Case-marginal still essentially flat.

## Confusion Analysis (force-detected, 651 real)

Top 8 confusions:

| GT | Pred | Count | Type |
|---|---|---|---|
| p | P | 31 | case |
| o | 0 | 23 | visual (font-specific: lowercase 'o' ≈ digit zero in MINYN) |
| y | Y | 22 | case |
| S | s | 20 | case |
| v | V | 16 | case |
| z | Z | 15 | case |
| s | S | 15 | case |
| V | v | 14 | case |

Case errors still dominate (as before), but new **font-specific** confusions appear: `o → 0` (23 errors) — the lowercase 'o' in the MINYN font is nearly identical to digit zero. This is intrinsic to the real font set and isn't fixable on the recogniser side without case-aware ensembling.

## Comparison to Earlier Approaches

| Setup | Real-eval word % | Real-eval char % |
|---|---|---|
| Model 1 PARSeq greedy | 31.18 | 74.37 |
| Model 1 PARSeq force5 | 33.18 | 80.29 |
| Model 1.5 ensemble (v1 PARSeq + char-CNN, α=0.70) | 36.41 | 81.89 |
| Multi-scale 3-CNN ensemble (v1 PARSeq + 3 char-CNNs, α=0.50) | 38.86 | 82.26 |
| **Model v2 PARSeq alone (force5)** | **39.94** | **84.28** |

**Model 1 v2 PARSeq alone beats the entire Model 1.5 ensemble pipeline** (which used 3 char-CNNs at multi-scale). The font swap is more powerful than any architectural / ensemble trick we tried.

## Limitations

1. **Lower synth val ceiling** (57% word vs Model 1's 77%) is *expected* but means we can't use synth val as a tight proxy for real-eval performance. Need to keep sanity-checking against real periodically.
2. **PARSeq training instability** — same wild bouncing as Model 1. Epoch 11 was the peak; epoch 14 was 11 pts below it. Retrospective best-checkpoint selection is required.
3. **Font-specific visual ambiguity** (e.g., MINYN's `o` ≈ `0`) is now in the synth distribution by design. Recogniser can't disambiguate these without external context.
4. **Eval set heterogeneity** — real eval has both 200×60 and 400×120 captchas. v2 synth is all 200×60 (a 400×120 mixed-scale variant was prototyped as v3 but ruled out — see "What was tried" below).

## What Was Tried, What Didn't Work

### Pseudo-labeling (`data/raw/pseudolabels.csv`)

Generated 114 high-confidence pseudo-labels from Model 1.5 ensemble on 600 unlabeled real captchas (threshold conf_joint ≥ 0.85), repeated 50× and mixed into PARSeq training. **Failed.** Val loss climbed away from baseline (1.85 → 2.35) as the model memorised the 114 unique pseudo-labels. Repeat factor was too aggressive; would need 1-3× repeat or fine-tuning from a good checkpoint instead of training from scratch. Not retried for v2 — left as a future experiment.

### v3 synth (supersample 2× + mixed-scale 200×60/400×120)

Rendered 100k captchas at 2× supersample then downsampled, with random per-image native target size. **Visually drifted from real distribution** — the supersample-then-downsample step produced sub-pixel-thin background lines and over-anti-aliased text edges that real GD-rendered captchas don't have. Visual side-by-side check (`data/_v2_v3_vs_real.png`) confirmed v2 (no supersample) is closer to real than v3. Pipeline was killed before training. Mixed-scale alone (no supersample) is still a worthwhile follow-up if the eval-set 400×120 portion specifically underperforms.

## Files

| Path | Purpose |
|---|---|
| `data/_mybb_fonts/` | MINYN___.ttf + edmunds.ttf (downloaded from MyBB GitHub) |
| `src/php_synth/generate_mybb_captcha.php` | extended renderer with `--fontset` flag |
| `data/synth_mybb_v2/` | 100k MyBB-fonts synth + bboxes.jsonl |
| `configs/parseq_v2.yaml` | PARSeq v2 training config |
| `checkpoints/parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt` | **best PARSeq v2 checkpoint** |
| `checkpoints/parseq_v2/lightning_logs/version_0/metrics.csv` | full training metrics |
| `data/eval/` | 651 real CryptBB captchas (eval-only, untouched) |

## Conclusion

A label-free CryptBB captcha solver achieving **39.94% word / 84.28% char** on 651 real captchas, trained only on PHP-rendered synthetic data using **MyBB's actual bundled fonts**.

The headline lesson: **distribution match dominates everything else**. We tried CNN ensembles, multi-scale CNNs, pseudo-labeling, TTA, case-aware decoding, Qwen2.5-VL-3B fine-tuning, supersample rendering — every architectural / training trick — and none of them moved the needle as much as a correct font set in the renderer. v1 → v2 = +6.76 pts word from a one-line config change.

**Next step (Model 1.5 v2):** stack the multi-scale char-CNN ensemble on top of v2 PARSeq. Documented in `model_one_v1.5_v2_results.md` — final headline **49.16% word / 86.86% char**.
