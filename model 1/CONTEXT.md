# CryptBB Captcha Solver — Session Context

Last session ended: 2026-05-03. **Pick up here.**

## TL;DR

- **Model 1 is shipped** — 31.18% word / 74.37% char on the 651-captcha real eval set with greedy decoding.
- **Decoder upgrade**: data-driven fixed-length forcing (`src/eval/case_decode.py`) lifts to **33.18% word / 80.29% char**, no retraining. Same checkpoint, smarter decoder.
- **Model 1.5 is shipped** — PARSeq + detector + char-CNN ensemble (α=0.70) lifts to **36.41% word / 81.89% char**, +3.23 word over Model 1 baseline. Still zero real labels in training. See `model_one_v1.5_results.md`.
- **Model 1 v2 is shipped** — same architecture as Model 1, but renderer uses MyBB's actual bundled fonts (only 2: MINYN + edmunds) instead of 32 Windows TTFs. **39.94% word / 84.28% char** on real eval. Synth-real gap closed by ~26 pts. See `model_one_v2_results.md`.
- **Model 1.5 v2 is shipped — current best**: PARSeq v2 + detector v2 + 3 char-CNN scales (64/96/128) on MyBB-fonts synth, ensemble at α=0.30 (CNN-heavy). **49.16% word / 86.86% char**, +12.75 word over Model 1.5 v1, +18 word over Model 1 baseline. See `model_one_v1.5_v2_results.md`.
- **Final pipeline (Model 1.5 v2)**: MyBB-fonts PHP synth → PARSeq v2 logits + detector v2→crops→3 char-CNN scales (avg) → log-prob mix at α=0.30 → force-5 argmax.
- **Best checkpoints** (all in `checkpoints/`): `parseq_v2/parseq-epoch=11-val_accuracy=57.260.ckpt`, `detector_v2/best.pt` (val IoU 0.7522), `charcls_{64,96,128}_v2/best.pt` (val_case 99.1-99.2%).
- **Selective prediction analyzed**: at 0.95 confidence threshold, model attempts 6.1% of captchas with 70% accuracy. PARSeq's confidence plateaus at ~70% — needs calibration work to push higher.
- **Full v1 writeup**: `model_one_v1_results.md`
- **All other directions ruled out for now** — see "Tried and failed since v1.5 v2" below. v1.5 v2 (49.16% / 86.86%) remains the deployed best.

## What's In Each File

### Core source
- `src/php_synth/generate_mybb_captcha.php` — port of MyBB's `captcha.php` algorithm. Generates 200×60 PNG captchas matching CryptBB exactly. CLI: `php generate_mybb_captcha.php --out DIR --n 100000 --seed 0`
- `src/render_synthetic.py` — original PIL renderer (deprecated; superseded by PHP version, kept for reference)
- `src/parseq/train.py` — PARSeq training. Reads `data/{data_dir}/labels.csv` + images. Uses albumentations augmentation.
- `src/parseq/infer.py` — batch inference. Now emits `conf_joint`, `conf_min`, `conf_mean` per prediction. Empty predictions get conf=0.
- `src/eval/score.py` — exact-match + char accuracy + top-N confusion table.
- `src/eval/analyze_confidence.py` — selective prediction analysis: skip/attempt/accuracy at thresholds, K-refresh solve rate, calibration plot.
- `src/cyclegan/`, `src/cleaner/` — deprecated tracks (kept for reference; do not retrain).

### Configs
- `configs/parseq.yaml` — current config. `data_dir: data/synth_mybb` (PHP-MyBB synth). 15 epochs, fp32, perm_num=1, batch=128, augmentation ON.

### Data on disk
- `data/synth_mybb/` — **100k** PHP-rendered synth + `labels.csv`. Active training set.
- `data/synthetic/` — old PIL synth (100k). Deprecated. Safe to delete if disk is tight.
- `data/synth_nobg/` — bg-free synth (100k). Deprecated.
- `data/translated/` — CycleGAN-translated synth. Deprecated.
- `data/raw/` — original 600 unlabeled CryptBB captchas (kept for reference; not used in current pipeline).
- `data/eval/` — **651 labeled real CryptBB captchas + `labels.csv`**. Filename = ground truth. The eval set we measure against.
- `data/eval/predictions_php_e10_conf.csv` — Model 1's predictions on the eval set with confidences.
- `data/eval/_confidence_analysis/` — calibration + precision-coverage plots.
- `data/cleaned_eval/` — cleaner-pipeline output on real (heuristic cleaner; deprecated).

### Checkpoints
- `checkpoints/parseq/parseq-epoch=10-val_accuracy=76.980.ckpt` — **best Model 1 checkpoint**.
- `checkpoints/parseq/last.ckpt` — last epoch (epoch 14, lower accuracy).
- `checkpoints/parseq/lightning_logs/` — training metrics CSV.
- `checkpoints/cyclegan*/` — old GAN checkpoints. Deprecated.

### Documentation
- `model_one_v1_results.md` — full v1 writeup: pipeline, failed approaches, hyperparameters, confusion analysis, selective-prediction addendum.
- `CONTEXT.md` — this file.
- `FUTURE_NOTES.md` — backlog of follow-up experiments.

## Pipeline (current, what works)

```
   render_mybb_captcha.php  ──►  100k (image, label) pairs (data/synth_mybb/)
                                              │
                                              ▼
                                  PARSeq training
                            (configs/parseq.yaml, 15 ep)
                                              │
                                              ▼
                              best ckpt: parseq-epoch=10
                                              │
                                              ▼
                              infer.py on data/eval/
                                              │
                                              ▼
                              31.2% word, 74.4% char on 651 real
```

No cleaner. No GAN. No real-data fine-tuning. Just PHP-distribution-matched synth + standard supervised PARSeq training.

## Methodology Constraints

- **No human labels during training**. Strictly enforced.
- **The 651 real captchas are eval-only**. Don't train on them.
- The PHP renderer = CryptBB's actual server algorithm. Distribution gap is near-zero (which is why this works).

## Compute Environment

- Windows 11, Python 3.9.13 venv at `.venv/` (created with `py -m venv`)
- NVIDIA RTX A4000 (16 GB VRAM)
- PyTorch 2.5.1 + CUDA 12.1
- XAMPP PHP 8.2.12 at `C:/xampp/php/php.exe` (used by the synth renderer)
- Anaconda DLL bundle path required for sqlite3: `C:/ProgramData/Anaconda3/Library/bin` — already added to `os.add_dll_directory` in `infer.py` and other scripts that need it.

## Selective Prediction Snapshot (Model 1 on 651 real)

```
threshold | shoot rate | accuracy on attempts
----------+------------+----------------------
0.70      | 29.2%      | 57.4%
0.75      | 24.1%      | 61.8%
0.80      | 20.1%      | 61.8%
0.85      | 15.8%      | 64.1%
0.90      | 11.1%      | 68.1%
0.95      |  6.1%      | 70.0%
no thr    | 100.0%     | 31.2%
```

PARSeq's confidence is somewhat informative but not strongly calibrated — accuracy on attempts plateaus at ~70%. To push higher requires temperature scaling, calibration training, or a better model.

## What's Failed (Don't Re-Try Without New Idea)

- **CycleGAN unsupervised translation** (v1, v2). Adversarial pressure consistently fights character-content preservation. Even the best (most-content-preserving) checkpoints hurt PARSeq accuracy when used for translation.
- **Heuristic cleaner** (saturation + darkness + Canny + morphology). Worked OK on real, but the PHP-MyBB synth made it irrelevant.
- **Bg-free synth + augmentation**. Word accuracy never broke 1.5% — augmentation introduced distortions the bg-free distribution couldn't generalize to.
- **Decoder-only pretraining (DPTR)**. Implemented but not run — for uniform-random alphanumeric captchas there's no language structure for the decoder to learn.

### Tried and failed since v1.5 v2 shipped (2026-05-03)

- **Pseudo-labeling on `data/raw/`** (114 high-conf labels, repeat=20-50, mixed into PARSeq retrain). Val loss climbed away from baseline (1.85 → 2.35) — overfit on the small unique pseudo-label set. Fixable in principle (lower repeat, fine-tune from epoch-10 ckpt instead of from scratch) but not retried.
- **Photometric TTA on PARSeq** (5 augs: brightness±, contrast±, blur). **Zero gain** (often −0.15 pts). PARSeq's training already uses heavy photometric aug; TTA at inference is redundant.
- **v3 synth (supersample 2× + mixed-scale 200×60/400×120)**. Visually drifted from real distribution (sub-pixel-thin bg lines, over-smoothed text edges). Killed before training. v2 (no supersample, 200×60 only) is closer to real.
- **Compbg synth** (LaMa-extracted real CryptBB bgs + PHP `--text-only` rendered text composited on top, 100k pool at `data/cryptbb_synth_compbg/`). PARSeq trained on it plateaued at **1.52% val_word / 78.87% val_NED** on synth val, **1.23% word / 72.62% char on real** — *below* even Model 1 v1's baseline. Char-level signal exists but every captcha has at least one wrong char (uniformly spread errors). Likely cause: alpha-composite step creates anti-aliasing artifacts at text/bg boundary that real captchas don't have, plus randomised bg sampling decouples text-bg interaction in a way that uniformizes per-position errors. Validates exactly why glyph-conditioned diffusion exists. See `model_one_v1.5_v2_results.md` for the complete trajectory.
- **Diffusion-based synth (GlyphControl, AnyText, QwenStyle, B-CvT)**. Paths investigated but not productive: GlyphControl zero-shot produces wrong-style outputs and drifts content (0/5 on PARSeq read-back); AnyText setup not finished; QwenStyle's HF demo had a runtime error and the model is 20B params (won't fit A4000); B-CvT is medical-imaging-specific and code release unclear. Low priority unless we have a concrete reason a specific approach will work.

## Sibling projects (cross-reference)

- **`E:/pranav/synthetic image generation/`** — project to build a generic synth pipeline that handles any captcha scheme without per-scheme PHP renderers. Architecture A (composite: real bg + rendered text) was implemented and evaluated as `compbg`, **failed** (1.23% word real). Architectures B (mask-aware GAN translation) and C (bg replacement around rendered text) remain unattempted. See that dir's `CONTEXT.md` for status.
- **`E:/pranav/model 2/`** — curriculum learning + variable-length captcha solver. Adds `--difficulty-mode` and `--length-mode` to the renderer. Negative result on its own (M1 v2 still wins), but the renderer's difficulty parameter (`d ∈ [0,1]`) is a useful tool. Easy/medium/hard pools at `model 2/data/synth_easy/` (50k, d=0.2), `synth_medium/` (50k, d=0.5), `synth_curriculum/` (100k, mixed including d=1.0).
- **`E:/pranav/tiny STR/`** — minimal-architecture experiment. ~5.67M-param ViT-style recognizer. Hits 31.03% real word with ensemble — close to Model 1 alone. Validates "PARSeq architecture is replaceable; data is load-bearing." Parked.
- **`E:/pranav/model 1 Dino attention/`** — failed diffusion track (GlyphControl + LoRA). Reference for what didn't work.

## Open threads (in rough priority order)

- **Cross-scheme generalisation**: solver pipeline only verified on CryptBB. Courier Market (~MyBB-similar 200×60), DarkBay (200×50, 6-char, 35-class, no source code), City (324×81, no source code). DarkBay blocked on data (only 1000 labeled, eval-only — no separate unlabeled pool). Courier/City have unlabeled pools (2510 / 2000) but no labeled eval.
- **Compbg follow-ups**: if revisiting the synthetic image generation project, try architecture B (mask-aware GAN translation with explicit text-pixel preservation loss) or architecture C (bg replacement around rendered text instead of randomised composite). The naive composite has a fundamental edge-artifact issue.
- **Pseudo-labeling done right**: lower repeat factor (1-3×), fine-tune from `parseq_v2/epoch=11` checkpoint instead of training from scratch. We have the infrastructure (`src/eval/pseudolabel.py`).
- **Case-aware decoding**: case errors still dominate top confusions (`p→P`, `s↔S`, `o↔0`, etc.). Per-character logit calibration learned from synth val is a possible direction.
- See `FUTURE_NOTES.md` for the longer backlog.

## FUTURE_NOTES.md backlog

Long-term ideas:
- Auto-tune renderer with VLM + coding agent
- VGG-perceptual cycle loss (deferred)
- Wider renderer manifold (Diversity #2)
- PARSeq decoder-only pretraining (low expected value for our task)
- Diffusion-based synth (alternative to PHP) — investigated, see "Tried and failed" above
- Self-training / pseudo-labeling on confident predictions — first attempt failed; see above
- Test-time augmentation — photometric tested, gave 0 gain. Geometric (rotate/scale + string-vote) untried.
- TPS rectification
- VLM benchmarks reference
- Case-folded PARSeq + CNN-only case head — ruled out by Model 1.5 v1 strategy G (case head doesn't generalise)

## Quick Commands (Cheatsheet)

Generate synth (single shot, 1k samples for testing):
```
php src/php_synth/generate_mybb_captcha.php --out data/test_synth --n 1000 --seed 42
```

Train PARSeq (uses configs/parseq.yaml):
```
.venv/Scripts/python.exe -m src.parseq.train --config configs/parseq.yaml > checkpoints/parseq.log 2>&1
```

Run inference on eval (with confidence):
```
.venv/Scripts/python.exe -m src.parseq.infer --ckpt "checkpoints/parseq/parseq-epoch=10-val_accuracy=76.980.ckpt" --src data/eval --out data/eval/predictions_php_e10_conf.csv --batch 64
```

Score predictions vs ground truth:
```
.venv/Scripts/python.exe -m src.eval.score --pred data/eval/predictions_php_e10_conf.csv --labels data/eval/labels.csv --top 15
```

Selective prediction analysis:
```
.venv/Scripts/python.exe -m src.eval.analyze_confidence --conf-field conf_joint --target-precisions 0.99 0.95 0.90 0.85 0.80 0.75 0.70
```

---

# Appendix — Full module map (added during 2026-06 exploration crawl)

The body above + `TRAINING_INFRASTRUCTURE.md` cover the flagship pipeline. This
appendix indexes **every** `src/` module so an agent can navigate, and flags the
one track (`qwenvl/`) that the body never mentions.

## Recognizer + ensemble (the deployed Model 1.5 v2)
- `src/php_synth/generate_mybb_captcha.php` — **the load-bearing artifact.** Line-by-line port of MyBB's `captcha.php`. 200×60, 5 chars, MyBB's 2 bundled fonts (`data/_mybb_fonts/`). Has `--difficulty-mode`/`--length-mode`/`--text-only`/`--charset`/`--width`/`--height` (used by Model 2 + synthetic-image-gen).
- `src/parseq/train.py` — PARSeq Lightning trainer. `build_parseq(cfg)`, `CHARSET`, albumentations aug. 23.8M params, 32×128 input, perm_num=1.
- `src/parseq/infer.py` — batch inference; emits `conf_joint/conf_min/conf_mean`. `ImageDir` dataset reused by eval scripts.
- `src/parseq/pretrain_decoder.py` — DPTR decoder-only pretraining. **Implemented, never run** (no language structure in random captchas → low expected value).
- `src/detector/model.py` — `CharBoxNet`, ~0.5-1.18M-param ConvNet → 5 bboxes (cx,cy,w,h), sorted-by-cx GIoU loss. `data.py`/`train.py`/`infer.py`/`visualize_*`.
- `src/charcls/model.py` — ResNet-18 multitask: 62-class letter head + 3-class case head {UPPER,LOWER,DIGIT} (`CASE_UPPER/LOWER/DIGIT` consts). Trained at 3 scales (64/96/128). `infer_fast.py` is the DataLoader-prefetch path used in eval.
- `src/eval/case_decode.py` — **the ensemble eval harness.** Collects PARSeq softmax, runs detector→char-CNN crops, mixes `α·logP_parseq + (1-α)·logP_cnn` per position. 7 decoding strategies (greedy / force-5 / case-marginal / ensemble-α-sweep / letter-override / case-flip-override). Optional `--tta`. This produces the headline 49.16% number at α=0.30.
- `src/eval/score.py` — exact-match + char acc + top-N confusion table.
- `src/eval/infer_with_conf.py` + `calibrate_temperature.py` + `selective_with_ecps.py` — the calibration/selective-prediction stack (see `analysis.md`). Temperature fit on the *other* model's training data (cross-pollination).
- `src/eval/analyze_confidence.py` — selective-prediction thresholds + calibration plots.

## Curriculum-teacher scoring (feeds Model 2)
- `src/eval/score_synth_difficulty.py` — runs the M1.5 v2 ensemble over a synth pool, emits per-sample NLL / P_correct / correct. **This is the "teacher" that defines Model 2's difficulty buckets.**
- `src/eval/bucket_synth.py` — turns those scores into easy/medium/hard NLL-tertile CSVs.
- `src/eval/pseudolabel.py` — M1.5 pseudo-labels on unlabeled `data/raw/`. **Tried, failed** (overfit on small pseudo-set; see CONTEXT body).
- `src/eval/build_labels.py`, `infer_synthgen.py`, `analyze_confidence.py` — supporting eval utilities.

## Deprecated / negative-result tracks (kept for reference — DO NOT retrain)
- `src/cyclegan/` — synth↔real CycleGAN translation (Zhu 2017 recipe). `train.py`/`models.py`/`dataset.py`/`translate.py`/`inspect_translate.py`. **Failed**: adversarial pressure destroys text content. Output `data/translated/` later also broke `tiny STR` cold-start.
- `src/cleaner/` — DINOv2-saliency + Canny mask cleaner (`pipeline.py`, `dinov2.py`, `saturation.py`, `batch.py`, `compare_eval.py`, `inspect.py`). Worked on real but made irrelevant by MyBB-fonts synth. The DINOv2-saliency idea resurfaces in `captcha regenerator/masking/` and `auto_synth Part C`.
- `src/render_synthetic.py` — original PIL renderer. Superseded by the PHP port (PIL version assumed a CycleGAN would close the style gap, which it never did).
- `src/qwenvl/` — ⚠ **NOT mentioned anywhere else in this CONTEXT.** A VLM-recognizer experiment: LoRA fine-tune **Qwen2-VL-2B** on PHP-MyBB synth (adapter on LM attention proj only, vision tower frozen). `train.py`/`infer.py`/`data.py`. Outcome not recorded in the results docs — **status unknown; check `checkpoints/` for a qwenvl run or treat as exploratory/unfinished.**

## Misc utilities
- `src/compare_real_vs_synth.py`, `compare_v2_synth_vs_real.py`, `php_synth/compare_php_vs_real.py`, `smoke_v3.py`, `inspect_data.py`, `preview_large.py`, `render_synthetic.py` — visual A/B and data-inspection helpers (outputs in `data/_*.png`).

## Results-doc reading order (model lineage)
1. `model_one_v1_results.md` — Model 1 v1 (32 Windows TTF fonts) → 31.18% w greedy.
2. `model_one_v1.5_results.md` — + ensemble (α=0.70) → 36.41% w.
3. `model_one_v2_results.md` — **font swap to MyBB's 2 real fonts** → 39.94% w. *The biggest single jump; closed ~26pt synth-real gap.*
4. `model_one_v1.5_v2_results.md` — v2 synth + ensemble (α=0.30, CNN-heavy) → **49.16% w / 86.86% c (flagship)**. Also documents the failed `compbg` experiment in detail.
5. `analysis.md` — selective prediction + temperature calibration across M1/1.5/2/2.5.

## The one insight this whole directory proves
**Data distribution-matching beats model sophistication.** Same PARSeq, swapping
32 generic fonts → MyBB's actual 2 fonts, lifted real accuracy +8.76pt (Model 1
v1→v2). The ensemble adds ~9pt more. Everything fancy (CycleGAN, cleaner,
diffusion synth, pseudo-labeling) failed; the boring distribution-matched PHP
renderer won. This is why `auto_synth/` exists — to get distribution-matching
*without* needing the target's source code.
