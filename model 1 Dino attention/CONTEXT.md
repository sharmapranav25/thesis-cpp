# model 1 Dino attention — CONTEXT

> Written during the 2026-06 exploration crawl (only `research.md` existed before).
> ⚠ **The directory name is misleading** — there is no DINO-attention work here.
> This directory is two things: (1) **off-the-shelf OCR/VLM baselines** on real
> CryptBB, and (2) a **glyph-conditioned diffusion synth** track (GlyphControl +
> LoRA). Both are essentially negative/reference results. Treat the folder name
> as a leftover label.

## Part 1 — OCR/VLM baselines on real CryptBB (the useful result)

Ran 4 pretrained recognizers zero-shot on the 649-image labeled real eval set
(`src/run_*.py`, shared harness `src/eval_common.py`, filename-stem = label).
Results in `results/summary.csv`:

| Model | Word % | Char % | ms/img |
|---|---|---|---|
| TrOCR-large-printed | 0.00–0.15 | 0.27–10.72 | 32–109 |
| GOT-OCR2.0 | 0.62 | 19.61 | 2298 |
| EasyOCR | 1.39 | 25.14 | 39 |
| **Qwen2-VL-2B** (best) | **2.47** | **27.49** | 390 |

**Headline: every off-the-shelf OCR/VLM is useless on CryptBB captchas
(≤2.5% word, ≤27% char).** This is the empirical justification for the whole
thesis — generic OCR can't read these, so you *must* train a recognizer on
distribution-matched synthetic data (which Model 1 does → 39.94%, Model 2.5 →
57.60%). Keep this table; it's the baseline the thesis beats by ~20-55×.

Per-model prediction CSVs: `results/{trocr_large_printed,got_ocr2,qwen2vl_2b,easyocr}.csv`.

## Part 2 — Glyph-conditioned diffusion synth (GlyphControl + LoRA)

The attempt to generate real-looking synth *with guaranteed labels* via diffusion
instead of CycleGAN (which had failed in `model 1/` — cycle loss doesn't protect
glyph identity). Approach (`research.md` has the full lit survey: GlyphControl,
TextDiffuser, AnyText, EasyText, UniGlyph, FreeText, Diff-CAPTCHA, etc.):

1. `src/diffusion_render.py` — render label → 5 evenly-spaced glyph bboxes → **GlyphControl** (SD-2.0 + glyph ControlNet, 512×512) → crop strip → 200×60. Glyph ControlNet hard-wires content; diffusion paints captcha style on top.
2. `src/lora_train.py` — LoRA-tune the SD-2.0 **U-Net** (attention projections only, ControlNet frozen) on the 600 real CryptBB images to absorb style (speckle bg, vivid hues, chromatic shadows). Checkpoints exist: `checkpoints/lora/lora_step{50,500,1000,1500,2000}.pt`.
3. `third_party/GlyphControl/` — vendored repo + pretrained `textcaps5K_epoch_40_model_ema_only.ckpt`.

**Outcome (recorded in `model 1/CONTEXT.md`, not here):** GlyphControl
zero-shot produces wrong-style outputs and *drifts content* — **0/5 on PARSeq
read-back**. The base model was trained on natural-scene text; CryptBB's heavy
speckle/color/shadow is too far out of distribution, and the style LoRA on only
600 images didn't close the gap. **Track abandoned.** This failure (plus
CycleGAN's) is what motivated the *split-the-problem* insight in `synthetic
image generation/` and `auto_synth/` (let a renderer guarantee content, handle
style separately and more cheaply than full diffusion).

## File map

| Path | Purpose |
|---|---|
| `research.md` | Glyph-conditioned diffusion literature survey + chosen pipeline + risks. The most reusable artifact here. |
| `src/run_{trocr,got,qwenvl,easyocr}.py` | OCR/VLM baseline runners (each loads one model, predicts on a dir). |
| `src/eval_common.py` | Shared load/score harness (filename-stem labels, case-sensitive exact + char + confusion). |
| `src/diffusion_render.py` | GlyphControl synth generator. |
| `src/lora_train.py` | Style-LoRA trainer for the GlyphControl U-Net. |
| `results/*.csv` | Baseline predictions + `summary.csv`. |
| `checkpoints/lora/lora_step*.pt` | Style-LoRA checkpoints (the failed track). |
| `third_party/GlyphControl/` | Vendored GlyphControl repo + weights. |
| `.venv-glyph/` | Separate venv (GlyphControl/diffusers deps conflict with model 1's). |

## Relationship to the thesis

- **Part 1 is the baseline floor** the entire recognizer line is measured against (generic OCR ≈ 0-2.5% word on CryptBB).
- **Part 2 is one of several failed "make synth look real" attempts** (alongside CycleGAN in `model 1/` and the compbg composite). Collectively they motivate the renderer-guarantees-content, style-handled-cheaply philosophy of `synthetic image generation/` and `auto_synth/`.
- Qwen2-VL-2B reappears as a fine-tune target in `model 1/src/qwenvl/` (also exploratory).

## Status

Parked. OCR baselines are done and final. Diffusion track abandoned (content
drift). No further work planned.
