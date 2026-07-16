# synthetic image generation — curriculum-driven captcha synth

> See `GOAL.md` for the research framing.
> See `plan.md` for the full multi-phase plan.
> See `mvp_plan.md` for the testable cut deliverable in 6–8 hours.

## What this project is

A curriculum-learning study on synthetic captcha training. The deliverable is a
generator that produces labeled synth captchas at controllable difficulty,
plus a 4-model comparison harness that measures how curriculum and feedback
affect a PARSeq recognizer's accuracy on real captchas.

The synth generator is **one component**. It is composed of:
- A **parametric renderer** (TTF fonts + per-char styling + bg sampling) for
  cheap easy→medium output.
- A **pixel refinement U-Net** for closing the realism gap when difficulty is
  high.
- A **few-shot font generator** (DG-Font / AGIS-Net) for schemes whose font
  isn't in our TTF pool.

These compose under a single difficulty knob `t ∈ [0, 1]`.

## Why this exists

`E:\pranav\model 1\` reached 49.16% word / 86.86% char on real CryptBB using
a label-free pipeline whose trick was a **distribution-matched PHP renderer**
(line-by-line port of MyBB's `captcha.php`). That trick only works because we
got lucky finding CryptBB's source captcha library. For schemes whose source
we don't have — DarkBay, City, Courier Market, future schemes — we need a
different way to produce labeled synth.

We tried CycleGAN (content drift), GlyphControl (distribution mismatch +
content fidelity), AnyText (same), QwenStyle (20B params, demo bug),
B-CvT (medical-imaging-specific). All tried to do **content + style at once**.
The user's insight: **split the problem** — let the renderer handle text
(content guaranteed correct), let the rest of the stack handle style.

Beyond that one-shot synth pipeline, the larger research question is: given
that synth has a quality-vs-difficulty trade-off, can a curriculum that
controls difficulty per-batch (and adapts to the student) train a better
recognizer than a single fixed synth distribution? That's what the 4-model
study (see `GOAL.md`) measures.

## Strict constraints (inherited from `model 1/`)

- **No human labels in training the recognizer.** Synth labels come from the
  renderer.
- **The 651 real CryptBB captchas at `model 1/data/eval/` are eval-only.**
  Any per-scheme training (renderer tuning, refinement net, FFG) for CryptBB
  uses only the disjoint unlabeled set at `model 1/data/raw/` (600 images).
- **The 1000 labeled DarkBay captchas at `DarkBay_Labeled-...\` are eval-only**
  for the DarkBay scheme. ⚠ Currently both the LaMa-extracted DarkBay bg pool
  (`data/darkbay_bg_lama/`, 1000 images) and the DINOv2 search anchor read
  from this set — eval contamination. To resolve before training a DarkBay
  recognizer: either source unlabeled DarkBay or accept the contamination and
  report it.
- **Per-scheme generality:** new scheme = new config + a folder of unlabeled
  real samples. No new code.

## Data sources per scheme

| Scheme | Real (unlabeled) | Labeled (eval-only) | Native size | Status |
|---|---|---|---|---|
| CryptBB | 600 in `model 1/data/raw/` | 651 in `model 1/data/eval/` | 200×60 | First validation case. Has 49.16%/86.86% baseline from Model 1 v2. |
| DarkBay | — (blocker) | 1000 in `DarkBay_Labeled-...` | 200×50, 6-char | Eval contamination above must be resolved before recognizer training. |
| Courier Market | 2510 in `Courier Market (live)-...` | — | ~200×60 | Untouched. Likely MyBB-based — try Model 1 v2 first if recognizer needed. |
| City | 2000 in `City (live)-...` | — | 324×81 | Untouched. |

## What's been built so far

### Generator components (parametric only)
- `src/extract/extract_bgs.py` (cv2.inpaint, kept for reference) and
  `src/extract/extract_bgs_lama.py` (LaMa, used). LaMa hallucinates ~20% on
  CryptBB but no ghost text. CryptBB output: `data/cryptbb_bg_lama/` (600
  PNGs, ~65% strict-clean / 80% clean+borderline).
- `src/extract/extract_bgs_darkbay_lama.py` — DarkBay equivalent. Output:
  `data/darkbay_bg_lama/` (1000 PNGs). ⚠ Sourced from the labeled eval set
  (see contamination note above).
- `model 1/src/php_synth/generate_mybb_captcha.php` — has `--text-only` mode
  (transparent RGBA, no bg primitives, no border) and `--width / --height /
  --captcha-len / --charset` flags.
- `src/composite/render_darkbay_text.py` — Pillow-based text-only renderer
  for DarkBay (200×50, 6-char). Hardcoded, predates the auto-tuner.
- `src/auto_tune/render_darkbay_v2.py` + `renderer_config.py` — config-driven
  parametric renderer. 16 knobs (font set, size range, rot, box prob, color
  ranges, position jitter, char spacing). Now has a `__main__` so it can
  render N samples from a saved config.
- `src/composite/compose.py` — generic alpha-compositor: text-only RGBA over
  sampled bg + light jitter + 1-px black border. Scheme-agnostic.

### Auto-tuner (search loop)
- `src/auto_tune/dinov2_search.py` — encodes real samples + candidate synth
  with DINOv2-small, scores configs by `1 - cos(real_anchor, synth_anchor)`,
  random search.
- DarkBay run (200 trials, ~70s, batch 48): best dist 0.189 at trial 51,
  plateaued. Saved as `configs/darkbay_renderer.json`. Sample comparison at
  `notebooks/04_darkbay_diff_v2.png` — bg colorways + boxed style match real;
  per-char vs per-image style mismatch is visible (real DarkBay tends to be
  uniformly all-boxed or all-plain; v2 mixes per-char).
- ⚠ DINOv2-cosine is a similarity metric, not a curriculum metric. It picks a
  single "looks-real" config, which is useful as the **hard anchor** of the
  difficulty axis but is *not* what the curriculum optimizes against.

### Refactor needed before MVP
- `render_darkbay_v2.py` hardcodes `W=200, H=50, N_CHARS=6, CHARSET=...`. New
  scheme = duplicate the file. Should accept a `SchemeSpec` dataclass.
- `dinov2_search.py` hardcodes `REAL_DIR`, `BG_DIR`, `OUT_CFG`. Should be CLI
  flags.
- `extract_bgs_*.py` are scheme-named for similar reasons.

### CryptBB-specific work (independent — predates the curriculum framing)
- 100k composite training set at `model 1/data/cryptbb_synth_compbg/` (2.1 GB,
  built from PHP --text-only + CryptBB bg sampling). User running training.
- New training configs in `model 1/configs/`: `parseq_compbg.yaml`,
  `detector_compbg.yaml`, `charcls_64/96/128_compbg.yaml`.
- `scripts/run_compbg_overnight.sh` — chains all 5 trainings + ensemble eval.
  ⚠ Two prior background launches hit silent exit 127 after PARSeq partially
  started. Run manually or under a real terminal session.

## What three viable architectures looked like (historic — kept for reference)

These were the three options on the table when this project started. The
chosen path is a hybrid: **A** (composite) for cheap easy/medium synth +
**B** (refinement) for the realism end of the difficulty axis.

- **A. Composite pipeline.** Render text on transparent bg via PHP, alpha-blend
  over extracted-from-real bg. *Risk: real text isn't a clean overlay.*
- **B. Mask-aware GAN translation.** PHP outputs `(synth, text_mask)`; GAN
  produces realistic image with text preserved exactly via L1 on text pixels.
  *Higher quality, more complex training.*
- **C. Renderer + bg replacement.** PHP outputs full captcha; GAN inpaints
  non-text regions. *In between A and B.*

## Reuse from `model 1/`

- `src/php_synth/generate_mybb_captcha.php` — text rendering (text-only mode).
- `src/parseq/`, `src/detector/`, `src/charcls/`, `src/eval/case_decode.py` —
  recognizer training and eval pipelines. Used by all 4 models in the study.
- `data/raw/` — 600 unlabeled CryptBB (in-bounds for refinement-net training).
- `data/eval/` — held-out comparison target.
- `.venv/` — shared Python env (Python 3.9.13, PyTorch 2.5.1+cu121). Reuse
  via absolute path or `PYTHONPATH`.

## Compute

Local RTX A4000 (16 GB). Same env as `model 1/`. One GPU = models train
sequentially in the comparison harness.

---

# Appendix — full project evolution (added during 2026-06 exploration crawl)

⚠ The body above documents the **original framing** (curriculum-driven generator
with a difficulty knob `t`, parametric renderer + DINOv2 auto-tuner). The project
**pivoted twice** after that. This is the most important directory in the thesis
for understanding *how `auto_synth/` came to be* — it's the bridge. Reading order
of the real evolution:

## Phase 1 (original) — curriculum generator with difficulty knob
- `src/generator/parametric.py` (ParametricGenerator), `refined.py` (RefinedGenerator: composes parametric + refiner under knob `t`, `alpha(t)=max(0,2t-1)`), `render.py`, `real_bg.py`.
- `src/auto_tune/` — `render_darkbay_v2.py` + `renderer_config.py` (16-knob config-driven renderer), `dinov2_search.py` (random search scoring configs by `1-cos(real,synth)` DINOv2; best DarkBay dist 0.189, plateaued), `difficulty.py`.
- **This `t`/curriculum machinery is now DEAD** — `v2_plan.md` says "difficulty knob permanently dropped." It lingers on disk unused (noted in `synth_v1/STATUS.md` item 5).

## Phase 2 — refinement GAN (`src/refine/`)
- `model.py` — mask-aware U-Net generator (input 4×H×W = RGB composite + text mask; 4 down/4 up + skips; tanh) + PatchGAN discriminator. `data.py`/`data_bg.py`/`train.py`.
- Trained **4 versions** (`mvp_results/refiner_v{1..4}_*`); v4 ran 15000 steps, stable GAN (G_adv~4, G_l1~0.07). Checkpoints `checkpoints/darkbay_refiner_v4_alpha/`.
- **Verdict: marginal.** `v2_plan.md` line: *"v1's refiner showed marginal lift; not load-bearing."* Refiner dropped from the go-forward plan. (It closes JPEG-noise/anti-alias/color-bleed gaps the renderer can't parameterize, but didn't move recognizer accuracy enough to justify it.)

## Phase 3 (THE PIVOT) — `src/synth_v1/` char-crop bank — **direct precursor to `auto_synth/`**
The insight that reframed the whole thesis: **don't render glyphs from fonts —
*recombine real char crops* pseudolabeled from the unlabeled real captchas.**
- `scheme.py` (SchemeSpec; only CryptBB wired), `probe.py` (auto-detect W/H/charset/length from raw captchas via PARSeq), `build_bank.py` (detector_v2 + parseq_v2 pseudolabel real → **per-char alpha-keyed RGBA crop bank**, case-encoded folders for Windows FS), `font_bank.py` (`CharBank.sample(ch)` with jitter), `compose.py` (scheme-agnostic: bank + LaMa bg → labeled synth), `generate.py` (CLI).
- **Result (`synth_v1/STATUS.md`):** 600 raw CryptBB → **2714 alpha-keyed crops across 61/62 chars**, 99.2% image keep rate. QA sheet `mvp_results/synth_v1_cryptbb_qa.png` — **visually matches real CryptBB.** This *proved* the char-crop-recombination + LaMa-bg recipe works.
- **The blocker it exposed:** `build_bank` uses `detector_v2` (hardcoded 5 boxes, CryptBB-trained) + `parseq_v2` (CryptBB-trained). So it only works on CryptBB. *"This is what blocks the any-captcha claim"* (STATUS item 4). Residual issues: faint rectangular bg halo on some crops (alpha-keying imperfect), PARSeq case-ambiguity leaking into bank labels.

## Phase 4 — `src/synth_v2/` scheme-agnostic perception (the generalization attempt)
Replace v1's CryptBB-specific perception with open-source, scheme-agnostic models:
- `perceive.py` — **CRAFT** (per-character region heatmap → boxes) + **Qwen2.5-VL-7B** (char strings) paired L2R, filtered by **Consensus Entropy** (dual-pass agreement, CE-OCR Zheng 2025). `smoke_{craft,dinov2_attn,dinov2_pca,qwen}.py` smoke tests.
- `v2_plan.md` is the detailed phased plan (CRAFT+Qwen→bank→render, gate at each phase, City as the "any captcha" test). `synth_v2/STATUS.md` is **empty** — v2 was planned in detail but not carried to a logged result here.
- **This is conceptually exactly `auto_synth/` Part A (alphabet/perception) + Part B (glyph bank) + Part C (LaMa bg).** `auto_synth/` is the cleaner re-implementation that finished what synth_v2 started — using CC+DINOv2 clustering / easyOCR / VLM for the label-free perception instead of CRAFT+Qwen, and running end-to-end on 4 schemes.

## The narrative this directory carries
1. Curriculum/difficulty-knob generator (Phase 1) → **dropped**.
2. Refinement GAN to close realism gap (Phase 2) → **marginal, dropped**.
3. **Char-crop bank recombination (Phase 3, synth_v1) → WORKS on CryptBB** — the key positive result and the conceptual seed of `auto_synth`.
4. Scheme-agnostic perception (Phase 4, synth_v2) → planned, then **superseded by `auto_synth/`** which executed the same idea to completion.

So: the difficulty-knob 4-model curriculum study (the original GOAL.md framing)
was **abandoned** in favor of the label-free char-bank synth line that became
`auto_synth/`. The 4-model curriculum study survived instead in the recognizer
directories (`model 1/2/3`) as M1→M2→M3.

## Doc index for this dir
- `GOAL.md`/`plan.md`/`mvp_plan.md`/`mvp_api.md` — original curriculum-generator framing (Phases 1-2). Largely historical now.
- `v2_plan.md` — the synth_v2 (CRAFT+Qwen) plan. The forward-looking doc; realized as `auto_synth/`.
- `src/synth_v1/STATUS.md` — **read this** — the char-bank positive result + its blocker.
- `notes.md`/`future.md` — deferred ideas (FFG/DG-Font, refiner-on-clean-pool, DarkBay contamination).
- `mvp_results/` — refiner v1-v4 logs + QA PNGs (`synth_v1_cryptbb_qa.png` is the one to look at).
