# MASTER CONTEXT — `e:/pranav` thesis

> Top-level map of the whole thesis, written during the 2026-06 exploration
> crawl. Every working directory has its own `CONTEXT.md` (and usually `GOAL.md`
> / results docs); **this file is the index + the narrative that ties them
> together** so you (or an agent) can see the whole shape at a glance and know
> where to dig.

## The thesis in one paragraph

**Recognize and synthesize dark-web-market text CAPTCHAs (CryptBB, DarkBay,
City, Courier Market, JoeyRetitling) under a strict no-human-labels rule** — a
recognizer may train only on renderer-labeled *synthetic* data; the real labeled
sets are eval-only. Two intertwined research lines: (1) a **recognizer arc**
(M1→M2→M3) studying how curriculum/adaptation affect a PARSeq recognizer, and
(2) a **synthesis line** chasing label-free synthetic data that matches a
scheme's distribution *without* needing that scheme's source code.

## The two foundational empirical facts (everything hangs off these)

1. **Generic OCR/VLMs can't read these captchas.** Off-the-shelf TrOCR /
   GOT-OCR2 / EasyOCR / Qwen2-VL-2B all score ≤2.5% word / ≤27% char on real
   CryptBB (`model 1 Dino attention/`). → you *must* train a bespoke recognizer.
2. **Data distribution-matching beats model sophistication.** Same PARSeq,
   swapping 32 generic fonts → the target's *actual* 2 fonts lifted real
   accuracy +8.76pt (`model 1/`). Every fancy idea (CycleGAN, diffusion,
   cleaners) failed; the boring distribution-matched renderer won.

## Recognizer arc — results on the 651 real CryptBB eval set

| Model | Idea | PARSeq alone (word) | + ensemble (word) | Dir | Status |
|---|---|---|---|---|---|
| **Model −1** (model 0) | generic 5-PHP-lib synth corpus (breadth) | ~1% (synth val) | — | `model 0/` | FAILED → motivated distribution-matching |
| **Model 1 / 1.5** | distribution-matched MyBB renderer + char-CNN ensemble | 39.94% | **49.16%** (M1.5 v2) | `model 1/` | shipped baseline |
| **Model 2 / 2.5** | static curriculum (teacher NLL buckets, 1/1/13) | 54.22% | **57.60%** (M2.5) | `model 2/` | **flagship** |
| **Model 3** | adaptive curriculum (student loss EMA, no teacher) | ~21% synth peak then collapse | — | `model3/` | FAILED (unstable; never reached M2 bar) |

Char accuracy tracks ~84-89%. Ensemble = PARSeq + bbox detector + 3-scale
char-CNN, log-prob mixed. Selective-prediction / calibration analysis in each
dir's `analysis.md`. **Recurring unsolved error: case confusion** (p→P, s→S…).

## Synthesis line — the evolution toward label-free, source-free synth

The synth problem: M1's win required porting MyBB's `captcha.php`. For schemes
whose source we *don't* have, we need another way to get distribution-matched
labeled synth. The attempts, in order:

1. **`captcha regenerator/`** — extract the captcha's *parts* (char mask + bbox)
   to reconstruct it. `char_extraction/` (otsu_cc / SAM / SWT / PSSTRNet method
   zoo + eval harness) — **every method OUT_OF_SCOPE** on all schemes; pretrained
   scene-text-removal nets output "all background" on adversarial captchas.
   `masking/` (VLM-color-gate → HSV or DINO-attention mask). Downstream
   `bg_syth`/`Font_syth` never started.
2. **`model 1 Dino attention/`** — glyph-conditioned **diffusion** synth
   (GlyphControl + style LoRA). Content drifts (0/5 PARSeq read-back). FAILED.
   (Also holds the OCR-baseline result, fact #1 above.)
3. **`tiny STR/`** — orthogonal: a 5.67M from-scratch recognizer proving
   *architecture is replaceable, data is load-bearing* (matches PARSeq char acc;
   `word≈char^5`; 31% real ensembled). Parked.
4. **`synthetic image generation/`** — the **bridge**. Started as a
   curriculum-generator with a difficulty knob `t` + parametric renderer +
   DINOv2 auto-tuner + refinement GAN (refiner = marginal, dropped; knob =
   dropped). Then **pivoted** to `synth_v1/` = **char-crop bank recombination**
   (pseudolabel real captchas → per-char alpha crop bank → recompose on LaMa
   bg). This **WORKED on CryptBB** and is the conceptual seed of auto_synth. Then
   `synth_v2/` = scheme-agnostic perception (CRAFT + Qwen + Consensus Entropy),
   planned then superseded.
5. **`auto_synth/`** — the **newest, most complete** synthesis system. Label-free
   captcha replication from unlabeled images via 3 orthogonal parts:
   **A** alphabet/length discovery (CC-segment + DINOv2 + KMeans, OR easyOCR
   direct, OR Qwen-VLM), **B** glyph bank (leakage-aware char-crop sampler),
   **C** background extraction (LaMa, with SAM / GroundingDINO / Qwen variants).
   Runs end-to-end on 4 schemes; **City synth ≈ indistinguishable from real**,
   Courier weakest (underline survives). ⚠ **Gold-standard test still NOT done:**
   train a recognizer on auto_synth output and eval on a labeled set
   (head-to-head vs M2.5's 57.60%).

## What worked vs what failed (the "I don't remember" answer)

**Worked:**
- Distribution-matched PHP renderer (MyBB port) → the whole recognizer line.
- char-CNN multi-scale ensemble (+~9pt word everywhere).
- Static teacher-bucket curriculum (M2, +8pt over M1).
- char-crop bank recombination synth (synth_v1, and auto_synth Parts A/B/C) — visually matches real.
- LaMa background extraction (with CC/SAM char masks).
- TinySTR minimal recognizer (matches PARSeq, validates "data > architecture").

**Failed / dead ends (don't re-try without a new idea):**
- Generic multi-library synth corpus (model 0, ~1% word).
- CycleGAN synth↔real translation (content drift).
- Glyph-conditioned diffusion / GlyphControl LoRA (content drift, 0/5).
- Pretrained scene-text-removal nets on captchas (output "all background").
- Composite synth (`compbg`: real bg + rendered text → edge artifacts, 1.23% word).
- Adaptive/self-paced curriculum (model 3 — spikes then collapses).
- Off-the-shelf OCR/VLMs as solvers (≤2.5% word).
- Refinement GAN (marginal lift, not load-bearing).
- char_extraction method zoo (all OUT_OF_SCOPE on IoU bar).

## Directory index

| Dir | One-liner | Where to start |
|---|---|---|
| `auto_synth/` | **Newest** label-free source-free synth (Parts A/B/C) on 4 schemes | `GOAL.md`, `RESULTS.md`, `CONTEXT.md` (+ code-map appendix) |
| `synthetic image generation/` | The bridge: curriculum-gen → char-bank (synth_v1) → CRAFT+Qwen (synth_v2) | `CONTEXT.md` evolution appendix, `src/synth_v1/STATUS.md`, `v2_plan.md` |
| `captcha regenerator/` | Part-extraction attack (char_extraction zoo + masking) — all parked | `CONTEXT.md`, `char_extraction/research.md` |
| `model 0/` | "Model −1": generic synth corpus, failed; **vendors third_party/parseq** | `CONTEXT.md` |
| `model 1/` | Distribution-matched recognizer + ensemble (M1/1.5, 49.16% flagship) | `CONTEXT.md`, `TRAINING_INFRASTRUCTURE.md` |
| `model 1 Dino attention/` | OCR baselines (the floor) + failed diffusion synth. Misnamed dir. | `CONTEXT.md`, `research.md` |
| `model 2/` | Static curriculum recognizer (M2/2.5, **57.60% best**) | `CONTEXT.md`, `notes.md` |
| `model3/` | Adaptive curriculum — implemented + run ~10x, FAILED | `context.md` (status block at top), `setup.md` |
| `tiny STR/` | 5.67M from-scratch recognizer (data>architecture proof) | `CONTEXT.md`, `notes.md` |
| `synth_data/` | 100k synth_v1 CryptBB output PNGs (data only) | — |
| `papers/` | One ref PDF (B-CvT skin-lesion style transfer; a rejected approach) | — |
| `recreate syth/`, `tmp/` | empty | — |
| `personal/` | **NON-THESIS** — keelo `kith-buysheet-agent` work tool + "private texting" chat app. Left untouched. | — |
| `*.zip` + `*-001/` dirs at root | The raw datasets: City (2000), Courier Market (2510), CryptBB unsolved (600), DarkBay labeled (1000, eval-only), JoeyRetitling (651) | — |

## Shared infrastructure

- **PARSeq** (Bautista & Atienza, ECCV 2022) vendored in `model 0/third_party/parseq` and `model 1/third_party/parseq`; reused by models 0/1/2/3.
- **Python venv:** `model 1/.venv/Scripts/python.exe` (Python 3.9.13, PyTorch 2.5.1+cu121) — shared by everything. sqlite DLLs via `os.add_dll_directory("C:/ProgramData/Anaconda3/Library/bin")`.
- **PHP renderer:** XAMPP `C:/xampp/php/php.exe` (the MyBB `captcha.php` port).
- **Hardware:** Windows 11, single RTX A4000 (16 GB).
- **Eval set of record:** 651 labeled real CryptBB at `model 1/data/eval/` — used by M1/M2/M3/tinySTR for apples-to-apples comparison.

## The single most important open thread

**Validate auto_synth downstream.** Everything points here: train a PARSeq on an
`auto_synth/data/<scheme>/synth_pool/` and eval on a held-out labeled set
(CryptBB `model 1/data/eval/` is the natural head-to-head vs M2.5's 57.60%). Until
that runs, auto_synth's quality rests only on visual A/B panels. All the pieces
exist (synth pools of 2000 each — bump to 10k+; PARSeq trainers in model 1/2).
