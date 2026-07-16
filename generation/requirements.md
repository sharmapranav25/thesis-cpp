# Requirements — CAPTCHA Synth Generation (`generation/`)

> The constraints any work in this folder must satisfy. The goal is a **general,
> source-free, label-free synth-generation pipeline**: point it at a folder of
> **unlabeled** captchas from *any* scheme and produce **labeled** look-alike
> training data (the label is known because *we* render the text), good enough to
> train a recognizer that scores well on real captchas.
>
> The failure catalogue for everything already tried is in
> [FAILED_SYNTHESIS_ATTEMPTS.md](FAILED_SYNTHESIS_ATTEMPTS.md) — read it before
> proposing a new approach; the "do-not-retry" list there is binding here.

---

## 1. Hard constraints (MUST)

These are the iron rules. A method that violates any of them is out of scope, no
matter how well it scores.

- **R1 — Source-free.** No access to the target captcha **scheme's own source code**
  and no use of the **scheme's own font files**. Matching a scheme by porting its
  generator (e.g. the MyBB `captcha.php` port + MyBB's 2 bundled fonts that powered
  Model 1/2) is **source-based** — that is the thing this pipeline must achieve
  *without*. Only the scheme's *unlabeled output images* may be used as input.

- **R2 — PHP / generic captcha libraries ARE allowed, as long as they are not
  captcha-scheme-specific.** *(new)* General-purpose rendering tools are fair game:
  PHP + PHP-GD primitives (`imagettftext`, sine-wave warp, noise/lines/arcs) and
  generic PHP captcha libraries (Securimage, Gregwar/Captcha, KCAPTCHA, Cool PHP
  Captcha, Simple PHP Captcha, etc.), driven with **generic fonts**. What is
  forbidden is any renderer, library, config, or font that is **specific to the
  target scheme** (the scheme's own captcha code or typeface). Rule of thumb: a tool
  you could point at *any* scheme = allowed; a tool that encodes *this* scheme = R1
  violation.

- **R3 — No labeled data is received at any part of synth generation.** *(new)* The
  pipeline ingests real captchas as **unlabeled images only**. No ground-truth or
  human-provided transcription of any real captcha enters generation at any stage.
  The only labels that ever exist are:
  1. labels **we assign** to synth we render ourselves (correct-by-construction), and
  2. **self-generated pseudo-labels** produced by our own models during
     self-training (model-generated, not *received* — this is allowed and is the
     route that worked).
  Real labeled sets are **held entirely outside generation** and used only for final
  scoring (see R6).

- **R4 — General across schemes.** The pipeline must be **point-at-a-folder**: swap
  the unlabeled input directory and it works on a new scheme with no scheme-specific
  code. Per-scheme hand-tuning is a smell; if a step needs it, it must degrade
  gracefully (e.g. auto-measured from the pool), not hard-code the scheme.

- **R5 — Local / open-source only, within the hardware budget.** No paid APIs, no
  cloud inference. Everything runs on a single **RTX A4000 (16 GB)**; models that
  don't fit (e.g. a 20B style model) are out. Open weights / local checkpoints only.

## 2. Success criterion (MUST)

- **R6 — Gold-standard downstream validation.** A synth pipeline is only "good" once a
  recognizer **trained on its synth output** is **scored on a held-out labeled real
  set**. Visual A/B panels are necessary but **not sufficient** (this is exactly the
  open gap in `auto_synth/`). The eval set of record is the **651 labeled real
  CryptBB** captchas (`model 1/data/eval/`); report word + char accuracy.
  - **Baselines to beat / contextualize** (651 real CryptBB): OCR/VLM floor ≤2.5% word;
    source-free+label-free VLM **30.4% word** (current best under these constraints);
    source-based Model 1 39.94%; Model 2.5 flagship **57.60% word** (uses the scheme
    font — the ceiling we're chasing source-free).
  - For a typical (non-pathological) scheme the same pipeline should land far higher
    (DarkBay reached **91.0% word** source-free).

## 3. Data rules (MUST — prevents label leakage)

- **R7 — Train only on unlabeled pools; never train on an eval set.** Pseudo-label and
  train strictly on the designated **unlabeled** pools (e.g. CryptBB 600 `unsolved`,
  DarkBay ~799 pool). The labeled evals (651 CryptBB, DarkBay 200) are **scored only,
  never trained on**. Verify train/eval disjointness before every run.

- **R8 — Do not use answer-key directories as labels.** `JoeyRetitling` is the CryptBB
  **eval answer key** and DarkBay-labeled is an eval set — never feed either in as
  labels or as training targets. Treat them as scoring material only. (See the data
  gotcha memo.)

## 4. Explicitly NOT allowed

- ❌ The target scheme's own captcha source / generator / config.
- ❌ The target scheme's own font files.
- ❌ Any human or ground-truth **labels** for real captchas, at any generation stage.
- ❌ Training or tuning on the labeled eval sets (or on `JoeyRetitling`).
- ❌ Paid/cloud model APIs; models that exceed 16 GB VRAM.

## 5. Explicitly allowed (for the avoidance of doubt)

- ✅ Generic PHP / PHP-GD rendering and generic PHP captcha libraries (R2), generic
  fonts, procedural backgrounds.
- ✅ General-purpose pretrained **open** models used as scheme-agnostic tools —
  font-general VLMs (Qwen2-VL), LaMa (inpainting), SAM/MobileSAM, GroundingDINO,
  DINOv2, CRAFT, easyOCR, etc. — provided they are not fine-tuned on scheme labels
  and fit the hardware. (Note their known failure modes in the catalogue.)
- ✅ Self-training / pseudo-labeling on the unlabeled real pools (self-generated
  labels, R3.2), subject to the minimum-pool floor (~89+ correct pseudo-labels).
- ✅ The unlabeled real captcha images themselves, as input and as self-supervision.

## 6. Environment (reference)

- **Hardware:** Windows 11, single RTX A4000 (16 GB).
- **Python:** shared venv `model 1/.venv/Scripts/python.exe` (3.9.13, PyTorch
  2.5.1+cu121). sqlite DLLs via `os.add_dll_directory("C:/ProgramData/Anaconda3/Library/bin")`.
- **PHP:** XAMPP `C:/xampp/php/php.exe` (for R2 renderers).
- **Caches/downloads on `E:`** — `HF_HOME=E:\hf_cache`.
- **Eval set of record:** `model 1/data/eval/` (651 real CryptBB), scored-only.

---

*Requirements R2 and R3 added per user direction; R1, R4–R8 and the environment
pulled forward from the established thesis constraints (source-free/label-free rule,
data-disjointness rules, hardware/stack). If a constraint here conflicts with an
older doc, this file wins for work under `generation/`.*
