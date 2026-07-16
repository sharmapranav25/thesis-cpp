# model 0 — CONTEXT

> Written during the 2026-06 exploration crawl (no prior notes existed here).
> This is the **earliest experiment in the thesis** and a clean negative result.

## What model 0 is

The first attempt at the core thesis idea: *train a CAPTCHA recognizer on
synthetic data only (no human labels), then read real captchas.* The strategy
here was **breadth** — generate a big, maximally-varied synthetic corpus by
dispatching across **5 off-the-shelf PHP captcha libraries**, hoping a PARSeq
recognizer trained on that generic mix would generalize.

**It failed.** This is the negative result that motivated Model 1's pivot to a
*distribution-matched* renderer (a line-by-line port of the target scheme's own
captcha library).

## The result (the one number to remember)

PARSeq (23.8M params, 15 epochs) peaked at **1.08% word accuracy on its own
synthetic validation split** (`val_accuracy=1.080` in the checkpoint name;
`checkpoints/parseq/lightning_logs/version_0/metrics.csv`). Char-level NED rose
to ~38 (so it learned *some* glyph shape) but word accuracy never left the
floor. It was **never even evaluated on real captchas** — synth val alone showed
the approach was dead.

**Why it failed:** the corpus is too heterogeneous to learn word-level mapping.
5 libraries × random image sizes (H 40-100, W 150-280) × variable length 4-8 ×
multiple charsets (incl. case-sensitive 62-char) × every distortion knob
randomized = enormous entropy, all squashed to a 32×128 input. Word accuracy ≈
(per-char acc)^length collapses. **Lesson carried into Model 1: match the target
scheme's actual distribution; don't train on a generic union of everything.**

## What's on disk

| Path | What |
|---|---|
| `generators/gen_corpus.php` | The corpus generator. Dispatches across securimage / gregwar / kcaptcha / coolcaptcha / simplecaptcha. Each lib has a `gen_<lib>()` fn that randomizes size, fonts, colors, distortions, and **forces a known label** (e.g. `SecurimageForced` subclass overrides `createCode()`). Writes sharded PNG/JPG + `labels.tsv` with a `params_json` (incl. `lib` field for later filtering). |
| `generators/cool_captcha_class.php`, `lib_common.php`, `_check_fonts.php` | Cool PHP Captcha wrapper (inlined to avoid side effects) + shared helpers (charset pool, rand_text, font pool, sharding) + font sanity check. |
| `vendor/` | The 5 vendored PHP captcha libraries (+ their fonts/backgrounds) and `_downloads/*.zip` originals. securimage, gregwar_captcha, gregwar_image, kcaptcha, coolcaptcha, mewebstudio_captcha, simplecaptcha. |
| `scripts/generate_corpus.ps1` | Orchestrator: launches 4 parallel PHP workers (25k each → 100k total, ~30 min). See `logs/orchestrator.log`. |
| `scripts/build_labels_csv.py` | Walks shards → unified `train_corpus/labels.csv` (`filename,label`, 100k rows). |
| `scripts/scan_corrupt.py`, `quick_infer.py` | Corpus QA + quick inference sanity check. |
| `src/parseq/train.py` | PARSeq Lightning trainer (the ancestor of `model 1`/`model 2` trainers — reads `labels.csv`, 5% val split). |
| `configs/parseq.yaml` | 384d / 12-enc / 1-dec PARSeq, `max_label_length: 8` (corpus is len 4-8), img 32×128, 15 epochs, bs 128. Same architecture later reused in Model 1 v3. |
| `configs/parseq_smoke.yaml` | Tiny smoke config. |
| `third_party/parseq/` | Full upstream PARSeq repo (Bautista & Atienza, ECCV 2022) vendored as a git submodule-style checkout. This same vendored copy is the recognizer used throughout the thesis. |
| `train_corpus/` | The 100k-image generic corpus + `labels.csv`. Sharded (`shard_NNNNN/`). |
| `checkpoints/parseq/` | The failed run. Best ckpt `parseq-epoch=14-val_accuracy=1.080.ckpt`. |
| `logs/` | Orchestrator + 4 worker logs + parseq train log/err. |

## Relationship to the rest of the thesis

- **Predecessor to everything.** Model 0 (generic multi-library synth) → fails →
  Model 1 (`model 1/`) pivots to a **MyBB `captcha.php` distribution-matched
  renderer** for the specific CryptBB scheme and immediately works (39.94% real
  word PARSeq-alone). That pivot is *the* foundational insight of the thesis.
- The vendored `third_party/parseq/` here is the same PARSeq the whole thesis
  uses. The PHP-generator pattern (force a known label into an existing captcha
  library, sweep its knobs) is reused and refined in `model 1/src/php_synth/`.
- Not part of the M1/M1.5/M2/M2.5/M3 numbering — think of it as "Model −1": the
  thing that proved breadth-without-distribution-match doesn't work.

## Status

Parked / superseded. Kept as the negative-result baseline. No further work
planned. If ever cited in the thesis: "naive generic synthetic corpus → ~1%
word on synth val, never generalizes → motivates distribution matching."
