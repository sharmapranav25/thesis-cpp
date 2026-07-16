# DATASETS — the captcha corpus (all schemes)

Every captcha directory available, with **which ones are safe to feed the GAN**
(unlabeled = source-free input) vs **which are eval-only** (labeled — training on
them is cheating). Machine-readable version: `configs/corpus.yaml`.

## ✅ SAFE unlabeled pools (use as the GAN's "real" domain)
| Scheme | Dir | Count | Size (W×H) | Chars (observed) | Scorable? |
|---|---|---|---|---|---|
| **CryptBB** | `CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023` | 600 | 200×60 | ~5, alphanumeric mixed-case | ✅ yes → eval below |
| **City** | `City (live)-20260503T045450Z-3-001/City (live)` | 2000 | 324×81 | ~5, digits 0-9 | ❌ no labeled eval (generate only) |
| **Courier Market** | `Courier Market (live)-20260503T045531Z-3-001/Courier Market (live)` | 2510 | 200×60 | ~4, digits 0-9 | ❌ no labeled eval (generate only) |

## ⛔ LABELED — EVAL ONLY (filenames ARE the labels; NEVER train / NEVER a GAN input)
| Set | Dir | Count | Size | Use |
|---|---|---|---|---|
| **CryptBB eval** | `model 1/data/eval` | 651 (+2× upscaled dupes, so sizes are 200×60 **and** 398×114) | filenames = 5-char labels | **the scoreboard** — score CryptBB here only |
| **DarkBay** | `DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled` | 1000 | 200×50 | filenames = 6-char labels | eval-only; ⚠ **NO clean unlabeled pool exists** → using it as a GAN input = contamination. DarkBay is blocked until disjoint unlabeled data is sourced. |

## 🚨 Critical guardrail — JoeyRetitling is NOT a scheme
`JoeyRetitling_extracted/JoeyRetitling` **and** `JoeyRetitling-20260502T012958Z-3-001/JoeyRetitling` are **byte-identical to the CryptBB eval set** (651/651 filenames match; filenames are the CryptBB labels). "Joey retitling" = someone renamed the CryptBB eval images to their labels.
- **Do NOT use JoeyRetitling as an unlabeled pool.** It is the CryptBB answer key.
- ⚠ Note: `auto_synth/data/joey_retitling*` and `auto_synth/data/cryptbb_unsolved*` were built treating these as two schemes — the joey_retitling artifacts are actually derived from **eval** data. Don't reuse joey_retitling artifacts for training.

## Which schemes can we actually *score*?
Only **CryptBB** has the clean setup: a disjoint unlabeled pool (600) **and** a
labeled eval (651). So CryptBB is the scoreboard. City & Courier can be
*generated* and visually checked (and used to prove generality of the pipeline),
but there's no labeled eval to score a recognizer on them. DarkBay is blocked.

## Practical implication for the build
- Develop + measure on **CryptBB** (train on 600 unlabeled → GAN → synth → PARSeq → eval on 651).
- Prove **generality** by running the *same code* on **City** and **Courier** (swap `real_dir` + canvas), judged visually / by FID since no labels exist.
- Skip **DarkBay** and **JoeyRetitling** as inputs (contamination / duplicate).
