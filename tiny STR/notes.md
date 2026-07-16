# tiny STR — running notes

Live summary of the in-progress experiment. See `CONTEXT.md` for the full pitch.

## Goal

Test whether a minimal task-specific architecture can match PARSeq on CryptBB-style
captchas. If it can, the conclusion is that PARSeq's machinery (AR decoder,
permutation training, refinement iterations) is overkill for fixed-length random
alphanumeric text, and the win is the data, not the architecture.

## Architecture (TinySTR)

Resolved the four open design questions in `CONTEXT.md` with the doc's leans:

| Question | Decision |
|---|---|
| Positional encoding | Learnable 2D (one per patch position) |
| Decoder depth | 1 cross-attn layer, no self-attn between queries |
| Bbox aux head | None |
| Per-position head | Shared `Linear(D → 62)` |

Final model:
- Input: 60×200×3
- Patch embed: Conv2d(3 → 256), kernel/stride 4×8 → 15×25 = **375 patch tokens**
- 6× transformer encoder blocks (embed_dim 256, 8 heads, MLP ratio 4)
- 5 learnable position queries → 1× cross-attention layer
- Shared head Linear(256 → 62), greedy argmax decode
- **5.67M parameters** (CONTEXT.md target was 3-5M; encoder dominates at ~4.7M)

Files: `src/model.py`, `src/data.py`, `src/train.py`.

## Training setup

- **Data:** `E:\pranav\model 1\data\synth_mybb\` — 100k PHP-MyBB synthetic renders,
  200×60 RGB, labels in `labels.csv`
- **Split:** val_frac=0.05, seed=0 (identical to PARSeq's split → directly comparable)
- **Augmentation:** mirrored exactly from `model 1/src/parseq/train.py`
  (ColorJitter / blur / GaussNoise / ImageCompression / Affine / CoarseDropout)
- **Optim:** AdamW, lr=5e-4 cosine with 5% warmup, weight_decay=1e-4
- **Other:** batch 128, fp16 AMP, 15 epochs, grad clip 5.0
- **Compute:** RTX A4000 (17 GB), 95k train / 5k val, 742 steps/epoch, 11130 total

Hard constraints honored: no human labels, no real-eval contamination, no backbone
pretraining, no fine-tuning on real data.

## Training trajectory

| Epoch | train_loss | val_loss | val_char | val_word | char^5 prediction | epoch_time |
|---|---|---|---|---|---|---|
| 1 | 20.65 | 20.57 | 2.05% | 0.00% | — | 282s |
| 2 | 20.49 | 20.33 | 3.10% | 0.00% | — | 403s |
| 3 | 20.19 | 19.34 | 5.33% | 0.00% | — | 292s |
| 4 | 15.67 | 9.77 | 41.30% | 1.16% | 1.20% | 142s |
| 5 | 9.05 | 5.24 | 66.18% | 12.54% | 12.51% | 142s |
| 6 | 5.94 | 3.19 | 78.42% | 29.74% | 29.6% | 358s |
| 7 | 4.19 | 2.14 | 85.25% | 45.36% | 45.0% | 483s |

PARSeq peak (synth val, ep10): **92.5% char / 77.0% word**

## Two findings already visible

### 1. Classic ViT cold start, then phase transition

Epochs 1-3 looked broken — loss essentially flat at random baseline (5·ln(62) ≈ 20.65),
char acc creeping from 2% to 5%. Considered restarting with lower LR and curriculum
(no augmentation for 1-2 epochs). Decision rule: if char_acc < 5% **and** val_loss ≥ 19.5
by end of epoch 3, restart. Both barely failed at ep3 (5.33% / 19.34) — let it ride.

Epoch 4 → cliff edge: val_loss dropped 9.6 in a single epoch, char acc went 5.33% → 41.3%.
This is the textbook ViT phase transition — flat until features form, then sudden
takeoff. The accelerating loss-decrease rate (epoch-to-epoch) was the early signal
that the model was *about* to take off, not stuck.

### 2. Per-position errors are statistically independent (as designed)

The architecture has no AR coupling — each of the 5 position queries attends to the
encoder independently. Prediction: `word_acc ≈ char_acc^5`. Observed:

| Epoch | char^5 | word_acc | Δ |
|---|---|---|---|
| 4 | 1.20% | 1.16% | −0.04 |
| 5 | 12.51% | 12.54% | +0.03 |
| 6 | 29.6% | 29.74% | +0.14 |
| 7 | 45.0% | 45.36% | +0.36 |

Within ~0.4 pts every epoch. This is a *clean architectural signal* — confirms the
"independent heads, no AR coupling" design works exactly as intended.

It also implies the upper bound on word_acc, given any char_acc level. If TinySTR
plateaus at PARSeq's char acc (92.5%), word acc ceiling is `0.925^5 = 67.7%`,
leaving a structural ~9-pt gap to PARSeq's 77.0%. **That gap is exactly the value
PARSeq's AR decoder buys** — using char-N to disambiguate char-N+1.

## What "PARSeq vs TinySTR" actually compares

Two channels of "transferred knowledge" matter:

| Channel | PARSeq-Model1 | TinySTR |
|---|---|---|
| **Pretrained weights from STR datasets** | No — random init | No — random init |
| **Architecture designed/validated against STR** | Yes — published recipe shaped by years of OCR research | No — every choice (5 queries, no AR, no EOS, no length head, fixed 60×200) is hardcoded for CryptBB |

Even with random weights, training PARSeq is *fitting the trainable parameters of a
shape that's already known to work on text-in-images broadly*. The shape itself
encodes prior knowledge from a wider data universe. TinySTR has none of that — its
architecture cannot represent variable-length text, cannot make EOS errors, and
spends every degree of freedom on "where in this 200×60 image is char N out of 5."

Hence the cleaner reading at the end:

- TinySTR matches PARSeq → task-specific minimalism is at least as good as the
  general STR prior; the heavy machinery isn't load-bearing for narrow distributions
- TinySTR beats PARSeq on word acc → PARSeq's variable-length / AR baggage is
  actively miscalibrated for CryptBB and *introduces* errors
- TinySTR loses → the general STR architectural prior carries useful structure we
  lost by stripping things out

## Status as of writing

Mid-training. 8 epochs left. Char gains decelerating (12 → 7 pts/ep) — encoder
approaching capacity. Projecting:
- char acc plateau: 90-92%
- word acc plateau: 60-65% (from `char^5`)
- gap-to-PARSeq word: ~12-17 pts (the AR-coupling premium)

Real-eval (651 CryptBB captchas) comes after training finishes — that's where the
Model 1 baseline (31.18% word / 74.37% char greedy) lives.

## Files

| Path | Purpose |
|---|---|
| `CONTEXT.md` | Original design doc, hypothesis, hyperparam targets |
| `notes.md` | This file — running summary of what's been done + observed |
| `src/model.py` | TinySTR architecture |
| `src/data.py` | Dataset + augmentation (mirrors PARSeq's recipe) |
| `src/train.py` | Training loop with cosine LR, fp16 AMP |
| `train.log` | Live training output |
| `checkpoints/best.pt` | Best val_word_acc checkpoint |
| `checkpoints/last.pt` | Most recent epoch checkpoint |

---

# Important findings

Compiled at end of session 2026-05-02. The mid-training narrative above stops at
epoch 7; this section is the post-hoc summary.

## Final training trajectory (full 15 epochs, with one resume)

| Epoch | val_loss | val_char | val_word | char^5 prediction | Δ from prediction |
|---|---|---|---|---|---|
| 1 | 20.57 | 2.05% | 0.00% | — | — |
| 2 | 20.33 | 3.10% | 0.00% | — | — |
| 3 | 19.34 | 5.33% | 0.00% | — | — |
| 4 | 9.77 | 41.30% | 1.16% | 1.20% | −0.04 |
| 5 | 5.24 | 66.18% | 12.54% | 12.51% | +0.03 |
| 6 | 3.19 | 78.42% | 29.74% | 29.6% | +0.14 |
| 7 | 2.14 | 85.25% | 45.36% | 45.0% | +0.36 |
| 8 | 1.63 | 88.24% | 53.88% | 53.42% | +0.46 |
| 9 | 1.29 | **90.54%** | 60.90% | 60.93% | −0.03 |
| 10 | 1.11 | 91.54% | 64.10% | 64.27% | −0.17 |
| 11 | 0.95 | **92.57%** | 67.96% | 67.97% | −0.01 ← matched PARSeq char acc |
| 12 | 0.87 | 93.17% | 70.06% | 70.05% | +0.01 |
| 13 | 0.81 | 93.65% | 71.78% | 71.78% | 0.00 |
| 14 | 0.78 | 93.82% | 72.34% | 72.61% | −0.27 |
| 15 | 0.78 | **93.88%** | **72.58%** | 72.84% | −0.26 |

Independence held to within ±0.5 pts every epoch — a clean architectural signal.

## Real-eval headline numbers (651 CryptBB captchas)

| Setup | Char % | Word % | gap vs PARSeq alone | gap vs Model 1.5 |
|---|---|---|---|---|
| TinySTR alone (greedy) | 73.17 | 22.27 | −1.20 / −8.91 | −8.72 / −14.14 |
| TinySTR + char-CNN α=0.50 | 77.84 | 30.88 | +3.47 / −0.30 | −4.05 / −5.53 |
| **TinySTR + char-CNN + TTA (α=0.50)** | **77.90** | **31.03** | **+3.53 / −0.15** | **−3.99 / −5.38** |
| PARSeq alone | 74.37 | 31.18 | — | −7.52 / −5.23 |
| Model 1.5 (PARSeq + char-CNN) | 81.89 | 36.41 | +7.52 / +5.23 | — |

## Important findings

1. **Architecture is replaceable.** A 5.67M task-specific transformer matches
   PARSeq's 23.80M general STR architecture on synth val char acc (93.88 vs
   92.50). PARSeq's machinery (AR decoder, perm training, refinement) is not
   load-bearing for character recognition on fixed-length random captchas.

2. **`word_acc ≈ char_acc^5` to within ±0.5 pts every epoch.** Per-position
   errors are statistically independent — exactly what the architecture
   predicts (5 independent cross-attention heads, no AR coupling). PARSeq's
   word_acc is +9.3 pts above its `char^5` value because its AR decoder
   correlates errors. **This 9-pt word-acc gap is structural** and irreducible
   without changing the architecture.

3. **CycleGAN-translated synth is unlearnable from cold init.** v4 (identical
   architecture to v1, only data changed) sat at random baseline for 7 epochs.
   The translation produces realistic-looking captchas but with noise
   distributions that an untrained ViT can't separate from signal during cold
   start. Curriculum (pretrain on raw synth → fine-tune on translated) is the
   standard fix and is the most promising untried experiment.

4. **Char-CNN ensemble closes most of the recognizer-only gap to PARSeq.**
   TinySTR + Model 1's char-CNN at α=0.50 lands at 31.03% real-eval word acc
   — within 0.15 pts of PARSeq alone, with 30% fewer recognizer params and
   3.5 pts higher real-eval char acc. The ensemble's best α is lower for
   TinySTR (0.50) than for PARSeq (0.70) because TinySTR is the weaker
   recognizer.

5. **TTA's value is largely absorbed by the CNN ensemble.** TTA=6 gave +0.77
   word pts on TinySTR alone but only +0.15 on the ensemble. The CNN already
   provides the same kind of multi-view per-position signal TTA does.

6. **Synth → real generalization is the dominant residual.** TinySTR drops
   20.71 pts char acc going from synth val (93.88%) to real eval (73.17%).
   PARSeq drops 18.13 pts for comparable architecture-induced reasons. This
   gap is bigger than any single architecture/training knob we have — it's the
   real bottleneck.

7. **Confusion residue is heavily case-dominated.** Top errors after the
   ensemble: `p→P (34)`, `s→S (20)`, `v→V (19)`, `o→O (17)`, `W→w (16)`. Same
   pattern Model 1.5 reported. These are likely in the irreducible visual
   ambiguity tier — pixel-level case is genuinely ambiguous in this captcha
   distribution.

## Gaps / open questions

The following were planned but not executed (reasons in parens):

1. **Curriculum: v1 weights → fine-tune on translated.** (Project paused before
   running.) Highest expected upside on real-eval generalization. Would test
   whether v1's already-formed features can adapt to the harder distribution
   that v4 couldn't learn from cold.

2. **Cleaner pipeline.** Code complete in `src/cleaner.py` and
   `src/train_cleaner.py`; smoke-tested but never trained against a real
   recognizer. Trains on the 600 unlabeled CryptBB at
   `E:\pranav\CryptBBCaptchas(unsolved)07272023-...\` (deduped against eval).
   Uses entropy minimization through frozen recognizer + L1 residual penalty.
   Would validate whether recognizer-supervised cleaning provides a
   meaningful domain bridge for synth-trained models on real captchas.

3. **Bigger TinySTR with proper init.** v2 (17.95M) was killed when the new
   query-self-attn layer was identified as a likely co-factor in cold-start
   failure. Zero-initializing the self-attn output projection (so the
   layer starts as identity, like ReZero / LayerScale) would let it train
   from a clean start. Expected gain: +1-3 pts on synth val char acc, smaller
   on real eval.

4. **Ablation of the CNN ensemble's value source.** The +8.6 pts word-acc
   gain from adding the CNN comes from some mix of: (a) the CNN being more
   accurate per-position than TinySTR on real data, (b) the CNN providing
   error-correlation that TinySTR alone lacks, (c) the CNN's case-aware
   features specifically helping case errors. We didn't tease these apart.
   Would need a controlled run swapping in CNN softmaxes only on positions
   where TinySTR is uncertain (vs always).

5. **Beating Model 1.5.** The ~5.4 pt word gap to Model 1.5 is the
   AR-coupling premium. To close it would require either (a) re-introducing
   AR or query-coupling in TinySTR (but doing it well — see gap #3), or
   (b) closing the synth→real gap so TinySTR's char acc rises enough that
   `char^5` overtakes Model 1.5's word acc. Either is a real research
   project, not a quick experiment.

6. **No domain-specific cleaner experiments tried.** The cleaner code is
   recognizer-supervised (entropy min). A different supervision signal
   (e.g., adversarial loss matching real distribution, or denoising
   autoencoder on the 600 unlabeled real captchas with simulated extra
   noise) might be more effective. Untested.

7. **No exploration of why CycleGAN-translated data fails cold.** We
   diagnosed it empirically (v1 vs v4 controlled comparison) but didn't
   look at the actual pixel distributions to understand *why*. Possibly the
   translated images have higher visual entropy than either raw synth or
   real captchas — the worst of both worlds for cold-start learning.

## Files added in later phases (not in original plan)

| Path | Purpose |
|---|---|
| `src/eval_ensemble.py` | TinySTR + char-CNN log-prob mix + TTA, reuses Model 1's pre-trained char-CNN |
| `src/cleaner.py` | TinyCleaner U-Net (~0.76M params) + entropy/confidence losses |
| `src/train_cleaner.py` | Cleaner training script with SHA256 dedup against eval — written, smoke-tested, not yet run |
| `eval_predictions.csv` | Predictions from the standalone (no-CNN) eval run |
