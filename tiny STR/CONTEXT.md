# tiny STR — minimal captcha recognizer from scratch

> **Status:** v1 trained, evaluated, and parked. See `notes.md` for the running
> log of findings. Cleaner pipeline written but not run; curriculum/v2 attempts
> failed and are documented under "What was tried" below.

## Why this exists

`E:\pranav\model 1\` uses **PARSeq** (Bautista & Atienza, ECCV 2022) as its primary
recognizer, hitting **31.18% word / 74.37% char** on real CryptBB captchas
(lifted to **36.41% / 81.89%** with the Model 1.5 PARSeq + char-CNN ensemble).
PARSeq works really well, but it's a sophisticated model: ViT encoder +
autoregressive transformer decoder + permutation training + iterative refinement.
About 24M params, ~1500 lines in `third_party/parseq/strhub/models/parseq/`.

The hypothesis tested in this directory: **most of PARSeq's machinery is overkill
for our task.** Random alphanumeric captchas of fixed length (5 chars, no
language structure) shouldn't need an autoregressive decoder, permutation
training, or refinement iterations. A vanilla transformer encoder with N learnable
position queries should hit ~95% of PARSeq's accuracy at ~25% of the model size.

## What we built

A fixed-length-output transformer recognizer:

```
input image (200×60×3)
        │
        ▼
patch embed (Conv 4×8 stride 4×8)
        │     output: 25 wide × 15 tall = 375 patch tokens, each dim 256
        ▼
+ learnable 2D positional embedding (one per patch position)
        │
        ▼
encoder: 6× transformer self-attention layers (embed_dim 256, 8 heads)
        │     output: 375 × 256
        │
        │     ┌──── 5 learnable position queries (5 × 256)
        │     │
        ▼     ▼
cross-attention decoder (1 layer, no self-attn between queries)
        │     queries attend over the 375 encoder tokens
        │     output: 5 × 256
        ▼
shared linear head Linear(256 → 62)
        │
        ▼
output: 5 × 62 logits  (one independent softmax per position over 0-9A-Za-z)
```

**5.67M parameters total.** ~150 lines of model code (`src/model.py`).

The four design questions in the original draft were resolved with their leans:
learnable 2D positional encoding, 1 cross-attn layer (no self-attn between
queries), no bbox aux head, shared per-position head.

## Headline results

### Synth val (5k held-out from training set; identical split to PARSeq)

| Model | Params | Char % | Word % |
|---|---|---|---|
| PARSeq peak (ep10) | 23.80M | 92.50 | 77.00 |
| **TinySTR (ep15)** | **5.67M** | **93.88** | 72.58 |

TinySTR **beats PARSeq on synth val char acc** (+1.38 pts) with 4.2× fewer
parameters. The word gap (−4.42 pts) is exactly the predicted AR-coupling premium:
TinySTR's per-position errors are statistically independent (`word ≈ char^5`
to ±0.4 pts every epoch — see `notes.md`), while PARSeq's AR decoder correlates
errors so they cluster on the same captchas, boosting word acc by ~9 pts beyond
the independence formula.

### Real eval (651 hand-labeled CryptBB captchas)

| Setup | Recognizer params | Char % | Word % |
|---|---|---|---|
| TinySTR alone (greedy) | 5.67M | 73.17 | 22.27 |
| TinySTR + char-CNN (α=0.50) | 16.87M | 77.84 | 30.88 |
| **TinySTR + char-CNN + TTA=6 (α=0.50)** | **16.87M** | **77.90** | **31.03** |
| PARSeq alone (greedy) | 23.80M | 74.37 | 31.18 |
| Model 1.5 (PARSeq + char-CNN) | 35.00M | 81.89 | 36.41 |

**With Model 1's pre-trained char-CNN ensembled in, TinySTR matches PARSeq alone
to within 0.15 word pts** while using 30% fewer recognizer params and beating
PARSeq's char acc by 3.5 pts. Gap to Model 1.5 is the AR-coupling premium
(~5 pts word) — same magnitude observed on synth val.

## Methodology constraints (honored throughout)

- **No human labels in training the recognizer.** Trained only on
  `model 1/data/synth_mybb/labels.csv` (100k PHP-MyBB synthetic captchas).
- **The 651 real captchas at `model 1/data/eval/` are eval-only.** Never used for
  training or hyperparameter tuning of the recognizer.
- **No backbone pretraining.** Trained from scratch.
- **No real-data fine-tuning of the recognizer.** Same rule.
- **Same 5% synth-val split (seed=0)** as PARSeq, so synth-val numbers are
  directly comparable head-to-head.

## What was tried

| Run | Architecture | Data | Result |
|---|---|---|---|
| **v1 (kept)** | 5.67M, no self-attn | synth_mybb | Trained successfully → headline numbers above |
| v2 | 17.95M (384/8/+selfattn) | translated | Stuck at random baseline, killed at ep4 |
| v3 | 6.46M (256/6/+selfattn) | translated | Killed before completion (suspicion shifted to data) |
| v4 | 5.67M, no self-attn (= v1) | translated | **Stuck for 7 epochs at random baseline** |

**Diagnosis from v4:** the CycleGAN-translated synth set is *not learnable from
cold init* for this architecture. v4 has the exact same model as v1 — only the
data differs — and it failed to break through after 7 epochs while v1 hit 41% char
acc by ep4. The translation produces realistic-looking captchas (good!) but with
high enough noise/distortion that an untrained ViT can't separate signal from
noise during cold start. Fine-tuning v1's already-formed features on translated
data (curriculum) was not attempted but is the obvious next experiment.

**Ensemble** (TinySTR + Model 1's char-CNN) was attempted and produced the
real-eval headline numbers above. Sweep alpha=[0.30, 0.50, 0.70, 0.90]; best
α=0.50 (vs Model 1.5's α=0.70 — TinySTR is the weaker recognizer so the CNN gets
more weight).

**TTA** (6 spatially-aligned variants: identity, bright±, contrast±, blur)
yielded only +0.15 word pts on the ensemble (+0.77 on TinySTR alone). The CNN
already does similar multi-view averaging per-position, so TTA's value is
absorbed.

## What's parked for later

All code/checkpoints are in place; these were not run before parking the project.

1. **Curriculum fine-tuning.** Resume from `checkpoints/best.pt` (v1 ep15) and
   fine-tune on `data/translated/` for ~5-10 epochs. v1's features should adapt
   to the harder distribution where cold training failed.
2. **Cleaner pipeline.** `src/cleaner.py` (TinyCleaner U-Net, 0.764M params) +
   `src/train_cleaner.py` (entropy-min through frozen recognizer + L1 residual
   penalty) are written and smoke-tested. Trains on the 600 unlabeled CryptBB
   screenshots at `E:\pranav\CryptBBCaptchas(unsolved)07272023-...\` (must dedup
   against the eval set first — script already does this via SHA256). At inference
   the cleaner pre-processes real images so the synth-trained recognizer sees
   something closer to its training distribution.
3. **Bigger TinySTR done right.** v2 (17.95M) failed cold-start on translated
   data, but the same architecture on synth_mybb would likely train fine and may
   gain 1-2 pts. Or zero-init the query self-attn output projection (residual
   makes it identity at init) to make the architectural addition trainable.
4. **Generalization improvements.** The 20.7-pt synth→real char-acc drop dwarfs
   any architecture knob we have. Better-matched synth data, domain-targeted
   augmentation, more synth data, or unsupervised domain adaptation would
   compound with the current setup.

## Files

| Path | Purpose |
|---|---|
| `CONTEXT.md` | This file — current state of the project |
| `notes.md` | Running log of training / findings / gaps |
| `src/model.py` | TinySTR + optional query self-attn layer |
| `src/data.py` | Dataset + augmentation (mirrors PARSeq's recipe exactly) |
| `src/train.py` | Training loop, AdamW + cosine + fp16, with `--resume` and arch flags |
| `src/eval.py` | Greedy + optional TTA + optional `--cleaner` flag |
| `src/eval_ensemble.py` | TinySTR + char-CNN log-prob mix + TTA |
| `src/cleaner.py` | TinyCleaner U-Net + entropy/confidence losses |
| `src/train_cleaner.py` | Cleaner training (dedup + entropy-min + L1) — not yet run |
| `checkpoints/best.pt` | v1 ep15: 93.88% char / 72.58% word on synth val |
| `checkpoints/last.pt` | Same as best.pt (last and best coincided at ep15) |

## Connection to other directories

- `E:\pranav\model 1\` — PARSeq + Model 1.5 ensemble (current production line)
- `E:\pranav\model 1 Dino attention\` — failed track on diffusion synth + GlyphControl LoRA; not relevant here
- `E:\pranav\tiny STR\` — this directory — minimal from-scratch alternative to PARSeq

## Hyperparameter values used (final)

| Setting | Value |
|---|---|
| Image size | 60 × 200 |
| Patch size | 4 × 8 (→ 15 × 25 = 375 patches) |
| Embed dim | 256 |
| Encoder depth | 6 |
| Encoder heads | 8 |
| Decoder cross-attn layers | 1 |
| Position queries | 5 |
| Output classes | 62 (`0-9A-Za-z`) |
| Params | **5.67M** |
| Loss | sum over 5 positions of cross-entropy |
| Optim | AdamW, lr 5e-4 cosine, 5% warmup, weight_decay 1e-4 |
| Batch size | 128 |
| Epochs | 15 |
| Augmentation | albumentations (color/blur/affine/erasing) — same as PARSeq |
| Train time | ~75 min on RTX A4000 (with one resume) |
