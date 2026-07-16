# Model 3 — Setup

How to bootstrap the Model 3 working directory and run the adaptive-curriculum training pipeline.

## Prerequisites

The following are already installed and working from the M1/M2 work — Model 3 reuses them rather than reinstalling.

| Component | Location / Version |
|---|---|
| Python venv | `E:/pranav/model 1/.venv/Scripts/python.exe` (Python 3.9.13, PyTorch 2.5.1+cu121) |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| OS | Windows 11 |
| PHP (renderer) | `C:/xampp/php/php.exe` (PHP 8.2.12) — only needed if we re-render synth |
| sqlite3 DLLs | `C:/ProgramData/Anaconda3/Library/bin` — must be added via `os.add_dll_directory` in any train/infer script |

If anything is missing, set up the M1 venv first per `E:/pranav/model 1/CONTEXT.md` § Compute Environment.

## Directory Layout (target)

```
E:/pranav/model3/
├── context.md                    ← project context (already created)
├── setup.md                      ← this file
├── requirements.txt              ← (TBD — likely `-r ../model 1/requirements.txt` or symlink)
├── configs/
│   └── parseq_adaptive.yaml      ← M3 training config
├── src/
│   ├── parseq/
│   │   ├── train.py              ← forked from `model 2/src/parseq/train.py`, adds adaptive sampler
│   │   └── infer.py              ← reused from M2 (or imported)
│   └── eval/
│       └── (reused from M1/M2)
├── checkpoints/
│   └── parseq_adaptive/          ← created at first train run
└── data/
    └── (none — we reuse model 2's pool and model 1's eval set)
```

We do NOT copy training data. We point the config at `E:/pranav/model 2/data/model2_training_data/` and the eval at `E:/pranav/model 1/data/eval/` directly.

## Step 1 — Bootstrap source tree

Fork M2's training infra. Run from `E:/pranav/model3/`:

```powershell
# Copy M2 sources we'll modify
New-Item -ItemType Directory -Force -Path src/parseq, src/eval, configs, checkpoints
Copy-Item "E:/pranav/model 2/src/parseq/train.py" src/parseq/train.py
Copy-Item "E:/pranav/model 2/src/parseq/infer.py" src/parseq/infer.py
Copy-Item "E:/pranav/model 2/src/eval/*.py" src/eval/
Copy-Item "E:/pranav/model 2/configs/parseq_arc_m2_teacher_v2.yaml" configs/parseq_adaptive.yaml
Copy-Item "E:/pranav/model 1/requirements.txt" requirements.txt
```

(Don't run yet — we'll only do this once we agree on the structure. For now this is the documented procedure.)

## Step 2 — Modify config

Edit `configs/parseq_adaptive.yaml`. The flagship recipe (v0.3) reproduces M2's
1/1/13 schedule but with **student-defined buckets** instead of teacher-defined
ones, so it tests whether the bucket-source change alone can match M2 holding
the schedule constant.

```yaml
data_dir: E:/pranav/model 2/data/model2_training_data
eval_dir: E:/pranav/model 1/data/eval
ckpt_dir: checkpoints/parseq_adaptive

# Architecture: vanilla PARSeq, same as M2
max_label_length: 5
augment: true                  # NON-NEGOTIABLE — carry over from M2's lesson

# Adaptive curriculum (v0.4 — full-epoch cycles)
curriculum:
  mode: adaptive
  cycle_size: 95000            # full training pool — matches M2's 742 batches/epoch
  buckets:
    trivial_quantile: 0.25     # bottom 25% AR-loss = trivial
    too_hard_quantile: 0.75    # top 25% AR-loss = too-hard
                                # middle 50% = learning band
  ema_alpha: 0.6               # per-sample loss EMA decay
  # SPCL hybrid (Jiang 2015 AAAI): teacher buckets warmup, then student
  # with a GRADUATED ramp toward heavy too_hard weighting.
  phases:
    - { until_cycle: 1, bucket_source: teacher, easy: 1.00, medium: 0.00, hard: 0.00 }   # cycle 0
    - { until_cycle: 3, bucket_source: teacher, easy: 0.50, medium: 0.50, hard: 0.00 }   # cycles 1-2
    - { until_cycle: 4, bucket_source: teacher, easy: 0.33, medium: 0.33, hard: 0.34 }   # cycle 3
    - { until_cycle: 7, bucket_source: student, trivial: 0.30, learning: 0.50, too_hard: 0.20 }   # cycles 4-6: gentle student
    - { until_cycle: 10, bucket_source: student, trivial: 0.20, learning: 0.40, too_hard: 0.40 }  # cycles 7-9: moderate
    - { bucket_source: student, trivial: 0.15, learning: 0.25, too_hard: 0.60 }                   # cycles 10+: mature

# Backtracking — thresholds in PERCENTAGE POINTS (val_accuracy is 0-100)
backtrack:
  val_drop_threshold: 1.5      # 1.5pp absolute val drop triggers rollback
  rollback_skip_too_hard: 1    # skip too-hard slice for N cycles after rollback
  max_consecutive_backtracks: 3

# Stopping criteria — min_delta also in percentage points
early_stop:
  plateau_window: 6            # cycles without ≥ min_delta gain
  min_delta: 0.5               # 0.5pp absolute val_word movement
  partition_stability_threshold: 0.05  # < 5% bucket boundary crossings
  max_cycles: 15               # hard cap
```

### Why the SPCL hybrid (v0.5)

Versions v0.1-v0.4 used student-bucketed sampling from cycle 0. They failed in textbook self-paced-learning ways:

- v0.1-v0.3 plateaued at 1.0-1.4% val_word — buckets at cold-start are noise (random-init model has no real "easy" signal)
- v0.4 (full-epoch, phased trivial-only warmup) peaked at 21.5% but the model overfit to 14,986 cold-start "trivial" samples and triggered 3 backtracks when fresh distribution was introduced

The literature has a name for this: SPL fails when the student is too weak to produce a meaningful difficulty signal. Standard fix: combine a predefined curriculum (instructor prior) with self-paced learning. Jiang et al, AAAI 2015, "Self-Paced Curriculum Learning" (SPCL) is the canonical citation.

**v0.5 SPCL hybrid:**

| Cycle | Source | Buckets used | Weights |
|---|---|---|---|
| 0 | teacher (M1.5 NLL tertiles from labels.csv) | easy / medium / hard | 1.00 / 0.00 / 0.00 |
| 1-2 | teacher | easy / medium / hard | 0.50 / 0.50 / 0.00 |
| 3 | teacher | easy / medium / hard | 0.33 / 0.33 / 0.34 |
| 4+ | **student** (per-sample AR loss EMA quantiles) | trivial / learning / too_hard | 0.15 / 0.25 / 0.60 |

During cycles 0-3 the student EMA is still updated on every batch (`LossEmaUpdateCallback` runs unconditionally). By cycle 4 the per-sample EMA has 4 cycles of meaningful content drawn from the deployment distribution, so the student-bucket handoff is well-conditioned instead of cold-started on noise.

This violates a strict "no teacher" reading of the M3 framework, but matches how M2 itself uses M1.5 as a teacher: the teacher only provides the static `bucket` column in labels.csv, and isn't running during the training loop. The adaptation axis (axis ✔ for M3) is preserved — cycles 4+ are fully student-driven.

### v0.6 fixes (2026-05-05): graduated student ramp + optimizer reset on backtrack

Run 6 (v0.5) hit a fragile 18.30% peak at cycle 5 then crashed to 2.78% at cycle 6 with three consecutive backtracks. Two structural causes:

1. **One-shot 15/25/60 at student handoff is too aggressive.** M2's mature `hard=0.60` worked because M2's hard bucket is M1.5's NLL-defined (a strong, stable signal). M3's student `too_hard` bucket is the still-warming student's own difficulty estimate, with bucket boundaries that drift each cycle. 60% weight on that target → unstable. Fix: graduated ramp 0.20 → 0.40 → 0.60 over 6 cycles so the student bucketing has time to stabilize.

2. **Backtrack restores model weights but not optimizer state.** Adam's running averages (exp_avg, exp_avg_sq) still reflected gradients from the regressing cycle, biasing the first post-backtrack steps away from the restored weights. Fix: `for opt in trainer.optimizers: opt.state.clear()` after `load_state_dict()` — Adam rebuilds momentum from fresh gradients on the restored weights.

## Step 3 — Add the adaptive sampler

Modify `src/parseq/train.py`:

1. Add an `AdaptiveCurriculumSampler` class. It maintains a `self.sample_loss_ema: dict[int, float]`, exposes `compute_buckets()` and `sample_indices(n)`.
2. Hook a per-sample loss callback into the training step that updates `sample_loss_ema` from each batch's per-sample CE.
3. Add `RebucketCallback(on_train_epoch_end)` that recomputes bucket boundaries from the current EMA and reseeds the sampler for the next cycle.
4. Add `BacktrackCallback(on_validation_end)` that compares current val_word to best-so-far and reverts weights if drop > threshold. **Do NOT call `pl_module.log()` from `on_validation_end`** — Lightning forbids it (raises `MisconfigurationException`). Use `print()` for diagnostics.
5. Add `SaturationStopCallback` checking the conjoined plateau + partition-stability conditions.
6. Each Lightning epoch corresponds to one mini-epoch — the custom sampler yields exactly `cycle_size // batch_size` batches per `__iter__`, so `len(train_loader) == cycle_size // batch_size` and `max_epochs` becomes the cycle cap.

### Per-sample loss must be from autoregressive decoding, not teacher forcing

PARSeq's stock `training_step` uses teacher forcing — the decoder receives ground-truth tokens at every position. Per-sample loss measured from teacher-forced logits is **not the right bucketing signal**: it can stay very low even when autoregressive decoding compounds errors and val_accuracy is stuck at 1%. A run that buckets samples by teacher-forced loss will see training EMA collapse from ~3.5 to ~0.1 with val_acc still under 2% (this happened in run 1/run 2 of M3 — bug fixed at v0.2).

The fix in `PARSeqWithLossLog.training_step`: after the standard teacher-forced loss is computed for backprop, run an explicit AR forward pass in eval mode + `torch.no_grad()`, compute per-sample CE on those logits, and stash that in `_last_per_sample_loss`. Cost: one extra forward pass per train step (~30-50% overhead). Worth it — the bucketing signal now matches what val_accuracy measures.

```python
with torch.no_grad():
    was_training = self.training
    self.eval()
    try:
        ar_logits = self(images)  # AR forward
    finally:
        if was_training:
            self.train()
    # ... compute per-sample CE on ar_logits vs ground truth ...
```

## Step 4 — Run training

```powershell
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.parseq.train `
    --config configs/parseq_adaptive.yaml > checkpoints/parseq_adaptive.log 2>&1
```

Expect ~1-2 min per mini-epoch. With 25-cycle cap that's ~25-50 min worst case before the saturation criterion fires; typical runs should stop earlier (10-15 cycles).

## Step 5 — Evaluate on real captchas

Once a `parseq-epoch=NN-val_accuracy=NN.NNN.ckpt` is written:

```powershell
# Inference + per-sample confidence
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.eval.infer_with_conf `
    --ckpt "checkpoints/parseq_adaptive/parseq-epoch=NN-val_accuracy=NN.NNN.ckpt" `
    --src "E:/pranav/model 1/data/eval" `
    --out data/eval/preds_model3.csv

# Score
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.eval.score `
    --pred data/eval/preds_model3.csv `
    --labels "E:/pranav/model 1/data/eval/labels.csv" --top 15

# Selective + ECPS
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.eval.selective_with_ecps `
    --pred data/eval/preds_model3.csv --name model3
```

## Step 6 — Optional ensemble (M3.5)

If we want to compare on the same footing as M2.5, combine M3's PARSeq output with M1.5 v2's detector + 3 char-CNN scales (already trained, lives in `E:/pranav/model 1/checkpoints/`). Use the existing α-mix code from M1.5 v2 with α searched on M2's val pool.

## Reference Numbers to Beat

| Model | Real word | Real char | Real word (with M1.5 ensemble) |
|---|---|---|---|
| M1 (PARSeq alone) | 39.94% | 84.28% | 49.16% (M1.5) |
| M2 (PARSeq alone) | 54.22% | 87.88% | 57.60% (M2.5) |
| **M3 (PARSeq alone)** | **TBD** (target ≥ 54%) | TBD | **TBD** (target ≥ 57%) |

Eval set is the same 651 real CryptBB captchas at `E:/pranav/model 1/data/eval/` for all three models.

## Common Pitfalls

- **`augment: false` will collapse training to ~1.30% val_word** as soon as the curriculum samples a small slice repeatedly. M2 documented this — don't repeat the experiment.
- **Per-sample EMA staleness.** If a sample isn't sampled for 5+ cycles its EMA value is from a much weaker model. Either refresh via a periodic full-pool inference pass (every 5 cycles, ~5 min) or down-weight stale samples in the quantile computation.
- **Validation set leakage.** Use a held-out *synth* val split for the in-loop val_word that drives backtracking and stopping. The 651 real captchas are eval-only — never used during training, even for stopping decisions.
- **Lightning best-checkpoint logic.** Default `ModelCheckpoint(monitor='val_accuracy')` saves the best ckpt over all cycles. Don't accidentally overwrite it during a backtrack — the backtrack should restore weights from the best ckpt, not save a worse one.
