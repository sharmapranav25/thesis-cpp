# Model 2 — Curriculum + Length-Aware Captcha Solver Session Context

Last session ended: 2026-05-04. **Pick up here.**

## TL;DR

- **🏆 Model 2.5 is the new flagship — 57.60% word / 89.04% char on 651 real CryptBB** with PARSeq curriculum + multi-scale CNN ensemble. **+8.44 pts word over Model 1.5 v2's 49.16%.** See `model_two_v2_curriculum_results.md`.
- **Recogniser**: `checkpoints/parseq_arc_m2_teacher_v2/parseq-epoch=03-val_accuracy=78.260.ckpt` (curriculum-trained PARSeq, peak val_word 78.26% on synth val — also crushes Model 1 v2's 57.26% peak).
- **PARSeq alone (no ensemble): 54.22% word / 87.88% char.**
- **Recipe that finally worked**: teacher-defined buckets (M1.5 v2's per-sample NLL on a fresh 100k synth pool) + 1/1/13 curriculum schedule + `augment: true`. Same data pool, smarter learning order.
- **Earlier negative result preserved for context**: every prior curriculum variant (d-knob buckets, scheduled sampling, fine-tune-from-converged) regressed or tied baseline. Documented in `model_two_v1_results.md`. The fix was teacher-defined buckets within the deployment distribution rather than human-defined difficulty across distributions.
- **Key diagnostic finding from this round**: `augment: false` + small-bucket curriculum = catastrophic overfit (val_word stuck at 1.30%). Same teacher-bucket setup with `augment: true` instantly recovered to 78.26%. Augmentation is non-negotiable when the curriculum's first stage is a small (~31k) sample subset shown with replacement.

## Research Arc

The overall plan progresses through a sequence of models, each isolating one mechanism. **Currently at Model 2.**

### Model 1 — Random Synthetic (No Curriculum, No Adaptation)
- Synthetic data with all distortions mixed
- No structure, no feedback
- Student: PARSeq
- ❌ No curriculum  ❌ No teacher
- **Purpose:** show naive synthetic fails
- **Expected:** ~10–30%

### Model 2 — Static Curriculum
- Predefined stages (easy → hard)
- Same schedule for all CAPTCHAs
- Student: PARSeq
- ✔ Curriculum  ❌ No adaptation  ❌ No teacher
- **Purpose:** show curriculum alone helps
- **Expected:** ~20–40%

### Model 3 — Adaptive Curriculum (No Teacher)
- Generator knobs adjusted using performance feedback
- Backtracking + guided search
- Student: PARSeq
- ✔ Curriculum  ✔ Adaptation  ❌ No teacher
- **Purpose:** show adaptation without external knowledge
- **Expected:** ~30–60%

## Differences From Model 1

(Current state — Model 2 v2, the working flagship recipe. The earlier Model 2 v1
approach with length head + variable length is documented in
`model_two_v1_results.md` as a negative result.)

| | Model 1 v2 (prior baseline) | Model 2 v2 (flagship) |
|---|---|---|
| Training pool | 100k fixed-5, all d=1.0 | 100k fixed-5, all d=1.0 (re-rendered, seed=1234) |
| Bucketing | none — shuffled | M1.5 v2 ensemble per-sample NLL → easy/medium/hard tertiles |
| Sampler | `shuffle=True` | `BucketCurriculumSampler` (per-epoch bucket weighting) |
| Schedule | n/a | 1/1/13 — ep 0 easy / ep 1 easy+medium / ep 2-14 proportional 0.15/0.25/0.60 |
| Augmentation | as set per run | **`augment: true`** (mandatory — `false` causes overfit collapse on small bucket) |
| Model | vanilla PARSeq | vanilla PARSeq (same architecture; *no* length head, *no* scheduled sampling) |
| Loss | char CE only | char CE only |
| `max_label_length` | 5 | 5 |
| Peak val_word | 57.26% (ep 11) | **78.26% (ep 3)** — ~4× faster to peak, +21 pts higher |
| Real-eval (PARSeq alone) | 39.94% word | **54.22% word** |
| Real-eval (+ M1.5 ensemble) | 49.16% word | **57.60% word / 89.04% char** ← Model 2.5 |

## What's In Each File

### Core source (modified vs Model 1)
- `src/php_synth/generate_mybb_captcha.php` — adds `--difficulty-mode` + `--length-mode` CLI flags. Plumbs `$d` through `draw_string_ttf`, `draw_dots`, and the background-pattern Bernoulli gate. Outputs 4-column `labels.csv` (`filename,label,difficulty,is_fixed_length`). The flagship run uses `--difficulty-mode fixed:1.0 --length-mode fixed:5` (so it behaves identically to Model 1's renderer; the variable-length plumbing was experimented with in Model 2 v1 and abandoned).
- `src/php_synth/merge_pools.py` — combines pools into one dir with bucket-prefixed filenames (`<bucket>_NNNNN.png`). Used to merge `synth_easy/medium/hard` into `synth_buckets` for the failed M2 v2 d-knob run.
- `src/parseq/train.py` — adds `CurriculumSampler`, `BucketCurriculumSampler`, `BucketCurriculumCallback`, `StageEndCheckpointCallback`, and (deprecated) `PARSeqWithLengthHead` / `PARSeqSchedSample`. `CaptchaDir` reads 2/4/5/6-column CSVs (legacy auto-padded to 5-tuple). `build_model()` defaults to vanilla PARSeq when length head + SS are disabled (the flagship setup).
- `src/parseq/infer.py` — emits per-position confidences and `conf_joint`. Length-head outputs are optional and default to none.
- `src/eval/score.py`, `src/eval/case_decode.py`, `src/eval/selective_with_ecps.py`, `src/eval/infer_with_conf.py` — scoring + multi-strategy decoding + selective-prediction analysis.

### Configs
- **`configs/parseq_arc_m2_teacher_v2.yaml`** — flagship config: 1/1/13 schedule, `augment: true`, `data_dir: data/model2_training_data`, vanilla PARSeq, 15 epochs, `ckpt_dir: checkpoints/parseq_arc_m2_teacher_v2`.
- `configs/parseq_arc_m2_teacher.yaml` — initial teacher attempt (3/3/9, `augment: false`). Failed.
- `configs/parseq_arc_m2.yaml` — d-knob bucket curriculum on `synth_buckets` (5/5/15 additive). Failed.
- `configs/parseq_bucket_curriculum.yaml`, `configs/parseq_m2_*.yaml` — earlier M2 v1 attempts (length head, scheduled sampling, fine-tune-from-converged). All failed; documented in `model_two_v1_results.md`.

### Data on disk
- **`data/model2_training_data/`** — 100k captchas (length=5, d=1.0, MyBB fonts, seed=1234). Active training set for Model 2 v2 flagship. Includes `labels.csv` (with `bucket` column from M1.5 NLL tertiles), `easy.csv` / `medium.csv` / `hard.csv` per-bucket subsets, `images/` subdir, and `labels_with_scores.csv` (raw NLL/P_correct/correct from M1.5 inference).
- `data/synth_buckets/` — 95k merged pool from `synth_easy` + `synth_medium` + `synth_hard` (d=0.2/0.5/1.0). Used by failed M2 v2 d-knob curriculum run; kept for reproducibility.
- `data/synth_fixed5/`, `data/synth_var37/`, `data/synth_curriculum/` — earlier M2 v1 pools (length-aware experiments). Stale; safe to delete to reclaim ~2 GB if needed.
- `data/eval/` — 651 real CryptBB captchas (same as Model 1). **Eval-only**.

### Checkpoints
- **`checkpoints/parseq_arc_m2_teacher_v2/parseq-epoch=03-val_accuracy=78.260.ckpt`** — Model 2 flagship recogniser. 78.26% val_word, 54.22% real word alone, 57.60% real word with M1.5 ensemble (Model 2.5).
- `checkpoints/parseq_arc_m2_teacher_v2/stage_end_*.ckpt` — stage-boundary snapshots (ep 0, 1, 14) for the flagship run.
- `checkpoints/parseq_arc_m2/` — failed M2 v2 with d-knob buckets (best 42.82% val, 10.45% real). Negative-result preserved.
- `checkpoints/parseq_arc_m2_teacher/` — failed first attempt of teacher curriculum (`augment: false`, 3/3/9 schedule). Best val 1.30% — degenerate overfit on small easy bucket.
- `checkpoints/parseq_runA/`, `parseq_m2_*` — earlier M2 v1 attempts (length head, scheduled sampling, etc.). All negative results documented in `model_two_v1_results.md`.

## Pipeline (Current — Model 2 v2 Flagship)

```
   generate_mybb_captcha.php
     --difficulty-mode fixed:1.0 --length-mode fixed:5     (seed 1234)
                  │
                  ▼
   data/model2_training_data/  (100k, all d=1.0, MyBB fonts)
                  │
                  ▼
   Model 1.5 v2 ensemble inference on every sample
     → per-sample (NLL, P_correct, correct)
                  │
                  ▼
   Bucket by NLL tertile (writes labels.csv with bucket column):
     easy   = bottom 33% NLL  (M1.5 nails these)
     medium = middle 33%
     hard   = top 33% NLL  (M1.5 struggles on these — 79.1% acc)
                  │
                  ▼
   PARSeq (vanilla, no length head, no SS) trained from scratch
   with BucketCurriculumSampler + augment: true:
     ep 0     : easy=1.0  medium=0.0  hard=0.0
     ep 1     : easy=1.0  medium=1.0  hard=0.0
     ep 2-14  : easy=0.15 medium=0.25 hard=0.60   (size-proportional)
                  │
                  ▼
   best ckpt by val_accuracy → epoch 3 @ 78.26% val_word
                  │
                  ▼
   case_decode.py with multi-scale char-CNN ensemble (α=0.50)
   on data/eval/  →  Model 2.5 predictions
                  │
                  ▼
   57.60% word / 89.04% char on 651 real CryptBB
```

Earlier curriculum pipelines (length-aware variable-length, d-knob buckets,
scheduled sampling, fine-tune-from-converged) are recorded in
`model_two_v1_results.md` for context but are no longer the active recipe.

## Methodology Constraints (carried over from Model 1)

- **No human labels during training**. Strictly enforced.
- **The 651 real captchas are eval-only**. Don't train on them.
- The PHP renderer at `d=1.0` matches CryptBB's actual server algorithm; lower-`d` captchas are an easier curriculum step, not a different distribution.

## Compute Environment

- Same machine as Model 1 (Windows 11, RTX A4000, PyTorch 2.5.1+cu121).
- **No `.venv/` in Model 2**. We're using Model 1's venv: `E:/pranav/model 1/.venv/Scripts/python.exe`. To bootstrap Model 2's own venv, run from Model 2: `py -m venv .venv && .venv/Scripts/pip install -r requirements.txt`.
- XAMPP PHP at `C:/xampp/php/php.exe`.
- `os.add_dll_directory("C:/ProgramData/Anaconda3/Library/bin")` is set in `train.py` and `infer.py` for sqlite3.

## Open Threads

- **Re-bucket with M2.5 itself** (recursive teacher bootstrapping). Current buckets came from M1.5 v2's NLL; using the new flagship M2.5's NLL as the teacher could yield a sharper curriculum signal.
- **Multiple seeds for the curriculum run.** Epoch 3 was a "lucky bounce" the model couldn't reproduce in 11 subsequent epochs (val_word swung 2.5% → 24.5% post-peak). Multiple seeds + ensembling across lucky-peak ckpts could bring further gains.
- **Move to Model 3** (adaptive curriculum). Static curriculum is now demonstrated to work; the next arc step is adaptive — use training loss to dynamically adjust generator knobs / bucket weights mid-run.
- **Case-confusion errors** still dominate top-15 confusions even at the new flagship (`p→P` 35, `y→Y` 30, `S→s` 26). Curriculum lifted overall accuracy but didn't structurally fix case ambiguity. Largest remaining single-axis improvement.

## Quick Commands

Render mixed pool from scratch:
```
C:/xampp/php/php.exe src/php_synth/generate_mybb_captcha.php \
    --out data/synth_fixed5 --n 50000 --seed 0 --difficulty-mode uniform --length-mode fixed:5
C:/xampp/php/php.exe src/php_synth/generate_mybb_captcha.php \
    --out data/synth_var37 --n 50000 --seed 1 --difficulty-mode uniform --length-mode range:3-7
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.php_synth.merge_pools \
    --src data/synth_fixed5 data/synth_var37 --out data/synth_curriculum
```

Train (Run A, curriculum on):
```
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.parseq.train --config configs/parseq.yaml > checkpoints/parseq_runA.log 2>&1
```

Infer + score:
```
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.parseq.infer \
    --ckpt "checkpoints/parseq_runA/parseq-epoch=NN-val_accuracy=NN.NNN.ckpt" \
    --src data/eval --out data/eval/predictions_runA.csv --batch 64
"E:/pranav/model 1/.venv/Scripts/python.exe" -m src.eval.score \
    --pred data/eval/predictions_runA.csv --labels data/eval/labels.csv --top 15
```

---

# Appendix — modules + docs not covered above (added 2026-06 crawl)

The body above documents the flagship (M2 v2 teacher-bucket curriculum). Two
source modules belong to the **abandoned M2 v1 variable-length routing track**
and aren't described above:

- `src/parseq/length_classifier.py` — standalone ResNet-18 binary classifier: `P(is_fixed_length)` from a 32×128 image. Trained on `synth_curriculum`'s `is_fixed_length` column. Config `configs/length_classifier.yaml`.
- `src/parseq/router.py` — end-to-end router: image → length classifier → dispatch to fixed-5 specialist (if `is_fixed_prob > 0.5`) vs a variable-length specialist (TBD/never built). Emits `predictions_router.csv`. This was the M2 v1 idea of *detecting* captcha length then routing to a specialist; **abandoned** once M2 v2 showed plain fixed-5 + teacher-bucket curriculum wins. Related configs: `parseq_fixed5_specialist.yaml`, `parseq_diag*.yaml`, `parseq_m2_*.yaml`, `parseq_runB/C.yaml`.

**Other docs in this dir (read these for depth):**
- `notes.md` — the richest working log. Per-epoch val curves for the *additive
  5/5/15 d-knob curriculum* (`parseq_arc_m2`), the augmentation-is-the-killer
  diagnosis, the "figure-ground separation is a distinct inductive bias"
  hypothesis (post-hard regression ep13→16 then recovery), and the implication
  for Model 3 (**decouple bg-density from char-distortion — the single `d` knob
  entangles them**).
- `TRAINING_INFRASTRUCTURE.md` — full build doc (mirrors model 1's).
- `model_two_v1_results.md` — all the M2 v1 negative results (length head,
  scheduled sampling, fine-tune-from-converged, d-knob buckets).
- `model_two_v2_curriculum_results.md` — the flagship M2.5 writeup (57.60% w).
- `analysis.md` — copy of the M1/1.5/2/2.5 selective-prediction + calibration analysis.

**The one structural lesson for Model 3** (from `notes.md`): the `d` knob
conflates background density and character distortion. They're separate
inductive biases (figure-ground vs glyph shape); ramp them independently.
