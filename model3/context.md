# Model 3 — Adaptive Curriculum (No Teacher) Session Context

> ⚠ **STATUS CORRECTION (added 2026-06 exploration crawl).** The line below
> ("design agreed, no code written yet") is STALE. Model 3 was **fully
> implemented and run ~10 times** after this doc was written. Everything from
> here down is the original *design* doc; the actual *outcome* is summarized in
> this block. Read `setup.md` for the full version history (v0.1→v0.7, SPCL
> hybrid, the AR-loss bucketing fix, optimizer-reset fix).
>
> **OUTCOME: the adaptive curriculum did NOT work.** Best ever ~21.5% synth
> val_word (run 5, ep 2) then immediate catastrophic collapse; most runs stuck
> at ~1%; main run's best preserved ckpt is 15.02% (ep 11). **Never reached a
> real-eval number worth reporting; never approached M2's 54% bar.** Self-paced
> / student-defined curricula are *unstable* here — they spike then collapse,
> and backtracking + SPCL teacher-warmup + graduated ramp + Adam-state reset all
> failed to stabilize. This strengthens M2's finding: the teacher's stable,
> pre-computed difficulty signal is what makes the curriculum work.
>
> Implementation lives in `src/parseq/train.py` (`AdaptiveCurriculumSampler`,
> `PARSeqWithLossLog` with explicit AR forward pass for bucketing, Loss-EMA /
> Rebucket / Backtrack / SaturationStop callbacks). 7 numbered run dirs under
> `checkpoints/` + the main `parseq_adaptive/`. Config: `configs/parseq_adaptive.yaml`.
> A key untried future lever: decouple the renderer's `d` knob (bg-density vs
> char-distortion) so the adaptive signal acts on a less-entangled axis.

Last session: 2026-05-04. **Status: design agreed, no code written yet.**
**(↑ outdated — see STATUS CORRECTION above.)**

## TL;DR

- Model 3 is the third step in the M1 → M2 → M3 research arc. It isolates the **adaptation** mechanism: the curriculum is no longer fixed (M2) and no longer derived from an external teacher's per-sample scores. Instead, the *student itself* decides which images to train on next, based on its own running per-sample loss.
- Same 100k synth pool as M2 (`E:/pranav/model 2/data/model2_training_data/`). Same architecture (vanilla PARSeq). Same eval set (651 real CryptBB at `E:/pranav/model 1/data/eval/`). What changes is **scheduling** and **stopping**.
- Design summary: short training cycles (~15-20k images each), per-sample loss tracked as an EMA during training (free signal — no extra inference pass), images bucketed by loss quantile into trivial / learning-band / too-hard, next cycle samples 70/15/15 from those buckets. Validation drop > 1.5% triggers rollback + skip-too-hard for one cycle (backtracking). Training ends when val accuracy plateaus AND bucket membership stabilises (curriculum saturation). Hard cap 25 mini-epochs.
- Target: match or exceed M2's **54.22% real word (PARSeq alone) / 57.60% with M1.5 ensemble (M2.5)** to demonstrate adaptive scheduling without a teacher closes the gap to teacher-defined static curriculum.
- Framework's expected band for M3 is ~30-60%. The bar is "show adaptation works", not necessarily beating M2.

## Research Arc

### Model 1 — Random Synthetic
- All distortions mixed, no structure, no feedback, no teacher.
- Result: 39.94% real word / 84.28% real char (PARSeq alone, MyBB-fonts synth).
- Lives at `E:/pranav/model 1/`.

### Model 2 — Static Curriculum
- Predefined easy → hard schedule from teacher-defined buckets (M1.5 ensemble's per-sample NLL on the deployment distribution).
- 1/1/13 schedule, vanilla PARSeq, augment=on.
- Result: 54.22% real word PARSeq alone / 57.60% with M1.5 ensemble (M2.5).
- Lives at `E:/pranav/model 2/`.

### Model 3 — Adaptive Curriculum, No Teacher (current)
- Student-driven scheduling via running per-sample CE loss EMA.
- Backtracking on validation regression, curriculum-saturation stop criterion.
- ❌ No external teacher. ✔ Curriculum. ✔ Adaptation.
- Expected: 30-60% real word.

## Differences From Model 2

| | Model 2 v2 (static curriculum) | Model 3 (adaptive curriculum) |
|---|---|---|
| Bucket signal | M1.5 ensemble per-sample NLL (teacher) | Student's own training-loss EMA (no teacher) |
| Bucket boundaries | Fixed once at start (NLL tertiles) | Recomputed every mini-epoch (rolling quantile thresholds) |
| Schedule | 1/1/13 fixed (ep 0 easy / ep 1 easy+medium / ep 2-14 proportional) | Dynamic 70/15/15 from learning-band / trivial / too-hard, every cycle |
| Cycle size | 100k full epoch (~5-10 min) | ~15-20k mini-epoch (~1-2 min) |
| Stopping | 15-epoch fixed budget, best ckpt by val | Plateau ≥ 5-7 cycles AND bucket-partition stable < 5% movement; cap at 25 cycles |
| Recovery from bad cycle | None | Backtrack: revert ckpt + skip too-hard for one cycle if val drops > 1.5% |
| Loss signal acquisition | Standalone inference pass over pool | Harvested from training forward pass (free); inference only on stale tail |
| Augmentation | `augment: true` (mandatory) | `augment: true` (carry over — M2 showed this is non-negotiable) |
| Architecture | Vanilla PARSeq | Vanilla PARSeq (unchanged) |
| Training pool | 100k fixed-5 d=1.0 (`model2_training_data/`) | Same pool, reused as-is |

## Pipeline (Planned)

```
   data/model2_training_data/  (100k MyBB-fonts synth, d=1.0, length=5)
                  │
                  ▼
   Initial cycle 0: random 15k subset (no buckets yet)
                  │  (during training, log per-sample CE → init EMA)
                  ▼
   After each mini-epoch:
     1. Update per-sample loss EMA from this cycle's forward passes
        (samples not seen this cycle keep their last EMA)
     2. Re-bucket by loss quantile:
          trivial      = bottom ~25%
          learning band= middle ~50%
          too-hard     = top ~25%
     3. Sample next 15-20k for next cycle:
          70% learning band, 15% trivial, 15% too-hard
     4. Run val on real-eval-style synth val split → val_word
     5. If val_word drops > 1.5% from prior best:
          revert weights to last good ckpt
          drop too-hard slice for the next cycle (80/20/0)
     6. Check stopping conditions
                  │
                  ▼
   Stop when (a) val plateau ≥ 0.5% over 5-7 cycles
              AND (b) < 5% of samples cross bucket boundaries
              OR cycle count = 25
                  │
                  ▼
   case_decode + (optionally) M1.5 ensemble on data/eval/  →  M3 / M3.5 results
                  │
                  ▼
   Compare against M2 (54.22%) and M2.5 (57.60%) on the same 651 real captchas
```

## Methodology Constraints (carried over)

- **No human labels during training**. Strictly enforced.
- **The 651 real captchas at `E:/pranav/model 1/data/eval/` are eval-only.**
- **No teacher signal.** Per-sample loss must come from the *current student during training*, not from M1, M1.5, M2, or any external scorer. Inference passes over the pool by the in-training student are fine (it's the same model, mid-training — that's introspection, not a teacher).
- M2's `augment: true` lesson carries: small bucket subsets WILL overfit catastrophically without augmentation. Keep it on.

## Compute Environment

- Same machine as M1/M2 (Windows 11, RTX A4000, PyTorch 2.5.1+cu121).
- **Reuse `E:/pranav/model 1/.venv/Scripts/python.exe`**. No separate venv for M3 unless package conflicts arise.
- XAMPP PHP at `C:/xampp/php/php.exe` (only needed if we re-render — current plan does not).
- `os.add_dll_directory("C:/ProgramData/Anaconda3/Library/bin")` required for sqlite3 in any train/infer script.

## What's Planned (Implementation Backlog)

1. Copy/adapt M2's `src/parseq/train.py` → add `AdaptiveCurriculumSampler` (replaces `BucketCurriculumSampler`).
2. Add per-sample loss EMA hook on the train step (no extra forward pass).
3. Add re-bucketing callback that fires at the end of each mini-epoch.
4. Add backtracking callback (val drop → revert + restrict next cycle).
5. Add curriculum-saturation early-stop callback (plateau + partition stability).
6. Wire mini-epoch length via `limit_train_batches` or a custom sampler that yields ~15-20k indices per epoch.
7. Reuse `src/eval/case_decode.py` and ensemble code from M1.5 v2 for M3.5 evaluation.

## Sibling Projects (Cross-Reference)

- `E:/pranav/model 1/CONTEXT.md` — full M1 / M1.5 v2 writeup, calibrated selective-prediction analysis.
- `E:/pranav/model 2/CONTEXT.md` — M2 flagship recipe, what failed, the research-arc framing.
- `E:/pranav/model 2/src/parseq/train.py` — source training infra to fork/adapt for M3.
- `E:/pranav/model 1/data/eval/` — the 651 real eval captchas.
- `E:/pranav/model 2/data/model2_training_data/` — the 100k synth pool we'll reuse.

## Open Questions

- **Stale-tail strategy.** Samples not seen in the last K cycles have outdated EMA. Options: (a) periodic full inference pass every N cycles to refresh; (b) implicit refresh — if the trivial/too-hard tail isn't sampled, its EMA staleness doesn't matter much; (c) decay weight of stale samples in quantile computation. Lean (b) by default, add (a) every 5 cycles as cheap insurance.
- **Cold-start cycle.** First mini-epoch has no loss signal yet. Default: random 15k for cycle 0, then bucket from cycle 0's training losses for cycle 1 onward.
- **Partition-stability metric.** Currently defined as "fraction of samples crossing bucket boundary between cycles". Alternative: rank correlation (Spearman) on per-sample loss between consecutive bucketings. Pick whichever is simpler to compute first.
- **Whether to also adapt the renderer.** The framework spec mentions "generator knobs adjusted using performance feedback". Current plan keeps the pool fixed and only adapts sampling weights. If results plateau early, second phase: bring the renderer's `d` knob into the loop and re-render harder samples on demand.
