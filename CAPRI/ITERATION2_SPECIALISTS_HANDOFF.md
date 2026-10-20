# Next session: train the iteration-2 hybrid with specialists

Written 2026-10-03. This document is the current handoff; older plans describe earlier experiments and contain superseded decisions.

## What the user wants

Improve the hybrid feature describer through specialist training, then merge the specialists into one hybrid. The describer examines a CAPTCHA family, predicts its visual/text/font features, and tells our Python renderer what synthetic training images to make. This is separate from training a final CAPTCHA solver.

The user initially chose a simple 100,000-image, 15-epoch baseline without specialists or adaptive curriculum. That baseline is finished. They now want the next session guided toward specialist training. This handoff does not launch training. The old 15-epoch instruction applied to the completed baseline; it is not automatically a budget for every specialist and merge phase.

Training labels must come from synthetic rendering. Real CAPTCHA answers must never enter training, grouping, calibration or model selection. Keep CryptBB, City and DarkBay separate throughout family inference and dataset generation. Do not average all 3,599 predictions into one family.

## Where to work

Project root: `E:\pranav\Final sem\sourcefree_synth`. Renderer: `E:\pranav\Final sem\captcha-effects`.

The chat workspace is `C:\Users\pranav\Documents\ChatGPT\thesis`. Starting there does not make project-relative paths work. Use quoted absolute paths or change to the project root. Project writes may need sandbox escalation; this is a tool permission boundary, not a reason to ask again about ordinary authorized edits.

Training Python: `E:\pranav\model 1\.venv\Scripts\python.exe`.
XGBoost Python: `E:\pranav\Final sem\sourcefree_synth\experiments_xgb\.venv\Scripts\python.exe` (Torch 2.5.1+cu121, XGBoost 2.1.4, CUDA verified).

Read these before implementation:

- `docs/ITERATION2_HANDOFF.md`, `docs/ITERATION2_PLAN.md`, `docs/ITERATION2_LOG.md`: background and baseline decisions. Older status statements such as “nothing generated or trained” are now stale.
- `docs/SPECIALISTS_V1.md`: historical specialist experiment, especially its decision log and final results. The opening table is the original plan, not the final execution.
- `feature_bank/iter2/model.py`, `data.py`, `losses.py`, `train.py`, `metrics.py`: current architecture and data/loss interfaces.
- `feature_bank/train/specialists.py`: old grouping, specialists and distillation machinery to adapt, not an iteration-2 trainer to run unchanged.
- `feature_bank/iter2/controller.py`, `mapping.py`: family aggregation and prediction-to-render mapping.

## Finished baseline and starting weights

Data/run root: `E:\pranav\Final sem\sourcefree_synth\runs\iter2_baseline_15_20261003_04`.

Use `training_config.yaml` in that root, not an assumed default configuration. Selected frozen baseline weights:

`baseline_training/epoch_15/ema.pt`

SHA256: `ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32`.

That file is a state dictionary. `epoch_15/raw.pt` is also available. Initialize the new experiment from the selected epoch-15 EMA weights; start fresh optimizer/scheduler/EMA state. Keep the original run and checkpoint unchanged. Baseline reports are under `baseline_training/final_report.json` and per-epoch `scores.json`.

Static corpus: train 100,000; steering 2,000; selection 2,000; calibration 2,000; seen_test 2,000; generalisation 2,500. Training is in ten 10,000-image shards under `iter2_h96/train_*`. Discover them through `data.open_split`, not hand-written shard assumptions. Held-out fonts stay held out when generating any new specialist samples.

The model shares a CNN stem, then separates into:

- CNN effect branch: `eff_block`, normalization, per-head pooling, 37 presence heads and 37 ordinal intensity heads.
- Transformer text branch: 16 presence heads, associated ordinal outputs, seven font outputs, categorical length 2–8, and a separate 63-class CTC character reader.

CTC learns to read a sequence; the length classifier predicts how many characters there are. They are different outputs. Specialist training should preserve both. No architecture expansion is required to start this experiment.

## What the latest XGBoost experiment actually showed

Completed output: `C:\Users\pranav\Documents\ChatGPT\thesis\iter2_xgb_01`.
Script: `C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_dataset_work\launch\iter2_xgb.py`.

All 107 boosted heads finished: 37 effect presence + 37 effect intensity; 16 text presence + 9 text intensity; 7 font outputs; length. Feature extraction took 291 seconds, fitting took 1,319 seconds. A two-class multiclass prediction shape error interrupted evaluation, was fixed using probability argmax, and evaluation resumed without retraining. The full pipeline subsequently completed and verified the baseline checkpoint hash unchanged.

Seen synthetic test results:

| Metric | Neural | XGBoost |
|---|---:|---:|
| Effect presence mean F1, equally calibrated | 0.5854 | 0.4667 |
| Text presence mean F1, equally calibrated | 0.5905 | 0.5907 |
| Effect intensity within ±1, true-positive-level rows | 0.8048 | 0.7383 |
| Length accuracy | 0.8495 | 0.8565 |
| Font category accuracy | 0.6870 | 0.7010 |
| Font ordinal within ±1 mean | 0.7870 | 0.6201 |

Presence thresholds were calibrated on calibration to maximize F1 subject to FPR ≤ 0.10. In `synthetic_results.json`, the neural summary F1 uses its original 0.5 threshold; the fair calibrated neural numbers above are means of each head's `calibrated_f1`. Do not compare the summary 0.4926 to calibrated 0.5854 as if they were different models.

Weak text heads remain weak. Calibrated neural → XGB F1: multicolor 0.1780 → 0.2296; font mixture 0.3520 → 0.3759; rotation 0.4946 → 0.4601; jitter 0.3267 → 0.3256; size variation 0.2558 → 0.2159. Exact names include `multicolor_text` and `font_mixture`.

Earlier XGBoost gained 0.6484 → 0.8348 on a different specialist-trained hybrid, different data and 45 scored heads. It tried both CNN and transformer vectors for each head, then chose a winner on selection; CNN won most heads, including many text layout features. This run restricted effects to shared-stem CNN vectors and text/font to transformer vectors, used an 800-tree cap instead of 2,000, and did not choose between routes. Current effect XGB bypassed the learned effect block. Therefore this experiment is not a faithful repetition, and its failure does not prove the baseline contains no feature information.

Optional diagnostic before expensive training: compare shared-stem CNN pooling, the processed effect vectors, and transformer vectors on the same selection set. Record route selection without touching test sets again. Do not turn this optional check into an endless prerequisite or assume it replaces the requested specialist experiment.

## Specialist plan for the next session

Implement a new iteration-2 specialist trainer/configuration. Port useful routines from the old trainer, while using `Iter2Model`, iteration-2 shards, current head names, CTC and font labels. The old trainer's slot/coverage losses, library pools and 46-head assumptions do not match this model. Do not run the old phase commands and call that iteration-2 specialist training.

Use the completed baseline as the shared start. No need to repeat its 15 epochs. Keep pools static and use fixed loss weights; no adaptive curriculum.

1. **Audit labels and measure task grouping.** Count actual positives/negatives and positive intensity levels for every head in the 100k training set. “Evenly allocated generation targets” does not mean every measured label has 50% prevalence. Define a task manifest for current presence, length and font outputs. Measure TAG on synthetic training batches, never real data. Use the shared stem for comparisons across branches; branch-local diagnostics can supplement this. Keep CTC as a preservation task rather than forcing it into an effect group. Select roughly 4–7 practical groups using measured affinity and document head coverage.
2. **Train presence specialists.** Each is a fresh copy of epoch-15 EMA. Train its assigned presence/categorical tasks, with 50% static targeted sampling and 50% shared training examples. Initially reuse the existing corpus with label-based sampling; render additional samples only where coverage is inadequate. Balance positive/negative cases per assigned feature, with diverse other features, fonts and text. Do not make the feature perfectly correlated with one font/background. Group assignment and sampler indexes must be saved.
3. **Merge presence specialists into one student.** Initialize from epoch-15 EMA. Use all synthetic ground-truth tasks plus distillation only from each teacher's assigned outputs. Keep baseline ordinal supervision active as an anchor so the already useful intensity predictions are not erased. Preserve CTC and length through synthetic ground truth and, where helpful, baseline teacher outputs on shared examples. Precompute teacher outputs only for training images; key them by image ID/hash and record teacher checkpoints.
4. **Train intensity specialists if needed.** Regroup ordinal tasks using their own masked ordinal losses; presence affinity is not evidence of intensity affinity. Target positive examples with balanced measured levels 1–10, especially weak text geometry/font properties. Initialize from the selected presence-merged student and retain presence/CTC/length anchors. Merge their ordinal predictions back into that student with true-label loss and ordinal distillation. Do not replace a good baseline intensity head with a weaker specialist merely because the phase finished.
5. **Choose and evaluate the final hybrid.** Select checkpoints using selection only, with per-head presence/intensity and explicit CTC/length preservation checks. Calibrate on calibration only. Write the selection rule before the run. Score seen_test and generalisation once after final selection; both sets have already been observed in earlier experiments, so call them reused benchmark sets, not untouched validation evidence.

Starting defaults for the new configuration, to document as proposed rather than historical facts: presence specialists 4 epochs each; presence merge 8 epochs; optional intensity specialists 4 epochs each and intensity merge 4 epochs. Define an epoch's image count explicitly (suggest 20k draws for each specialist; state the merge sample count separately). Four short specialist epochs are not four passes through all 100k images. If merge needs extension, judge on steering/selection and log it; do not promise a training time before measuring smoke-run throughput and teacher-cache time.

Start with AdamW, bf16, batch ≤128 with the existing padded-pixel budget, 80% GPU-memory cap, gradient clip 1, weight decay 0.05, warm-up EMA. Suggested fine-tuning peaks: specialists 2e-4; merge 3e-4; intensity 1e-4, cosine floor 1e-5. Use warmup around 100 steps, capped to a small fraction of actual phase steps. These are starting settings, not measured optimal values.

## Implementation details that must be right

- Mask unassigned head losses before averaging. An all-head mean with most heads zero silently shrinks specialist gradients. Null labels (-1) are unknown, not absent.
- Train the assigned branch and shared stem. Freezing only a classifier does not protect CTC from shared-stem changes; preserve reading explicitly and evaluate it every epoch. Unassigned outputs must not receive accidental specialist target gradients.
- Effect/text ordinal labels train only where level >0; presence handles absence. Font ordinal zero is a valid value. Report per-head counts as well as losses.
- Current `losses.compute` computes all tasks. It needs a proper task-mask/preservation interface for specialists; summing selected scalar groups alone is insufficient for selecting heads inside groups.
- Warm up EMA with `min(decay, (1+t)/(10+t))`. Initialize it from the loaded specialist/student weights. In the old short phases, fixed decay 0.999 left EMA close to initialization.
- TAG lookahead must restore parameters exactly, ignore missing-label tasks, and not update optimizer/EMA. Old epsilon 0.5 overshot; sweep 0.1, 0.03, 0.01 and require positive self-affinity for at least 90% of eligible tasks. If no epsilon passes, fix the measurement instead of clustering noise.
- Use old group-size repair fixes, not the initial broken rebalancing. Save affinity, cosine, counts, chosen epsilon, grouping objective and groups.
- Do not average specialist weight dictionaries to merge. Train a student with ground-truth supervision and assigned-teacher distillation.
- Preserve checkpoint hashes, split manifests and no-leak assertions. Give every phase a resume checkpoint containing raw/EMA weights, optimizer, scheduler step, EMA step and sampler state. Check interrupted/resumed execution in the smoke run.

## Historical lessons to carry forward

Earlier specialists broke several “always absent” predictions, but merge needed longer than its original four epochs (four plus eight extension epochs). The two-epoch intensity phase failed; its existence is not evidence that two epochs suffice. Windows can spill VRAM into RAM and become very slow without a clear OOM. Keep pixel-budgeted batches and the memory cap, including grouping and teacher inference. Historical supervisor authority was scoped to that old overnight run; do not treat its document as permission for unrelated deployment or messaging.

## Acceptance checks and reporting

Before full launch, run a small end-to-end smoke: load baseline exactly; verify assigned gradients and label masks; measure valid TAG; train two small specialist groups; generate aligned teacher cache; merge; save/resume; evaluate. Use training subsets or dedicated smoke data, not test-set tuning.

Every epoch write raw and EMA checkpoints; per-feature presence/ordinal losses and label counts; CTC/length/font losses; steering and selection per-head F1, recall/FPR and positive-level within ±1. Compare each specialist against baseline on the same assigned heads and each merged head against its teacher. Overall loss alone is not success. Report regressions as well as improvements.

At the end, infer the complete real sets separately: CryptBB 600, City 2,000, DarkBay 999. The existing source list is in `C:\Users\pranav\Documents\ChatGPT\thesis\iter2_full_target_recreation_01\predictions.jsonl`; use source paths, not its old predictions as training targets. Run the current controller/mapping to produce 100 previews per family. Show the sheets in chat with absolute image paths and assess family resemblance, not just renderer usability.

Current XGB previews are in `iter2_xgb_01/dataset` under the chat workspace. All 300 rendered, but strict training readability passed City 73/100, CryptBB 96/100, DarkBay 67/100. They remain visibly diverse in fonts/backgrounds and are not an approved 100k solver-training corpus. DarkBay requested blocks in 83 previews but measured them in 51; mapping/render interactions remain an additional issue. Better feature training does not automatically fix uncontrolled colors, placement or combinations. Keep model prediction errors and renderer mapping errors separate in the final assessment.

Deliver a final hybrid checkpoint, reproducible configuration/commands, data and teacher manifests, per-head comparison report, all three family previews, and a concise updated log. Do not launch final CAPTCHA-solver training as part of this task. Be direct in status replies; measure elapsed time and remaining work rather than promising “quick” or “40 minutes.”
