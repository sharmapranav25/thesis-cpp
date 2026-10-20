# Iteration-2 specialist run

Requested 2026-10-03 through `docs/ITERATION2_SPECIALISTS_HANDOFF.md`.

Completed 2026-10-04. Selected **intensity merge epoch 4**; the final model is `run_20261003_01/final/hybrid.pt` (SHA256 `198f89c05877a2e5053215ed53cc3e9be33d4a2a9dabc94f6e7842af415f6ccf`). All five presence specialists, eight presence merge epochs, three intensity specialists and four intensity merge epochs finished. Training settings stayed fixed.

The full report is [REPORT.md](C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/run_20261003_01/final/REPORT.md). On the same selection set, effect F1 .4883 → .5387, text F1 .5024 → .5457, CTC exact .4160 → .4465, length .8450 → .8825, font category .6925 → .7440 and font ordinal within ±1 .7785 → .7957. All seven font attribute metrics improve, but these include F1 and within-one scores rather than an all-attributes-exact accuracy. Held-out-font category improves only .5656 → .5724 on the reused generalisation benchmark. Of 107 applicable outputs, 94 improve and eight regress; regressions are preserved in the report/CSV. This compares the completed pipeline against the saved 15-epoch baseline; there is no equal-compute continued-simple control to isolate the benefit of specialization from extra training.

Whole-family inference is complete: CryptBB 600, City 2,000, DarkBay 999. There are 100 verified previews per family. Their appearance remains imperfect, especially CryptBB's short/narrow strings. Strict training readability passes City 77/100, CryptBB 68/100 and DarkBay 61/100. These are preview artifacts, not an approved solver-training corpus.

The trainer is `launch/specialists.py`. It reads the original training configuration and frozen epoch-15 EMA at `E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04`. New datasets/checkpoints/reports are stored here; the baseline is never edited.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\launch\specialists.py' --smoke --out 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\smoke_06'
& 'E:\pranav\model 1\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\launch\specialists.py' --out 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\run_20261003_01'
```

The second command also resumes an interrupted run. Completed phases are reused; partial phases restore raw/EMA weights, optimizer, schedule/EMA steps, draw/cursor state and all RNG states. Teacher caches are hash-checked on reuse. Do not reuse an output directory with changed training code or configuration.

## Proposed settings, fixed before full launch

- Every presence specialist starts from the frozen baseline, with a fresh optimizer/schedule/EMA. Four epochs, each **20,000 draws**, half targeted and half shared. Targeted tasks and measured values receive equal draw opportunities. Pools are saved indexes into the original synthetic training shards.
- AdamW, bf16 network execution, batch at most 128, the existing padded-pixel budget, 80% GPU allocator cap, clip 1, weight decay .05, warm-up EMA, cosine floor 1e-5. Presence peak 2e-4; presence merge 3e-4; intensity phases 1e-4. Warmup 100 steps, capped at 10% of planned phase steps.
- Presence merge: eight epochs, **100,000 unique training images per epoch**, all true-label losses and assigned-output distillation. Font tasks retain the baseline's group averaging. CTC and length remain supervised throughout. No averaging of weight dictionaries.
- Optional intensity stage: use selection to identify measured-positive ordinal heads below .75 within-one. At least 16 weak tasks triggers separately measured ordinal TAG groups, four specialist epochs of 20k draws, and four merge epochs of 100k draws. All presence tasks, CTC and length anchor intensity specialists. Unsupported measured levels are explicitly reported; no claims of balanced nonexistent levels.
- Checkpoint choice uses selection only. CTC exact and length may each decline at most .02; mean ordinal within-one at most .02; no more than 10% of ordinal heads may decline by more than .10. Assigned teachers that are weaker than their starting model are replaced by that starting model's output for those heads. The starting checkpoint is always eligible. The final intensity candidate must preserve presence mean within .01 and satisfy original-baseline preservation gates too.
- Presence thresholds maximize calibration F1 subject to FPR <= .10. Seen-test and generalisation are **reused benchmark sets**, evaluated after final selection. Real answers never enter training, grouping, calibration or selection. Real inference reads only family/index/source from the old source listing and keeps the three families separate.

## Smoke evidence and adjustments

`smoke_06/smoke_report.json` records the passing end-to-end check: masked/assigned gradients, exact baseline load, unknown-font masks, font level zero, two presence specialists, separate ordinal grouping and two intensity specialists, aligned teacher caches, both merges and save/resume. Interrupted/resumed versus uninterrupted raw weights have maximum absolute difference **0**. Smoke scoring uses distinct synthetic training rows; none of the benchmark sets is scored.

The proposed .1/.03/.01 normalized TAG steps overshot one task on the pretrained stem. Backtracking to .003 satisfied the unchanged >=90% positive self-affinity requirement in smoke. Full grouping repeats validation on full-training batches and records its own epsilon. Measurement uses FP32 with TF32 disabled, CNN stem parameters only, restores parameters exactly, and never steps the optimizer or EMA. Missing-task pairs are recorded through counts, with zero affinity only for unobserved pairs.

CUDA CTC backward is nondeterministic in the installed Torch build. CTC loss/backward runs on CPU while the model remains on CUDA; deterministic CUDA kernels and complete RNG restoration made the resume replay exact. Earlier failed smoke directories are retained as debugging evidence.

## Artifacts

`run_20261003_01/configuration.json` contains the actual configuration and selection rule. `data_manifest.json`, `task_manifest.json`, `label_audit.json`, and `sampler_indexes.npz` record source splits, image hashes, label coverage and pools. Each phase contains saved draw plans, raw/EMA checkpoints, resume state, per-head losses/counts, selection/steering reports and selected-checkpoint provenance. Group directories hold epsilon sweeps, affinity, cosine, counts and complete assignments. Teacher manifests identify every source checkpoint and aligned training row.

`final/hybrid.pt`, `final/REPORT.md`, per-head JSON/CSV, `merged_teacher_comparison.csv`, calibration and reused benchmark reports describe the selected hybrid. `benchmark_comparison.json` reads the saved historical baseline report rather than scoring it again. `real_inference`, `profiles`, and `dataset` contain 3,599 separate-family predictions and 100 fresh previews per family. Preview renderer usability and stronger training readability are reported separately. No final CAPTCHA solver was trained.

The original preview command produced 299 images and stopped after CryptBB row 25/prototype 415 exhausted six render attempts. `launch/repair_previews.py` retained the same prototype and continued fresh render seeds; it succeeded on zero-based attempt 8. Original failure reports/labels are retained alongside the exact retry seed. The trainer completion check now verifies complete counts, labels, uniqueness and saved image pixel hashes; it rejects a partial report even when the report file exists. This correction did not alter any training settings or model weights.

```powershell
& 'E:\pranav\model 1\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\launch\repair_previews.py' --dataset 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\run_20261003_01\dataset' --additional-attempts 24
& 'E:\pranav\model 1\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_specialists_work\launch\report.py'
```

The dataset sheets are `dataset/CryptBB_generated.jpg`, `dataset/DarkBay_generated.jpg` and `dataset/City_generated.jpg`. Full report artifacts live at `C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/run_20261003_01/final`.

Live status: `run_20261003_01/status.json`; detailed events: `events.jsonl`; console: `full_run.log`.
