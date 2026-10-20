# Model 3 handoff: adaptive curriculum for DarkBay

Prepared 6 October 2026 for a new session. This is a handoff, not a record of a completed Model 3 experiment. No Model 3 data generation or training has started.

## Objective

Develop Model 3 using an **adaptive curriculum**: training difficulty should respond to the reader's learning progress, rather than following Model 2's fixed epoch schedule. Preserve the pixel-only target-conditioning workflow and establish a reproducible protocol before launching training.

## Confirmed scope: curriculum is the only change

The user clarified that **Model 3 changes only the curriculum**. Keep Model 2's existing `Darkbay_ds2_noFont-intel` images, exact 95,000/5,000 training/validation split, three controllers, no-font-intelligence setup, reader architecture, augmentation, optimizer, initialization from scratch, seeds 48/123/456, 20-epoch budget and synthetic checkpoint-selection method. Reuse the frozen Model 1 teacher ranking for the initial difficulty baseline; adapt the curriculum thereafter using the training pool.

**Deferred to a later loop:** font intelligence, a separate validation-generation controller, and 100,000 optimization images plus 5,000 additional validation images. Do not implement these or regenerate the dataset for Model 3.

The user has now agreed to rescore the full training pool after each epoch using the current reader, smooth the difficulty estimates and use the updated ranking for the next epoch's sampling. The exact score formula, smoothing coefficient and sampling limits remain implementation choices to specify and freeze before training. Save Model 3 outputs separately and preserve Model 2 as the fixed-curriculum comparison.

## Read first

- [Complete DarkBay methodology through Model 2](C:/Users/pranav/Documents/ChatGPT/thesis/docs/DARKBAY_THROUGH_MODEL_2.md): primary record, including planned differences, epoch tables, loss histories, evidence links and limitations.
- [Workspace instructions](C:/Users/pranav/Documents/ChatGPT/thesis/AGENTS.md).
- [Code and model organization](C:/Users/pranav/Documents/ChatGPT/thesis/docs/CODE_AND_MODELS.md).
- [CAPRI workflow](C:/Users/pranav/Documents/ChatGPT/thesis/CAPRI_WORKFLOW.md).
- [Maintenance log](C:/Users/pranav/Documents/ChatGPT/thesis/docs/MAINTENANCE_LOG.md).
- [Machine paths](C:/Users/pranav/Documents/ChatGPT/thesis/configs/local.json) and [model registry](C:/Users/pranav/Documents/ChatGPT/thesis/models/registry.json).

Workspace: `C:/Users/pranav/Documents/ChatGPT/thesis`.

Some README passages predate completed reader testing. Use the complete methodology, saved protocols and receipts for actual results. Model 1/2 in this handoff mean the DarkBay reader experiments, not the older external repositories with those names.

## Current state: completed, no training to resume

Model 2 finished all three seeds and all 20 epochs. The training monitor `monitor-darkbay-ds2-curriculum-run` was paused after completion. Verify processes if necessary, but do not restart the completed DS2 job or reuse its output directory for Model 3.

All original models, recipes, datasets, prediction files and analyses are preserved. No font-intelligence module has yet been integrated into this pipeline. GLM-OCR was a separately requested external benchmark, not a CAPRI component.

## Starting pipeline

1. The selected **simple neural hybrid**, epoch 15 EMA, sees 100 DarkBay reference images as pixels. It provides feature predictions and initial renderer settings. It is not the specialist model or an XGBoost replacement. Its CTC branch is not decoded into target text.
2. Deterministic code resolves incompatible feature heads, clamps settings and retains raw predictions and changes.
3. Length six came from 72/100 hybrid votes under a 60% majority rule. It was not manually supplied from DarkBay labels. Current code stops without a majority; the proposed uniform-length fallback is not implemented.
4. Three controller arms—custom appearance, original DINO ViT-S/8, and their calibrated mixture—refine both text and background from pixels. They select settings using fresh synthetic strings, not the real target answers.
5. The 100 entries per arm give 300 recipe entries, with 179 distinct parameter signatures. They are not 300 proven independent families. All three arms stay in the rendering mixture.
6. Renderer-generated strings are the synthetic labels. Target strings, character boxes, textures and known renderer settings are not supplied to controllers.

Inputs: [100-reference manifest](C:/Users/pranav/Documents/ChatGPT/thesis/configs/targets/darkbay.json).

Recipes and candidate evidence: [entries](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/entries.json), [controller history](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/controller_history.json), [length decision](C:/Users/pranav/Documents/ChatGPT/thesis/capri_darkbay_workflow/length_decision.json).

Do not introduce VLMs into the new pipeline without a new explicit user instruction. Original DINO is active; SAM, DINOv3 and Meta-Sim2 experiments are not selected. The mixed arm reuses existing comparators and has no separate trained checkpoint.

## Model 1 baseline

Dataset: `Darkbay_ds1_noFont-intel`, 100,000 synthetic images from the frozen recipe mixture. All 100,000 were used for optimization; no separate training-time validation set. Three readers, seeds 48/123/456, trained from scratch for 20 epochs with no augmentation. Fixed epoch 20 selection.

Reader architecture: SVTRv2LNConvTwo33 encoder plus PARSeq decoder, 20,076,927 parameters. RGB 32×128 bicubic preprocessing, normalization 0.5/0.5, case-sensitive 62-character alphabet, autoregressive decoding with one refinement, no forced six-character output.

Batch 128, AdamW learning rate 0.00035, weight decay 0.05, OneCycleLR with two warmup epochs, BF16 training. Exact configurations are saved per seed.

Root: [Model 1 run](C:/Users/pranav/Documents/ChatGPT/thesis/runs/parseq_darkbay_100k_20261005).

## Model 2 baseline

Dataset: `Darkbay_ds2_noFont-intel`, 100,000 fresh renders from the **same** controller recipe bank, with a different seed base. Exact RGB audits found no DS1 or target overlap. This is fresh rendering, not a new independently fitted target domain.

The frozen three-seed Model 1 epoch 20 teacher scored known synthetic strings using NLL from averaged per-position character distributions. Tertiles produced easy 33,334 / medium 33,333 / hard 33,333 images.

The completed run used 95,000 training / 5,000 validation images. The user clarified afterward that training should have remained 100,000 with validation additional, but accepted this run as final. Never retrospectively claim Model 2 trained on 100,000 optimization images.

Fixed curriculum, sampling with replacement:

| Epoch | Easy | Medium | Hard |
|---|---:|---:|---:|
| 1 | 100% | 0% | 0% |
| 2 | 50% | 50% | 0% |
| 3–20 | 15% | 25% | 60% |

95,000 draws per epoch; final partial minibatch dropped. Training augmentation includes color jitter, blur, noise, compression, affine transforms and dropout. Validation is unaugmented. Same reader architecture and optimizer as Model 1; three seeds from scratch, not fine-tuned Model 1 weights.

All 20 checkpoints per seed are saved. Synthetic validation selects the best checkpoint independently: seed 48 epoch 18, seed 123 epoch 20, seed 456 epoch 18. Validation never updates weights. Real interim tests did not choose checkpoints or change the curriculum.

Root: [Model 2 run](C:/Users/pranav/Documents/ChatGPT/thesis/runs/Darkbay_ds2_noFont-intel).

## Results to compare against

Real evaluation uses the same 900 historical DarkBay images, excluding the 100 controller-reference RGB hashes from the 1,000-image source. It is repeatedly examined historical data, not an untouched benchmark.

| Reader | Model 1 correct /900 | Model 2 correct /900 |
|---|---:|---:|
| Seed 48 | 492 | 566 |
| Seed 123 | 439 | 520 |
| Seed 456 | 474 | 524 |
| Ensemble | 496 (55.11%) | 565 (62.78%) |

Ensembling is whole-string majority; if all disagree, use highest mean character confidence, with seed order for exact ties. Averaging independently trained Model 1 weights failed (0/900); do not confuse that with prediction ensembling.

Model 2's best observed interim seed-48 score was epoch 15 at 607/900, but it was not selected on that real score. The shared epoch-19 ensemble scored 567/900. Final Model 2 is the synthetic-selected 18/20/18 ensemble above.

### Corrected shoot-rate procedure

Learn emitted-character confusion risks on 2,500 synthetic validation images, calibrate thresholds on another 2,500, freeze the rule, compute and freeze real attempt decisions, then score using real labels. Never select a threshold by real accuracy.

Saved method: majority-string mean-confidence support multiplied by the product of Laplace-smoothed emitted-character reliabilities. Predeclared synthetic accuracy goals 90/95/98/99%; maximize calibration coverage with at least 100 attempts and a descriptive Wilson 95% lower bound meeting the goal. The bound is not a post-selection guarantee.

| Model | Score threshold, rounded | Real correct / attempts | Accuracy | Shoot rate |
|---|---:|---:|---:|---:|
| 1 | 0.2411 | 496/900 | 55.1% | 100% |
| 1 | 0.3003 | 494/884 | 55.9% | 98.2% |
| 1 | 0.6254 | 353/543 | 65.0% | 60.3% |
| 1 | 0.9197 | 232/339 | 68.4% | 37.7% |
| 2 | 0.2302 | 565/900 | 62.8% | 100% |
| 2 | 0.5865 | 535/772 | 69.3% | 85.8% |
| 2 | 0.8730 | 411/530 | 77.5% | 58.9% |

Use full precision from saved policies for actual decisions. The two looser Model 2 goals share a threshold. These scores differ from the earlier joint-confidence sweeps. Preserve that distinction.

Model 2 reused its checkpoint-selection validation set for policy calibration. Model 3 retains the same validation set and discovery/calibration partitions for comparability; disclose the same dependence. Do not add or replace data in this loop.

## Agreed adaptive training procedure

The image dataset stays fixed; difficulty rankings change as the reader learns. Do not replace the full rescore with a small probe or keep the original teacher ranking unchanged throughout training.

1. Initialize each seed's reader from scratch as in Model 2. Start epoch 1 from the existing Model 1 teacher ranking and initial easy-stage sampling. Do not use Model 2 weights or change the initial teacher.
2. Train an epoch using the current sampling distribution and the unchanged Model 2 augmentation and optimizer settings. Retain 95,000 replacement draws per epoch and the same partial-batch handling.
3. At the end of the epoch, use **that seed's current reader** to score all **95,000 training images** against their known synthetic labels. Use deterministic, unaugmented preprocessing, evaluation mode and no gradients. This pass must not update weights, optimizer, scheduler or normalization statistics. Restore training mode afterward.
4. Smooth new difficulty estimates with their previous values, then update the ranking/buckets and sampling distribution for the **next** epoch. An image can move from hard to easy as it becomes familiar. Maintain coverage across difficulty groups after initialization and cap persistent hard-example dominance; do not silently delete images or use real labels to decide which examples are learnable.
5. Evaluate the separate 5,000-image synthetic validation set each epoch for reporting and checkpoint selection. **Validation does not determine difficulty rankings or adaptive sampling.** Real images do not enter either process.
6. Save all 20 epoch checkpoints, synthetic validation results and adaptive state. Seed 48, 123 and 456 each maintain their own student-derived rankings; do not share one seed's ranking with the others.

After epoch 20, there is no next training epoch. A final full-pool scoring pass may be retained for the agreed per-epoch diagnostic history, but must not create an extra training epoch.

### Numeric choices to freeze before the run

Per-epoch full-pool rescoring is agreed. The difficulty formula, teacher-to-student score normalization, smoothing coefficient, bucket boundaries, minimum sampling mass and update bounds are not yet numerically specified. Define them in the saved protocol and smoke-test them before production training. NLL against known synthetic labels is a candidate consistent with Model 2; do not call it an already selected formula. If using an exponential moving average, specify its initialization and coefficient. Teacher ensemble and early-student scores can have very different scales, so explicitly define how the first update handles that transition.

Keep the initial easy-stage behavior and all non-curriculum settings matched to Model 2. Subsequent rankings/sampling are adaptive; do not silently revert to Model 2's fixed 50/50 and 15/25/60 schedule. Choose routine implementation details within this scope, record them, and never tune them on real DarkBay results.

### Reproducibility and resume state

Record per-image IDs, raw and smoothed difficulty, bucket membership, sampling probabilities, intended/actual bucket draws, scoring checkpoint hash and epoch. Preserve RNG state so the scoring pass does not accidentally alter augmentation or sampling randomness; use separate RNG streams or save/restore them. Check that no validation ID appears in the scoring or optimization pool. Verify finite scores and valid sampling probabilities before starting the next epoch; fail explicitly instead of silently falling back to a different curriculum.

Checkpoint optimizer/scheduler, sampler RNG and adaptive ranking state together. Resume must use the matching epoch state and must not repeat an update or advance the scheduler during rescoring. Measure training and rescoring time separately: equal epochs do not mean equal total compute.

## Model 3 validation, testing and analysis protocol

### Synthetic validation and checkpoint selection

Reuse the exact Model 2 5,000 validation IDs and deterministic preprocessing. No augmentation, gradient updates, adaptive ranking or sampler decisions use validation. Select each seed's checkpoint using the same highest synthetic whole-string accuracy rule and tie behavior as Model 2; verify the implementation rather than inventing a new tie rule. Retain final epoch 20 independently. Report selected epoch, validation accuracy and loss histories for all seeds. Do not select a checkpoint by real accuracy.

### Real DarkBay testing

Use the existing 900-image manifest at `runs/parseq_darkbay_100k_20261005/evaluation_darkbay/image_manifest.json`, with the same 100-reference exclusion, RGB hashes, preprocessing, charset and decoding. No target length forcing, case conversion or answer repair. Freeze each selected reader's predictions and the whole-string ensemble predictions before reading the real scoring labels. Report each seed and the ensemble as correct/900, exact accuracy and CER. Compare final synthetic-selected Model 3 with the final synthetic-selected Model 2 ensemble (565/900), and Model 1 (496/900), clearly stating selection differences.

For user-requested interim status, compare saved checkpoints at the same epoch and seed; use a shared-epoch ensemble when all three are available. These are diagnostics on an already examined historical set, not inputs to training, curriculum or selection. Do not promise an exhaustive per-epoch real sweep unless separately requested.

### Synthetic-derived shoot-rate analysis

After checkpoint selection, infer the selected readers on the 5,000 synthetic validation images and retain the exact Model 1/2 discovery/calibration partitions: 2,500 discovery and 2,500 calibration. Reuse the corrected policy procedure described above: learn character reliability on discovery, choose thresholds on calibration, freeze the policy, freeze real attempt decisions without labels, and only then score on real answers. Recompute Model 3's own reliability and thresholds; do not reuse Model 2's numeric thresholds as if calibrated for Model 3.

Report plain-reader/ensemble results first, then actual score threshold, correct/attempted, accuracy and shoot rate. Retain the 90/95/98/99% synthetic goals internally for like-for-like calibration; do not present them as promised real accuracy. Keep any confidence-only sweep explicitly separate. Disclose the reused checkpoint-selection validation set and historical real-test status.

### Completion deliverables

Save the frozen training protocol, unchanged-input hashes, all adaptive histories and epoch checkpoints, validation curves, selected/final checkpoint receipts, frozen real predictions, synthetic policy and real decision freezes, results tables and wall-clock/compute overhead. Include a Model 2 versus Model 3 comparison and a short methodology update. Higher real accuracy is an experimental question, not a completion requirement; report regressions honestly without changing the protocol to improve the test result.

## Deferred work: outside Model 3

Font intelligence, a dedicated validation controller and the 100k-training-plus-5k-validation allocation remain later-loop ideas. Do not implement, search for fonts, build a fourth controller or generate a replacement corpus as part of this curriculum-only experiment.

## Implementation locations

| Component | Workspace-relative path |
|---|---|
| Per-reference hybrid cleanup and controller search | src/capri_controller/pipeline.py |
| Three controller objectives | src/capri_controller/controllers.py |
| Comparators and inferred regions | src/capri_controller/features/ |
| Rendering and glyph controls | src/capri_controller/rendering/ |
| Corpus export and audit | src/capri_controller/corpus.py |
| DS2 orchestration and scoring | src/capri_controller/training/ds2.py |
| Curriculum sampler and augmentation | src/capri_controller/training/ds2_data.py |
| Final real evaluator | src/capri_controller/training/ds2_evaluate.py |
| Synthetic-derived policy | tools/synthetic_shoot_policy.py |
| DS2 policy inference | tools/ds2_synthetic_shoot_policy.py |

Reader Python: `E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe` (the folder name does not mean XGBoost is active). OpenOCR: `E:/pranav/Final sem/Models/level 1/upstream/OpenOCR`. Local package overlay: `.runtime/parseq_packages`. Renderer: `E:/pranav/Final sem/captcha-effects`. Use saved configuration rather than guessing dependencies or changing the global environment.

Keep new implementation under `src/capri_controller`, new experiments under a fresh `runs/` directory, and update the CLI where appropriate. Do not reuse the noFont-intel dataset name for data generated with font intelligence. Preserve DS1/DS2 and their checkpoints. Record exact protocol/configuration, data manifests, hashes and outputs before making performance claims.

## First actions for the receiving session

1. Read the methodology and completed DS2 sampler, configuration and split manifests.
2. Present a concise concrete adaptive-curriculum design, clearly marking the open update-rule choices. Ask only for decisions that materially affect the experiment and cannot be inferred from the user's instructions.
3. Implement the curriculum in new files/paths with split, sampler and training smoke checks before a long run.
4. Verify reuse of the exact Model 2 dataset and 95,000/5,000 split by hashes. Ensure no validation IDs enter optimization or adaptive training-pool scoring.
5. Train with a saved adaptive decision history and all epoch checkpoints. Use synthetic criteria for selection; real tests remain diagnostics.
6. Perform frozen-prediction real testing, prediction ensembling and synthetic-derived shoot analysis; compare with the recorded baselines and disclose changed factors.

Creating this handoff does not launch Model 3 or authorize messaging a new chat. The receiving session should follow the user's instruction there on when to start the production run.

## Communication preferences

The user prefers short updates and copy-paste result tables. For requested status, show current seed/epoch, latest completed checkpoint's real score and a clearly identified baseline, plus synthetic validation. Use ensemble results once corresponding checkpoints for all seeds exist; do not label one seed as an ensemble. Keep automated monitoring quiet during ordinary progress. Report accuracy as correct/attempted and shoot rate separately; the user prefers actual score thresholds to the ambiguous label “target.” Show requested images directly in chat using absolute paths because the user may be on a phone.
