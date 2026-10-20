# Model 3: frozen curriculum-only protocol

Prepared 6 October 2026 before production training. Machine-readable protocol and input hashes: `runs/Darkbay_model3_adaptive_20261006/protocol.json`. Completion and scores must be read from saved receipts; this document does not claim completed training.

Reuse Model 2's existing `Darkbay_ds2_noFont-intel` LMDB and exact 95,000/5,000 split. No new data, font intelligence, controller, target answer, or target length is introduced. Seeds 48/123/456 independently initialize from scratch and run 20 epochs. All non-curriculum configuration is copied from the corresponding Model 2 YAML and checked for equality after excluding only output paths and curriculum metadata. Augmentation and OpenOCR training steps are reused unchanged. Batch128, 95,000 replacement draws, drop24, 742 optimizer updates per epoch, AdamW/OneCycleLR and BF16 match Model 2.

## Numeric rule fixed before training

1. Epoch1 uses the existing frozen Model1 teacher buckets and exactly the same sampling RNG and draws as Model2. Initialize smoothed difficulty from training-only teacher NLL ordinal percentiles `(rank+0.5)/95000`; break exact ties by image ID.
2. After each epoch, the current seed reader rescans all95,000 training images in evaluation mode, without augmentation or gradients, FP32 batch128. Use free autoregressive predictions and one refinement. Renormalize the62 character probabilities after excluding special tokens. Difficulty is summed synthetic-character `-log(p_true+1e-9)`, with negative numerical roundoff clamped to zero. Labels provide synthetic scoring positions only; test decoding remains unrestricted. Preserve all model buffers and global/model/augmentation RNG states.
3. Convert fresh NLL to training-only ordinal percentiles. Update `D = 0.5 * D_previous + 0.5 * new_percentile`. This removes teacher/student scale differences at the first update without using validation. Rerank by smoothed difficulty then ID and split into31,667/31,667/31,666 easy/medium/hard images.
4. For each image, bounded difficulty risk is `1-exp(-min(NLL,50))`. For each current bucket, average this risk. Propose bucket mass `0.2 + 0.4*(bucket_risk+1e-6)/(sum_bucket_risks+3e-6)`. Cap mass at0.5 and redistribute any excess equally among uncapped groups. Interpolate from previous mass toward the proposal so no mass changes by more than0.1. The first adaptive transition uses a uniform reference distribution instead of the initial easy-only mass, restoring all-pool coverage. Every subsequent group has at least20% and at most50%; no image is discarded.
5. Round mass times95,000 to integer bucket counts, place the residual in the largest-mass bucket, sample uniformly within each group with replacement, then shuffle. Use independent NumPy generator `seed + epoch*100003`, matching Model2's sampler stream construction. Record every drawn ID and both pre-drop and optimized bucket counts.

No threshold or curriculum decision is tuned against real DarkBay results. Validation is never supplied to the scorer or adaptive sampler. Epoch20 rescoring is diagnostic and does not add another training epoch.

## Selection, recovery and audit

Reuse Model2's5000 unaugmented validation images. Highest whole-string accuracy selects each seed; OpenOCR's `>=` rule selects the later epoch on an exact tie. Save all20 epoch checkpoints and the final checkpoint independently. Record actual mean training loss, validation NLL, accuracy, and separate training/validation/rescoring durations.

Each committed epoch checkpoint atomically stores model, optimizer, scheduler, BF16 scaler, sampler/adaptive state, Python/NumPy/PyTorch/CUDA RNG, decoder permutation RNG, and augmentation RNG. Archive the same checkpoint bytes with a receipt linking their SHA256 to the scoring weights and per-image adaptive NPZ. Resume only from a committed boundary, never reapply its difficulty update. A smoke test intentionally interrupts and resumes at that boundary. Source hashes are frozen after smoke repairs and before production.

Initial and per-epoch adaptive records retain image IDs, raw NLL, normalized and smoothed difficulty, bucket membership and next-epoch sampling probability. Validate finite values, probability sums and split disjointness; fail explicitly on inconsistent state.

## Testing and calibration

After all three seeds finish, hash-verify synthetic-selected checkpoints and infer on the existing900-image DarkBay manifest with its100-reference exclusion. Freeze all reader and whole-string majority ensemble predictions before reading scoring labels. With three different strings, choose the reader with greatest mean character confidence, retaining seed order48/123/456 for exact ties. Report exact case-sensitive accuracy and CER.

Infer the selected readers on the same5000 synthetic validation images. Reuse the exact prior2500 discovery/2500 calibration ID partitions. Learn Laplace-smoothed emitted-character reliabilities on discovery. Score a prediction by the product of character reliabilities times mean confidence support for the winning string. Choose maximum calibration coverage among thresholds with at least100 attempts and descriptive Wilson95 lower accuracy bound meeting each predeclared90/95/98/99% synthetic goal. Freeze policy, then real label-free attempt decisions, then score real labels. Report actual thresholds, correct/attempted, accuracy and shoot rate. Unavailable policies remain unavailable.

Model2 is the curriculum-only comparator. Model1 differs in corpus, augmentation, split and fixed epoch20 selection. The historical real set has been repeatedly examined. Reuse of checkpoint-selection validation for shoot calibration creates dependence, and the searched Wilson bounds are descriptive rather than post-selection guarantees. Equal epochs entail additional Model3 inference compute.

## Commands

Use the Python configured in `configs/local.json`:

```powershell
& $capriPython capri.py darkbay-ds3 --stage prepare
& $capriPython capri.py darkbay-ds3 --stage smoke --stop-after 1
& $capriPython capri.py darkbay-ds3 --stage smoke --resume
& $capriPython capri.py darkbay-ds3
```

The intentional first smoke stop raises `InterruptedError` after saving a complete boundary. Production orchestration runs all seeds sequentially, then real evaluation, synthetic inference, shoot-policy calibration, comparison report and completion verification. Existing baseline artifacts are read-only inputs. Production completion requires `status.json` stage `complete` and `completion_verification.json`; launch alone is not completion.

## User amendment: current checkpoint is epoch20

After reviewing seed48 real epoch results on6 October2026, the user designated the final epoch as the checkpoint for now. Model3 evaluation, ensembling and synthetic-calibrated shoot analysis therefore use each seed's immutable epoch20 checkpoint. Original synthetic-selected checkpoints and training receipts remain preserved. The training curriculum and20-epoch budget are unchanged. This evaluation-selection amendment follows real-test inspection and is recorded separately in `checkpoint_selection_override.json`; the earlier predeclared selection description above remains the historical protocol.
