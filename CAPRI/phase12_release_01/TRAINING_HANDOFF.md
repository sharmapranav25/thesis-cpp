# Next-session guide: Model1 PARSeq from generated data

## Scope and current state

This session prepared the generator only. Train PARSeq in the next session. Use Phase1+Phase2 release01; count inference and all Phase3experiments are excluded by user request. Do not restart the old continuous automation. Preserve imported/vendor code, old datasets and all historical experiments. The old Model1/1.5/2/2.5 code is background reference, not a required template. Old target-specific fonts/generators, teacher models, ensembles and checkpoints must not leak into the new experiment.

## Order per family

CryptBB -> City -> Courier -> DarkBay -> Outlined -> Roncuevas -> Pymlex -> Sloth. Prepare and train a separate fresh PARSeq model for each family, one at a time. Common source architecture is fine; previous family weights are not. No separate model per length/bucket: each family's one model covers all21cells.

1. Read README and verify release. Establish real style-fit vs evaluation manifests before generating. Pixel-hash audit cross-boundary duplicates, record all historical exposure. Existing profiles are convenient exposed development inputs, not untouched evaluation material. If appropriate new boundaries are needed, profile only the approved fit list.
2. Establish trustworthy evaluation labels without using them to train, select or calibrate. CryptBB has651historically used labels in the old Model1/2data/eval; this is exposed evaluation. Other families' labels must be audited. Screenshot filenames are not labels. No invented accuracy for unlabeled or preview-only sets.
3. Generate210-image preview for each chosen mode. Inspect all21cells, glyph visibility/retries and font coverage. Keep generic lengths2–8 and three content buckets for every family, including numeric-looking families. No real label-derived whitelist.
4. Generate approximately100k training images and separate synthetic validation/test sets. Balanced21cells. Baseline and diverse are separate experiment arms; equal seed/compute budgets if comparing. `diverse` is bounded augmentation of Phase1+Phase2, not a guarantee against overfitting. Use explicit different seeds and audit partitioned text, pixels, manifests and real reference overlap.
5. Build a plain PARSeq data loader for relative filename,label CSV; preserve case/leading zeros. RGB resize32x128 and normalization compatible with chosen PARSeq implementation. Never filter real labels silently to inflate scores. Original PARSeq Apache-2.0 architecture source is available under old model2/third_party/parseq, but a fresh audited upstream implementation is acceptable. Record source version/license/hash either way.
6. Train from random initialization. The proposed config in `parseq_proposed_config.json` passed a CUDA forward-only smoke test (RTXA4000,23,803,071parameters); no optimizer/data-loop test or training was executed. Starting hypothesis: AdamW lr3e-4, weightdecay.01, dropout.1, batch64, permutation training,15epoch budget, warmup10%, early stopping on synthetic validation with patience3after minimum3epochs. Verify framework optimizer semantics before claiming these exact settings. Record seeds, time, training/validation curves, selected checkpoint and planned ablations.
7. Do not force underfitting: aim to learn characters while resisting renderer shortcuts. Test earlier checkpoints, bounded augmentation, regularization and synthetic style-held-out validation. Choose checkpoints on synthetic validation only. A final real score cannot choose the best checkpoint/epoch without becoming model-selection data; freeze selection first.
8. Freeze model and evaluate each family's reserved labeled real split. Autoregressive EOS decoding, no force5, real length, character whitelist, target-specific teacher or test-label correction. Report exact string accuracy, CER via total edit distance/total ground-truth characters, case-sensitive results, supported-label coverage, sample counts, failures and exposure. Include per-length/bucket diagnostic only after freezing; never feed those labels back into training. Report accuracy unavailable where reliable labels do not exist.
9. Save model/config/manifest hashes, provenance, results and failure analysis in new numbered runs; update thesis papers with exact ideas implemented and deviations. Then move to next family. Do not claim all-family generalization from tiny Pymlex/Sloth previews.

## Why old headline results are not clean controls

Model1.5v2 used100kMyBB-font matched samples; Model2.5 used teacher curricula and forced-five/ensemble evaluation on651CryptBBexamples. Model3's self-paced approach was reported unstable. Those results are useful background but not equivalent to general-library synthesis with generic type/length priors. Reusing target-matched data or selecting from repeatedly evaluated real labels would invalidate this new comparison.

## Responsibilities still outstanding

No100kcorpus generated, no PARSeq training loop implemented/run, no comprehensive real-label audit and no new recognition results. Those are the next session's work. The deliverable here is a verified generation package, asset/profile provenance, manifest contract and guide. No claim that synthetic visual quality or recognition accuracy is solved.
