# Maintenance log

## 2026-10-10 — Bounded smaller-hybrid controller previews

User authorized first6 per scheme from five existing100-image sets, using all100 references. Added `controllers --preview`, preview-smaller-hybrid and review-smaller-hybrid, plus orchestration/review modules under src/capri_controller. Preview-only no-majority fallback follows the previously documented equal-observed-length rule with seeded shuffled cycles; all3 comparators/objectives/candidate searches unchanged. Standard mode still stops without60/100 agreement; raw predictions are now preserved before that stop. Corpus generation refuses preview artifacts. Saved-histogram and59/60 boundary checks passed; rejection of preview corpus export verified. Documentation and single handoff updated; font appendix preserved.

DarkBay18 outputs reused after exact-input/all-checkpoint verification; City18,CryptBB18,Outlined18 newly selected. SimpleCaptcha failed on input6; recovered15 saved choices by deterministic replay and original3-seed validity checks. No new search, dropped target, forced real length, VLM or training. Final review on E under decoder diagnostics/smaller_hybrid_controllers_20261010/review shows87 outputs and one explicit failed row. Only40/87 pass strict readability; SimpleCaptcha0/15 and Outlined1/18. No fidelity/accuracy improvement claim: the gallery is real-versus-controller-generated, not hybrid-before-versus-controller-after. Audited recorded source hashes, finite saved scores, all100 model count agreement and output/input pixel nonidentity. Checkpoints and original artifacts remain intact; training stays stopped. Workspace inventory refreshed separately.

## 2026-10-08 — Post-freeze scheme count scoring

Added score-decoder-schemes command and decoder_scheme_scoring.py, with City annotation location in configs/local.json. Responded to request for every count accuracy using existing frozen predictions only. CryptBB62/100; SimpleCaptcha17/100 from publisher's exactly-five-character specification, saved/hash-pinned. City original-file SHA256 matched51 annotations,33correct=64.71%; conditional five-character scoring across100 gives63%, explicitly not fully per-image verified (49unmatched). Prior DarkBay831/900 retained. Source annotation hashes, matched/unmatched IDs and unchanged prediction hash are saved in count_scoring.json. Updated the single handoff and preview report; no inference, training, model selection or rendering changes. Fixed-length results do not demonstrate variable-length generalization.

## 2026-10-08 17:21 UTC — 100-per-scheme CAPRI tests and genuine procedural previews

Added test-decoder-schemes and render-decoder-schemes commands, three modules outside frozen decoder code, and machine dataset paths in configs/local.json. User requested CryptBB/City/SimpleCaptcha tests and renderer examples, limited to100 images each. Fixed final epoch8 EMA, pixel-hash-ranked selection, unchanged model/gallery/thresholds; all300 predictions frozen before CryptBB labels. CryptBB exact count62/100, compared with100/100 posthoc constant-five baseline. City/SimpleCaptcha unscored without matched verified annotations. Font/localization accuracy unmeasured; no reader or VLM.

Generated24 fresh random-text previews using predicted count/boxes/fonts/colours/effects; first4 inputs per set,2fixed seeds. Preserved all outputs;15/24 pass renderer acceptance,9fail contrast/visibility/clipping criteria. Adapter defaults and inverse-render limitations recorded, no controller fitting or authenticity claim. Final report is under E-backed decoder diagnostics/real_three_schemes_100_epoch8_20261008_v2/preview_verified/report.html.

Validation: independent300-image hash/finite/output audit, checkpoint/source checks and score reproduction passed. Preserved initial failure evidence; corrected Windows venv-wrapper duplicate detection, <=1e-6 resize roundoff clamping (observed2.384e-7 overshoot), and NumPy-integer preview JSON serialization. Corrected24PNG exports and recipes exactly match pre-label originals. Frozen training sources unchanged; training remains stopped and heartbeat paused. Updated the single handoff, preserving its font appendix; workspace inventory refreshed separately.

## 2026-10-08 16:53 UTC — Final CAPRI pilot tested on real DarkBay

User requested testing real CAPTCHAs. Added `capri.py test-decoder-real` and `src/capri_controller/decoder_real_evaluation.py`; existing frozen decoder sources and active models are unchanged. Evaluated final epoch8 EMA on all900 opaque historical DarkBay images, with pixel hash checks, unchanged preprocessing and fixed thresholds. Saved and hash-froze all count/box/font/effect predictions before reading labels for count scoring. No training, VLM, forced target length, answer decoding or checkpoint/threshold selection.

Count head831/900 (92.33%); fixed-threshold detected-region count837/900 (93.00%). Ground-truth lengths are899 six and one seven, so an after-scoring constant-six diagnostic is899/900 (99.89%): no demonstrated variable-length generalization from this set. Real font/box accuracy remains unmeasured without annotations. Outputs, first12 fixed-order box examples and report are under the E-backed decoder diagnostics/real_darkbay_epoch8_20261008. Independent audit verified hashes and reproduced scores; inverse box geometry passed across three aspect-ratio cases. Updated single handoff; inventory completed with zero errors. Training/heartbeat stay stopped/paused.

## 2026-10-08 11:48 UTC — Bounded CAPRI pilot complete; heartbeat paused

Pilot reached its frozen eight-epoch maximum at11:45:49UTC after4h17m26s and20,494 updates; no extension. Final fixed DEV: count1,807/2,000 (90.35%), localization F195.8819%, font top1 100/10,018 (0.99820%), top5 373/10,018 (3.72330%). Frozen final verdict mixed/inconclusive: font gain misses the1-point criterion and absolute quality is very weak; count calibration deteriorated. No real-target/final-test claim.

All8 epoch checkpoints and40 recoveries preserved on E with raw/EMA and complete resumable state, independently hash/CPU verified during supervision. Final epoch8 and recoveries19,500/20,000 passed full finite-state checks; completed epoch hashes rechecked. Final SHA256 4d054e6287ddef82bd95b002fb7320298b9d94bdea5b141925469c0ab1538840. No pilot/original trainer remains, frozen sources unchanged, resources healthy. Updated main run status and the single handoff; supervision heartbeat confirmed PAUSED. Original full training remains stopped. No code, model selection, source protocol, dataset or active-checkpoint change. Further training requires new authorization.

## 2026-10-08 11:17 UTC — Bounded CAPRI epoch 7 verified

Verified epoch 7 at step 17,933 and recoveries 17,500/18,000 by checksum and CPU full-state checks. Fixed DEV count 90.5%, localization F1 95.7831%, font top1 1.18786% and top5 3.86305%. Font top1 gain reaches the modest fixed criterion, but absolute retrieval remains very weak; count accuracy and calibration worsened. Final planned epoch 8 is running. Updated handoff and supervision receipt; no source, model, protocol, deadline or original full-training stop changes.

## 2026-10-08 10:47 UTC — Bounded CAPRI epoch 6 verified

Epoch 6 at step 15,372 and recoveries 15,000/15,500 passed checksum and CPU full-state checks, including finite raw/EMA and optimizer tensors. Fixed DEV count remains 90.75%, localization F1 95.5929%, font top1 1.03813% and top5 3.88301%. Font quality remains very weak and its gain remains below the fixed 1-point criterion; count calibration worsened further. Epoch 7 progresses with finite loss. Updated single handoff and supervision receipt; source hashes, process identities and resources passed. No code, protocol, budget, model or original stop changes.

## 2026-10-08 10:17 UTC — Bounded CAPRI epoch5 verified

Epoch5 at step12,808 and recoveries12,500/13,000 passed checksum and CPU full-state checks, including finite raw/EMA and optimizer tensors. Fixed DEV count90.75%, localization F195.5616%, font top10.93831%/top53.67339%; font retrieval remains weak and count NLL/ECE worsened. Epoch6 is progressing with finite loss. Process identities, frozen sources and resources passed; handoff and supervision receipt updated. No training changes, deadline extension, full-run restart or active-model replacement.

## 2026-10-08 09:47 UTC — Bounded CAPRI epoch4 verified

Verified epoch4 raw/EMA and full finite resume state at step10,250, plus recoveries10,000/10,500. Fixed DEV count remains90.4%, localization F195.3403%, font top10.77860% and top53.19425%; font quality remains very weak and count calibration metrics continue worsening. Epoch5 is running with finite losses. Process identities, frozen sources, disk and checkpoint integrity passed. Updated single handoff and supervision receipt; no code, model, protocol, deadline or original full-run stop changes.

## 2026-10-08 09:18 UTC — Bounded CAPRI epoch3 verified

Verified epoch3 atstep7,685 and recoveries through8,000 by SHA256 and full CPU state checks; raw/EMA and optimizer state are finite. Fixed DEV count accuracy90.4%, localization F195.3378%, end-to-end font top10.56898%/top52.49551%; font quality remains very weak and count calibration metrics worsened slightly. Epoch4 is progressing. A finite pre-clipping gradient spike was transient; existing norm1 clipping remains unchanged. Updated single handoff and supervision receipt. No source/protocol/deadline change, full-run restart, active-model replacement or new training process.

## 2026-10-08 08:48 UTC — Bounded CAPRI epoch2 DEV result

Recorded epoch2 EMA results on the fixed2,000-image DEV subset: count1,804/2,000 (90.2%), localization F1 95.0111%, font end-to-end top1 43/10,018 (0.42923%), top5 179/10,018 (1.78678%). Font retrieval remains very weak. Verified recovery5,500 by SHA256 and CPU reload with finite raw/EMA and full resume state; rechecked prior checkpoint hashes, process identities, frozen sources and resources. Epoch3 is progressing without alerts. Updated the single handoff and supervision receipt; no protocol, code, deadline, active model or original full-run stop changed.

## 2026-10-08 08:36 UTC — Bounded CAPRI pilot epoch2 saved

Verified epoch2 (step5,123) and all recoveries through5,000 by SHA256 and CPU reload, including raw/EMA and complete finite resume state. Supervision found no process, frozen-source or resource alerts. Updated the single handoff; DEV evaluation of epoch2 is still running, so epoch1 remains the latest quality evidence. No training code, protocol, deadline, active model or original full-run stop changed.

## 2026-10-08 08:17 UTC — Pilot epoch1 verified; count improves, font retrieval remains weak

- Independently verified/reloaded the first full20k-image epoch at update2,564: raw/EMA and complete finite resume state, SHA `6e6f36dd2106624f71f089f8b9602d3b9bb07e2bbaeba2ea340b72ea1c3a4a4c`. Recoveries through3,000 also verified; no changed frozen sources or process/resource alerts.
- Fixed2kDEV EMA results: count1,801/2,000 (90.05%,+5.65pp), localization F1 95.0126% (initial93.8240%). Font top1 remains only28/10,018 (0.2795%); top5 1.0381%, all1,453 candidate families represented. Recorded improvements and material font limitation without declaring pilot success.
- Epoch2 continues within the unchanged time cap. Updated single handoff with font appendix preserved; no code, selection, architecture, data-location or protocol change.

## 2026-10-08 07:46 UTC — Bounded pilot recoveries500/1,000 verified

- Confirmed supervisor/trainer/four-worker identities, finite current loss/gradient and unchanged frozen pilot sources; no alerts. Epoch1 remains in progress, zero completed epochs and no post-training DEV result yet.
- Independently SHA256-verified and CPU-loaded both510,630,301-byte checkpoints; raw/EMA and full optimizer/scheduler/scaler/sampler/RNG/execution state are present and finite. Saved pilot supervision receipt and updated handoff with font appendix preserved.
- Six-hour target/eight-hour cap unchanged. No code, model-selection, data-location or training-protocol changes; original full-corpus training remains stopped.

## 2026-10-08 07:28 UTC — User-authorized bounded six-to-eight-hour pilot launched

-07:36UTC actual joint optimization confirmed: step101, finite losses/gradients, AMP16. Initial held-out count84.4%, localization F1 93.824%, font end-to-end top1 0.13975% across1,453 represented candidate families. These establish the before-training reference, not pilot success. Checkpoints will start at update500 and remain distinct from the stopped original warmup.

-07:32UTC readiness update: subset covers1,332TRAIN families/37effects/all7counts; actualGPU allocation5.615GiB; original/optimized FP16 loss/gradient comparison passed within declared tolerances. The96-step fixed31-image learning gate reduced localization loss62.03%, followed by an exact initial-weight reset. Initial DEV evaluation is running; no pilot quality result yet.

- Added isolated pilot launcher/watchdog, paired stratified subset reader/audit, batched localization losses and held-out font retrieval evaluation. Original450k trainer remains stopped; initialization uses its verified raw step20,000 with new joint optimizer.
- CPU/CUDA analytic loss, assignment and gradient equivalence checks passed. Actual model GPU/learning gates must pass before pilot optimization.20kTRAIN/2kDEV, unchanged model/input resolution; sparse all-font TRAIN+DEV evaluation gallery, no real/calibration/test answers.
- Launched `capri.py decoder timed-pilot` with an independent8hour watchdog and5.5hour optimization cutoff. Everyepoch/500updates full checkpoints and final partial save; fixed quality-regression guards. Target around06:28Pacific, hard stop08:28Pacific today.
- Changed existing heartbeat to pilot-only supervision every15minutes, pausing on completion/failure/stop. No automatic full-run restart, scaling, specialists or active-model replacement. Handoff updated with font appendix preserved. Results/readiness remain pending; do not claim pilot quality yet.

## 2026-10-08 07:12 UTC — Bounded stopped-model runtime investigation

- Added `decoder/performance_probe.py` and `capri.py decoder profile-stopped`. Checks the training stop/processes and original hashes; performs no optimizer step, resume, checkpoint or stopped-status overwrite. Reports live under the E-backed CAPRI diagnostics.
- Four eight-image TRAIN batches with two warmups/three repeats measured83–116ms forward,215–220ms matching/loss and273–363ms backward. Moving targets to GPU once did not materially improve total time. The single profiled batch exposed many small tensor operations/scalar reads; profiler overhead is explicitly distinguished from stage timings.
- Verified the earlier PARSeq seed48 duration6843.90s for100k x20epochs, batch128/base32-high input; comparison confirms the user's concern while documenting CAPRI's larger images/dense localization workload. No unmeasured speedup or fixed20-epoch CAPRI plan claimed.
- Handoff records the insufficient throughput gate, measurements, limits and next optimization candidates. Frozen baseline code/weights, training stop and paused heartbeat remain intact. Workspace catalog refreshed.

## 2026-10-08 02:43 UTC — User requested training stop

- Immediately stopped and verified the CAPRI trainer, four workers and launcher; paused its automatic supervision heartbeat. No restart authorized.
- Preserved every saved artifact and reverified recovery 20,000 SHA256/full-state receipt. Last reported update 22,251 was later than the saved recovery; subsequent updates were unsaved. No completed epochs.
- Recorded the stop and preserved prior progress in E-backed `user_stop.json`, marked run/phase status stopped, retained the stopped lock, and updated the single handoff while preserving its font appendix. No code/model/location change.

## 2026-10-08 02:40 UTC — Audited held-out DEV font library and recovery 20,000

- Added separate DEV gallery generation, full audit and pixel-only dataset adapter (`decoder/dev_gallery.py`, `dev_gallery_data.py`) through four `capri.py decoder` stages. Prepared 3,060 clean references across 188 files/121 DEV families on E; existing TRAIN references and 500k corpus remain unchanged. Gallery is selection-only, never optimization; no calibration/test images accessed.
- Pilot 32/32 accepted; all 24 archives/3,060 images and labels audited, eight renders exactly replayed, 32 reader examples and complete batching checked. Wrong splits/altered pixel identities refused. Preserved 204 duplicate images with provenance; no cross-family exact-pixel duplicates within this DEV bank. Full snapshot `2188b0104bbff31498e0e2167c97ff42d9564c2706a5a88dae81cbb3a03a7f6a`.
- First pilot safely rejected the corpus-era storage-helper hash. New gallery binds the exact documented E-migration helper while enforcing original rendering dependencies; original corpus protocol/failure evidence preserved. Two CPU workers finished; no additional GPU job.
- Recovery 20,000 verified: raw/EMA plus complete resume state, SHA `6bee2dbcee7a965f24998855f343f545f2dfd075b2d3d0233409fa5cfb49d7e5`. Supervision: one trainer, five recoveries, zero completed epochs, no alerts; frozen 141 training sources unchanged. Updated handoff with font appendix preserved and refreshed inventory.
- Limits: data/audit readiness only, no trained font retrieval result. DEV extraction, TRAIN distractor integration and retrieval metrics remain, plus production joint launcher/budget/GPU checks and later specialist/distillation/intensity phases. Active models/controllers unchanged.

## 2026-10-08 02:07 UTC — Epoch-aware joint-phase integration and complete CPU resume checks

- Added `decoder/joint_phase.py` and `capri.py decoder joint-phase-tests`. JointPreparation schedules deterministic TRAIN references using epoch/global step; JointObjective combines existing task losses and explicit auxiliary count/localization weights. The later-phase engine freezes preparation/objective contracts, prepares once before AMP retry, and checks contract/scene-field preservation.
- Fifteen generic engine and six tiny joint-phase CPU check groups passed with CUDA hidden. Interrupted two-epoch runs and forced overflow retries exactly matched complete saved state/reference choices; diagnostic epoch checkpoints persist and completed reruns are idempotent. Reports/checkpoints remain under E-backed CAPRI diagnostics, distinct from actual training epochs.
- All 141 frozen live warmup sources match; PID 12920 continues with finite loss/gradients at update 19,751. Latest supervision verified four recoveries, no alerts and zero completed epochs. No GPU competition or data regeneration.
- Updated the single handoff while preserving the font appendix; refreshed catalog. Full-model GPU readiness/profiling, production launcher/checkpoint initialization, explicit training budget/weights and held-out font-gallery evaluation are still required before the joint phase can launch. No reader/controller/active-model change.

## 2026-10-08 01:35 UTC — Deterministic font-reference pairing and subbatch integration

- Added `decoder/reference_sampling.py`, checks and `capri.py decoder reference-sampling-tests`. Stateless reference choices bind original corpus family IDs, reference snapshot, seed, epoch, explicit mode/budget and scene identity. Bounded omissions are reported; exact-shape collating avoids new padding. Extended joint font forward to multiple reference subbatches with repeated-specimen refusal.
- Validation: six CPU sampling check groups passed, including preserved RNG/retry/resume behavior and real existing TRAIN I/O (36,492 references verified, one 1,000-image corpus shard, 20 decoded reference selections). Nine joint font-objective CPU groups passed, including subbatch evidence/order invariance. CUDA hidden throughout; reports saved under E-backed CAPRI diagnostics.
- Warmup remains live and unchanged at update 17,726 / 139,104 TRAIN images. Existing bounded AMP handling recovered two additional overflows; scale 128 and finite progress. Four recoveries verified, no alerts, zero completed epochs, all 141 frozen source hashes match.
- Single handoff/font appendix preserved; catalog refreshed. Epoch-aware loop wiring, GPU readiness, reference frequency/phase budget and actual downstream training remain pending. No data regeneration, reader/controller or active-model change.

## 2026-10-08 01:05 UTC — Joint scene/reference font-loss integration

- Added `decoder/font_training.py`, CPU checks and the `capri.py decoder font-training-tests` entry point. Existing 96 task terms are retained; the font contrastive term gains gradients from known TRAIN reference embeddings selected from predicted regions. Labels associate predictions only after pixel-only forward.
- Excluded duplicate pixel/glyph pairs and cross-family negatives within pixel-identical specimens without merging whole families. Unsupported/missing evidence remains explicit and finite; no count/crop oracle or character reader is introduced.
- Eight analytic CPU check groups passed with CUDA hidden. The refined ambiguity check verifies that alternate glyph positions in an identical specimen cannot create conflicting positive/negative family supervision. Report under E-backed `runs/capri_decoder_500k_20261007/diagnostics/font_training_20261008`. GPU memory/throughput, deterministic reference sampling, phase hyperparameters/budget and actual joint training remain pending.
- All 141 frozen warmup sources match; live trainer PID 12920 continues with finite losses. Four recovery snapshots verified, zero completed epochs, no alerts. Single handoff/font appendix preserved and workspace catalog refreshed; no corpus or active-model change.

## 2026-10-08 00:53 UTC — CAPRI E-drive migration complete and training resumed

- Moved the complete CAPRI run to `E:/pranav/CAPRI/runs/capri_decoder_500k_20261007`, with the original C path preserved by a junction and mapping in `configs/local.json`. SHA256-verified 19,028 files / 24.35 GiB before switching. Removed only the verified redundant C backup after a successful resumed optimizer update. C free space is now 65.32 GiB; E has 5,843.43 GiB free.
- Fixed a background PowerShell compatibility failure before the first switch: replace unavailable Get-FileHash with .NET SHA256. Confirmed identical manifest digest; added exact prior-stop/checkpoint guards for quiescent retry, re-copied/re-hashed all files, and retained first-failure evidence. Python/PowerShell syntax validation passed; the full retry completed and exercised the corrected path.
- Training resumed from recovery 15,000 under PID 12920 / four workers. Finite update 15,001 verified, then progress beyond 15,026. Four recovery snapshots verified, zero completed epochs, no supervision alerts. CPU checkpoint loading verified AMP amendment and consumed sampler state; all 141 original frozen source hashes still match.
- Added configured storage mapping to inventory JSON/catalog so logical workspace totals are not mistaken for C-drive usage. Refreshed catalog after cutover. Single handoff updated; font appendix preserved. No architecture/corpus/active-model change. Warmup and later font/specialist/distillation/intensity phases remain incomplete; runtime profiling and total budget remain pending.

## 2026-10-08 00:31 UTC — Read-only CAPRI throughput observation

- Added `decoder/runtime_observer.py` and `capri.py decoder observe-training` for bounded existing-process/driver telemetry. The 30-second observation wrote only to a separate run folder; no CUDA model, process interruption or live-run mutation.
- Measured mean GPU utilization 37.22% (median 22%), trainer approximately 1.04 CPU cores, loaders 0.13–0.15 cores each. Coarse sampled telemetry cannot establish a causal bottleneck; operator profiling and a measured overall schedule are still required before later production phases.
- Single handoff records the user's runtime concern and unfinalized total budget. Frozen trainer and active migration worker untouched; no source-run diagnostics near cutover. Font appendix preserved.


## 2026-10-08 00:02 UTC — AMP retry protection for later CAPRI phases

- Added `decoder/optimization.py` and integrated bounded same-batch overflow recovery into the independent phase engine. Retry source/policy bind to future phase protocols; scheduler/EMA/sampler advance only on successful optimizer updates.
- Validation: twelve CPU phase-engine fixture groups passed, including full-epoch exact clean-versus-retry state and persistent-overflow failure without state advancement. Corrected an overly specific fixture batch-count assertion to use the actual paired sampler count. No GPU diagnostics while the live trainer runs.
- Original 141 baseline source hashes and live recovery amendment unchanged. E migration worker remains waiting for checkpoint 15,000 and owns the restart. Added handoff guard against diagnostic writes near/during its quiescent copy/verification window. No new production training phase or trained quality claim; single handoff/font appendix preserved.


## 2026-10-07 23:58 UTC — User-requested CAPRI data migration to E (in progress)

- Copied the full CAPRI run to E:/pranav/CAPRI/runs/capri_decoder_500k_20261007, retaining the live C copy until checkpoint-safe cutover. Initial robocopy completed 23.89 GiB with zero failures.
- Added `capri.py decoder migrate-storage` and bounded native PowerShell cutover/cleanup. Worker waits for recovery 15,000 by default, verifies checkpoint/process identity, stops the trainer, synchronizes/hash-verifies every file, creates the old-path junction, updates machine storage configuration, resumes training and verifies progress before deleting the redundant verified C backup. Failure retains data for inspection.
- Updated decoder path resolution to preserve frozen logical paths and allow only the configured physical mapping. Actual E-junction tests and ten CPU phase-engine checks passed; no original frozen baseline source was changed.
- Migration remains pending; journal under runs/capri_storage_migration_20261007. Worker owns trainer restart while active. Single handoff updated; font appendix preserved. No controller, reader, architecture, corpus or active-model changes. Refresh catalog and record final verification when migration completes.


## 2026-10-07 23:30 UTC — Recover CAPRI AMP gradient overflow without skipped samples

- Supervision found localization training stopped at 23:04 on a nonfinite gradient norm, after logged update 10,251. Preserved failure evidence; verified update-10,000 recovery and absence of live duplicate trainer.
- Added `decoder/amp_recovery.py`, fault-injection checks, and verified amendment dispatch through the existing `capri.py decoder train-baseline` command. Retry the same batch/RNG/buffers at half loss scale (reset growth tracker), bounded to nine attempts; no optimizer/scheduler/EMA/cursor advance until finite success. Forward nonfiniteness and persistent failure still stop. Future checkpoints bind amendment identity.
- Validation: tiny CPU and CUDA retry matched clean raw/EMA/optimizer/scheduler/scaler/RNG exactly; persistent failure preserved model/optimizer/scheduler/EMA while reducing only the loss scale, and stopped within bound. Original 141 source hashes/protocol/checkpoints unchanged. Actual replay verified at 23:35: update 10,298 overflowed in initial_reference.weight, retried scale 512 to 256 on the same batch, then continued through update 10,301 with finite gradients/no alerts. Receipt amp_recovery/resume_verification.json. Full-model CUDA continuation is not claimed bit-exact.
- Restarted hidden at 23:29:27, PID 16572, from verified update 10,000. New logs/launch receipt preserve earlier failure logs. No data regeneration, architecture/objective change, VLM, reader/controller/Model4 change or active checkpoint replacement. Single handoff updated; font appendix preserved.


## 2026-10-07 23:01 UTC — CAPRI joint DEV evaluation

- Added `decoder/joint_evaluation.py` and `capri.py decoder joint-evaluation-tests` to support later full-head joint/specialist evaluation. The evaluator binds DEV dataset/batching/source/threshold identities and reports 96 supported task losses plus separate count/localization/font-attribute/effect/intensity/appearance metrics.
- Validation: eleven CPU fixture groups passed with CUDA hidden, including denominator/missing-label correctness, partial coverage, exact traversal, wrong split/mutation refusal and failure restoration. Added explicit evaluation-contract checksum after a tiny-fixture batch-limit mutation retained the same batch indices. All 141 live baseline source hashes unchanged.
- Limits: no real DEV inference, selection thresholds/guard limits, trained retrieval gallery, teacher routing or new training phase was executed. Contrastive loss is not font retrieval accuracy. Localization training continues; no VLM, reader/controller/Model4 edits, corpus regeneration or active checkpoint replacement. Single handoff maintained; font-inventory appendix preserved byte-for-byte.


## 2026-10-07 22:31 UTC — CAPRI predicted-region font extraction

- Added independent `decoder/font_extraction.py` and `capri.py decoder font-extraction-tests`. Full TRAIN-reference extraction uses pixel-only forward and predicted count-independent region selection, then library-label overlap auditing/association. It records missing coverage and preserves identical-pixel family ambiguity for the existing font index.
- Validation: eleven CPU fixture groups passed with CUDA hidden, including duplicate/empty handling, strict provenance/split boundaries, index round-trip/ties and RNG/mode restoration on injected failure. All 141 frozen baseline source hashes unchanged; live localization training remains healthy with two verified recoveries and no completed epoch.
- Limits: actual font-trained checkpoint extraction and coverage/publication gates remain unrun. Caller must bind loaded model and frozen thresholds/source hashes to the index identity. No production index, trained font metric, corpus regeneration, GPU duplicate, VLM, reader/controller/Model4 changes or active checkpoint replacement. Single handoff updated; appended font inventory preserved byte-for-byte.


## 2026-10-07 22:05 UTC — CAPRI subsequent-phase checkpoint engine

- Added `decoder/phase_engine.py`, CPU fault-injection checks and `capri.py decoder phase-engine-tests` for resumable multi-epoch full-corpus phases. This prepares later joint/specialist work while preserving the already-running localization trainer.
- Full epoch/recovery snapshots retain raw/EMA, optimizer, scheduler, scaler, consumed sampler and RNG/execution state. EMA validation can fail and retry without another training pass or checkpoint replacement; settings/hash mismatches fail closed.
- Validation: ten CPU fixture groups passed with CUDA hidden, including exact continuation across mid-epoch/final-batch/later-epoch interruptions, scaler growth, validation failure and tampering. Fixed optimizer tuple/list serialization comparison revealed by the first fixture run. All 141 live baseline source hashes unchanged. Update-5,000 real recovery hash verified by supervision; no completed real epoch yet.
- Limits: generic engine only, not connected to a new production stage. Full-model GPU readiness, joint/font metrics/objectives, measured specialist groups, targeted sampling, distillation and intensity refinement still pending. No corpus regeneration, GPU duplicate, VLM, reader/controller/Model4 change or model-registry replacement. Single handoff updated; appended font inventory bytes preserved.


## 2026-10-07 — Measured specialist grouping and DEV-only teacher-routing gates

- Added `decoder/specialist_selection.py` and `selection_checks.py`, exposed via `capri.py decoder specialist-selection-tests`. Adapted the inspected historical spectral TAG grouping and per-task teacher-improvement policy; directed affinity must be TRAIN-only, adequately observed for every pair, and pass the 90% positive-self gate. Missing evidence is not fabricated as zero. Explicit size limits are repaired deterministically, without dropping groups.
- Teacher routing checks frozen synthetic DEV dataset/evaluation identity, raw/EMA consistency, evidence support, finite scores and actual checkpoint hashes. Only assigned tasks with strict improvement and passing preservation guards can route to a specialist; otherwise baseline remains. No real group, teacher, metric threshold or training budget was chosen.
- Ten CPU fixture groups passed, including missing evidence, target-order invariance, invalid splits, preservation regression, ties, nonfinite scores and tampered files. CUDA hidden for tests; 141 running-baseline source hashes independently verified unchanged. Live PID 8416 still training epoch 1, with finite losses and no duplicate job. Single handoff updated; joint training, real affinity measurement, specialist runs and trained font indexing remain incomplete.


## 2026-10-07 — Specialist loss/distillation contracts and TRAIN reference reader

- Added `decoder/specialist_tasks.py`, `distillation.py`, `specialist_checks.py`: 96 explicit loss terms over existing heads, with real evidence counts, assigned-task-only group means and detached teacher imitation aligned through synthetic glyph identity. Each effect presence/intensity output is independently addressable; absent intensity, missing font metadata and padded pixels are masked. No task groups, teachers, phase budgets or active model were selected.
- Added `decoder/reference_data.py`: complete immutable TRAIN reference snapshot reader with archive/plan/font/pixel identity checks and shared pixel-only preprocessing; known library labels are a separate structure. Existing 36,492 specimens are reused without generation or corpus changes.
- Added `capri.py decoder specialist-contract-tests` and `reference-data-tests`. Validation: nine CPU analytic specialist groups passed, including original aggregate-loss equivalence, query permutation, detached teacher and unassigned gradient exclusion; full reference archives verified, 75 decoded samples across all shards, all examples covered once. No CUDA diagnostic ran alongside the trainer. Frozen active-training source hashes were independently checked unchanged.
- The running count/localization epoch continues. These additions do not yet implement trained font extraction/indexing, joint-baseline GPU readiness, actual TAG grouping, DEV teacher selection, specialist training or final distillation runs. Those remain required and must receive phase-specific protocols/checkpoints before launch. Single handoff updated and historical DINOv3 status explicitly separated from the live DINOv2 run.


## 2026-10-07 — Authorized DINOv2 integration and real count/localization trainer

- User clarified that the earlier DINOv2 discussion meant to proceed. Replaced the missing DINOv3 dependency for CAPRI Decoder with official pretrained DINOv2 ViT-S/14 with four registers. Downloaded public weights and pinned unmodified official source revision `85a24602099d397264d5b30461ad7f3bfd726ca1`; kept source/license/asset hashes. Strict `weights_only=True` loading passes. Initial newer source is retained but unused because its annotations require newer Python. Updated machine paths and the existing supervision heartbeat; active controllers/registry unchanged.
- Added physical-coordinate 14-patch alignment into the existing stride-16 fusion grid, preserving white padding and glyph aspect ratio. Total model 42,904,702 parameters, including the unchanged 11,132,160-parameter CNN. Added commands `acquire-dinov2`, `training-tests`, `train-baseline` via `capri.py`.
- Added one-full-pass 450k TRAIN count/localization baseline stage, seeded exact-shape pair-preserving sampling, TRAIN-only initial loss-scale estimates, frozen DINO and secondary heads, AdamW/AMP/EMA, full-DEV EMA diagnostics, immutable epoch checkpoints and separate intermediate recovery snapshots. No calibration/test/real label tuning. Initial stage deliberately ends at a saved boundary; joint baseline, font index, measured TAG specialists, assigned-output distillation and intensity refinement remain to implement/complete.
- Validation: strict pretrained load; actual full-model checkpoint pre-step state and continuation within measured native CUDA grid-sampling drift; 16 real TRAIN pilot updates; physical patch-grid analytic check; maximum 128x768 batch8 backward peak 5.34 GiB; trainer state/cursor reload. Final loader/recovery integration verification passed, including Windows multiprocessing and sampler/RNG separation. Training launched at 20:50:59 UTC, PID 8416; first actual finite-loss optimizer update at 20:53:24 UTC. Epoch 1 is in progress; no completed actual training epoch yet. The frozen protocol contains all 450k TRAIN images in 57,333 batches. At 20:55 UTC: 126 actual optimizer updates / 987 images, finite losses; update-100 full recovery snapshot saved and independently SHA-verified, no alerts. Epoch remains in progress. Workspace inventory refreshed: 166,869 files, 120,247,437,049 bytes, zero errors. Diagnostic checkpoints live outside actual `training/` and are not reported as trained epochs.


## 2026-10-07 CAPRI Decoder reference layout recovery and audit

Completion verified at20:36 UTC: all36,492 planned references /73shards /1,121,419,032bytes, covering all2,052 TRAIN font files and1,332 outline-alias groups. Final independent audit checked every archive hash/CRC and label and1,170 decoded image/mask samples; three accepted specimens used the additional size fallbacks. Preserved670 cross-family identical-image groups for ambiguity-aware retrieval/calibration, without merging fonts or changing splits. References contain30,073 reported unique pixel hashes; duplicate library views are expected and separate from the unique500k corpus. No rendering job remains live; no model training or embedding extraction has occurred. Full results and limits are in the single handoff and immutable audit reports.

The clean font-reference job stopped after 26,500 committed references because tall jsMath-cmex10 glyphs in one six-character specimen reached the canvas edge. Added diagnosis and a separately verified operational retry amendment: three smaller-size settings follow the original six, preserving font, axes, text and all renderer acceptance gates. Original renderer/reference-generator sources and base protocol remain unchanged; a process-local initializer extends the retry table only after the amendment's source/protocol hashes pass. The stage dispatcher selects this verified recovery when present. Old failures, partial data and committed shards remain preserved.

Validation reproduced the failure, found a valid size fraction0.4 result at attempt6, and compared all 96 saved pilot payloads byte-for-byte with extended-retry outputs. Restarted the same four-worker command only after verifying no active process, with separate logs and a launch receipt binding the amendment hash. Progress passed the old failure to 32,000 committed references. Added an independent reference audit for all archive hashes/CRCs and labels, planned font/axis/text coverage, duplicate-hash ambiguity and decoded pixel/mask samples including all fallbacks. Completion/audit results are recorded in the single handoff and run reports.

No font/count substitution, gate relaxation, main corpus change, model/backbone switch, controller/reader change, active-checkpoint replacement or actual training occurred. DINOv3 weights and remaining integration are still pending. Appended handoff font inventory preserved; workspace catalog refreshed after verification.

## 2026-10-07 CAPRI Decoder clean library reference pilot and render launch

Added `reference_render.py`, `reference_generation.py` and three explicit decoder stage commands for a clean TRAIN reference pilot, storage/recovery checks and full reference rendering. Reuses unchanged hash-verified renderer, fonts and calibration; bounded retries only change canvas/size/spacing while preserving font/axes/text. Labels, PNGs, masks and boxes are checked before acceptance. Published 500-reference shards have CRC/hash receipts and fixed specimen order; completed archives are preserved, partial/orphan files retained, rejection fails explicitly, and free space must remain above 30 GiB. Supervision now tracks this separate job's process/progress/failure. No decoder model, controller, reader, active checkpoint or 500k corpus change.

Validation: all 96 pilot specimens accepted on first attempt, deterministic payload replay and serialization checks passed; reviewed the style contact sheet. One identical-pixel Noto pair already shares the same outline alias group. The 16-image storage diagnostic passed an intentional stop/resume, preserved partial data and unchanged completed restart. Started the full 36,492-reference job with four workers only after confirming no duplicate process. Verified live lock PID18776, two committed shards/1,000 references, archive SHA/CRC and all first-1,000 instance font/axis/count labels; stderr empty. Launch, logs, fixed protocol and ongoing progress are under the decoder run; single handoff records paths and recovery rules.

This job renders library specimens, not additional training-corpus examples or trained embeddings. Full-reference completion/coverage and model-based region extraction remain pending. Zero actual training epochs; approved DINO weights and remaining integration still required. Appended font inventory preserved; catalog refreshed as a running-work snapshot.

## 2026-10-07 CAPRI Decoder font-reference index infrastructure

Added a replaceable visual font index, immutable hash-verified NPY/JSON publication, strict embedding identity (checkpoint/raw-or-EMA/preprocessing/pipeline/library hashes), diagnostic-index guard and deterministic TRAIN specimen planning. Retrieval uses nearest specimen cosine grouped by alias-aware family, preserves per-query candidates and optional quality-weighted aggregation/mixture evidence, and labels every score uncalibrated. New commands: `capri.py decoder retrieval-tests` and `prepare-font-references`. No backbone/model/head/corpus/controller/reader changes and no real target labels.

Validation: analytic known neighbours, style/alias family grouping, mixed families, quality, absent-family queries, empty/zero evidence, deterministic ties/order, nonfinite/zero rejection, duplicate/cross-split metadata refusal, identity mismatch, save/load, overwrite, tamper and diagnostic guards passed. Created a TRAIN-only plan for all 2,052 font files: 36,492 six-character default/axis-endpoint specimens. Font hashes and an independent complete-plan validation confirmed all recorded ASCII coverage, unique IDs, correct family/split provenance and in-range axes. Plan and validation live under `runs/capri_decoder_500k_20261007/font_reference_plans`; no specimen images or trained embeddings were generated.

Limits: analytic retrieval tests establish plumbing, not font accuracy; max cosine is affected by gallery size/diversity, and quality/rejection policies require exact-bank calibration. Reference rendering/predicted-region extraction and split-isolated evaluation galleries remain pending. No actual decoder training epochs; approved DINOv3 weights and remaining training integration are still required. Single handoff updated with these boundaries and appended font inventory bytes preserved. Catalog refreshed.

## 2026-10-07 CAPRI Decoder masked font and appearance supervision

Added six matched-instance font auxiliaries (category, stroke class, intrinsic italic, nominal weight, stored x-height and intrinsic width) on a separate style projection from localized spatial/query features. Model core is now 42,447,742 parameters; CNN/DINO are unchanged. Added explicit unknown/nonfinite/variable-instance validity masks, actual applied weight-axis labels, conservative lowercase evidence for x-height, and separate task support counts. `appearance.py` also supplies observed character RGB and underlying-background losses; partial padded cells are unpremultiplied and coverage-weighted to avoid false dark-edge targets. Missing RGB does not become a zero-colour target. Inputs remain pixels only; no frozen corpus sources, controller, reader, active checkpoint, or pretrained architecture substitution.

Validation through `capri.py decoder appearance-tests`, `data-tests`, `objective-tests`, `model-tests` and `model-resume-tests`: CPU/CUDA missing-label zero gradients/support, matched-only gradients/permutation, known colour/background values and padding invariance passed. A 53-image/262-glyph TRAIN audit recorded actual metadata support. Expanded model checks exercised nonzero real-label gradients in all six attribute heads, finite AMP matching/losses and frozen/partial DINO behavior. Its 563,937,559-byte checkpoint restored raw/EMA/optimizer exactly before continuation, including sampler/RNG/modes/precision/scaler/scheduler; the following GPU update retained small documented nondeterminism (raw 5.88968e-6 versus 3.24287e-6 no-file control). This is random-DINO diagnostic recovery, not baseline or specialist training. Older core test reports were preserved in `checks_history/`.

Limitations: inherited font metadata can be heuristic; the fields describe library style, not uniquely recoverable renderer settings. Background targets are pre-photometric under-layers. Retrieval/prototypes, palette/geometry summaries, measured task weights and phase budgets remain to implement/freeze, and approved DINOv3 weights are still required. No held-out labels or real target answers used. Single handoff updated with exact semantics, parameter counts and remaining work; appended font inventory bytes preserved. Catalog refreshed.

## 2026-10-07 CAPRI Decoder full-core checkpoint recovery

Extended decoder checkpoints to schema v2: exact optimizer parameter ordering/groups, trainable-parameter policy, model/pretraining configuration, per-module mode and numerical execution settings are saved and validated/restored. Scheduler/scaler presence mismatches are refused; finite scheduler and metrics checks supplement raw/EMA/optimizer/scaler checks. Existing v1 files remain supported. Added `model.resume_configuration()` and `capri.py decoder model-resume-tests`; no architecture, corpus, controller, reader or active model selection change.

Validation: full 42.09M core with partial DINO gradients, AdamW/AMP/scaler/scheduler and EMA saved a verified 558,113,959-byte immutable checkpoint. Raw, EMA and optimizer restored bit-for-bit before continuation; sampler/RNG/modes/precision/scaler/scheduler restored and wrong unfreezing/order were refused. The next loss matched, but CUDA continuation was not bit-exact: raw maximum difference 4.39724e-6 versus 3.91807e-6 for a no-file live-memory replay. Initial nominal raw tolerance failed for both; the documented empirical ten-times-control bound passed. Do not describe this as exact end-to-end GPU continuation. A test-only optimizer counter alias was fixed by cloning the replay snapshot. Previous failed diagnostics are preserved. CPU and tiny CUDA exact continuation and existing nonfinite/overwrite guards passed.

These were two logical diagnostic updates with replay, random DINO and one TRAIN shard, not baseline or specialist training. Actual saved training epochs remain zero. Approved DINO weights, remaining appearance/font tasks and final training/readiness contracts are still required. Single handoff updated preserving the appended font inventory bytes; workspace catalog refreshed.

## 2026-10-07 CAPRI Decoder completed corpus and visual model core

The 500,000-image corpus finished at 17:33 UTC. Final independent audit verified all 500 archive hashes, receipt-level global pixel uniqueness, split/coverage/count contracts, all 2,514 indexed font files, and 14,400 sampled TRAIN labels; no alerts. Held-out coverage was inspected through production receipts, without exploratory held-out label sampling. Generation is complete and must not be relaunched.

Implemented `decoder/model.py` and `attention.py`: larger residual CNN, actual DINOv3 ViT-S/16, multiscale fusion/deformable memory, iterative 16-query character decoder, visible masks/boxes/count, region-based font embeddings and initial supporting appearance/effect outputs. Core parameter count is 42,085,362, including 11,132,160 CNN parameters. Missing pretrained weights are refused outside explicit random architecture diagnostics. Default frozen DINO and controlled final-four-block unfreezing are supported. No reader, controller, VLM, active checkpoint or corpus source changes.

Validation via `capri.py decoder model-tests`: analytic CPU/CUDA sampling and gradients, masked sampling, aspect-preserved/detached font crops, synthetic Hungarian matching and intermediate losses, finite GPU AMP gradients, input/padding/batch checks and partial-DINO maximum-canvas backward passed. Batch-composition checks isolate TF32 rounding by disabling TF32 and restore flags afterward; initial TF32 mask-logit differences are documented. Diagnostic batch 2, 128x768 used 1.45–1.52 GiB allocated peak, excluding optimizer/EMA and full training state. Zero optimizer steps and zero decoder training epochs; approved weights, remaining appearance/font-attribute contracts, retrieval prototypes, full-model resume and measured training/specialist schedule remain outstanding. Single handoff records exact scope; catalog refreshed.

## 2026-10-07 CAPRI Decoder deterministic retry recovery

The renderer stopped after 212,000 committed images because the seven-character jsMath-cmex10 pair at TRAIN indices 212026/212027 exhausted 80 attempts. Added TRAIN-only failure diagnosis and bounded continuation of the same deterministic attempt stream to 400 attempts, plus explicit writer-revision provenance verification. Frozen rendering/sampling sources, fonts, labels, quality thresholds, quotas and corpus protocol remain unchanged. The original writer, failure logs and partial shard are preserved; an operational amendment records original/current writer and recovery hashes. Failures now mark corpus progress failed. Commands are `capri.py decoder diagnose-render-failure`, `recovery-tests` and the existing `generate-corpus` resume.

Validation: reproduced original exhaustion, found the first accepted pair at attempt 110, verified unchanged font/count and exported mask/PNG checks, and compared four ordinary pairs' full encoded payloads byte-for-byte against the original writer. Independent audit verified 212 archive hashes / 212,000 unique images, all 2,052 TRAIN files and 1,565 families, balanced counts and 6,784 sampled TRAIN labels. No held-out label analysis. Restarted only after verifying no live generation process; the resume command revalidates every committed archive. A finite 400-attempt budget can still exhaust and must be diagnosed if it does. No decoder training, controller/reader changes or model-selection changes. Single handoff updated; catalog refreshed after this change.

## 2026-10-07 CAPRI Decoder specialist-affinity preparation

Audited the previous iteration's actual TAG lookahead and assigned-loss implementation. Added functional-copy directed task-affinity measurement, companion gradient cosine, explicit missing-evidence/support counts, validated epsilon selection and assigned-task/group objective masking under the maintained decoder package. Only synthetic TRAIN batches are accepted. The measurement does not update live parameters, buffers, existing gradients, optimizer or EMA, and restores RNG/module modes/TF32 after exceptions. Added `capri.py decoder affinity-tests`.

Validation: 11 analytic CPU/GPU fixture checks passed, covering positive/negative/orthogonal transfer, missingness, count aggregation, rejection of invalid steps or unrepresented tasks, exact state preservation, injected-failure cleanup and excluded-task gradients. No real decoder baseline was measured and no groups, training weights or epoch budget selected. Actual per-head task adapters/shared CNN parameters still require the completed model. The running corpus and frozen renderer/font sources were preserved. Single handoff updated; appended inventory unchanged.

## 2026-10-07 Independent CAPRI Decoder corpus audit

Added `capri.py decoder audit-corpus` and a read-only audit module. It snapshots published receipts, independently hashes completed archives, verifies cross-shard uniqueness/count balance/font split membership, reports library coverage and storage projections, and measures accepted count associations on a deterministic TRAIN-only sample. Reports are immutable files under the active decoder run's `audits/`; no running-job, renderer, sampler, font manifest, model selection or controller change.

Validation on 71,000 published images: 71 archive hashes checked, zero duplicate/split/continuity failures, counts 2–8 within two examples of each other, all1,565 train families and2,051/2,052 train font files covered. The2,272-image train sample passed quality/coverage checks and showed small descriptive count associations with canvas size/layout; this does not establish statistical independence or model accuracy. One Lato style remains unsampled at this partial stage. Requested severity acceptance is mildly uneven and realized coverage remains measured rather than exactly balanced. No held-out-label exploratory analysis or training was performed. Single handoff updated; appended font inventory unchanged.

## 2026-10-07 CAPRI Decoder assignment, loss and metric preparation

Added separate decoder objective primitives for one-to-one character assignment, visible-region localization, count/cardinality/consistency, font-family contrastive learning and conditional effect severity. Task losses are returned separately; final training coefficients remain unfrozen. Added specialist query alignment through synthetic glyph identities and count/localization metrics that expose duplicate/missed detections and count-head disagreement. Stable log-softmax NLL avoids understating confidently wrong counts. Exposed `capri.py decoder objective-tests`.

Validation: 11 analytic fixture groups passed, including query/target permutation invariance, teacher alignment, padding exclusion, finite CPU/CUDA-AMP gradients, absent-intensity zero gradient, contrastive anchor validity, invalid/capacity rejection and duplicate/miss detection. These are framework checks, not model training or final-test evaluation. Actual decoder integration and measured specialist affinity remain required. Production renderer, sampler, font manifest, controller/reader workflow and active model registry are unchanged. The live 500k corpus job was checked before work; no duplicate process was launched. Single handoff updated with the exact implementation limits.

## 2026-10-07 CAPRI Decoder data loading, CUDA recovery and live supervision

Added maintained decoder shard loading with a pixel-only input boundary, shared aspect-preserving preprocessing, normalized visible/core boxes and per-instance masks, conditional intensity validity and frozen family identities. Added deterministic exact-shape batch plans that preserve declared font/content pairs and resume from the trainer's consumed-batch cursor. Dataset snapshots cannot grow silently; full training requires a complete corpus, and calibration/final-test access requires frozen selection records. Added `data-tests`, `checkpoint-cuda-tests` and `supervise` stage commands; `decoder status` now reflects live locks/progress rather than just the last diagnostic stage event. No frozen renderer/sampler/font source or active model selection changed.

Validation: data tests passed on one completed TRAIN shard (1,000 rows, 53 full sample checks, 270 batches), including all-count coverage, input/preprocessing equality, coordinate round trips, pair integrity, reproducible resume and split/padding guards. A tiny CUDA/AMP checkpoint diagnostic reproduced the next optimizer update exactly and restored raw/EMA/scaler/RNG and the real batch-plan cursor. Nonfinite EMA/optimizer state was refused. This is infrastructure testing, not CAPRI Decoder training. Live supervision verified eight workers, advancing corpus receipts and no alerts. DINOv3 pretrained weights remain missing; full-model losses, matching, GPU memory and end-to-end restart still need validation. Details remain in the single decoder handoff.

## 2026-10-07 CAPRI Decoder 500k corpus readiness and supervised launch

Completed additional official/archive/historical/Debian font acquisition: 1,133/1,139 requested inventory entries resolved, six explicitly unresolved. Indexed 2,514 usable files / 1,907 family names / 1,646 alias-aware split groups. Extended the independent decoder renderer to preserve multicolour backgrounds through hollow/thin/translucent operations, keep decoration behind glyph cores, and derive boxes from exact exported mask values. Sampling retains family/count anchors through rejection and keeps auxiliary font/content siblings within splits.

Added resumable 1,000-image ZIP corpus shards with global pixel deduplication, full source/label provenance, CRC/SHA verification, atomic publication and receipt-based recovery. No package installation; the legacy Python lacks SQLite DLL resolution, so the bounded uniqueness index is reconstructed from durable receipts. Every pilot image/mask/font hash, family split, sibling relationship and intensity bin passed audit; 4,096 unique pilot images cover every family and all 37 effect heads. The 40-image storage test passed injected interruption and unchanged-shard resume. CPU full-epoch checkpoint continuation previously passed. Native target input remains pixels; no VLM, reader training, controller changes or active model replacement.

Production launched 7 October 09:22 UTC via `capri.py decoder generate-corpus --workers 8`, under `runs/capri_decoder_500k_20261007`, with 450k/20k/10k/20k train/dev/calibration/test and a 30 GiB checkpoint reserve. A 30-minute supervision heartbeat is active. Accepted distributions are measured; effect/intensity joint balance is not guaranteed. Difficult synthetic cases can remain ambiguous, requiring uncertainty evaluation. DINOv3 approved pretrained weights are still missing; model training is not running. All source changes and remaining work are tracked in the single decoder handoff; font inventory suffix is unchanged.

## 2026-10-07 CAPRI Decoder acquisition, instance renderer and checkpoint foundation

Added separate `capri.py decoder` stage commands and maintained `src/capri_controller/decoder` modules for pinned font acquisition/indexing, per-instance rendering, deterministic paired sampling, bounded pilot export and atomic full epoch checkpoints. Added decoder-specific machine paths to `configs/local.json`; no active model selection or existing controller/reader workflow changed. New outputs are under `runs/capri_decoder_500k_20261007`. User authorized immediate work, periodic supervision, a500k decoder corpus and specialist training, with no VLMs. A30-minute thread heartbeat is active.

Validation: renderer regression groups passed, the256-image pilot accepted256 unique images without duplicates, and a CPU save/resume test reproduced the next optimizer update exactly while refusing epoch overwrite. Pilot review led to stronger quality gates and paired retries; a revised pilot is being checked. Font acquisition reached1088/1139 inventory entries, with exact historical file identity unproven and unresolved entries reported. Initial font index must be refreshed after final acquisition. DINOv3 official pretrained weights are access-gated and not locally available; only architecture instantiation was checked. No500k production generation or CAPRI Decoder training launched at this point. Remaining work is recorded in the single CAPRI Decoder handoff.

## 2026-10-06 Supplied font inventory requires acquisition

Clarified the active CAPRI Decoder handoff: the user's appended font inventory lists libraries that must be downloaded, not locally installed assets. Historical paths are references only. Added source/license/hash tracking and explicit handling of unavailable fonts; preserved the appended inventory. No fonts downloaded in this update.

## 2026-10-06 Handoff narrowed to CAPRI Decoder

Rewrote `docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md` in place as the single active CAPRI Decoder handoff, preserving its link. Scope is exclusively the next CNN/transformer scheme-understanding hybrid: heads, font libraries/font intelligence, character count, supervision, training and evaluation. Controller redesign, data-allocation changes and reader training remain out of scope; Model4 stays on hold. No code or model changed.

## 2026-10-06 Upstream hybrid/font/controller redesign handoff

Created `docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md` for the user's return to upstream design. Records user-reported Model3 completion and Model4 hold without inventing Model3 results. Restores deferred font intelligence, validation-controller and100k-plus5k planning, separately identifies character-count majority/fallback work, and notes that the user will provide font libraries. Includes actual active hybrid/controller context, pixel-only boundaries, proposed font/library and head audits, design agenda and artifact paths. No training or workflow implementation was changed.

## 2026-10-06 Model 3 per-epoch self-ranking and evaluation handoff

Updated the Model3 handoff after the user accepted full training-pool rescoring after each epoch. The current reader independently reranks95k fixed training images per seed with unaugmented no-gradient inference, smooths difficulty and updates next-epoch sampling; validation never controls rankings. Added numeric decisions still to freeze, RNG/resume and adaptive-state requirements, same-split synthetic selection, frozen900-image real testing, ensemble comparisons and corrected synthetic-derived shoot analysis. Curriculum remains the only experimental change. No Model3 training started.

## 2026-10-06 Model 3 scope corrected: adaptive curriculum only

User confirmed that font intelligence, a fourth validation controller and100k optimization images are deferred to a later loop. Updated the Model3 handoff and methodology's future-work section accordingly. Model3 reuses Model2's exact DS2 data and95k/5k split, initial teacher ranking, architecture, augmentation, optimizer, seeds, initialization and epoch budget; only curriculum changes. Adaptive rule details remain to be specified. This supersedes the earlier next-run allocation note. No training launched.

## 2026-10-06 Model 3 adaptive curriculum handoff

Created `docs/MODEL_3_ADAPTIVE_CURRICULUM_HANDOFF.md` for a receiving session. It records the completed baseline and results, user-agreed font intelligence/dedicated validation controller/100k-plus5k changes, and separates these commitments from proposed adaptive-curriculum choices. Includes implementation pointers, evaluation boundaries, calibration limitations, preservation requirements and communication preferences. No Model3 run or new chat was started.

## 2026-10-06 Next-run methodology changes recorded

Added a planned-differences section to `docs/DARKBAY_THROUGH_MODEL_2.md`: integrate font intelligence, use another controller dedicated to validation generation, and optimize on100000 training images with an additional5000 validation images. Recorded the user's clarification that the completed95k/5k Model2 allocation differs from the intended100k-plus-validation design but is accepted as the final current run. Historical methods and results remain unchanged. Future controller/font-intelligence implementation details remain unspecified; no new training or generation was started.

## 2026-10-06 Complete DarkBay through Model 2 methodology record

Created `docs/DARKBAY_THROUGH_MODEL_2.md` and linked it from README. The record traces actual active checkpoint provenance, upstream feature/mask/encoder training, 100 pixel references, model-derived length, per-image cleanup, three controller objectives/search/selection, frozen300-entry recipe mixture, both100k datasets, reader architecture and training differences, curriculum/augmentation, validation/selection, historical real tests, failed weight averaging, epoch diagnostics, corrected synthetic-calibrated policies and separate GLM benchmark. Tables of full DS1 epoch results/losses and DS2 validation/losses/interim results are extracted from saved artifacts, with absolute evidence links and snapshot hashes. It distinguishes historical confidence sweeps from corrected policies, discloses reused validation for DS2 policy calibration, and does not claim unverified original reference-sampling details or interpret “tab3” as a distinct model. No model, dataset or result was changed.

## 2026-10-06 DS2 synthetic-calibrated shoot analysis

Added `tools/ds2_synthetic_shoot_policy.py` to infer the three synthetic-selected checkpoints (epochs18/20/18) on all5000 unaugmented synthetic validation images using the same FP32 batch64 decoder as real evaluation. Verified pixel and checkpoint hashes and froze each seed's predictions with receipts. Extended `tools/synthetic_shoot_policy.py` with an optional Model2 path while preserving DS1's default method and outputs. Used exactly the same2500 discovery/2500 calibration IDs, character reliability score, majority/mean-confidence ensemble, predefined90/95/98/99% targets, minimum100 attempts and descriptive Wilson95 lower-bound criterion as DS1. No weights changed. This validation set also selected DS2 checkpoints, so calibration is not independent of model selection.

Policy frozen before reading real predictions; real attempt decisions frozen before accessing real labels. Independently checked split identity/disjointness, freeze hashes,900 unique real IDs, all scores/decisions and result counts. Output `runs/Darkbay_ds2_noFont-intel/synthetic_validation_shoot_policy`; inference receipts in `synthetic_policy_inference`. Policy SHA270059e6c591932d89012d8a6b47d3af4a92f4bfffb539f6066df8a67bad2989.

Synthetic discovery2428/2500; calibration baseline2429/2500 (97.16%). Targets90/95 share a threshold and yield real565/900 (62.78%,100% shoot). Target98: synthetic2409/2444 (98.57%); real535/772 (69.30%,85.78% shoot). Target99: synthetic2312/2325 (99.44%); real411/530 (77.55%,58.89% shoot). Compared with DS1's99% target real232/339 (68.44%,37.67% shoot), DS2 attempts more with higher observed precision; synthetic targets are not real guarantees. No real-guided threshold adjustment. Prior real-confidence sweep remains distinct and preserved.

## 2026-10-06 DS2 overnight run completed and verified

All three seeds completed20 epochs; pipeline completed06:35:58 local. Verified dataset LMDB and sample/curriculum hashes, selected and final checkpoint hashes, all60 epoch checkpoint files, all60 validation records, all four frozen real prediction hashes and independently recomputed correct counts. Saved `runs/Darkbay_ds2_noFont-intel/completion_verification.json`. Dataset100000 fresh unique images, zero recorded DS1/target overlap; buckets easy33334/medium33333/hard33333; train95000/validation5000.

Synthetic-selected checkpoints: seed48 epoch18 validation96.64%; seed123 epoch20 validation96.62%; seed456 epoch18 validation96.58%. Real DarkBay singles566/900 (62.89%),520/900 (57.78%),524/900 (58.22%); ensemble565/900 (62.78%), versus DS1 epoch20 ensemble496/900 (55.11%), +69 correct and +7.67 points. This ensemble combines independently synthetic-selected epochs, not uniformly epoch20. Interim real scores did not determine selection.

The saved final shoot_rate.json is the original protocol's fixed real-confidence sweep, NOT the later clarified synthetic-derived rejection policy. At threshold0.99 it reports178/209 (85.17%,23.22% shoot rate). Preserve this diagnostic with that distinction; no synthetic-calibrated DS2 rejection result is claimed. Different augmentation, validation and data prevent attributing improvement solely to curriculum. Completed run and artifacts preserved; hourly monitor paused after completion.

## 2026-10-06 DS2 status: all-seed epoch19 comparison

Seeds48/123 completed20 epochs; seed456 was training epoch20. Evaluated the latest shared completed epoch19 for all three seeds with the existing interim protocol. Real correct/900:568,514,535 respectively; synthetic validation96.46%,96.62%,96.54%. Froze a whole-string-majority ensemble with existing mean-confidence tie-break and seed48/123/456 exact-tie order before scoring real labels. Ensemble567/900 (63.00%), versus DS1 epoch19 ensemble502/900 (55.78%): +65 correct, +7.22 points. Corresponding DS1 singles493,437,477/900. Prediction source hashes verified and ensemble freeze/summary saved under `runs/Darkbay_ds2_noFont-intel/interim_evaluation/ensemble_epoch_19`. No real-guided selection or training changes.

## 2026-10-06 DS2 status: interim epoch17

Verified orchestrator running at seed48 epoch18. Evaluated latest completed epoch17 with the existing interim protocol: real DarkBay577/900 (64.11%) versus DS1 seed48 epoch17 494/900 (54.89%), +83 correct and +9.22 percentage points. Synthetic validation96.16% on5000 images. Real accuracy recovered from epoch16 but remains below epoch15's607/900. Results in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_17`. Training and synthetic-validation checkpoint selection unchanged; seeds123/456 remain queued.

## 2026-10-06 DS2 status: interim epoch16

Verified orchestrator running at seed48 epoch17. Tested latest completed epoch16 using the existing interim protocol: real DarkBay560/900 (62.22%) versus DS1 seed48 epoch16 488/900 (54.22%), +72 correct and +8.00 percentage points. Synthetic validation96.32% on5000 images. Real accuracy decreased from epoch15's607/900 despite improved synthetic validation; no training or checkpoint-selection changes were made based on real results. Results in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_16`. Seeds123/456 remain queued.

## 2026-10-06 DS2 status: interim epoch15

Verified orchestrator running at seed48 epoch16. Evaluated latest completed epoch15 with the existing interim protocol: real DarkBay607/900 (67.44%) versus DS1 seed48 epoch15 453/900 (50.33%), +154 correct and +17.11 percentage points. Synthetic validation96.00% on5000 images. Highest DS2 interim real score tested so far; single reader, with seeds123/456 still queued. Results saved in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_15`. Training and synthetic-validation checkpoint selection unchanged.

## 2026-10-06 DS2 status: interim epoch13

Confirmed seed48 epoch13 saved and epoch14 running. Evaluated epoch13 using the existing interim protocol: real DarkBay564/900 (62.67%), versus DS1 seed48 epoch13 478/900 (53.11%), +86 correct and +9.56 percentage points. Synthetic validation95.56% on5000 images. This is the highest of the DS2 interim checkpoints tested so far; real results do not affect checkpoint selection or training. Results in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_13`. Seeds123/456 remain queued.

## 2026-10-06 DS2 status: interim epoch12

Verified the orchestrator remains running at seed48 epoch13 and evaluated completed epoch12 with the existing frozen interim protocol. Real DarkBay526/900 (58.44%), versus DS1 seed48 epoch12 406/900 (45.11%): +120 correct, +13.33 percentage points. Synthetic validation95.06% on5000 images. Latest real score is below the previously tested epoch10 result548/900; this diagnostic does not change training or synthetic-validation checkpoint selection. Other seeds remain queued. Results saved in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_12`.

## 2026-10-06 GLM-OCR synthetic-derived shoot policy

Added `src/capri_controller/training/glm_shoot_policy.py` and `tools/glm_shoot_policy.py` for the explicitly requested external GLM benchmark. This does not add a VLM to CAPRI. Used the same 5000 held-out synthetic validation images and 2500 discovery / 2500 calibration partition as the DS1 policy. Official pinned GLM-OCR, original RGB, Text Recognition prompt, BF16 SDPA, greedy generation, batch8, max64 tokens. Saved generated-token probabilities; heuristic score multiplies joint non-special-token confidence by discovery-derived Laplace-smoothed emitted-character reliability. No internal-space, case or length correction. Character analysis includes insertions, substitutions and deletions.

Predefined 90/95/98/99% targets require at least100 calibration attempts and descriptive Wilson95 lower bounds. Only90% was supported: threshold0.5500554611137143, synthetic calibration695/754 (92.18%,30.16% shoot rate). Calibration without filtering1588/2500 (63.52%); discovery1639/2500 (65.56%). Policy frozen before real inference, real decisions frozen before loading real labels. Unchanged policy on900 historical real DarkBay images:160/174 (91.95%,19.33% shoot rate); no filter645/900 (71.67%). Higher targets were unavailable, not zero-accuracy policies. Six synthetic outputs reached the token limit; none real. Batch8 changed28 real predictions compared with prior batch1 but aggregate accuracy remained645/900.

Outputs: `runs/glm_ocr_synthetic_shoot_policy`. Independently verified all prediction/policy/decision hashes, unique5000/900 IDs, recomputed real scores, decisions and160/174 count. Policy SHA256:36ec9f64ef74cfbf3856ab909548ca45d14f41a3393652cde871e1ec201f1d87. Synthetic top errors included extra spaces/formatting and O-to-0 substitution. No threshold tuning used real labels. Post-selection bounds are descriptive and the real set was previously examined. DS2 training remained running; GLM peak allocated GPU memory about2.44GB.

## 2026-10-06 DS2 status: interim epoch10

While seed48 trained epoch11, tested completed epoch10 using the existing interim protocol. Real DarkBay:548/900 (60.89%), versus DS1 seed48 epoch10 426/900 (47.33%): +122 correct, +13.56 percentage points. Synthetic validation94.36% on5000 images. Output: `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_10`. Training and synthetic-validation checkpoint selection remain unchanged; this is a single-seed historical-test diagnostic.

## 2026-10-06 DS2 status: interim epoch9

While seed48 trained epoch10, tested the latest completed epoch9 using the existing interim evaluator. Real DarkBay:517/900 (57.44%) versus DS1 seed48 epoch9 473/900 (52.56%): +44 correct, +4.89 points. DS2 synthetic validation93.10% on5000 images. Saved in `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_9`. This single checkpoint exceeds DS1's previously observed best same-seed score506/900, but is an interim historical-test result and does not alter model selection. Training continues unchanged.

## 2026-10-06 DS2 status: interim epoch8

During seed48 epoch9, evaluated latest completed epoch8 under the existing frozen interim protocol. DS2 scored485/900 (53.89%) on real DarkBay versus DS1 seed48 epoch8 371/900 (41.22%): +114 correct, +12.67 percentage points. DS2 synthetic validation was92.96% on5000 images. Saved at `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_8`. Training and synthetic-validation checkpoint selection remain unchanged; other seeds are queued.

## 2026-10-06 Synthetic-derived DS1 shoot policy, then real test

Implemented the user's clarified procedure in `tools/synthetic_shoot_policy.py`. Used only the5000 DS2 synthetic validation images and frozen DS1 epoch20 teacher outputs, not DS1 training data. Deterministic split:2500 discovery images for predicted-character reliability/confusion analysis and2500 separate calibration images for thresholds. Levenshtein alignment counts correct/substituted/inserted emitted characters and deletions. Score is the product of Laplace-smoothed emitted-character reliability times the supporting models' mean sequence confidence divided by three. This is a risk-ranking heuristic, not a calibrated success probability; synthetic true-label NLL and p_correct never enter inference features. This differs from the earlier joint-confidence-only real-test sweep.

For predefined90/95/98/99% synthetic targets, selected maximum calibration coverage with >=100 attempts and a descriptive Wilson95 lower bound meeting the target. Bounds after threshold search are not formal post-selection guarantees. Wrote and hashed the policy before accessing real predictions; wrote and hashed all real attempt decisions before loading real labels. Original prediction hashes and synthetic manifest hash verified; real no-filter ensemble reproduced496/900. No model weights, training or real-guided threshold adjustments occurred. Outputs in `runs/parseq_darkbay_100k_20261005/synthetic_validation_shoot_policy`.

At synthetic99% target: calibration1611/1619 (99.51%,64.76% shoot rate); unchanged real policy232/339 (68.44%,37.67% shoot rate). At98% target: synthetic2286/2319 (98.58%); real353/543 (65.01%). At95% target: synthetic2370/2472 (95.87%); real494/884 (55.88%). At90% target: synthetic2375/2500 (95.00%); real496/900 (55.11%). The rejection policy improves attempted real precision but synthetic precision does not transfer. Both synthetic and historical-real evidence remain saved, with no replacement of previous reports.

## 2026-10-06 Status comparison preference and epoch6 test

User requested that future status answers automatically include the latest completed DS2 checkpoint's real DarkBay score compared with DS1 at the identical epoch/seed. Recorded this preference in AGENTS.md, retaining quiet routine heartbeat behavior and forbidding test-informed training changes. Latest completed checkpoint at inspection: seed48 epoch6 (epoch7 in progress). Interim evaluation returned432/900 =48.00%, versus DS1 seed48 epoch6 350/900 =38.89%: +82 correct, +9.11 percentage points. DS2 synthetic validation at epoch6 was90.84% on5000 images. Predictions, stable checkpoint hash and source image hashes were verified by the interim evaluator; results saved under `interim_evaluation/seed_48_epoch_6`. Training remains unchanged.

## 2026-10-06 DS2 interim epoch5 test

User requested testing the latest saved weights while training continued. At inspection, seed48 was training epoch6; epoch5 was the newest completed immutable checkpoint. Added `tools/evaluate_ds2_checkpoint.py` and tested that checkpoint on the same900 historical DarkBay images with identical preprocessing and decoding. Checkpoint hash was stable, all image RGB hashes checked, and predictions frozen before scoring. Result:427/900 =47.44% real exact accuracy, CER12.55%; corresponding synthetic validation accuracy88.82% on5000 images. Saved under `runs/Darkbay_ds2_noFont-intel/interim_evaluation/seed_48_epoch_5`. GPU memory capped at20%; training continued without configuration or checkpoint-selection changes. This interim test must not be used to tune the curriculum or select a checkpoint.

## 2026-10-05 GLM-OCR external baseline

Completed and independently recounted: **645/900 = 71.67%** case-sensitive whole-CAPTCHA accuracy, versus DS1 epoch20 ensemble496/900 =55.11% (+149 correct, +16.56 percentage points). Raw and edge-whitespace-trimmed exact-match counts are identical; no prediction hit the64-token limit. Corpus CER13.18%. All900 predictions and their SHA256 were verified after scoring. Inference took about295seconds; examples show the first12 images in fixed manifest order. DS2 continued generation and packing during this test.

User explicitly requested downloading and testing GLM-OCR on DarkBay. Downloaded official `zai-org/GLM-OCR` revision `2e85a62840ccac27daa451df36c736c4636b8628` to `models/external/GLM-OCR`; recorded repository metadata, sizes, hashes, and verified the published LFS weight hash. Sources: https://huggingface.co/zai-org/GLM-OCR and https://huggingface.co/docs/transformers/model_doc/glm_ocr. This is an explicitly requested pretrained multimodal baseline, separate from the no-VLM CAPRI controllers and DS2 pipeline; its pretraining data overlap is unknown.

Created isolated Python3.12 environment `.runtime/glm_ocr` with CUDA PyTorch2.6.0/torchvision0.21.0 and Transformers5.17.0; dependency snapshot `maintenance/requirements-glm-ocr.txt`. The existing DS1/DS2 environment is unchanged. Added `tools/download_glm_ocr.py`, `training/glm_ocr.py`, and `tools/evaluate_glm_ocr.py`. Run outputs are in `runs/glm_ocr_darkbay_20261005`. GPU allocation is capped at30% to allow DS2 to continue; observed peak tensor memory was about2.2GB during inference.

Fixed protocol: same900 real images excluding100 controller references; original RGB images with official processor; prompt `Text Recognition:`; greedy decoding with64-token cap; case-sensitive exact match after edge-whitespace trimming only. Raw answers and truncation flags are retained. All predictions are frozen and hashed before scoring labels are read. No fine-tuning, prompt search, case repair or target length supplied. Per-image journal supports continuation after infrastructure failure; completed evaluation refuses overwrite.

## 2026-10-05 DS2 curriculum job resumed and launched

Launch confirmed: orchestrator PID41664 at23:01 local time; first5000 images generated successfully. An hourly thread follow-up, `monitor-darkbay-ds2-curriculum-run`, monitors completion/failure and remains quiet on ordinary progress. It must be disabled after completion or if the user stops the job. Current status and launch receipt take precedence over this historical launch note.

User resumed the previously stopped job. Added the maintained `capri.py darkbay-ds2` command and `training/ds2.py`, `ds2_data.py`, `ds2_evaluate.py` with thin launch/readiness tools. The staged runner generates 100k fresh images under `runs/Darkbay_ds2_noFont-intel`, using the same 300 frozen recipes and controller proportions as DS1. `corpus.py` now accepts `CAPRI_CORPUS_SEED_BASE`, defaulting to its original value; DS2 uses10000000000 and verifies seed identity when reusing shards. DS1 artifacts and defaults are preserved. Full RGB audit plus DS1/target duplicate checks gate subsequent stages.

Ranking uses the three frozen DS1 epoch20 models, equal per-character probability averaging, and synthetic-answer NLL tertiles following the earlier Model2 teacher-bucket design. Earlier real-test epoch sweeps do not promote new teachers. A deterministic bucket/controller-stratified5000-image synthetic validation holdout is excluded from optimization. Three fresh readers (seeds48/123/456) use the DS1 architecture/optimizer and20epochs: easy only, then50/50 easy/medium, then15/25/60 easy/medium/hard. Augmentation copies Model2 v2 operations/parameters with explicit seeding. Every epoch is saved, validation runs every epoch, and best synthetic-validation checkpoints are selected; final epoch20 is retained too. This differs from DS1's all100k/no-validation/final-checkpoint protocol and cannot isolate curriculum alone. Post-training real evaluation and shoot-rate reporting are automatic; no target labels enter generation, ranking or training.

Validation before launch: fresh-seed renders across the three recipe ranges succeeded; sampler transitions and deterministic draws checked; disposable3-step GPU preflight exercised augmentation, training, validation and best/final checkpoint saving with finite weights. Preflight used a small DS1 fixture and is clearly separated from DS2 production. A further smoke test on actual DS2 is required automatically before full training. New job status, per-stage logs, launch PID, protocol and readiness receipts are saved in the run directory. Original stop occurred before any dataset generation; this entry documents the subsequent user-authorized start.

## 2026-10-05 DS1 saved-epoch diagnostic

At the user's request, tested all 57 saved checkpoints (epochs 2–20, seeds 48/123/456) on the same frozen 900-image real DarkBay set. Added `training/evaluate_epochs.py` and `tools/evaluate_parseq_epochs.py`; outputs, individual predictions, hashes, CSV/text results, and plot are in `runs/parseq_darkbay_100k_20261005/epoch_evaluation_darkbay`. Epoch 1 has no separately saved checkpoint. All checkpoints were loaded strictly, checked for finite tensors and recorded epoch, and hashed. All target RGB hashes were verified. All checkpoint and same-epoch ensemble predictions were frozen before scoring labels were loaded. Epoch-20 predictions reproduced the original seed outputs exactly; ensemble count reproduced 496/900. DS2 generation and training remain stopped.

Observed peaks: seed48 epoch14 506/900 (56.22%); seed123 epoch14 455/900 (50.56%); seed456 epoch19 477/900 (53.00%). Same-epoch majority-vote ensemble peaked at epoch19, 502/900 (55.78%), versus epoch20 496/900 (55.11%). Saved final logged smoothed training losses alongside the real-test curve. The trajectory is noisy and does not show a common, established overfitting onset across seeds. This is retrospective analysis of already examined real test data; peak selection would be test-informed and is not independent validation. No deployed checkpoint or earlier baseline was replaced.

This records changes to the maintained CAPRI code, selected models and workspace organization. Add dated entries with the reason, affected paths, verification and any unresolved limitations. The generated workspace catalog is an inventory, not a substitute for this change record.

## 2026 10 05 PARSeq training preparation

Requested: train on the completed 100k DarkBay corpus using the newer thesis methodology. Prepared `runs/parseq_darkbay_100k_20261005` with the fixed 20-epoch schedule, seeds 48/123/456, and final `latest.pth` selection. This applies the newer benchmark's completion rule to its PARSeq configuration; the primary benchmark architecture was NRTR.

Converted all 100,000 images losslessly from the 300 ZIP shards into LMDB, retaining every sample and the 33,334/33,333/33,333 controller mixture. Verified source ZIP hashes, decoded RGB hashes, synthetic text/spec consistency and uniqueness. Saved per-sample provenance and a hashed data receipt. No target labels or evaluation dataset enter the trainer, and low strict-readability samples were not silently removed.

Added preparation/training modules under `src/capri_controller/training`, plus `capri.py prepare-parseq` and `capri.py train-parseq`. The runner requires an explicit pretrained checkpoint or explicit `--from-scratch`; it does not use OpenOCR's silent missing-file fallback. It seeds the decoder's otherwise independent NumPy permutation generator, checks pretrained encoder coverage, and requires final epoch20 with finite weights. Separate seed processes preserve one-model-per-GPU execution.

Installed official SHA256-verified `lmdb==2.0.0` and `rapidfuzz==3.9.7` wheels into workspace-local `.runtime/parseq_packages`. Global environments and external source repositories were unchanged. Set data-loader workers to zero for Windows LMDB compatibility and made all charset paths and space-token settings consistent.

The complete loader initializes with 100,000 images. A discarded forward/backward readiness check passed at batch128, RGB32x128, using about7.2GiB GPU memory. It performed **zero optimizer steps**, saved no model weights and did not start the requested training.

At the end of preparation, the required `scene-text-base.pth` was absent from the thesis's staged inputs and searched local model locations. The run initially waited for that checkpoint or a change to initialization.

### Training launch

The user subsequently instructed “Start the training.” Proceeded with random initialization, explicitly disclosed before launch because the pretrained checkpoint remains unavailable. This is a change from the thesis's pretrained initialization, not a claim of exact benchmark reproduction. The fixed20-epoch schedule, seeds48/123/456, complete100k corpus and final-checkpoint rule remain unchanged.

Launched `capri.py train-parseq --from-scratch` as a hidden background process, one seed at a time. The launch receipt records the orchestrator PID and timestamp in `runs/parseq_darkbay_100k_20261005/launch.json`. Verified seed48 on CUDA at epoch1, optimizer step50, finite reported loss3.829776. Seeds123 and456 follow automatically if the preceding run completes successfully. No real-target labels or evaluation enter training. Completion is not yet claimed; consult `status.json`, per-seed logs and final completion receipts.

### Completed training and requested real evaluation

All three seeds completed epoch20 and15,640 optimizer steps each. Verified the hashes of their final `latest.pth` checkpoints. Total recorded training time was approximately5hours43minutes. Initialization was from scratch, as disclosed at launch.

Added `training/evaluate_parseq.py` and the `capri.py test-parseq` command. It selects all unique images from the existing labeled DarkBay folder except the100 controller-reference RGB hashes. This yielded900 images; no synthetic training-image hash overlaps. Evaluation preprocessing matches training: RGB, bicubic32x128, normalization(.5,.5). The full saved62-character decoder is retained, with no enforced output length, case correction, ensemble or best-seed selection.

Each final checkpoint was hash-verified and loaded strictly. All three image-only prediction files were frozen and hashed before parsing filename labels for scoring. Exact case-sensitive string scores were492/900 (54.67%) for seed48,439/900 (48.78%) for seed123, and474/900 (52.67%) for seed456. Mean52.04%, sample standard deviation2.99percentage points. Character error rates were9.79%,11.28% and10.33%. The original labels contain899 six-character answers and one seven-character answer (`0UHHAA1`); retained without repair or exclusion.

These are historical project images, not a newly collected untouched benchmark. The evaluation excludes this run's controller inputs but does not assert that earlier experiments never inspected these images. No retraining or selection followed the scores. Artifacts are under `runs/parseq_darkbay_100k_20261005/evaluation_darkbay`, including the protocol, prediction freeze, per-image results, summary, examples and HTML report.

## 2026 10 05 Controller code and model consolidation

### Completed changes

- Created `src/capri_controller` and extracted the active implementation from the old MVP dependency chain. The package uses direct relative imports and no longer imports old MVP or learned-controller experiment modules.
- Separated comparator features, network definitions, renderer controls, workflow orchestration, corpus generation, reporting and inventory code. Reduced the encoder's dependency on the old training grammar to its checkpoint-compatible dimension schema.
- Added the root `capri.py` CLI with `status`, `models`, `verify`, `inventory`, `controllers`, `generate`, `audit` and `report` commands. New controller runs require a new output directory.
- Centralized machine-specific dependencies in `configs/local.json` and copied the existing 100-image DarkBay input manifest to `configs/targets/darkbay.json`.
- Created four hash-verified active model copies under `models/active`, with roles, original paths and expected hashes in `models/registry.json`. The selected neural models did not change.
- Added a metadata-only historical checkpoint catalog and a categorized workspace catalog. The initial scan found 125,380 files and 295 checkpoint files; inventory totals change as artifacts are added.
- Moved 34 root-level logs into `archive/root_logs_20261005`, verifying every moved file's hash. Exact moves are recorded in `maintenance/root_log_moves_20261005.json`.
- Added the root README, architecture/model guide, future-run convention and cache/weight/run Git ignore rules. Existing historical folders, models and reports remain available.
- Recorded the tested dependency versions in `requirements-controller.txt` and `maintenance/environment_versions.json`. No packages were installed or changed. Added workspace maintenance instructions so subsequent changes update this log.
- Corrected the new pipeline's diversity statistic to exclude sampled text, made the gallery heading generic, and removed the old assertion against a specific DarkBay histogram. The live model votes still determine length; the 60% rule and no-majority failure remain unchanged.

### Verification

- All four centralized checkpoints match their recorded SHA256 hashes; both external source directories are present.
- Separate legacy and maintained processes produced exactly identical hybrid descriptions on three target inputs and identical pixels/text for nine renderer outputs across all three controllers.
- All 26 saved custom/DINO feature arrays match exactly, with maximum difference zero.
- The maintained renderer passed a two-worker Windows multiprocessing check. The maintained process imported no old MVP modules.
- All source files used for extraction still match their original hashes. Original checkpoint files were retained.
- The maintained CLI decoded and audited all 100,000 existing corpus images successfully, including unique RGB hashes, text/spec consistency, model-derived length and controller allocation.
- A separate two-worker export test saved four fresh images under a configured run directory, verified PNG/label integrity and confirmed identical resume behavior.
- Inspection/report commands passed, all package sources parse, and attempting to fit controllers into the completed run directory is rejected before fitting starts.
- Regression artifacts: `maintenance/validation/summary.json`, plus the separate legacy/current results and feature arrays. Extraction provenance: `maintenance/extraction_20261005.json`.

Final command verification and the full corpus audit are recorded in `maintenance/final_checks.json`. The final inventory contains 125,406 files and 295 checkpoint files, with zero scan errors. Maintained source hashes are in `maintenance/maintained_source_hashes_20261005.json`.

### Limits and next maintenance work

This cleanup consolidates the active workflow and indexes the broader workspace. It does not reorganize every old experiment internally: those directories and root-level historical helper scripts still contain fixed paths. They are indexed and preserved rather than silently broken. Root logs are the only relocated historical files.

The renderer, feature-model source, font assets and target pixels still depend on local external paths. Historical checkpoints are inventoried by path and size, not deduplicated or deleted. Rejected controller experiments remain available as evidence. The frozen `CAPRI` releases were not modified.

No full controller refit or new 100,000-image corpus was needed for this structural change. Regression validation covers representative inputs, model features and rendering; it does not prove downstream solver performance. The existing DarkBay corpus still has only 36.3% strict readability. Cleanup does not improve that quality metric.
# Equal-weight PARSeq average — 2026-10-05

At the user's request, averaged the fixed epoch-20 reader checkpoints for seeds 48, 123 and 456 with equal one-third weights. Added `src/capri_controller/training/average_parseq.py` and `tools/average_parseq.py`; outputs are in `runs/parseq_darkbay_100k_20261005/equal_weight_average`. Original checkpoints and previous evaluations are preserved. Run with the configured Python and `tools/average_parseq.py`; the script refuses to overwrite an existing experiment.

Verified all three source SHA256 hashes, architecture/decoder agreement, state keys/shapes/dtypes, finite floating tensors, equal nonfloating buffers, exact save/reload equality and strict model loading. Floating tensors were averaged with float64 accumulation and cast back to their original dtype. No optimizer states were averaged, and no fine-tuning or normalization recalibration was performed. The single averaged model used the same 900-image manifest, preprocessing and decoder as the individual evaluations; predictions were frozen before scoring labels were loaded.

Result: **0/900 (0.0%)** case-sensitive exact match; CER **253.95%** (insertions can make CER exceed 100%). Predicted lengths were 6 for 484 images and 26 for 416 images. The individual checkpoints scored 54.67%, 48.78% and 52.67%. These runs started independently from scratch; direct parameter averaging does not align their independently learned internal representations. This experiment failed and does not replace any selected model. It remains a follow-up on historical DarkBay data after individual scores were known. Averaged checkpoint SHA256: `65e398581897a11b39823501e6649d981237649c7c73083a84b0e4c80b5ffb96`.
# PARSeq prediction ensemble — 2026-10-05

Combined the three original frozen reader predictions at the user's request. Added `src/capri_controller/training/ensemble_parseq.py` and launcher `tools/ensemble_parseq.py`. Run outputs are preserved in `runs/parseq_darkbay_100k_20261005/prediction_ensemble`; existing output directories are rejected. The rule was specified before scoring: whole-string majority vote, highest saved sequence confidence when all three strings differ, fixed seed order 48/123/456 for exact confidence ties. No retraining, weight changes or additional inference occurred.

Validated source prediction SHA256 hashes, 900 unique matching image IDs, and frozen ensemble predictions before loading scoring labels. Result: **496/900 = 55.11%** case-sensitive whole-string accuracy; CER **9.74%**. Three-way agreement on 510 images, two-way agreement on 313, confidence tie-break on 77. This is four more correct than the best individual seed (492/900), a 0.44 percentage-point increase. Confidence is not calibrated across seeds, and this remains a follow-up evaluation on historical DarkBay data after earlier results were known. No ensemble rule was selected by comparing alternatives on these labels.
# DarkBay shoot-rate analysis — 2026-10-05

Reproduced the confidence-threshold metrics from `E:/pranav/model 1/src/eval/selective_with_ecps.py` at the user's request. Added `training/shoot_rate.py` and `tools/shoot_rate.py`; outputs are under `runs/parseq_darkbay_100k_20261005/shoot_rate_analysis`. All three fixed readers were rerun solely to recover products of emitted character probabilities (the previously saved OpenOCR confidence was a character mean). Every replayed string matched its frozen original prediction. Checkpoint and source prediction hashes were verified. No forced length, retraining or answer changes were introduced.

For the existing majority-vote ensemble, confidence is the sum of joint confidences of models supporting its frozen winning string divided by three; disagreeing models contribute zero. This explicit adaptation rewards agreement but is not calibrated probability or the old experiment's probability-level ensemble. Confidence predictions and the rule were saved before scoring labels were loaded. Used the old 13 thresholds from 0 to .99; saved all seed and ensemble counts, shoot rates, attempted accuracy, ECPS, and a plot. Ensemble baseline remains 496/900. At .90, attempts 398/900 and correct 293/398 (73.62%); at .99, attempts 269/900 and correct 211/269 (78.44%).

This is a descriptive historical-test sweep, not held-out threshold calibration. ECPS counts all images encountered, including skipped images, so increasing rejection cannot improve it for fixed predictions; rejecting can improve accuracy among attempts. No operating threshold or model was selected from these labels. Undefined ratios are null. All original artifacts remain intact.
# Saved DarkBay results and font-intelligence plan — 2026-10-05

Saved the full ensemble correct/attempted table in `docs/DARKBAY_SHOOT_RATE_RESULTS.md`. The user subsequently confirmed these results are **without font intelligence**, with integration planned next; the results document now explicitly records that baseline. Counts match the saved shoot-rate summary. Documentation only; no model or results changed.
# Training dataset named Darkbay_ds1_noFont-intel — 2026-10-05

Assigned the user's exact name `Darkbay_ds1_noFont-intel` to the complete 100,000-image corpus used by all three PARSeq seeds. Added `capri_darkbay_workflow/dataset_identity.json`, recording corpus and training LMDB paths and their existing provenance hashes; linked the name in README and the DarkBay results document. Identity values were checked against the training data receipt. Kept storage paths and frozen manifests unchanged to preserve training provenance and existing scripts. No images, models or scores changed.

## 2026-10-06 Model 3 adaptive curriculum implementation

Added `training/ds3.py`, `ds3_data.py`, `ds3_checks.py`, `ds3_evaluate.py`, `ds3_policy.py` and `capri.py darkbay-ds3`. New outputs are isolated under `runs/Darkbay_model3_adaptive_20261006`. Model2 corpus, exact95k/5k split and all non-curriculum YAML settings are reused; original data, code, checkpoints and release artifacts are preserved. No new controller, font intelligence or data was introduced. Numeric choices were fixed in the saved protocol before training: training-only NLL percentile difficulty,0.5 EMA, adaptive tertiles,20–50% bucket mass and0.1 update limit after initialization. See `docs/MODEL_3_ADAPTIVE_PROTOCOL.md`.

Behavior checks verified exact Model2 epoch1 draws, exclusion of validation, finite fail-closed scoring, sampling bounds and restart sampling. GPU smoke training was deliberately interrupted at epoch1 and resumed through epoch2; model tensors, scheduler, adaptive state and PyTorch RNG matched an uninterrupted smoke exactly. A smoke-discovered augmentation RNG traversal error was repaired; the failed smoke is retained separately. The copied shoot-policy procedure exactly reproduced Model2 policy, thresholds, discovery/calibration IDs, real decisions and results using existing baseline predictions. Full-pool GPU scoring readiness and production status are recorded separately; this entry does not claim completed Model3 training or real results.

Each completed epoch saves optimizer, scheduler, scaler, RNG and adaptive state atomically, plus immutable epoch weights and scoring hash receipts. All selected reader predictions and ensemble predictions freeze before real scoring. The synthetic-derived shoot policy and label-free real attempt decisions freeze before scoring. Known limits: historical repeatedly examined real set, calibration validation reused for checkpoint selection, and added full-training-pool scoring compute. The orchestration includes all three seeds, real testing, ensembling, synthetic shoot calibration, comparisons, curves, verification and final catalog refresh.

### Model 3 production launch

All95000 images passed the full-pool GPU score check with finite values and unchanged model weights (187.1seconds). Launched `capri.py darkbay-ds3` as a hidden background process at2026-10-06 09:32local; launch receipt records PID14012. Verified seed48 epoch1 on CUDA,20,076,927 parameters,742 batches per epoch and the initial easy-only distribution. `launch_verification.json` contains readiness receipts and matching frozen code/protocol hashes. Full training, real results and shoot-policy results remain pending; the sequential runner performs these stages automatically.

## 2026-10-06 Requested Model 3 all-epoch real diagnostics

At the user's explicit request to test all epochs, added `training/ds3_epoch_diagnostics.py`. It tests committed Model3 epoch checkpoints on the existing900-image DarkBay manifest and obtains same-seed/epoch Model2 results through the existing evaluator, reusing hash-verified saved tests. Model1 comparisons use its saved epoch sweep (epoch1 unavailable). Prediction freezes precede label scoring. The sweep checks for newly committed checkpoints before finishing, but does not wait for future training epochs. Outputs: `runs/Darkbay_model3_adaptive_20261006/all_epoch_diagnostics`, with CSV, Markdown, summary and execution status.

This module is not imported by training; all frozen production source hashes still match. Architecture, curriculum, sampling, optimization and synthetic checkpoint selection remain unchanged. Evaluations use a separate process with the existing20% CUDA allocator cap and can compete for GPU time; they do not pause or alter training. Historical-test comparisons remain descriptive and must not be used to select checkpoints. Completion counts and results are recorded in the sweep status and report.

The requested sweep completed21 available Model3 checkpoints: seed48 epochs1–20 and seed123 epoch1, with matching Model2 tests and saved Model1 comparisons. Verification is in `all_epoch_diagnostics/verification.json`. Seed48 synthetic-selected epoch19 scored563/900 (62.56%); epoch20 scored569/900 (63.22%). All frozen training source hashes remain unchanged. Unfinished future checkpoints were not tested.

## 2026-10-06 User designates Model 3 final epoch20

The user explicitly instructed: “Yeah for now the last epoch is checkpoint.” Recorded `checkpoint_selection_override.json` and the reader experiment selection in `models/registry.json`. Evaluation now resolves immutable epoch20 archives for all three seeds, including the ensuing ensemble and synthetic shoot-policy inference. Added `ds3_selection.py` and updated only the evaluation dispatcher/report. Existing synthetic-best checkpoints, seed completion receipts, original protocol and pre-amendment source manifest remain preserved. Updated source hashes with a separate amendment receipt and code snapshot. Training, curriculum, optimizer and data remain unchanged; no running process was restarted.

Verified the resolver on completed seed48: selects epoch20 and its actual validation metrics, retains original epoch19 selection provenance, checks the epoch20 hash and leaves best.pth unchanged. Seeds123/456 remain pending. This is a user selection amendment after real seed48 epoch results were viewed, not the original predeclared synthetic selection. The final report records the distinction.

## 2026-10-06 Model 3 interruption recovery

At21:08local, no Python training process remained and the GPU was idle. Seed456 logs ended at18:27:22 during epoch11, step7900, without a Python traceback; cause is unconfirmed. Seeds48/123 were already complete. Verified seed456 atomic latest checkpoint against its epoch10 receipt (step7420), protocol hash, optimizer/scheduler/scaler/RNG/adaptive state, and frozen code hashes. Preserved the stale lock, prior status/log and interrupted epoch11 sampling records under `recovery_20261006_2108`.

Relaunched `capri.py darkbay-ds3` as a hidden host process outside the sandbox at21:09:34, receipt `resume_launch_20261006_2108.json` (PID27708). The unchanged runner skips completed seeds48/123 and resumes seed456 at epoch11 from the committed epoch10 boundary; interrupted epoch11 steps are replayed. No dataset, curriculum, selection, model or training settings changed. The final epoch20 evaluation/ensemble/policy stages remain queued. Verification receipt: `resume_verification_20261006_2108.json`.

## Model 3 completed adaptive run — Darkbay_model3_adaptive_20261006

All three seeds completed20epochs with60 full95000-image rescoring passes. Ensemble selection: Final epoch20 designated by user after seed48 real epoch results were examined. Ensemble: 569/900; Model2:565/900; Model1:496/900. Full checkpoint/prediction/policy/decision hash verification, validation/loss curves, compute times and policy tables are saved in `runs/Darkbay_model3_adaptive_20261006/RESULTS.md`. Only curriculum changed in training relative to Model2; the user later changed evaluation checkpoint designation. Real data remain historical and validation was reused for policy calibration.

### Model 3 final matched-epoch comparison and registry completion

Verified completion of all60 checkpoints and rescoring passes, epoch20 evaluation selection and synthetic-policy/real-decision freezes. Recorded the completed reader experiment in the registry, preserving the prior registry snapshot. Added a matched-epoch20 Model2 prediction ensemble from hash-verified saved predictions, freezing the combined predictions before scoring: 572/900, compared with Model3 569/900 and Model1 496/900. Model2 original synthetic-selected565/900 remains separate. Final comparison and cross-seed spread are in `epoch20_final_comparison.json`.

## 2026-10-06 CAPRI Decoder design audit

Updated the single `docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md` with current/proposed architecture diagrams, audited baseline training and head mappings, font-library/retrieval design, count uncertainty and evaluation plans. Verified the active hybrid hash, cache-linked sampler/text-renderer/font-table hashes, training count/slant labels and cached glyph supervision; reused saved simple/specialist metrics. Preserved the appended font inventory. Designs remain proposals; supplied-font ingestion, calibration thresholds, loss weights and training budget are unresolved. No active code, model selection, machine paths, controllers or training runs changed; no inventory refresh was needed for this documentation-only update.
