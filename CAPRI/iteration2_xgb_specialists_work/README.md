# Specialist encoder with XGBoost replacements

Requested 2026-10-04: fit XGBoost heads and remove the CNN and transformer neural feature classifiers.

Completed: all **107** boosted outputs, **3,599** real-image predictions and **300** corrected renderer previews. Results and per-head regressions are in [run report](C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_xgb_specialists_work/run_20261004_02/REPORT.md). Original specialist and simple checkpoints are preserved.

User requested all four comparisons: [simple, simple + XGBoost, specialist, specialist + XGBoost](C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_xgb_specialists_work/run_20261004_02/FOUR_MODEL_COMPARISON.md). Same benchmark labels verified across every split; calibrated effect/text presence compared across all four; the old simple italic booster replayed at .5 to match the other font reports. No new fit or threshold selection. Both XGBoost experiments use different effect-feature routes and round/ordinal-weighting protocols, so this compares the actual saved runs rather than isolating only specialist training.

Font results are mixed. Seen-test category accuracy improves over simple **68.70% → 73.80%** (specialist neural 73.50%), while mean font ordinal within-one declines **78.69% → 77.18%** (specialist neural 79.81%). Held-out-font category improves **56.56% → 58.32%** (specialist neural 57.24%); font ordinal mean is **75.67% → 76.65%**, below specialist neural 76.91%. The report compares all seven font outputs, including the x-height/slant regressions. This replacement is not an overall font improvement over the specialist neural heads.

Source: completed specialist hybrid, intensity merge epoch 4, SHA256 `198f89c05877a2e5053215ed53cc3e9be33d4a2a9dabc94f6e7842af415f6ccf`. The source weights stay frozen. The new inference model removes the effect/text prediction MLPs, categorical length, category/stroke/italic and font ordinal prediction MLPs. CTC prediction layers are absent from the exported feature encoder; their original weights are an optional separate diagnostic sidecar. CTC strings are never inputs to a booster or a family profile.

The exported encoder has **2,179,108** neural parameters, versus 3,662,050 in the source model. The actual saved encoder state contains no removed prediction modules. Its CNN and transformer query vectors exactly matched the original model in an FP32 comparison (maximum absolute difference 0 for both). Tree files add their own storage; fewer neural parameters does not establish a smaller total bundle.

There are **107 replacements**: 37 effect presence and 37 effect intensity from each learned CNN effect query; 16 text presence and nine measured text intensity from each transformer query; seven font attributes and length from their transformer queries. The trained CNN effect block, position/normalization and learned effect pooling are retained. This avoids the earlier XGBoost experiment's bypass of the effect branch. Seven binary text features have no measured ordinal labels and receive no invented intensity booster.

Fit uses the same static 100,000 synthetic training images. Selection supplies early stopping; calibration supplies effect/text presence thresholds (max F1, FPR <= .10). Font italic uses .5, matching saved neural comparators. Unknown labels are excluded, effect/text intensity trains on measured positive levels, and font zero remains valid. Each ordinal regression compares exactly two training-weight options: inverse measured-level counts and unweighted. Select higher within-one on selection, then lower selection MAE; exact ties retain the first candidate. CNN/transformer weights are frozen; real images and their answers never enter fitting, calibration or selection. No neural fallback is chosen when a booster is weaker: requested outputs use XGBoost, and regressions are reported.

The initial balanced-only pass (`run_20261004_01`) stopped after 81 completed heads, before any benchmark scoring, when selection showed severe slant regression (.579 versus neural .917) and weaker x-height. The second pass (`run_20261004_02`) preserves those results and reuses unchanged classification models, balanced candidates and aligned features with hash checks. This selection-only adjustment was recorded before evaluating any replacement on seen_test/generalisation. It does not change neural training or encoder weights.

Fixed tree settings: CUDA histogram training, maximum 2,000 rounds, learning rate .05, depth 6, subsample .8, column subsample .8, lambda 1, min-child-weight 1, seed 42, eight CPU threads, 50-round early stopping. Stored models are sliced to their selected boosting rounds. Binary classification is used for a categorical target with two observed classes, then decoded through saved original class IDs; this avoids the earlier two-class multiclass shape bug.

The final models select a median of **323** rounds and a maximum of **1,578**; none selects the 2,000-round limit. Extra identical fitting passes would rebuild the same models. These fixed settings have not had an exhaustive parameter search.

```powershell
& 'E:\pranav\Final sem\sourcefree_synth\experiments_xgb\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\launch\fit_xgb.py' --out 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\run_20261004_02' --reuse-fits-from 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\run_20261004_01'
```

The command resumes completed extraction stages and completed head models. `protocol.json` must match on resume. Feature rows are keyed by split/shard/index/image hash; width-sorted extraction writes back to original row positions. Boosters and their metadata are saved per head. Test sets are reused benchmarks already seen in prior experiments; their results are read only after all fitting is complete, and are never used to choose a head/round/configuration.

```powershell
& 'E:\pranav\Final sem\sourcefree_synth\experiments_xgb\.venv\Scripts\python.exe' 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\launch\predict.py' --bundle 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\run_20261004_02\bundle' --image 'path-to-image.png' --out 'prediction.json'
```

The inference command requires the project Python sources and the recorded Torch/XGBoost environment. CUDA bf16 extraction with float16 cached vectors matches the training path. A CPU inference option is available and uses FP32 neural arithmetic; its tree decisions may differ near splits. There are no neural feature classifiers in this command's loaded model or forward pass.

Run outputs: `bundle/encoder.pt`, optional reader sidecar, `bundle/heads.json` and `bundle/boosters`; label masks/counts and fit provenance in `head_specs`; both ordinal candidate files in `head_candidates`; aligned features/manifests; per-head benchmark reports; real-family predictions, profiles and 100 previews each. Runtime: `run_20261004_02/status.json`; console: `second_pass.log`. No CAPTCHA-solver training is started.

Passing checks: classifier modules absent; exact learned feature equivalence; font zero retained/unknown masked; effect ordinal absence masked; all 107 tasks covered; two-class nonzero category-ID decoding and exact saved booster replay (`replacement_checks.json`).

Final bundle verification checks both source checkpoint hashes, every feature-cache and booster hash, all task-to-query routes, exact cached benchmark-score replay, command-line output structure and all 300 preview hashes/lengths/prototypes (`run_20261004_02/verification.json`). Bundle tree files and the optional reader occupy about **291 MiB**; the reduced neural parameter count does not make this a smaller storage package.

Inference QA corrected italic decoding to probability >= .5 before export to JSON. The original inference/profiles/previews are archived under `run_20261004_02/inference_initial_italic_decode`. Corrected inference and previews were rebuilt without changing any fitted model or benchmark score. An italic-positive real image passes both the regression check and published project CLI.

The published command is `python -m feature_bank.iter2.xgb_replacements.predict` from the project directory with the same `--bundle`, `--image`, `--out`, and optional `--device` arguments. Exact runtime files and checksums are included under `bundle/runtime` and `bundle/checksums.json`.
