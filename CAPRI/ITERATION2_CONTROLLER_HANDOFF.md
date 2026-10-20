# Iteration 2 controller handoff

Updated 4 October 2026. The specialist training plan, specialist XGBoost replacement and controller pilot tests are complete. The next session should explore ideas for a controller that takes feature-model starting settings and improves the renderer's resemblance to a target family. Start by discussing the design and evidence below; there is no unfinished training job to resume from this session.

The user cares most about **all font attributes together**, wants comparisons against the **simple hybrid by default**, and wants realistic visual assessment alongside metrics. They asked for existing neural training to remain unchanged when discussing font results. They want prompt reporting of results and dislike unnecessary confirmation questions or long report preparation before seeing an outcome. The latest task is to preserve this handoff for a new session, then explore controller ideas. A large GAN, a production-default change and new solver training are not the agreed next action.

## Locations and runtime

| Purpose | Absolute location |
|---|---|
| Main project | `E:/pranav/Final sem/sourcefree_synth` |
| Renderer | `E:/pranav/Final sem/captcha-effects` |
| Chat workspace and new run artifacts | `C:/Users/pranav/Documents/ChatGPT/thesis` |
| Neural training Python | `E:/pranav/model 1/.venv/Scripts/python.exe` |
| XGBoost and controller pilot Python | `E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe` |

The tested pilot environment is Python 3.9.13, Torch 2.5.1+cu121 and XGBoost 2.1.4, on an NVIDIA RTX A4000 with about 16 GiB. Use absolute paths or explicitly set the main-project working directory. The chat workspace is a different directory. The companion `ITERATION2_CONTROLLER_HANDOFF_ASSETS.json` records verified model paths, checkpoint hashes, phase checkpoints, source files and reports. Historical work elsewhere in the workspace is not a substitute for these completed iteration-2 runs.

## The four feature models

| Variant | Actual weights or bundle | What to load |
|---|---|---|
| Simple hybrid | `E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04/baseline_training/epoch_15/ema.pt` | Selected original 15-epoch EMA state dictionary; 3,662,050 neural parameters. Architecture/configuration is in that run's `training_config.yaml`. |
| Simple hybrid with XGBoost | `C:/Users/pranav/Documents/ChatGPT/thesis/iter2_xgb_01` | The same frozen simple checkpoint plus `heads.json` and all 107 JSON files under `boosters/`. This legacy experiment has no separate exported `encoder.pt`. Feature layout and checkpoint hash are in `feature_manifest.json`; protocol and cached features are alongside it. |
| Specialist hybrid | `C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/run_20261003_01/final/hybrid.pt` | Selected intensity-merge epoch 4; same 3,662,050-parameter neural architecture. This is the completed specialist model, not an individual teacher. |
| Specialist hybrid with XGBoost | `C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_xgb_specialists_work/run_20261004_02/bundle` | `encoder.pt`, `heads.json` and all 107 files in `boosters/`. The 2,179,108-parameter encoder physically excludes neural feature classifiers and CTC prediction modules. `optional_ctc_reader.pt` is a separate diagnostic sidecar; boosted feature inference does not load or execute it. |

Simple EMA SHA256: `ae73375ead99071d55c6394f7c4eb7c4d27c0a3285f8f15d8d84c9df43bc6b32`.

Specialist hybrid SHA256: `198f89c05877a2e5053215ed53cc3e9be33d4a2a9dabc94f6e7842af415f6ccf`.

Exported specialist encoder SHA256: `0d7d6aa2cf30494234aef8bed2926a08d1378927013aac747606b49d642b61d4`.

The hybrid shares a CNN stem. Its CNN effect branch predicts 37 presence and 37 intensity outputs. Its transformer branch predicts 16 text presence outputs, nine supported text intensity outputs, seven font attributes and length; the neural hybrid also has a separate CTC reader. The seven font attributes are category, stroke class, weight, width, x height, slant and italic.

The two XGBoost experiments use different feature routes. Legacy simple XGBoost uses 960-dimensional pooled CNN-stem features for effects, and 160-dimensional transformer task queries for text/font/length. Specialist XGBoost uses the trained CNN effect queries, 128 dimensions each, plus the specialist transformer queries. Do not pass the new encoder's features to the old simple trees. New tree fitting allows 2,000 rounds with 50-round early stopping and selection between balanced/unweighted ordinal regressions; old fitting allowed 800 rounds with 30-round stopping and balanced ordinals. These are four actual saved runs, not a controlled backbone-only ablation. The new bundle is about 291 MiB including trees and optional reader; removing neural heads did not make the entire package smaller.

From the main project, the published specialist XGBoost predictor is:

```powershell
& 'E:\pranav\Final sem\sourcefree_synth\experiments_xgb\.venv\Scripts\python.exe' -m feature_bank.iter2.xgb_replacements.predict --bundle 'C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_xgb_specialists_work\run_20261004_02\bundle' --image 'path-to-image.png' --out 'prediction.json' --device cuda
```

## Work completed before the controller tests

The plan in `E:/pranav/Final sem/sourcefree_synth/docs/ITERATION2_SPECIALISTS_HANDOFF.md` was executed. Five presence specialists each completed four epochs, followed by eight presence-merge epochs. Sixteen weak ordinal tasks triggered three intensity specialists, each with four epochs, followed by four intensity-merge epochs. Selection chose intensity-merge epoch 4. Original simple weights and the fixed neural training settings were preserved.

All phase models and resume states remain under `C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_specialists_work/run_20261003_01`: `presence_g1` through `presence_g5`, `presence_merge`, `intensity_g1` through `intensity_g3` and `intensity_merge`. Each phase has `epoch_*/raw.pt`, `epoch_*/ema.pt`, `selected.json` and `resume.pt`. Teacher caches are in `presence_teachers/` and `intensity_teachers/`. Full checkpoint paths are in the companion asset inventory.

Grouping was based on task affinity measured on the **shared CNN stem**. Buckets can contain CNN-effect and transformer tasks together; they were not separately clustered into pure CNN and pure transformer buckets. Architectural prediction routes remain distinct. Exact assignments and task names are in `presence_groups/report.json`, `intensity_groups/report.json` and `task_manifest.json`. CTC was an excluded grouping task and a preservation anchor. Specialist versus simple gains also include additional training; there is no equal-compute continued-simple control.

Specialist inference covered CryptBB 600, City 2,000 and DarkBay 999, kept separate. The completed run has 3,599 predictions and 100 verified previews per family. The 300-preview dataset is evidence for inspection, not a demonstrated solver-training corpus. Main specialist report: `iteration2_specialists_work/run_20261003_01/final/REPORT.md` under the chat workspace.

The specialist XGBoost replacement completed all 107 outputs, physically removed CNN/transformer neural feature classifiers in the new export, and verified exact FP32 query-feature equivalence with the original model. Saved rounds have median 323 and maximum 1,578; none hit 2,000. All 3,599 real predictions and 300 previews were regenerated after fixing italic decoding to probability >= .5. Use `run_20261004_02`; `run_20261004_01` is an incomplete balanced-only pass, and `inference_initial_italic_decode/` contains archived incorrect inference, not the current outputs. Original checkpoints stayed intact. No solver was trained.

The authoritative four-model report is `C:/Users/pranav/Documents/ChatGPT/thesis/iteration2_xgb_specialists_work/run_20261004_02/FOUR_MODEL_COMPARISON.md`, with `four_model_comparison.json` and `four_model_per_head.csv`. Copies of the report and replacement documentation are in the main project's `docs/ITERATION2_FOUR_MODEL_COMPARISON.md` and `docs/ITERATION2_XGB_REPLACEMENTS.md`.

| Synthetic benchmark metric | Simple | Simple XGB | Specialist | Specialist XGB |
|---|---:|---:|---:|---:|
| Seen-test font category accuracy | 68.70% | 70.10% | 73.50% | 73.80% |
| Seen-test mean font ordinal within one bin | 78.69% | 62.01% | 79.81% | 77.18% |
| Held-out-font category accuracy | 56.56% | 56.40% | 57.24% | 58.32% |
| Held-out-font mean font ordinal within one bin | 75.67% | 63.58% | 76.91% | 76.65% |

Specialist neural has the strongest aggregate font ordinal result; specialist XGBoost leads category but regresses some other attributes. Simple XGBoost has severe slant regression. The ordinal mean averages weight, width, x height and slant, excluding category/stroke/italic. No joint all-seven-font-attributes-exact metric was saved. Benchmarks are reused synthetic sets, not labelled real-family font accuracy. No feature head was removed merely because the user questioned its usefulness, and no model-size experiment established that a larger model is needed.

## Renderer demonstrations and starting profiles

Manual target-reference renders for all three families are in `C:/Users/pranav/Documents/ChatGPT/thesis/manual_family_renderer_20261004`, with `*_recipe.json`, images and `render_examples.py`. A separate fresh simple-hybrid-assisted trial is in `baseline_guided_family_renderer_20261004`, with full-family predictions, target/before/after images, adjustment JSON and `RESULT.md`.

That qualitative trial found a useful starting point for City, partial help for DarkBay and little overall reduction in work for CryptBB. It used one preselected reference per family, existing familiarity and manual controls; it was not a timed blind study. Some manual drawings supplied explicit geometry/palette controls beyond the standard recipe defaults. This demonstrates renderer expressivity under manual control, not automatic controller quality.

Main-project controller code is `feature_bank/iter2/controller.py`; its measured-label recipe correction is `feature_bank/iter2/mapping.py`. Family profiles preserve prediction co-occurrence and draw fresh text from the own font bank. They exclude real answers, CTC strings, target font files and old per-image recipes. Predictions can still be wrong, and coarse heads omit important colour, font identity and geometry details.

Useful cached predictions under the chat workspace: simple in `iter2_full_target_recreation_01/predictions.jsonl`; specialist in `iteration2_specialists_work/run_20261003_01/real_inference/predictions.jsonl`; simple XGB in `iter2_xgb_01/real_inference/predictions.jsonl`; specialist XGB in `iteration2_xgb_specialists_work/run_20261004_02/real_inference/predictions.jsonl`. The automated controller pilot used **specialist neural** initial profiles, compared with a simple neural baseline. It did not use the XGBoost models for initialization.

## Automated controller experiments

All scripts are in `C:/Users/pranav/Documents/ChatGPT/thesis/metasim_controller_pilot_work/launch`: `pilot.py`, `pilot_render.py`, `diagnostics.py`, `write_report.py`, `feedback_ablation.py` and `feedback_ablation_report.py`.

These are global black-box population searches over renderer settings, inspired by Meta-Sim. They do not train a GAN, a discriminator or an image-conditioned neural policy. A saved controller is a **JSON vector of settings**, not a new neural checkpoint. The hybrid supplies starting descriptions; a separate appearance objective judges generated distributions. The differentiability of a learned critic alone does not provide gradients through this renderer.

The 15 controls cover background and foreground gray blending/luminance, gray effect-palette probability, size, horizontal scale, spacing, jitter, variable-weight fraction, synthetic bold stroke, one of eight own-bank font choices, curve retention, band level and band slope. Colour and slope hooks are installed only in pilot worker processes. Original renderer files were not edited, and labels were measured after rendering. Neutral replay was pixel-exact. Strings and glyph counts are preserved. Changing operations can advance the renderer's shared random stream differently, so matching seeds does not hold every incidental colour/style draw fixed.

Appearance feedback uses cached **DINO ViT-S/8**, frozen and never retrained: code at `C:/Users/pranav/.cache/torch/hub/facebookresearch_dino_main`, weights at `C:/Users/pranav/.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth`. It has about 21.7 million parameters, with 384-dimensional L2-normalized image embeddings. Inputs resize to 320 by 64 with ImageNet normalization and CUDA bf16. The statistics channel has 48 colour, intensity, edge and row/column projection features. Each channel uses multi-RBF kernel **MMD squared** over image distributions, not paired pixel similarity. Statistics scaling and kernel bandwidths are determined only from fitting images. The independent grayscale/HOG sliced-Wasserstein audit is excluded from fitting and selection.

The original objective was 0.6 DINO MMD² + 0.4 statistics MMD², plus 0.8 times the unusable fraction and 0.15 times strict-readability failures. Pure-channel tests set appearance weights to 1/0 or 0/1, keeping validity rules. Both channels are still computed for diagnostic reporting, but only the named channel affects a pure search's fitness and selection. Therefore statistics-only runtime includes DINO diagnostics and is not a CPU-only benchmark.

### Original 60 and 40 experiment

Run: `C:/Users/pranav/Documents/ChatGPT/thesis/metasim_controller_pilot_work/run_20261004_01`. Read `REPORT.md`, `results.json`, `protocol.json`, `selected_controls.json` and `comparison.jpg`.

City had 2,000 files and 1 exact pixel duplicate. Of 1,999 unique images, 256 formed fitting, 128 selection and 512 test subsets. Both starting profiles used only their own model predictions on the same 256 fitting images. Search used eight rounds of eight candidates, with 24 common recipes per candidate: 1,536 search renders, plus recipe-mapper, selection and evaluation calls. Top three fitting policies and neutral were compared on 48 fresh recipes against selection. Candidates had to preserve usability and strict readability within five percentage points of neutral. Policies were saved before test scoring. Final comparisons used 96 synthetic images per variant and shared specialist strings/seeds.

The selected 60/40 controller reduced appearance distance from simple 0.42837 and specialist 0.42495 to 0.39400: 8.02% and 7.28% respectively. Strict readability was simple 85.4%, specialist 76.0%, adjusted 77.1%; all 96 texts in each main group were complete and passed basic usability. Execution took 234.7 seconds, excluding development. Original font keys were unchanged; adjustments mostly changed width, size and effects. Visual family resemblance remained weak. Text-only post hoc controls improved the proxy 2.85% but reduced strict readability to 69.8%; this did not establish font accuracy. Diagnostics are in `diagnostics.json` and `diagnostic_comparison.jpg`.

### Requested 100 percent experiments

Run: `C:/Users/pranav/Documents/ChatGPT/thesis/metasim_controller_pilot_work/feedback_ablation_20261004_02`. Read `REPORT.md`, `results.json`, `reductions_vs_simple.json`, `font_details.json`, `pixel_statistics.json`, `verification.json` and `comparison.jpg`.

DINO-only and statistics-only each ran a separate 1,536-render search with identical starting recipes, first-round candidates, budget and guards. They shared the original fitting/selection sets. Both policies were fixed before evaluating **512 additional City images**, disjoint by pixel hash and source path from every subset of the earlier pilot. Each variant has 96 synthetic images. Total execution was 143.5 seconds, excluding setup/report writing. The original 60/40 policy and baseline images were reused and scored against the same new target subset; they were not retrained.

| Settings on the fresh evaluation set | DINO MMD² lower is better | Statistics MMD² lower is better | Common 60/40 score | Independent audit | Strict readability |
|---|---:|---:|---:|---:|---:|
| Simple start | 0.38317 | 0.52074 | 0.43820 | 2.8579 | 85.4% |
| Specialist start | 0.37100 | 0.53075 | 0.43490 | 2.8601 | 76.0% |
| Original 60/40 policy | 0.35347 | 0.48121 | 0.40457 | 2.2911 | 77.1% |
| 100% DINO policy | 0.27792 | 0.66629 | 0.43327 | 2.4157 | 88.5% |
| 100% statistics policy | 0.35802 | 0.49251 | 0.41182 | 2.3980 | 77.1% |

Against simple, pure DINO improves DINO discrepancy 27.47% but worsens statistics discrepancy 27.95%. Pure statistics improves its statistics discrepancy 5.42%. The original mixed policy has the lowest common score and independent audit in this small experiment. Compare each metric column across policies; the raw pure objectives have different scales. New baseline numbers differ from the first report because the real evaluation images changed.

Both pure policies choose Reddit Mono throughout, changing glyph font keys in 90/96 samples; the other six already used it. DINO adds stronger vertical jitter, shrinks text and flattens bands. In the displayed examples, digit shapes and placement appear closer to City, but backgrounds become too gray. Nearly white pixels at the feedback resize occupy 0.05% of DINO renders versus 58.96% of real images. This explains a concrete aspect of the mismatch without proving the contribution of each statistic. Real font accuracy and OCR accuracy remain unmeasured.

All 480 saved PNG hashes, paired strings/seeds, label integrity, exact channel weights and source/checkpoint preservation passed verification. Bootstrap intervals are in `results.json`; pure DINO's common-score change versus specialist spans zero. There is one search seed per choice, City only, no downstream solver result and no evidence that the present metric weighting is generally optimal. **591 unique City images remain unused by these controller fitting/selection/evaluation objectives**; respect recorded split manifests if reserving another evaluation set.

Controller settings live in:

- Mixed: `metasim_controller_pilot_work/run_20261004_01/selected_controls.json` under the chat workspace.
- DINO-only: `metasim_controller_pilot_work/feedback_ablation_20261004_02/dino_only/selected_controls.json`.
- Statistics-only: `metasim_controller_pilot_work/feedback_ablation_20261004_02/stats_only/selected_controls.json`.
- Combined record: `metasim_controller_pilot_work/feedback_ablation_20261004_02/selected_all_controls.json`.

## Next session controller discussion

The user suggested a GAN, then Meta-Sim, to update initial renderer settings. The working conclusion is that a controller and visual feedback are useful, while a large new training run is not yet justified. The hybrid remains useful for initialization and possible soft constraints; these tests do not make it obsolete. The following are **ideas to explore, not completed methods or automatic launch instructions**:

1. Decide whether to learn family-level style distributions, per-image adjustments, or a sequential policy that observes a render and updates a recipe. The current pilot is only a global transformation of starting recipes.
2. Improve the objective before increasing model size. Investigate font/typography features, background-versus-text scoring, colour constraints and domain-specific feedback. Inspect the DINO examples; retaining white backgrounds while using its useful font/placement changes is an open hypothesis, not a validated fix. Consider how to measure all seven font attributes without using the same uncertain predictions as ground truth.
3. Expose missing renderer controls: palette, band position/width/pixel geometry, font identity, axis settings and glyph placement. Use explicit categorical choices for fonts and effect types; the pilot treated font selection as a rounded scalar and searched only eight candidates.
4. Compare inexpensive search and learned controllers: mixed categorical/continuous search, a small surrogate, imitation of successful adjustments, or a Meta-Sim-style parameter-distribution model. A discriminator/GAN-style critic is another candidate; it need not replace the hybrid or turn generation into unconstrained pixel editing. Plan how a black-box renderer would be optimized, and preserve known strings and labels.
5. Use fair renderer-call budgets, multiple seeds, sufficient recipe samples, strict content/readability checks and a reserved evaluation set. Test CryptBB and DarkBay separately after resolving the City design. Real CAPTCHA answers must not enter controller fitting or feature-model training. Font gains, synthetic readability, distribution-distance gains and real recognition accuracy are different results.

Primary research discussed in the session: [Meta-Sim](https://arxiv.org/abs/1904.11621), [Meta-Sim2](https://arxiv.org/abs/2008.09092), [SPIRAL](https://proceedings.mlr.press/v80/ganin18a.html) and [SimGAN](https://arxiv.org/abs/1612.07828). Verify relevant methods from primary sources when choosing an implementation; none was fully reproduced here.

Suggested first prompt for the new session: **Read docs/ITERATION2_CONTROLLER_HANDOFF.md, inspect the latest City comparison images, and help me explore controller ideas that improve typography and target-family style while keeping the trained hybrid as a starting point. Compare proposed tests to the simple baseline and distinguish appearance proxies from font accuracy.**
