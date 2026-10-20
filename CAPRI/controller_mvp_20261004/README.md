# Pixels-only controller MVP

The frozen simple hybrid supplies the initial renderer recipe distribution. A comparison network internally predicts soft text masks from pixels. A bounded search then adjusts renderer controls to match text, background and whole-image appearance across a target image set. The target needs no labels, strings, boxes, masks, texture names or font files.

## Run

From `C:/Users/pranav/Documents/ChatGPT/thesis`, in PowerShell:

```powershell
& 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' controller_mvp_20261004/run.py --targets 'PATH_TO_UNLABELED_IMAGES' --out controller_mvp_20261004/my_run
```

The existing environment name contains `xgb`; this MVP does not load or use XGBoost. GPU is preferred; CPU inference is supported. Inputs need at least 60 unique images. The output directory must be new. Defaults: 128/64/up-to-256 target fit/selection/test images, 18/36/96 synthetic renders per candidate/evaluation, and 12 renderer processes. Smaller input sets receive proportionally smaller target splits. Large datasets leave images beyond those limits unused.

For an explicit split, pass `--splits manifest.json` instead of `--targets`. Each of `fit`, `selection`, and `test` is a list of `{source, pixel_sha256}`; the hash is SHA256 of decoded RGB pixel bytes. Images with duplicate hashes across splits are rejected. Filenames are used only as paths and never parsed for answers.

Set `CAPRI_PROJECT` and `CAPRI_RENDERER` if the existing project roots move. The encoder dependency is `../learned_controller_20261004/training/encoder.pt`; its architecture and training code remain in that directory. This MVP is a local integration, not a standalone redistributable package.

## What changes

- Text: train-bank font choice, size, width, spacing, stroke weight, rotation and jitter multipliers; explicit RGB palettes and single/random/alternating glyph color assignment.
- Background: explicit base RGB, effect palette, effect presence and intensity. All supported under-text effect types can be proposed, even when the hybrid initially predicted them absent.
- Interaction: the original renderer still composes under-text, glyph, over-text and post effects. The whole-image comparison and renderer validity penalties prevent optimizing the two regions entirely independently.

Length, content-group distribution, layout type and canvas dimensions remain hybrid estimates. Rotation/jitter multipliers cannot introduce variation when the corresponding initial value is zero. The current palette initializer extracts modal colors; it does not learn arbitrary spatial color fields. Search is limited to the implemented renderer's capabilities.

## Comparison and supervision

The internal mask encoder was already trained on synthetic renderer images with exact masks, paired scheme contrastive learning, and auxiliary synthetic recipe objectives. Only its image embedding and soft mask are used here. Neither its recipe-classification outputs nor the earlier learned correction network is used. No new hybrid training occurs.

Real and synthetic images take the SAME predicted-mask path. Text comparison uses soft-mask-weighted color distributions and geometry. Background comparison uses complementary soft weights, color distributions, gradients and texture differences. The whole-image embedding provides a fifth equally weighted channel. A grayscale/orientation audit is calculated separately and never selects candidates. It is an independent descriptor, not an independent ground-truth judge.

Exact synthetic masks and strings are saved as generator labels and used in checks; they are not supplied as real-target information. The controller can inspect synthetic validity because it controls the renderer. Real mask accuracy is unknown without annotations. The development mask gallery is a qualitative inspection, not proof of real localization accuracy.

## Search and evaluation

Candidate proposals depend on fit images only. Each search phase compares candidates using shared synthetic recipes/seeds. The best three fit candidates are checked on separate selection images and fresh rendered strings. An update requires no more than 5% degradation in either text or background selection distance, no reduction in completeness, and at most 3 percentage points reduction in renderer usability. Selection optimizes the combined appearance score plus incompleteness/unusability penalties. The final frozen policy is evaluated once on test pixels and 96 fresh paired baseline/refined renders, also reported in three seed blocks.

`selected_controls.json` is saved and hashed BEFORE test images are loaded. `history.json` retains every evaluated candidate, selection check and exception. `results.json` reports text and background components separately. `report.html` shows the first 16 test examples without hand selection. `baseline/labels.json` and `refined/labels.json` contain the actual generated strings, glyph boxes, measured effects, controls and image hashes.

The demo uses City development images plus 256 pixels excluded from all previous controller/ablation target splits. `prepare_demo.py` records that provenance. Family names choose files only; there are no family-specific renderer presets. This establishes a one-family MVP, not generalization to all families.

## Checks and development runs

```powershell
& 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' controller_mvp_20261004/check_contracts.py
```

Checks cover exact baseline replay, effect-palette isolation from canvas and implicit text colors, explicit multicolor glyph rendering, deterministic same-pixel comparison and rejection of leaking splits. `inspection/masks.jpg` shows development target and synthetic mask predictions.

`run_01` and `run_02` were interrupted before final test evaluation while palette-hook isolation was repaired. They are incomplete development artifacts, not reported results. `run_03` is the first completed evaluation candidate. It reuses only the identical initial fit/selection recipes from run 02 after verifying hybrid descriptions, seed and render counts. It reruns the search with the repaired hooks.

Success requires visual improvement of both text and background as well as held-out evidence. Lower optimization scores alone do not establish reconstruction quality, exact font matching or OCR accuracy.
