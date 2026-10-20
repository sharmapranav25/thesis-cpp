# Controller MVP, iteration 2

The target interface remains pixels only. The frozen simple hybrid initializes recipes; the existing synthetic-supervised encoder predicts text probability internally. Neither target character strings nor target masks, boxes, textures, fonts or renderer labels are used.

## Changes

The new comparator measures inferred glyph-region aspect ratio, occupancy, stroke thickness, normalized projections and gradient directions, plus estimated component counts and layout. These connected components are imperfect inferred regions, not recognized characters or annotated boxes. Empty masks produce finite diagnostic features rather than a crash.

Background structure now measures edge orientations at three scales, directional differences over six offsets, and local variation at three scales. Dilated predicted text and uncertain boundaries are excluded from these background measurements. Color and global appearance remain separate channels. Descriptor dimensions are scaled by target fit variation with a floor to prevent nearly constant coordinates dominating.

New controls permit synthetic length 2–8, absolute rotation and vertical jitter (including when initial values are zero), additional stroke boldness, random mixtures of training-bank fonts, richer per-character palettes, and a distribution of image background colors. Background search can remove several unwanted effects together and select existing renderer block layouts. No family-specific rendering logic was introduced.

## Experiment

1. Starting from the saved MVP result, render exactly 96 candidates. Both old and new comparators rank those **same pixels**, with four finalists checked on selection images. This isolates comparator choice within this fixed candidate pool.
2. Starting from the new-comparator winner, search 96 additional candidates over layout, background and variation. This adds both controls and computation; do not attribute its gain solely to the controls.
3. Freeze all policies before test inference, then evaluate four arms: original MVP, old-comparator continuation, new-comparator continuation, and expanded controls. Each gets 72 renders using paired seeds. Length changes generate fresh synthetic strings, so strings need not remain identical across arms.

All changes are evaluated on development targets used previously, with fresh test pixels excluded from prior feature-ablation and MVP splits. The final fresh test sizes are City 128, CryptBB 24, DarkBay 128. CryptBB has exhausted its remaining unused pool; its small test is exploratory. Historical development use prevents claiming a pristine study of every design choice.

No perception retraining is performed in this iteration. Mask uncertainty and connected-component counts are saved for inspection. Improved scores cannot establish real mask accuracy or exact font matching. The independent whole-image grayscale/orientation audit is not used for selection.

## Run

From `C:/Users/pranav/Documents/ChatGPT/thesis`:

```powershell
& 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' controller_mvp_v2/experiment.py --previous controller_mvp_20261004/run_03 --splits controller_mvp_v2/City_split.json --out controller_mvp_v2/new_city_run --workers 4
```

Use the corresponding prior MVP run and split for other targets. Outputs must be new directories. The prior run supplies hybrid descriptions and prepared fit/selection recipes; the split must represent those same development images. Project/runtime dependencies are shared with MVP 1. No XGBoost is used despite the environment directory name.

`checks.py` verifies exact MVP replay on 12 images across all three families, 12 length edits, empty-mask behavior and identical-image distance. `results.json` includes all four arms under both judges. `history.json` records candidate failures and selection eligibility. `frozen_policies.json` is hashed before test. Generated folders retain actual text labels, masks, recipe parameters and pixel hashes. `preview.png` displays the unselected first eight test rows as target / MVP 1 / new comparator / expanded controls, preserving aspect ratios.

The controller uses a finite search, not a newly trained adjustment network or MetaSim-2 implementation. Whole-family scalar controls are still used for most geometry; uncertainty estimates are diagnostics, not calibrated confidence.
